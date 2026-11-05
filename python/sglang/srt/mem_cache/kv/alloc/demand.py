"""What an allocation takes from each sub-pool, and what a request needs."""

from __future__ import annotations

import enum
import math
from typing import Callable, Iterable, Mapping, Optional

import msgspec

from sglang.srt.utils.common import ceil_align


class KVDemand(msgspec.Struct, frozen=True):
    """What an allocation takes: tokens of the token sub-pool, tokens of each
    window sub-pool by its name, and state slots.

    Admission and the take need every window named (`checked`): a window left
    out is an error, not a window that takes nothing, so no window's share is
    guessed. An eviction request may name only the sub-pools it is short of
    (`completed` fills the rest with zero)."""

    full_tokens: int = 0
    window_tokens: Mapping[str, int] = msgspec.field(default_factory=dict)
    state_slots: int = 0

    @classmethod
    def same_tokens(cls, allocator, tokens: int, *, state_slots: int = 0) -> KVDemand:
        """Every token sub-pool takes ``tokens``: the new tokens of an extend
        or decode step, which every window keeps."""
        return cls(
            tokens,
            {name: tokens for name in allocator.window_pool_names()},
            state_slots,
        )

    def checked(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand, once it names exactly the windows listed."""
        names = set(window_pool_names)
        if set(self.window_tokens) != names:
            raise ValueError(
                f"a demand must name every window pool: it names "
                f"{sorted(self.window_tokens)}, the allocator keeps {sorted(names)}"
            )
        if self.full_tokens < 0 or self.state_slots < 0:
            raise ValueError(f"negative demand {self}")
        if any(t < 0 for t in self.window_tokens.values()):
            raise ValueError(f"negative window demand {self}")
        return self

    def completed(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand with every listed window named, those it leaves out
        taking nothing; a window it names that is not listed is an error."""
        names = tuple(window_pool_names)
        unknown = set(self.window_tokens) - set(names)
        if unknown:
            raise ValueError(
                f"no window pools {sorted(unknown)}; the allocator keeps "
                f"{sorted(names)}"
            )
        windows = {name: int(self.window_tokens.get(name, 0)) for name in names}
        return msgspec.structs.replace(self, window_tokens=windows).checked(names)

    def is_empty(self) -> bool:
        return not (
            self.full_tokens > 0
            or self.state_slots > 0
            or any(t > 0 for t in self.window_tokens.values())
        )


def retained_window_tokens(
    sliding_window_size: int, page_size: int, *, supports_prefix_sharing: bool = True
) -> int:
    """The rows behind a request's frontier its window retirement keeps
    (`retire_window_rows`): the window, and for a prefix-sharing cache the
    larger of it and a page, rounded up to whole pages, so a key ending at a
    page boundary still has a full window of live rows."""
    if not supports_prefix_sharing:
        return sliding_window_size
    return ceil_align(max(sliding_window_size, page_size), page_size)


def request_demand(
    input_tokens: int,
    max_new_tokens: int,
    *,
    windows: Mapping[str, Optional[int]],
    page_size: int,
    prefill_retire_lag: int,
    eviction_interval: int,
    checkpoint_interval: int = 0,
    decode_retire_lag: int = 0,
    chunk_size: Optional[int] = None,
    state_slots: int = 0,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
    cached_tokens: int = 0,
) -> KVDemand:
    """What one request still needs to place ``input_tokens`` prompt tokens,
    in chunks of at most ``chunk_size``, and then decode ``max_new_tokens``:
    the page-ceiled prompt, its decode and one page of allocation rounding in
    the token sub-pool (plus ``extra_tokens`` its caller charges on top), a
    bound on what each window sub-pool of ``windows`` (name to retained
    window) holds, and ``state_slots``.

    The window bound covers every phase at once rather than each exactly: the
    request's own rows, or else the window it retains (whole pages, as a
    prefix-sharing cache keeps it) plus the larger of what decode keeps past
    the window (up to ``eviction_interval`` rows less one, and the window
    behind the latest state checkpoint every ``checkpoint_interval`` rows)
    and a prefill chunk with the rows the scheduler retires behind it
    (``prefill_retire_lag``) and, under state checkpoints, the tail it keeps
    past the latest one (a whole chunk, when the chunk takes a branch
    checkpoint early in it); then the rows a decode step allocates ahead of its
    processed length (``decode_retire_lag``: the overlap scheduler's next step
    or a speculative reserve; none without a decode), less the window rows a
    cached prefix of ``cached_tokens`` holds (allocated already: every phase
    counts them once, among the window it retains), then one page of rounding
    and the pages a host hit restores. The last prefill's chunk is still live when the
    first decode allocates ahead (that step retires nothing), so the chunk and
    the rows ahead add up. The bound only grows with the chunk.

    One function prices a request wherever its demand is asked: the memory
    clip of a lone request in an empty pool, its admission, and the
    commitment a chunked request keeps until its last chunk. Priced alike,
    a request the clip lets through is one its chunks can finish."""
    hits = host_hit_lengths or {}

    if (
        prefill_retire_lag < 0
        or decode_retire_lag < 0
        or eviction_interval < 1
        or checkpoint_interval < 0
    ):
        raise ValueError(
            f"retirement lags {prefill_retire_lag} and {decode_retire_lag}, "
            f"interval {eviction_interval} and checkpoint interval "
            f"{checkpoint_interval}"
        )
    decodes = max_new_tokens > 0
    # Rows a decode keeps past the window, and rows a decode step allocates
    # ahead of its processed length: none without a decode.
    retention = max(eviction_interval, checkpoint_interval) if decodes else 0
    ahead = decode_retire_lag if decodes else 0
    rows = input_tokens + max_new_tokens
    last_chunk = input_tokens if chunk_size is None else min(input_tokens, chunk_size)

    def window_bound(name: str, window: Optional[int]) -> int:
        hit = hits.get(name, 0)
        hit_pages = -(-hit // page_size) * page_size if hit > 0 else 0
        held = rows + ahead
        if window is not None and window > 0:
            # TODO: this bound is deliberately conservative. Window rows are
            # retired by scheduler-side rules (maybe_evict_swa /
            # retire_window_rows) keyed on the processed length; once
            # retirement moves into the memory layer and keys on the
            # allocated frontier, derive this bound from that one policy.
            # FIXME: this assumes a state checkpoint lands every
            # checkpoint_interval rows, and that the prefix-sharing cache takes
            # each one, retiring the window behind it. Two skips break that,
            # and window retirement then keeps an older checkpoint's floor (or,
            # with none, retires nothing), more rows than this counts:
            # (A) lazy Mamba tracking skips a checkpoint when no second
            # tracking slot is free, even after eviction
            # (`mamba_lazy_prealloc_at_boundary`), and keeps the older
            # `mamba_last_track_seqlen`;
            # (B) taking a checkpoint into the tree needs a replacement state
            # slot; when none is free after eviction,
            # `MambaComponent.prepare_for_caching_req` returns a cache length
            # of 0 and `checkpoint_into_tree` retires no window row.
            # (A chunk shorter than the checkpoint grid takes no checkpoint
            # either; the scheduler cuts every chunk on that grid there,
            # `mamba_prefill_chunk_grid`.)
            retained = retained_window_tokens(window, page_size)
            in_prefix = min(retained, cached_tokens)
            # A chunk keeps its tail past the latest checkpoint: retirement
            # keeps the window behind the checkpoint. One that ends between
            # checkpoints keeps up to an interval less a row; one that takes a
            # branch checkpoint early in it instead (a full-KV match reaching
            # past the latest reusable state, `branch_checkpoint`) up to the
            # whole chunk.
            tail = (
                max(checkpoint_interval - 1, last_chunk) if checkpoint_interval else 0
            )
            beside = max(prefill_retire_lag + last_chunk + tail, retention - 1)
            # Every phase's peak counts the cached prefix's window rows once,
            # among the window it retains; they are held already.
            held = min(held, retained + beside + ahead - in_prefix)
        return held + page_size + hit_pages

    return KVDemand(
        -(-input_tokens // page_size) * page_size
        + max_new_tokens
        + max(0, ahead - 1)
        + page_size
        + extra_tokens,
        {name: window_bound(name, window) for name, window in windows.items()},
        state_slots,
    )


class DemandPurpose(enum.Enum):
    """Why a request's demand is asked. Each purpose builds the inputs of
    `request_demand` by its row of `DEMAND_RULES`."""

    # A lone request in an empty pool: whether it can ever run. The memory
    # clip of an arriving request and a decode node's capacity check.
    CEILING = "ceiling"
    # A prefill's admission, its final shape check and its commitment.
    ADMIT = "admit"
    # The rows a selected admission realizes before its host load-back.
    LOAD_BACK = "load_back"
    # A decode node's preallocation of a transferred request.
    DECODE_PREALLOC = "decode_prealloc"


class StateSlots(enum.Enum):
    # What acquire allocates for a request that holds no state yet.
    FRESH = "fresh"
    # What the request still lacks.
    PENDING = "pending"


class DemandRule(msgspec.Struct, frozen=True, kw_only=True):
    """How one purpose differs from the others. Everything else (the chunk,
    the windows, the new window, the token sub-pool's rounding page) is
    built alike for every purpose."""

    # The decode priced is the scheduler's estimate, at most
    # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION tokens. A ceiling prices all of
    # ``max_new_tokens``: the generation it lets through must fit.
    estimates_decode: bool
    state_slots: StateSlots
    # Whether cached entries the prefix cache could evict count as room. A
    # load-back reclaims first and asks what is free then; a ceiling asks an
    # empty pool.
    credits_evictable: bool


DEMAND_RULES: Mapping[DemandPurpose, DemandRule] = {
    DemandPurpose.CEILING: DemandRule(
        estimates_decode=False,
        state_slots=StateSlots.FRESH,
        credits_evictable=False,
    ),
    DemandPurpose.ADMIT: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=True,
    ),
    DemandPurpose.LOAD_BACK: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=False,
    ),
    DemandPurpose.DECODE_PREALLOC: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=True,
    ),
}


def decode_estimate_tokens(max_new_tokens: int) -> int:
    """The decode the scheduler expects of a request that may generate
    ``max_new_tokens``: at most SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION."""
    from sglang.srt.environ import envs

    return min(max_new_tokens, envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get())


class DemandInputs(msgspec.Struct, frozen=True, kw_only=True):
    """The inputs of `request_demand` for one purpose (`demand_inputs`)."""

    purpose: DemandPurpose
    input_tokens: int
    max_new_tokens: int
    windows: Mapping[str, Optional[int]]
    page_size: int
    prefill_retire_lag: int
    eviction_interval: int
    checkpoint_interval: int
    decode_retire_lag: int
    chunk_size: Optional[int]
    state_slots: int
    host_hit_lengths: Mapping[str, int]
    extra_tokens: int
    cached_tokens: int

    @property
    def rule(self) -> DemandRule:
        return DEMAND_RULES[self.purpose]

    def demand(self) -> KVDemand:
        return request_demand(
            self.input_tokens,
            self.max_new_tokens,
            windows=self.windows,
            page_size=self.page_size,
            prefill_retire_lag=self.prefill_retire_lag,
            eviction_interval=self.eviction_interval,
            checkpoint_interval=self.checkpoint_interval,
            decode_retire_lag=self.decode_retire_lag,
            chunk_size=self.chunk_size,
            state_slots=self.state_slots,
            host_hit_lengths=self.host_hit_lengths,
            extra_tokens=self.extra_tokens,
            cached_tokens=self.cached_tokens,
        )


def demand_inputs(
    purpose: DemandPurpose,
    input_tokens: int,
    max_new_tokens: int,
    *,
    windows: Mapping[str, Optional[int]],
    page_size: int,
    chunk_limit: Optional[int],
    prefill_retire_lag: int,
    eviction_interval: int,
    checkpoint_interval: int = 0,
    decode_retire_lag: int = 0,
    state_slots: int = 0,
    cached_tokens: int = 0,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
) -> DemandInputs:
    """The inputs every site that prices a request builds alike: the
    ``input_tokens`` it still places in chunks of at most ``chunk_limit``,
    the ``windows`` it keeps (sub-pool name to retained window), one rounding
    page in the token sub-pool (``extra_tokens`` are the caller's charges on
    top of it), the scheduler's window retirement (``prefill_retire_lag``,
    ``eviction_interval``, as `request_demand` takes them) and the window rows
    its ``cached_tokens`` already hold. The purpose's row of `DEMAND_RULES`
    sets what differs:
    the decode estimate and which ``state_slots`` the caller counts (and, for
    the caller to apply, whether eviction is credited). ``chunk_limit`` is the
    chunk the caller admitted, so a request's admission, shape check and
    commitment price the same chunk."""
    rule = DEMAND_RULES[purpose]
    if purpose is DemandPurpose.CEILING and (cached_tokens or host_hit_lengths):
        raise ValueError("a ceiling prices a request with nothing cached")
    if rule.estimates_decode:
        max_new_tokens = decode_estimate_tokens(max_new_tokens)
    return DemandInputs(
        purpose=purpose,
        input_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        windows=dict(windows),
        page_size=page_size,
        prefill_retire_lag=prefill_retire_lag,
        eviction_interval=eviction_interval,
        checkpoint_interval=checkpoint_interval,
        decode_retire_lag=decode_retire_lag,
        chunk_size=chunk_limit,
        state_slots=state_slots,
        host_hit_lengths=dict(host_hit_lengths or {}),
        extra_tokens=extra_tokens,
        cached_tokens=cached_tokens,
    )


def largest_fitting_chunk(
    fits: Callable[[int], bool], chunk_limit: int, granule: int
) -> int:
    """The largest multiple of ``granule`` at most ``chunk_limit`` for which
    ``fits`` holds, or 0 when none does. A request's demand only grows with
    its chunk (`request_demand`), so ``fits`` is monotone and a binary search
    finds it."""
    low, high = 0, max(0, chunk_limit) // granule
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid * granule):
            low = mid
        else:
            high = mid - 1
    return low * granule


def chunk_granule(page_size: int, align: Optional[int] = None) -> int:
    """The step a truncated chunk is cut on: one page, or the least common
    multiple of a page and the scheduler's truncation alignment where it
    sets one. A chunk limit off this grid truncates to a shorter chunk, and
    one below it to none."""
    return page_size if not align else math.lcm(page_size, align)


def smallest_admissible_chunk(
    chunk_size: Optional[int], page_size: int, align: Optional[int] = None
) -> Optional[int]:
    """The smallest chunk admission may cut a prompt to: the grid step
    (`chunk_granule`), when the configured ``chunk_size`` reaches it; 0 when
    it does not, since no positive chunk of the grid is admissible then, as
    `largest_fitting_chunk` finds none. None without chunking. Since demand
    only grows with the chunk, a request fits at some chunk
    `largest_fitting_chunk` searches exactly when it fits at this one."""
    if chunk_size is None:
        return None
    granule = chunk_granule(page_size, align)
    return granule if chunk_size >= granule else 0
