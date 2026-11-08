"""Capacity under pipeline parallelism: the first stage decides for every stage.

Each pipeline stage keeps its own arena. Its bytes, and the bytes of one unit
of each sub-pool, follow the stage's layers, so the same requests fit on one
stage and not on another. Every stage runs the same logical operations in the
same order, so the units each sub-pool holds are the same on every stage.

The first stage answers every capacity question for all of them: it prices
each other stage on that stage's geometry with the units it holds itself
(`simulated_snapshot`), and a demand fits only where it fits on every stage.
The later stages ask the same questions in the same order and take the first
stage's answers (`StageCapacity`), so they admit, evict, retract and take
exactly what it decided; each answer it relays names its question, and a
later stage that asks another one has diverged.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Deque, Dict, Iterator, Mapping, Optional, Tuple

import msgspec

from sglang.srt.mem_cache.kv.alloc.pricing import BandSnapshot, ChainSnapshot


class BandGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """One band of a stage's chain as its allocator built it: what pricing
    reads that is not a count."""

    name: str
    grow: str
    page_bytes: int
    entry_bytes: int
    page_size: int
    num_pages: int
    min_page: int
    lazy: bool
    num_virtual_pages: int
    # Mints its own virtual ids (the token owner, the state slots); a window
    # binds the owner's.
    id_owner: bool


class StageGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A stage's arena bytes and its bands in chain order. ``window_tokens``
    is the stage's nominal window share, the sizing's window tokens."""

    total_bytes: int
    page_size: int
    bands: Tuple[BandGeometry, ...]
    window_tokens: int = 0

    def sub_pools(self) -> Tuple[str, ...]:
        return tuple(sorted(band.name for band in self.bands))


def stage_geometry(
    snapshot: ChainSnapshot, *, id_owners: Tuple[str, ...], window_tokens: int
) -> StageGeometry:
    """The geometry of the stage whose pricing snapshot is ``snapshot``."""
    return StageGeometry(
        total_bytes=snapshot.total_bytes,
        page_size=snapshot.page_size,
        bands=tuple(
            BandGeometry(
                name=b.name,
                grow=b.grow,
                page_bytes=b.page_bytes,
                entry_bytes=b.entry_bytes,
                page_size=b.page_size,
                num_pages=b.num_pages,
                min_page=b.min_page,
                lazy=b.lazy,
                num_virtual_pages=b.num_virtual_pages,
                id_owner=b.name in id_owners,
            )
            for b in snapshot.bands
        ),
        window_tokens=window_tokens,
    )


def simulated_snapshot(
    geometry: StageGeometry,
    live: Mapping[str, int],
    used_ids: Mapping[str, int],
) -> ChainSnapshot:
    """A stage's pricing snapshot for the units another stage holds: ``live``
    pages per band and ``used_ids`` virtual pages per id owner. Every end is
    compacted, every float empty of holes, and no page is pending or fixed:
    with the moves `feasible` allows, a stage's own snapshot of the same
    counts serves exactly the same demands. Its virtual ids are its own
    count less the ids in use."""
    bands = []
    for g in geometry.bands:
        pages = live[g.name]
        e = g.page_bytes
        if g.grow == "down":
            low, high = g.num_pages - pages, g.num_pages
        else:
            low, high = g.min_page, g.min_page + pages
        free_ids = 0
        if g.id_owner:
            free_ids = g.num_virtual_pages - g.min_page - used_ids[g.name]
        floats = g.grow == "float"
        bands.append(
            BandSnapshot(
                name=g.name,
                grow=g.grow,
                page_bytes=e,
                entry_bytes=g.entry_bytes,
                page_size=g.page_size,
                num_pages=g.num_pages,
                min_page=g.min_page,
                lazy=g.lazy,
                blocked=False,
                holes=0,
                live=pages,
                span=pages,
                low_byte=low * e,
                high_byte=high * e,
                free_virtual_pages=free_ids,
                num_virtual_pages=g.num_virtual_pages,
                low_page=low if floats else 0,
                high_page=high if floats else 0,
            )
        )
    return ChainSnapshot(
        tuple(bands),
        geometry.page_size,
        geometry.total_bytes,
        {b.name: i for i, b in enumerate(bands)},
    )


class StageCapacityDivergence(RuntimeError):
    """A later stage asked a capacity question the first stage did not, or
    cannot realize what the first stage decided."""


class StageCapacityRecord(msgspec.Struct, frozen=True, kw_only=True):
    """What the first stage relays for one scheduling round: its answers in
    order, each with its question; with checks on, the round's outcome and,
    per later stage, the counts it priced that stage at."""

    decisions: Tuple[Tuple[str, Any, Any], ...]
    outcome: Any = None
    counts: Optional[Dict[int, Any]] = None


class StageCapacity:
    """The capacity questions of one stage's chain within a scheduling round.

    On the first stage (``stage == 0``) every answer prices every stage:
    ``others`` holds the geometry of each later stage, by stage index. While
    a round is open the outermost questions and their answers are recorded.
    On a later stage a round replays the first stage's record: each question
    takes the next answer, which must answer the same question. Outside a
    round, and for reads that only observe (`observing`), a later stage
    answers from its own chain.
    """

    def __init__(self, stage: int, others: Mapping[int, StageGeometry]):
        if stage and others:
            raise ValueError("only the first stage prices the other stages")
        self.stage = stage
        self.others: Dict[int, StageGeometry] = dict(others)
        self._recording: Optional[list] = None
        self._replay: Optional[Deque[Tuple[str, Any, Any]]] = None
        self._depth = 0
        self._observing = 0

    @property
    def decides(self) -> bool:
        return self.stage == 0

    @property
    def in_round(self) -> bool:
        return self._recording is not None or self._replay is not None

    def begin(self, decisions: Tuple[Tuple[str, Any, Any], ...] = ()) -> None:
        """Open a round: record on the first stage, else replay ``decisions``."""
        if self.in_round:
            raise RuntimeError("a capacity round is already open")
        if self.decides:
            self._recording = []
        else:
            self._replay = deque(decisions)

    def end(self) -> Tuple[Tuple[str, Any, Any], ...]:
        """Close the round; the first stage's answers in order. A later stage
        must have taken every answer."""
        if self.decides:
            recorded, self._recording = self._recording, None
            return tuple(recorded or ())
        left, self._replay = self._replay, None
        if left:
            raise StageCapacityDivergence(
                f"stage {self.stage} asked {len(left)} capacity question(s) "
                f"fewer than the first stage; the next was {left[0][:2]}"
            )
        return ()

    @contextmanager
    def observing(self) -> Iterator[None]:
        """Reads that decide nothing (metrics, load reports, logs), which a
        stage may make where another does not: neither recorded nor
        replayed."""
        self._observing += 1
        try:
            yield
        finally:
            self._observing -= 1

    def decide(self, op: str, key: Any, answer: Callable[[], Any]) -> Any:
        """The answer to question ``(op, key)``: replayed on a later stage
        inside a round, else ``answer()``, recorded on the first stage when
        it is not asked inside another recorded question."""
        if self._observing:
            return answer()
        replay = self._replay
        if replay is not None:
            if not replay:
                raise StageCapacityDivergence(
                    f"stage {self.stage} asked {(op, key)} past the first "
                    "stage's last capacity answer"
                )
            asked_op, asked_key, value = replay.popleft()
            if asked_op != op or asked_key != key:
                raise StageCapacityDivergence(
                    f"stage {self.stage} asked {(op, key)} where the first "
                    f"stage asked {(asked_op, asked_key)}"
                )
            return value
        if self._recording is None or self._depth:
            return answer()
        self._depth += 1
        try:
            value = answer()
        finally:
            self._depth -= 1
        self._recording.append((op, key, value))
        return value
