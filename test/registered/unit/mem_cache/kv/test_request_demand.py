"""A request's demand: the window bound covers the rows the scheduler's own
window retirement keeps live (a property over random requests, and the cases
reviews found as instances of it), grows with the chunk, and the chunk grid
admission searches."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    ReqKvInfo,
    ScheduleBatch,
    branch_checkpoint,
    prefill_checkpoint_boundary,
)
from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.mem_cache.kv.alloc.demand import (
    chunk_granule,
    largest_fitting_chunk,
    request_demand,
    smallest_admissible_chunk,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    branching_checkpoint_depth,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 128
WINDOW = 4096
CHUNK = 1024
PROMPT = 8192


class _WindowCache(RadixCache):
    """One window whose rows the scheduler's own retirement frees, counting
    every row it frees."""

    def __init__(self, *, window, page_size, rows, shares_prefixes, checkpoints=False):
        self.disable = False
        self.checkpoints = checkpoints
        self.sliding_window_size = window
        self.page_size = page_size
        self.shares_prefixes = shares_prefixes
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(rows).reshape(1, -1)
        )
        allocator = Mock(
            spec=[
                "free_swa_segment",
                "free_group_begin",
                "free_group_end",
                "page_size",
                "storage",
                "window_pool_names",
                "target_window_pool",
                "get_kvcache",
            ]
        )
        allocator.page_size = page_size
        allocator.storage = None
        allocator.window_pool_names = lambda: ("swa",)
        allocator.target_window_pool = lambda: "swa"
        allocator.get_kvcache = lambda: SimpleNamespace(request_window=None)
        self.token_to_kv_pool_allocator = allocator
        kv_storage_of(self.req_to_token_pool, allocator).bind_window_retention(
            window, None
        )

    def supports_swa(self):
        return True

    def supports_prefix_sharing(self):
        return self.shares_prefixes

    def swa_retain_floor(self, req):
        # As `UnifiedRadixCache.swa_retain_floor`: keep the window behind the
        # latest state checkpoint.
        checkpoint = req.kv.mamba_last_track_seqlen
        if not self.checkpoints or checkpoint is None:
            return None
        return checkpoint - self.sliding_window_size

    def freed(self) -> int:
        calls = self.token_to_kv_pool_allocator.free_swa_segment.call_args_list
        return sum(len(call.args[0]) for call in calls)


def _retired_peak(
    prompt,
    new_tokens,
    *,
    window,
    page_size,
    chunk,
    overlap,
    shares,
    interval,
    checkpoint_grid=0,
    pages=False,
    reserve=None,
    full_kv_hit=0,
):
    """The most window rows one request holds while the scheduler runs its
    chunks and decode steps (`ScheduleBatch.maybe_evict_swa` before each
    allocation), as live rows: allocated (whole pages with ``pages``) less
    retired. Under ``overlap`` a decode step is allocated before the result
    of the step before it is processed, so the request's length trails its
    allocated rows by that pending token. With ``checkpoint_grid``, each
    prefill chunk tracks the state checkpoint the scheduler's own decision
    gives it (`prefill_checkpoint_boundary`: none for a chunk shorter than the
    grid; with a ``full_kv_hit`` past the request's state, none reusable here,
    the branch checkpoint `branching_checkpoint_depth` picks, which the chunk
    holding it tracks instead, `branch_checkpoint`), the prefix-sharing tree
    takes the request up to its latest checkpoint and retires the window
    behind it, as `checkpoint_into_tree` does; a checkpoint lands on each multiple of the grid the decode
    reaches, and retirement keeps the window behind the latest one. With a speculative
    ``reserve``, each decode step allocates through its committed length plus
    the reserve (`page_aligned_decode_alloc_lens`) and accepts one token, the
    least a step commits, which keeps the most rows allocated ahead."""
    cache = _WindowCache(
        window=window,
        page_size=page_size,
        rows=prompt + new_tokens + (reserve or 0) + 2 * page_size + 1,
        shares_prefixes=shares,
        checkpoints=bool(checkpoint_grid),
    )

    def live(allocated):
        if pages:
            allocated = -(-allocated // page_size) * page_size
        return allocated - cache.freed()

    req = SimpleNamespace(
        seqlen=0,
        extend_batch_idx=0,
        decode_batch_idx=0,
        kv=ReqKvInfo(req_pool_idx=0),
    )
    batch = ScheduleBatch(
        reqs=[req],
        tree_cache=cache,
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        forward_mode=ForwardMode.EXTEND,
        enable_overlap=overlap,
    )
    allocated = peak = 0
    with (
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(interval),
        envs.SGLANG_OPT_RELEASE_PREFILL_SWA.override(False),
        patch(
            "sglang.srt.managers.schedule_batch.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=chunk),
        ),
    ):
        branching = (
            branching_checkpoint_depth(full_kv_hit, 0, checkpoint_grid)
            if checkpoint_grid
            else None
        )

        def take_into_tree(chunk_start, end):
            # A prefix-sharing cache takes each chunk into the tree once it
            # ran; the rows behind the window it keeps are the tree's, not
            # the request's (evictable, admission credits them). With state
            # checkpoints the tree takes the request only up to its latest
            # checkpoint, which the chunk's own tracking decision moves
            # (`checkpoint_into_tree` retires behind it, the window behind
            # the checkpoint kept), and nothing without one.
            if not checkpoint_grid:
                batch._evict_swa(req, end)
                return
            boundary = prefill_checkpoint_boundary(
                chunk_start, end - chunk_start, checkpoint_grid
            )
            if boundary is not None:
                # The scheduler checks the branch on any state chunk; the page
                # is the finest one, so the replay takes every branch it can.
                branch = branch_checkpoint(chunk_start, branching, end, page_size)
                req.kv.mamba_last_track_seqlen = boundary if branch is None else branch
            cached = req.kv.mamba_last_track_seqlen or 0
            if cached > 0:
                batch._evict_swa(req, cached - 1)

        for idx, start in enumerate(range(0, prompt, chunk)):
            req.extend_batch_idx = idx
            batch.prefix_lens = [start]
            if shares:
                if idx > 0:
                    take_into_tree(start - chunk, start)
            else:
                batch.maybe_evict_swa()
            allocated = min(prompt, start + chunk)
            peak = max(peak, live(allocated))
        if shares:
            take_into_tree((prompt - 1) // chunk * chunk, prompt)
        # The prefill emits the first token; each decode step writes one row.
        # Under overlap the latest token is still pending when the next step
        # is allocated.
        batch.forward_mode = ForwardMode.DECODE
        if reserve is not None:
            committed = prompt
            for step in range(new_tokens - 1):
                req.decode_batch_idx = step
                req.seqlen = committed if overlap else committed + 1
                if checkpoint_grid:
                    req.kv.mamba_last_track_seqlen = (
                        committed // checkpoint_grid * checkpoint_grid
                    )
                batch.maybe_evict_swa()
                kv = SimpleNamespace(
                    kv_allocated_len=allocated, kv_committed_len=committed
                )
                _, nxt, _ = page_aligned_decode_alloc_lens(
                    [SimpleNamespace(kv=kv)], reserve=reserve, page_size=page_size
                )
                allocated = nxt[0]
                peak = max(peak, live(allocated))
                committed += 1
            return peak
        for step in range(new_tokens - 1):
            req.decode_batch_idx = step
            req.seqlen = allocated if overlap else allocated + 1
            if checkpoint_grid:
                req.kv.mamba_last_track_seqlen = (
                    allocated // checkpoint_grid * checkpoint_grid
                )
            batch.maybe_evict_swa()
            allocated += 1
            peak = max(peak, live(allocated))
    return peak


class TestWindowDemandUnderRetirement(CustomTestCase):
    """The window demand bounds the rows the scheduler keeps live: retiring
    a chunk behind the frontier the overlap scheduler keeps for the prefill
    still running, and decode retiring once an interval slid out."""

    def _check(
        self,
        prompt,
        new_tokens,
        *,
        page_size,
        chunk,
        lag,
        interval,
        decode_lag=0,
        **kw,
    ):
        peak = _retired_peak(
            prompt,
            new_tokens,
            window=WINDOW,
            page_size=page_size,
            chunk=chunk,
            interval=interval,
            **kw,
        )
        demand = request_demand(
            prompt,
            new_tokens,
            windows={"swa": WINDOW},
            page_size=page_size,
            chunk_size=chunk,
            prefill_retire_lag=lag,
            decode_retire_lag=decode_lag,
            eviction_interval=interval,
        ).window_tokens["swa"]
        self.assertGreaterEqual(demand, peak)
        return peak

    def test_an_overlapped_chunked_prefill_keeps_a_chunk_more(self):
        # No prefix sharing: the overlap scheduler retires a chunk behind.
        peak = self._check(
            PROMPT,
            1,
            page_size=PAGE,
            chunk=CHUNK,
            lag=CHUNK,
            interval=128,
            overlap=True,
            shares=False,
        )
        self.assertEqual(peak, WINDOW + 2 * CHUNK)

    def test_decode_keeps_an_eviction_interval_more(self):
        for shares in (False, True):
            with self.subTest(shares=shares):
                peak = self._check(
                    1,
                    2 * WINDOW,
                    page_size=1,
                    chunk=CHUNK,
                    lag=0,
                    interval=128,
                    overlap=False,
                    shares=shares,
                )
                self.assertEqual(peak, WINDOW + 128)

    def test_an_overlapped_decode_keeps_its_pending_row(self):
        # One-row pages leave no rounding to hide the row allocated ahead of
        # the pending result; the prefill lag alone does not cover it.
        for shares in (False, True):
            with self.subTest(shares=shares):
                peak = self._check(
                    1024,
                    2 * WINDOW,
                    page_size=1,
                    chunk=CHUNK,
                    lag=CHUNK,
                    decode_lag=1,
                    interval=128,
                    overlap=True,
                    shares=shares,
                )
                self.assertEqual(peak, WINDOW + 128 + 1)


class TestWindowDemandUnderPageRetention(CustomTestCase):
    """REGRESSION: a prefix-sharing cache keeps the window rounded up to whole
    pages and frees whole pages only. Under overlap, a window that is not a
    whole number of pages (4095 rows, 128-row pages) kept a page more than the
    demand, which counted the raw window: 34 pages live against 33 priced."""

    def _pages(self, window, page, interval, *, shares):
        peak = _retired_peak(
            1024,
            8192,
            window=window,
            page_size=page,
            chunk=1024,
            overlap=True,
            shares=shares,
            interval=interval,
            pages=True,
        )
        demand = request_demand(
            1024,
            8192,
            windows={"swa": window},
            page_size=page,
            chunk_size=1024,
            prefill_retire_lag=0 if shares else 1024,
            decode_retire_lag=1,
            eviction_interval=interval,
        ).window_tokens["swa"]
        return -(-peak // page), -(-demand // page)

    def test_a_window_short_of_whole_pages(self):
        peak_pages, demand_pages = self._pages(4095, 128, 1, shares=True)
        self.assertEqual(peak_pages, 34)
        self.assertGreaterEqual(demand_pages, peak_pages)

    def test_the_demand_covers_whole_page_retention(self):
        for window in (4000, 4095, 4096, 4097):
            for interval in (1, 128):
                for shares in (False, True):
                    with self.subTest(window=window, interval=interval, shares=shares):
                        peak_pages, demand_pages = self._pages(
                            window, 128, interval, shares=shares
                        )
                        self.assertGreaterEqual(demand_pages, peak_pages)


class TestDecodeRowsAllocatedAhead(CustomTestCase):
    """REGRESSION: a short request's decode demand was its whole length when
    that was below the window, which dropped the rows a speculative decode
    step allocates ahead (a DFLASH step reserves two 32-row blocks): 131
    window rows priced against 192 allocated. Those rows sit on top of the
    bounded length, in the token sub-pool and every window alike."""

    def _allocated(self, prompt, page_size, reserve):
        """The rows the first decode step allocates after the prefill."""
        req = SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=prompt, kv_committed_len=prompt)
        )
        _, nxt, _ = page_aligned_decode_alloc_lens(
            [req], reserve=reserve, page_size=page_size
        )
        return nxt[0]

    def test_a_speculative_reserve_beyond_a_short_request(self):
        for prompt, page_size in ((128, 1), (100, 16)):
            with self.subTest(prompt=prompt, page_size=page_size):
                reserve = 64
                demand = request_demand(
                    prompt,
                    2,
                    windows={"swa": WINDOW},
                    page_size=page_size,
                    prefill_retire_lag=0,
                    eviction_interval=128,
                    decode_retire_lag=reserve,
                )
                allocated = self._allocated(prompt, page_size, reserve)
                self.assertGreater(allocated, prompt + 2)
                self.assertGreaterEqual(demand.window_tokens["swa"], allocated)
                self.assertGreaterEqual(demand.full_tokens, allocated)

    def test_the_first_decode_beside_the_last_prefill(self):
        """REGRESSION: the first decode step retires nothing, and a long
        prompt's last prefill left its chunk and the window and lag behind it
        live; the speculative reserve was priced only on steady decode: 6145
        window rows against 6208 live."""
        demand = request_demand(
            PROMPT,
            2,
            windows={"swa": WINDOW},
            page_size=1,
            chunk_size=CHUNK,
            prefill_retire_lag=CHUNK,
            eviction_interval=128,
            decode_retire_lag=64,
        ).window_tokens["swa"]
        self.assertGreaterEqual(demand, WINDOW + 2 * CHUNK + 64)
        peak = _retired_peak(
            PROMPT,
            2,
            window=WINDOW,
            page_size=1,
            chunk=CHUNK,
            overlap=True,
            shares=False,
            interval=128,
            reserve=64,
        )
        self.assertEqual(peak, WINDOW + 2 * CHUNK + 64)
        self.assertGreaterEqual(demand, peak)

    def test_a_prefill_only_request_reserves_no_decode_rows(self):
        args = dict(
            windows={"swa": WINDOW},
            page_size=PAGE,
            prefill_retire_lag=0,
            eviction_interval=128,
        )
        without = request_demand(PROMPT, 0, **args)
        with_lag = request_demand(PROMPT, 0, decode_retire_lag=64, **args)
        self.assertEqual(with_lag, without)


class TestWindowDemandBoundsTheRetiredPeak(CustomTestCase):
    """The window demand bounds, in whole pages, the most rows the
    scheduler's own retirement keeps live, over random requests: prompt and
    generation lengths, windows on and off the page grid, page sizes 1, 16
    and 64, chunks, the overlap scheduler, prefix sharing with state
    checkpoints, eviction intervals and speculative reserves. The retirement
    is `ScheduleBatch.maybe_evict_swa` (its first decode step retires
    nothing), the decode allocation `page_aligned_decode_alloc_lens`."""

    CASES = 160
    ON_GRID = True

    def _case(self, rng):
        page = rng.choice((1, 16, 64))
        window = max(
            1,
            rng.choice((page, 2 * page, 8 * page, 16 * page))
            + rng.choice((-1, 0, 0, 1)),
        )
        chunk = page * rng.randint(1, max(1, 1024 // page))
        overlap = rng.random() < 0.5
        shares = rng.random() < 0.5
        grid = page * rng.choice((0, 0, 1, 4, 16)) if shares else 0
        if grid and self.ON_GRID:
            # Every cut lands on the checkpoint grid (`mamba_prefill_chunk_grid`;
            # a configured chunk below it is refused at startup).
            chunk = grid * max(1, chunk // grid)
        prompt = rng.randint(1, 2048)
        # A full-KV match past the request's state: a branch checkpoint.
        hit = rng.choice((0, 0, rng.randint(0, prompt))) if grid else 0
        spec = rng.random() < 0.5
        reserve = rng.randint(1, 2 * rng.choice((4, 8, 32))) if spec else None
        return dict(
            prompt=prompt,
            new_tokens=rng.randint(1, 320),
            window=window,
            page_size=page,
            chunk=chunk,
            overlap=overlap,
            shares=shares,
            interval=rng.choice((1, 16, 128)),
            checkpoint_grid=grid,
            reserve=reserve,
            full_kv_hit=hit,
        )

    def _demand(self, c):
        if c["reserve"] is not None:
            ahead = c["reserve"]
        else:
            ahead = 1 if c["overlap"] else 0
        return request_demand(
            c["prompt"],
            c["new_tokens"],
            windows={"swa": c["window"]},
            page_size=c["page_size"],
            chunk_size=c["chunk"],
            prefill_retire_lag=(c["chunk"] if c["overlap"] and not c["shares"] else 0),
            decode_retire_lag=ahead,
            eviction_interval=c["interval"],
            checkpoint_interval=c["checkpoint_grid"],
        ).window_tokens["swa"]

    def test_the_demand_covers_the_peak(self):
        rng = random.Random(20261108)
        for i in range(self.CASES):
            c = self._case(rng)
            page = c["page_size"]
            peak = _retired_peak(pages=True, **c)
            demand = self._demand(c)
            peak_pages, demand_pages = -(-peak // page), -(-demand // page)
            with self.subTest(case=i, **c):
                self.assertGreaterEqual(demand_pages, peak_pages)

    def test_only_chunks_off_the_checkpoint_grid_break_the_bound(self):
        """Chunks anywhere on the page grid, below the checkpoint grid
        included: every case whose peak the demand misses cuts its prompt
        off the checkpoint grid, which the scheduler now refuses (a chunk
        below the grid) or aligns (a cut between checkpoints)."""
        rng = random.Random(20261109)
        self.ON_GRID = False
        broken = []
        off_grid = 0
        for _ in range(4 * self.CASES):
            c = self._case(rng)
            page, grid = c["page_size"], c["checkpoint_grid"]
            on_grid = not grid or c["chunk"] % grid == 0 or c["chunk"] >= c["prompt"]
            off_grid += not on_grid
            peak = _retired_peak(pages=True, **c)
            if -(-self._demand(c) // page) < -(-peak // page):
                broken.append((on_grid, c))
        self.assertGreater(off_grid, 0)
        self.assertTrue(broken, "no off-grid case broke the bound")
        self.assertEqual([c for on_grid, c in broken if on_grid], [])

    def test_an_early_branch_checkpoint_keeps_the_whole_chunk(self):
        """REGRESSION: a full-KV match past the request's latest reusable
        state makes the chunk holding the branch point track it instead of its
        own checkpoint, so the tree retires only behind the branch while the
        next chunk starts past the whole chunk (prompt 4096, chunk 1024,
        window 512, page 16, checkpoints every 64, a branch at 64: at least
        124 pages live against a 101-page demand). The bound keeps a whole
        chunk past the latest checkpoint."""
        c = dict(
            prompt=4096,
            new_tokens=1,
            window=512,
            page_size=16,
            chunk=1024,
            overlap=False,
            shares=True,
            interval=1,
            checkpoint_grid=64,
            reserve=None,
            full_kv_hit=100,
        )
        self.assertEqual(branching_checkpoint_depth(100, 0, 64), 64)
        self.assertEqual(branch_checkpoint(0, 64, 1024, 64), 64)
        peak = -(-_retired_peak(pages=True, **c) // 16)
        self.assertGreaterEqual(peak, 124)
        self.assertGreaterEqual(-(-self._demand(c) // 16), peak)

    def test_chunks_below_the_checkpoint_grid_retire_no_window_row(self):
        """REGRESSION: chunks shorter than the checkpoint grid track no
        checkpoint, so a prefix-sharing tree takes none of the prompt and
        retires no window row (prompt 4096, window 512, page 16, chunk 32,
        grid 64: 256 pages live against a 41-page demand). The scheduler
        keeps such a configuration from running (`mamba_prefill_chunk_grid`)."""
        c = dict(
            prompt=4096,
            new_tokens=1,
            window=512,
            page_size=16,
            chunk=32,
            overlap=False,
            shares=True,
            interval=128,
            checkpoint_grid=64,
            reserve=None,
        )
        # The production steps the replay chains: a chunk below the grid
        # tracks no checkpoint, so the request has none, and the tree takes
        # an unfinished request only up to its latest checkpoint (none: a
        # cache length of 0, so `checkpoint_into_tree` retires no row).
        from sglang.srt.mem_cache.unified_cache.components.mamba import (
            MambaComponent,
        )

        self.assertIsNone(prefill_checkpoint_boundary(64, 32, 64))
        unfinished = SimpleNamespace(kv=SimpleNamespace(mamba_last_track_seqlen=None))
        self.assertEqual(
            MambaComponent.prepare_for_caching_req(
                SimpleNamespace(cache=SimpleNamespace(enable_mamba_extra_buffer=True)),
                req=unfinished,
                insert_params=None,
                token_ids_len=96,
                is_finished=False,
            ),
            0,
        )
        peak = _retired_peak(pages=True, **c)
        self.assertEqual(-(-peak // 16), 256)
        self.assertLess(-(-self._demand(c) // 16), 256)
        aligned = dict(c, chunk=64)
        peak = _retired_peak(pages=True, **aligned)
        self.assertGreaterEqual(-(-self._demand(aligned) // 16), -(-peak // 16))

    def test_a_shared_chunk_ending_between_checkpoints(self):
        """REGRESSION: with prefix sharing and state checkpoints, a chunk that
        ends on a page but between checkpoints keeps its tail past the latest
        checkpoint (window 512, page 16, chunk 1008, checkpoints every 64:
        48 rows), and retirement keeps the window behind that checkpoint."""
        c = dict(
            prompt=4096,
            new_tokens=2,
            window=512,
            page_size=16,
            chunk=1008,
            overlap=True,
            shares=True,
            interval=1,
            checkpoint_grid=64,
            reserve=None,
        )
        peak = _retired_peak(pages=True, **c)
        self.assertGreaterEqual(-(-self._demand(c) // 16), -(-peak // 16))


class TestCachedWindowRowsCountOnce(CustomTestCase):
    def test_a_cached_prefix_is_credited_whichever_phase_peaks(self):
        """REGRESSION: the window rows a cached prefix holds were credited
        only against the prefill's peak; when decode's floor (the window plus
        the rows kept past it) set the bound, a request with a cached prefix
        was charged those held rows again, more than the same request priced
        from an empty pool. Every phase counts them once."""
        args = dict(
            input_tokens=256,
            max_new_tokens=4,
            windows={"swa": 64},
            page_size=16,
            chunk_size=64,
            prefill_retire_lag=0,
            eviction_interval=128,
            decode_retire_lag=1,
        )
        cold = request_demand(**args).window_tokens["swa"]
        cached = request_demand(cached_tokens=1024, **args).window_tokens["swa"]
        self.assertEqual(cold - cached, 64)


class TestWindowDemandGrowsWithTheChunk(CustomTestCase):
    """Admission searches the chunk grid for the largest chunk that fits,
    which holds only while a smaller chunk never needs more: a long cached
    prefix and a retirement lag must not charge more rows than the prompt
    adds."""

    def _demand(self, chunk, **kw):
        return request_demand(
            chunk_size=chunk, windows={"swa": WINDOW}, page_size=PAGE, **kw
        ).window_tokens["swa"]

    def test_the_search_finds_the_whole_prompt_behind_a_long_prefix(self):
        args = dict(
            input_tokens=512,
            max_new_tokens=1,
            prefill_retire_lag=CHUNK,
            eviction_interval=128,
            cached_tokens=2 * WINDOW,
        )
        for chunk in range(PAGE, 513, PAGE):
            with self.subTest(chunk=chunk):
                self.assertEqual(self._demand(chunk, **args), 512 + PAGE + 1)
        self.assertEqual(
            largest_fitting_chunk(
                lambda chunk: self._demand(chunk, **args) <= 768,
                chunk_limit=512,
                granule=PAGE,
            ),
            512,
        )

    def test_the_demand_never_falls_as_the_chunk_grows(self):
        for prompt in (512, 1024, 4096, PROMPT):
            for cached in (0, PAGE, WINDOW // 2, WINDOW, 2 * WINDOW, 3 * WINDOW):
                for lag, ahead in ((0, 0), (CHUNK, 0), (CHUNK, 64)):
                    demands = [
                        self._demand(
                            chunk,
                            input_tokens=prompt,
                            max_new_tokens=2,
                            prefill_retire_lag=lag,
                            decode_retire_lag=ahead,
                            eviction_interval=128,
                            cached_tokens=cached,
                        )
                        for chunk in range(PAGE, prompt + 1, PAGE)
                    ]
                    with self.subTest(
                        prompt=prompt, cached=cached, lag=lag, ahead=ahead
                    ):
                        self.assertEqual(demands, sorted(demands))


class TestWindowDemandUnderCheckpointRetention(CustomTestCase):
    """A cache that keeps state checkpoints keeps the window behind the latest
    one: decode holds up to the checkpoint interval past the window, which
    the demand must cover, page rounding included."""

    def test_the_window_behind_the_latest_checkpoint(self):
        window, page, interval, grid = 4096, 128, 128, 512
        prompt, new_tokens = 1024, 8192
        peak = _retired_peak(
            prompt,
            new_tokens,
            window=window,
            page_size=page,
            chunk=1024,
            overlap=False,
            shares=True,
            interval=interval,
            checkpoint_grid=grid,
            pages=True,
        )

        def demand(checkpoint_interval):
            return request_demand(
                prompt,
                new_tokens,
                windows={"swa": window},
                page_size=page,
                prefill_retire_lag=0,
                eviction_interval=interval,
                checkpoint_interval=checkpoint_interval,
                chunk_size=1024,
            ).window_tokens["swa"]

        self.assertGreater(peak, window + interval)
        self.assertGreaterEqual(demand(grid), peak)

    def test_without_checkpoints_the_eviction_interval_bounds(self):
        peak = _retired_peak(
            1024,
            8192,
            window=4096,
            page_size=128,
            chunk=1024,
            overlap=False,
            shares=True,
            interval=128,
            pages=True,
        )
        self.assertGreaterEqual(
            request_demand(
                1024,
                8192,
                windows={"swa": 4096},
                page_size=128,
                prefill_retire_lag=0,
                eviction_interval=128,
                chunk_size=1024,
            ).window_tokens["swa"],
            peak,
        )


class TestChunkGrid(CustomTestCase):
    def test_the_smallest_chunk_is_on_the_grid(self):
        self.assertIsNone(smallest_admissible_chunk(None, 4))
        self.assertEqual(smallest_admissible_chunk(16, 4), 4)
        self.assertEqual(smallest_admissible_chunk(64, 4, 6), 12)
        self.assertEqual(smallest_admissible_chunk(384, 128, 192), 384)

    def test_a_limit_below_the_grid_admits_no_chunk(self):
        # lcm(128, 192) = 384: no positive chunk of the grid fits in 256,
        # as the search over that grid finds none either.
        granule = chunk_granule(128, 192)
        self.assertEqual(smallest_admissible_chunk(256, 128, 192), 0)
        self.assertEqual(largest_fitting_chunk(lambda _: True, 256, granule), 0)
        self.assertEqual(smallest_admissible_chunk(8, 4, 64), 0)


if __name__ == "__main__":
    unittest.main()
