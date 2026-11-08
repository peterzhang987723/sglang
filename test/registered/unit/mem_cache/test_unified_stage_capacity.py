"""Pipeline stages under unified memory: the first stage decides every stage's
capacity, pricing each later stage on that stage's own geometry."""

import random
import unittest
from contextlib import contextmanager
from typing import List

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.scheduler_components.pp_capacity_relay import (
    PPCapacityRelay,
)
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.alloc.stage_capacity import StageCapacityDivergence
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# One byte budget, two layer mixes: the first stage holds mostly full
# attention, the second mostly recurrent state.
_STATE_STAGES = (
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=tuple(range(6)),
        state_layer_ids=(6,),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
    ),
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=(7,),
        state_layer_ids=tuple(range(8, 16)),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
        start_layer=7,
    ),
)
# With a sliding window (a float between the ends): the first stage mostly
# full attention, the second mostly window layers.
_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=tuple(range(4)),
        swa_layer_ids=(4,),
        state_layer_ids=(5,),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
    ),
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=(6,),
        swa_layer_ids=(7, 8, 9, 10),
        state_layer_ids=(11, 12),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
        start_layer=6,
    ),
)


def _stages(geometries):
    allocators = [
        build_unified_bundle(g).token_to_kv_pool_allocator for g in geometries
    ]
    shared = [a.stage_geometry() for a in allocators]
    for stage, allocator in enumerate(allocators):
        allocator.install_stage_capacity(stage, shared)
    return allocators


@contextmanager
def _alone(allocator):
    """The allocator answering for its own stage only."""
    capacity, allocator.stage_capacity = allocator.stage_capacity, None
    try:
        yield allocator
    finally:
        allocator.stage_capacity = capacity


def _fits_alone(allocator, demands) -> bool:
    with _alone(allocator):
        return allocator.units_fit(demands)


def _admits_alone(allocator, demand, evictable) -> bool:
    with _alone(allocator):
        return allocator.admits(demand, evictable=evictable)


def _in_lockstep(allocators, op):
    """Run ``op`` on every stage in one capacity round: recorded on the first
    stage, replayed on the others; every stage's result."""
    first = allocators[0].stage_capacity
    first.begin()
    results = [op(allocators[0])]
    decisions = first.end()
    for allocator in allocators[1:]:
        allocator.stage_capacity.begin(decisions)
        results.append(op(allocator))
        allocator.stage_capacity.end()
    return results


def _probe_demands(allocator, rng, count=12) -> List[dict]:
    owner, windows, state = allocator.sub_pool_roles()
    probes = []
    for _ in range(count):
        pages = rng.randrange(0, 80)
        demands = {owner: pages}
        for window in windows:
            demands[window] = min(pages, rng.randrange(0, 8))
        if state is not None:
            demands[state] = rng.randrange(0, 24)
        probes.append(demands)
    return probes


class TestStageCapacity(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_first_stage_decides_where_the_stages_disagree(self):
        """On empty arenas of one size, a FULL-heavy demand fits only the
        state-heavy stage and a state-heavy demand only the FULL-heavy one;
        the first stage refuses both, for both stages."""
        a0, a1 = _stages(_STATE_STAGES)
        owner, _, state = a0.sub_pool_roles()
        only = {0: [], 1: []}
        for full in range(0, 480, 8):
            for slots in range(0, 48, 4):
                demands = {owner: full, state: slots}
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                self.assertEqual(a0.units_fit(demands), fit0 and fit1, demands)
                if fit0 != fit1:
                    only[0 if fit0 else 1].append(demands)
        self.assertTrue(only[0] and only[1], only)
        for demands in only[0] + only[1]:
            tokens = demands[owner] * a0.page_size
            demand = KVDemand(tokens, {}, demands[state])
            self.assertFalse(a0.admits(demand), demands)
            self.assertNotEqual(
                _admits_alone(a0, demand, None), _admits_alone(a1, demand, None)
            )

    def _check_lockstep_simulation(self, geometries, seed):
        """The stages take and free the same units in the same order, the
        later one replaying the first one's decisions. At every step the first
        stage prices the later one at exactly the counts it holds, and its
        answers are the conjunction of the stages' own."""
        stages = _stages(geometries)
        a0, a1 = stages
        rng = random.Random(seed)
        owner, windows, state = a0.sub_pool_roles()
        tokens, slots = [], []
        disagreements = refusals = 0
        for _ in range(160):
            roll = rng.random()
            if roll < 0.4:
                n = rng.randrange(1, 24) * a0.page_size
                taken = _in_lockstep(stages, lambda a: a.alloc(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is None:
                    refusals += 1
                else:
                    tokens.append(taken)
            elif roll < 0.6 and state is not None:
                n = rng.randrange(1, 6)
                taken = _in_lockstep(stages, lambda a: a.alloc_state_slots(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is not None:
                    slots.append(taken)
            elif roll < 0.85 and tokens:
                pair = tokens.pop(rng.randrange(len(tokens)))
                for allocator, ids in zip(stages, pair):
                    allocator.free(ids)
            elif slots:
                pair = slots.pop(rng.randrange(len(slots)))
                for allocator, ids in zip(stages, pair):
                    allocator.mamba_allocator.free(ids)
            self.assertEqual(a0.simulated_stage_counts()[1], a1.stage_counts())
            for demands in _probe_demands(a0, rng, count=4):
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                disagreements += fit0 != fit1
                self.assertEqual(a0.units_fit(demands), fit0 and fit1, demands)
                evictable = {
                    name: rng.randrange(0, 16) * (1 if name == state else 4)
                    for name in (owner, *windows, state)
                    if name is not None
                }
                demand = KVDemand(
                    demands[owner] * a0.page_size,
                    {w: demands[w] * a0.page_size for w in windows},
                    demands.get(state, 0),
                )
                self.assertEqual(
                    a0.admits(demand, evictable=evictable),
                    _admits_alone(a0, demand, evictable)
                    and _admits_alone(a1, demand, evictable),
                    (demands, evictable),
                )
        self.assertGreater(disagreements, 0)
        self.assertGreater(refusals, 0)
        for allocator in stages:
            self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_lockstep_simulation_state_chain(self):
        self._check_lockstep_simulation(_STATE_STAGES, seed=0)

    def test_lockstep_simulation_window_chain(self):
        self._check_lockstep_simulation(_WINDOW_STAGES, seed=1)


class _Channel:
    """A FIFO standing in for the point-to-point channel to the next stage."""

    def __init__(self):
        self.items = []

    def send(self, data):
        self.items.append(data)
        return []

    def recv(self):
        return self.items.pop(0)


def _relays(allocators):
    channels = [_Channel() for _ in allocators]
    relays = []
    for stage, allocator in enumerate(allocators):
        relays.append(
            PPCapacityRelay(
                allocator=allocator,
                is_last_stage=stage == len(allocators) - 1,
                send=channels[stage].send,
                recv=(channels[stage - 1].recv if stage else None),
                commit=lambda work: None,
            )
        )
    return relays


def _round(relays, op, *, inject=None):
    """One scheduling round on every stage in turn, each closing at the same
    point; ``inject(stage, allocator)`` runs right before a stage's close."""
    for stage, relay in enumerate(relays):
        relay.round_begin(None)
        relay.relay()
        op(relay.allocator)
        if inject is not None:
            inject(stage, relay.allocator)
        relay.round_end(lambda: ("outcome",))
        relay.relay()


class TestStageCapacityRelay(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_agreement_check_fails_loudly_on_a_mismatch(self):
        """A later stage whose counts differ from what the first stage priced
        it at fails its round at WARN and STRICT, and passes at OFF."""
        for level, raises in (
            (InvariantCheckLevel.OFF, False),
            (InvariantCheckLevel.WARN, True),
            (InvariantCheckLevel.STRICT, True),
        ):
            with (
                self.subTest(level=level),
                envs.SGLANG_INVARIANT_CHECK.override(int(level)),
            ):
                relays = _relays(_stages(_STATE_STAGES))
                _round(relays, lambda a: a.alloc_state_slots(2))
                held = []

                def drift(stage, allocator):
                    if stage == 1:
                        with _alone(allocator):
                            held.append(allocator.alloc_state_slots(1))

                if raises:
                    with self.assertRaisesRegex(
                        AssertionError, "stage_capacity_agreement"
                    ):
                        _round(relays, lambda a: a.alloc(8), inject=drift)
                else:
                    _round(relays, lambda a: a.alloc(8), inject=drift)
                self.assertEqual(len(held), 1)

    def test_a_different_question_diverges_at_every_level(self):
        """A later stage that asks what the first stage did not ask, or asks
        fewer questions, fails at once, whatever the check level."""
        with envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.OFF)):
            a0, a1 = _stages(_STATE_STAGES)
            owner, _, _ = a0.sub_pool_roles()
            a0.stage_capacity.begin()
            a0.units_fit({owner: 3})
            a0.units_fit({owner: 4})
            decisions = a0.stage_capacity.end()
            a1.stage_capacity.begin(decisions)
            a1.units_fit({owner: 3})
            with self.assertRaises(StageCapacityDivergence):
                a1.units_fit({owner: 5})
            a1.stage_capacity.end()
            a1.stage_capacity.begin(decisions)
            a1.units_fit({owner: 3})
            with self.assertRaises(StageCapacityDivergence):
                a1.stage_capacity.end()

    def test_records_follow_requests_through_a_middle_stage(self):
        """Three stages: the middle one forwards the first stage's record, and
        the last stage takes exactly what the first stage decided, though on
        its own it would take more."""
        with envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT)):
            stages = _stages(_STATE_STAGES + (_STATE_STAGES[0],))
            relays = _relays(stages)
            taken = [[] for _ in stages]

            def take(allocator, n):
                stage = stages.index(allocator)
                taken[stage].append(allocator.alloc_state_slots(n) is not None)

            for n in (16, 16, 16, 1):
                _round(relays, lambda a: take(a, n))
            self.assertEqual(taken[0], taken[1])
            self.assertEqual(taken[0], taken[2])
            self.assertIn(False, taken[0])
            with _alone(stages[2]):
                self.assertTrue(stages[2].alloc_state_slots(1) is not None)
            for relay in relays:
                self.assertEqual(relay._send.__self__.items, [])


class _LoopDone(Exception):
    pass


def _loop_worker(rank, path, level):
    """One pipeline stage running the real PP event loop over gloo, with the
    scheduling and the model stood in: every round admits requests by the
    stage's capacity answers and takes their units, and requests that fit
    no empty arena are dropped when they arrive."""
    import math
    import os
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    os.environ["SGLANG_INVARIANT_CHECK"] = str(level)
    from sglang.srt.managers import scheduler_pp_mixin
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.runtime_context import get_parallel

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=60),
    )
    # One rank per stage: the attention TP broadcast is the identity.
    scheduler_pp_mixin.attn_cp_tp_broadcast_pyobj = lambda data: data
    geometry = _STATE_STAGES[rank % 2]
    allocator = build_unified_bundle(geometry).token_to_kv_pool_allocator
    page = allocator.page_size
    rounds = 24

    class Stage(SchedulerPPMixin):
        ingest_requests = Scheduler.ingest_requests

        def __init__(self):
            self.token_to_kv_pool_allocator = allocator
            self.tree_cache = None
            self.recv_skipper = None
            self.uocr_runtime = None
            self.device = "cpu"
            self.spec_algorithm = NS(is_none=lambda: True)
            self.pp_group = NS(
                is_first_rank=rank == 0,
                is_last_rank=rank == 2,
                cpu_group=dist.group.WORLD,
            )
            self.world_group = NS(cpu_group=dist.group.WORLD)
            self.pause_state = NS(
                holds_inputs=lambda: False,
                take_inputs=lambda reqs, stop_at_pause: list(reqs),
                may_launch=lambda: True,
            )
            self.request_receiver = NS(recv_requests=self._recv_requests)
            self.metrics_reporter = NS(record_scheduler_active=lambda start: None)
            self.invariant_checker = NS(run=lambda *args: None)
            self.new_token_ratio_tracker = NS(current=0.5)
            self.chunked_req = None
            self.waiting_queue = []
            self.held = []
            self.round = 0
            self.log = []

        def increment_scheduler_iterations(self):
            self.round += 1

        def _poll_timeout_aborts(self):
            return []

        def _recv_requests(self, local_reqs):
            if rank:
                return self._pp_recv_pyobj_from_prev_stage()
            # Requests of growing, then shrinking prompts; some fit no
            # FULL-heavy arena even empty.
            r = self.round
            return [(f"r{r}-{i}", (r * 37 + i * 101) % 360 + 8) for i in range(2)]

        def process_input_requests(self, reqs):
            for rid, tokens in reqs:
                fits = allocator.admits(KVDemand(tokens, {}, 1), empty_pool=True)
                self.log.append(("arrive", rid, tokens, fits))
                if fits:
                    self.waiting_queue.append(NS(rid=rid, tokens=tokens))

        def get_next_batch_to_run(self, running_batch, last_batch):
            if self.held and self.round % 3 == 0:
                req, ids, slots = self.held.pop(0)
                allocator.free(ids)
                allocator.mamba_allocator.free(slots)
                self.log.append(("finish", req.rid))
            while self.waiting_queue:
                req = self.waiting_queue[0]
                pages = math.ceil(req.tokens / page)
                if not allocator.admits(KVDemand(pages * page, {}, 1)):
                    self.log.append(("full", req.rid))
                    break
                ids = allocator.alloc(pages * page)
                slots = allocator.alloc_state_slots(1)
                assert ids is not None and slots is not None, req.rid
                self.waiting_queue.pop(0)
                self.held.append((req, ids, slots))
                self.log.append(("admit", req.rid))
            return NS(running_batch=running_batch, batch_to_run=None)

        def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *a):
            return None, None, None

        def _pp_commit_proxy_send_work(self):
            pass

        def on_idle(self):
            if self.round >= rounds:
                raise _LoopDone()

    stage = Stage()
    try:
        with get_parallel().override(
            pp_size=3,
            pp_rank=rank,
            tp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_rank=0,
            pp_async_batch_depth=0,
            world_group=NS(cpu_group=dist.group.WORLD),
        ):
            try:
                stage.event_loop_pp()
            except _LoopDone:
                pass
            stage._pp_commit_comm_work(stage.send_req_work)
            stage._pp_commit_comm_work(stage._pp_capacity_relay._send_work)
        logs = [None] * 3
        dist.all_gather_object(logs, stage.log)
        assert logs[0] == logs[1] == logs[2], logs
        kinds = {entry[0] for entry in logs[0]}
        assert {"admit", "full", "finish"} <= kinds, logs[0]
        arrivals = [e for e in logs[0] if e[0] == "arrive"]
        assert not all(e[3] for e in arrivals), arrivals
        # The state-heavy stage alone would keep requests the first stage
        # drops for the FULL-heavy stages; it follows the first stage.
        if rank == 1:
            with _alone(allocator):
                alone = [
                    allocator.admits(KVDemand(e[2], {}, 1), empty_pool=True)
                    for e in arrivals
                ]
            assert alone != [e[3] for e in arrivals], arrivals
    finally:
        dist.destroy_process_group()


class TestStageCapacityInThePipelineLoop(CustomTestCase):
    def test_three_stages_schedule_alike_under_the_pp_loop(self):
        """The real PP loop over three gloo ranks (FULL-heavy, state-heavy,
        FULL-heavy): each round's capacity record follows its requests on the
        same channel through the middle stage, and every stage arrives,
        admits, defers and finishes the same requests, the agreement checks
        passing at every round."""
        import tempfile

        import torch.multiprocessing as mp

        for level in (0, 2):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as tmp:
                mp.spawn(
                    _loop_worker,
                    args=(tmp + "/rendezvous", level),
                    nprocs=3,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
