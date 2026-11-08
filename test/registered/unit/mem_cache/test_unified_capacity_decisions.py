"""Capacity decisions over a unified pool ask a demand, never a scalar.

Sub-pools of a unified pool share bytes, and a request's demand mixes
per-token units (tokens, window rows) with per-request ones (state slots), so
no single "units available" says whether it fits. The pool's scalar reads are
estimates for observers; reading one inside a capacity decision is a
violation the invariant checks catch. The prefill budget's token-count
questions (the ignore-EOS staged finish, priority preemption, dLLM block
sizes) price demands instead.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_prefill_memory_budget import _cache, _shared_allocator

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.mem_cache.capacity import capacity_decision, create_prefill_budget
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand, WindowRetirement
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_tri_pool,
    largest_take,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def setUpModule():
    patcher = patch(
        "sglang.srt.mem_cache.prefill_budget.window_retirement",
        return_value=WindowRetirement(0, 0, 1),
    )
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


@capacity_decision
def _decide(read):
    return read()


class TestEstimatesAreObserveOnly(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        _, self.allocator = build_tri_pool(page_size=4)
        owner, (window,), state = self.allocator.sub_pool_roles()
        self.reads = {
            "available_size": self.allocator.available_size,
            "owner_available_size": self.allocator.owner_available_size,
            "full_available_size": self.allocator.full_available_size,
            "window_available_size": lambda: self.allocator.window_available_size(
                window
            ),
            "window_capacity_and_available": (
                lambda: self.allocator.window_capacity_and_available(window)
            ),
            "usage": self.allocator.usage,
            "state slots": self.allocator.state_slots.schedulable_available_size,
        }

    def test_strict_checks_refuse_an_estimate_inside_a_decision(self):
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            for name, read in self.reads.items():
                with self.subTest(read=name):
                    read()  # an observer may read it
                    with self.assertRaisesRegex(AssertionError, "observe-only"):
                        _decide(read)

    def test_warn_logs_and_off_stays_silent(self):
        read = self.reads["full_available_size"]
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.WARN):
            with self.assertLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.OFF):
            with self.assertNoLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())

    def test_demand_questions_pass_inside_a_decision(self):
        allocator = self.allocator
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            self.assertTrue(
                _decide(
                    lambda: allocator.ensure_capacity(
                        KVDemand.same_tokens(allocator, 4, state_slots=1)
                    )
                )
            )
            self.assertTrue(_decide(lambda: allocator.state_slots.slots_fit(1)))

    def test_estimate_bounds_every_take_and_keeps_the_census_exact(self):
        allocator = self.allocator
        owner, (window,), state = allocator.sub_pool_roles()
        self.assertIsNotNone(allocator.alloc(12))
        self.assertIsNotNone(allocator.state_slots.alloc(2))
        for name in (owner, window, state):
            with self.subTest(sub_pool=name):
                # Bytes no live page holds bound what any take finds.
                self.assertGreaterEqual(
                    allocator.estimated_units_available(name),
                    largest_take(allocator, name),
                )
                capacity, free = allocator.usage()[name]
                self.assertEqual(capacity - free, allocator.held_units(name))
        self.assertEqual(allocator.held_units(owner), 12)
        self.assertEqual(allocator.held_units(state), 2)

    def test_state_slots_fit_is_the_take(self):
        slots = self.allocator.state_slots
        room = largest_take(self.allocator, self.allocator.sub_pool_roles()[2])
        self.assertTrue(slots.slots_fit(room))
        self.assertFalse(slots.slots_fit(room + 1))
        self.assertIsNotNone(slots.alloc(room))
        self.assertFalse(slots.slots_fit(1))


class TestPrefillBudgetPricesDemands(CustomTestCase):
    def setUp(self):
        self.allocator = _shared_allocator(pool_tokens=256)
        self.addCleanup(reset_context)
        self.budget = create_prefill_budget(self.allocator, _cache())
        self.request = dict(
            extend_input_len=12,
            max_new_tokens=4,
            chunk_limit=16,
            state_slots=0,
            cached_tokens=0,
        )

    def test_the_budget_has_no_remaining_token_count(self):
        self.assertTrue(self.budget.prices_by_demand)
        for name in ("remaining_total", "remaining_current", "remaining_window_tokens"):
            with self.subTest(read=name), self.assertRaises(TypeError):
                getattr(self.budget, name)

    def test_ignore_eos_prices_the_worst_stage(self):
        page = self.allocator.page_size
        # Running requests (tokens left, tokens held), fewest left first.
        states = [(4, 16), (8, 16), (20, 8)]
        # Stages: (4 + 1) * 3 = 15, then (8 + 1) * 2 - 16 = 2, then
        # (20 + 1) - 32 < 0; the first is the worst.
        peak = 15
        request = self.budget.request_demand(
            12, 4, chunk_limit=16, state_slots=0, cached_tokens=0
        )
        for paged_input in (16, 64, 4096):
            with self.subTest(paged_input=paged_input):
                self.assertEqual(
                    self.budget.ignore_eos_decode_fits(
                        states,
                        paged_input=paged_input,
                        reserve_tokens=1,
                        **self.request,
                    ),
                    self.budget._fits(
                        paged_input + peak,
                        request.window_tokens,
                        request.state_slots,
                        now=True,
                    ),
                )
        self.assertTrue(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=2 * page, reserve_tokens=1, **self.request
            )
        )
        self.assertFalse(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=1 << 20, reserve_tokens=1, **self.request
            )
        )

    def test_preemption_releases_the_fewest_candidates_that_admit(self):
        # Running requests' pending decode is what the budget holds back:
        # twice the pool between them.
        owner = self.allocator.sub_pool_roles()[0]
        released = [largest_take(self.allocator, owner) / 2] * 4
        self.budget.total_offset += sum(released)
        candidates = [SimpleNamespace(rid=i) for i in range(len(released))]
        request = dict(self.request, extra_tokens=0, total_tokens=16)

        def admitted(count):
            demand = self.budget.request_demand(
                12, 4, chunk_limit=16, state_slots=0, extra_tokens=0, cached_tokens=0
            )
            return self.budget._fits(
                demand.full_tokens - sum(released[:count]),
                demand.window_tokens,
                demand.state_slots,
            )

        count = self.budget.preemptions_needed(candidates, released, **request)
        expected = next((c for c in range(1, len(released) + 1) if admitted(c)), None)
        self.assertEqual(count, expected)
        # One release is not enough here, and all of them are.
        self.assertIsNotNone(count)
        self.assertGreater(count, 1)
        self.assertIsNone(self.budget.preemptions_needed([], [], **request))

    def test_largest_chunk_is_the_largest_admitted(self):
        def fits(tokens, now):
            return self.budget._chunk_fits(
                tokens,
                0,
                chunk_limit=None,
                host_hit_lengths=None,
                state_slots=0,
                now=now,
            )

        self.budget.reserve(12, 4, chunk_limit=16)
        for now in (False, True):
            with self.subTest(now=now):
                largest = self.budget.largest_chunk_tokens(4096, now=now)
                self.assertGreater(largest, 0)
                self.assertTrue(fits(largest, now))
                self.assertFalse(fits(largest + 1, now))


class TestPreemptionCreditsStateSlots(CustomTestCase):
    """Preempting a request gives its state slots back, which no token count
    stands for: a request short only of a state slot is admitted once one is
    given back."""

    def test_a_held_state_slot_is_credited(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        _, allocator = build_tri_pool(page_size=4)
        cache = _cache()
        cache.supports_mamba = lambda: False
        budget = create_prefill_budget(allocator, cache)
        self.assertTrue(budget.has_state_slots)
        candidates = [SimpleNamespace(rid=0), SimpleNamespace(rid=1)]
        request = dict(
            total_tokens=8,
            extend_input_len=4,
            max_new_tokens=4,
            chunk_limit=None,
            state_slots=1,
            extra_tokens=0,
            cached_tokens=0,
        )

        # A pool whose tokens all fit and whose state slots are all taken.
        def fits(full_tokens, window_tokens, state_slots=0, **_):
            return state_slots == 0

        with patch.object(budget, "_fits", side_effect=fits):
            with patch.object(type(budget), "request_held_state_slots", return_value=0):
                self.assertIsNone(
                    budget.preemptions_needed(candidates, [0.0, 0.0], **request)
                )
            with patch.object(type(budget), "request_held_state_slots", return_value=1):
                self.assertEqual(
                    budget.preemptions_needed(candidates, [0.0, 0.0], **request), 1
                )


if __name__ == "__main__":
    unittest.main()
