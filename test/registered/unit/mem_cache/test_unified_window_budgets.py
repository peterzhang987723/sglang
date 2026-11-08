"""Every window of a unified pool is budgeted by the same rule.

A decode preallocation is priced against the scheduler's budget of the owner
and of every window sub-pool, by name: what a window's budget holds back for
running requests is charged beside the request, so a further window that
cannot hold its running requests' growth refuses the request even when the
target's window could.
"""

import unittest

from test_unified_window_sub_pools import _storage

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEveryWindowBudget(CustomTestCase):
    def _allocator(self):
        storage = _storage((16, 8))
        return storage.allocator

    def _fits(self, allocator, held_back):
        """A one-page request beside what each window's budget holds back
        (a pool sharing memory counts its budgets from zero)."""
        page = allocator.page_size
        names = allocator.window_pool_names()
        demand = KVDemand(page, dict.fromkeys(names, page), 0)
        return allocator.prealloc_fits(
            demand,
            evictable={},
            full_budget_tokens=0,
            window_budget_tokens={name: -tokens for name, tokens in held_back.items()},
        )

    def _room(self, allocator, window):
        """The most tokens ``window`` can hold back beside the request."""
        page = allocator.page_size
        names = allocator.window_pool_names()
        low, high = 0, allocator.size_swa // page + 1
        while low < high:
            mid = (low + high + 1) // 2
            windows = dict.fromkeys(names, page)
            windows[window] += mid * page
            if allocator.admits(KVDemand(page, windows, 0), evictable={}):
                low = mid
            else:
                high = mid - 1
        return low * page

    def test_a_further_window_holding_back_its_growth_refuses(self):
        allocator = self._allocator()
        target = allocator.target_window_pool()
        (further,) = [n for n in allocator.window_pool_names() if n != target]
        page = allocator.page_size
        room = self._room(allocator, further)
        self.assertGreater(room, 0)
        # Holding nothing back, or all the room it has, the further window
        # admits the request ...
        self.assertTrue(self._fits(allocator, {}))
        self.assertTrue(self._fits(allocator, {further: room}))
        # ... and holding back one page more for running requests' growth
        # refuses it, though the target's window could serve.
        self.assertFalse(self._fits(allocator, {further: room + page}))
        # A budget above zero holds nothing back.
        self.assertTrue(self._fits(allocator, {further: -page}))

    def test_each_window_is_charged_its_own_holdback(self):
        allocator = self._allocator()
        page = allocator.page_size
        names = allocator.window_pool_names()
        # Every window's holdback is charged beside the request in it.
        for held in (page, 3 * page, allocator.size_swa):
            self.assertEqual(
                self._fits(allocator, dict.fromkeys(names, held)),
                allocator.admits(
                    KVDemand(page, dict.fromkeys(names, page + held), 0),
                    evictable={},
                ),
            )


class TestSharedBytesServeEitherSide(CustomTestCase):
    """REGRESSION: a decode preallocation was refused when its owner tokens
    exceeded the owner's budget alone, though reclaiming cached window
    entries frees bytes the owner can take in a shared arena: with every row
    cached and nothing free, a request needing window bytes to become owner
    room was refused before any reclaim, every pass. One admission question
    decides it."""

    def test_window_entries_make_owner_room(self):
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
            setup_allocator_context,
        )

        setup_allocator_context()
        allocator = build_unified_bundle(
            UnifiedGeometry(
                chain="swa",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                page_size=4,
                full_tokens=256,
                swa_tokens=256,
                context_len=256,
            )
        ).token_to_kv_pool_allocator
        (window,) = allocator.window_pool_names()
        held = 0
        while allocator.alloc(4) is not None:
            held += 4
        owner = allocator.full_attn_allocator.sub_pool_name
        # Every row is a cached entry the prefix cache could evict.
        evictable = {owner: held, window: held}
        # More owner tokens than evicting the owner's own entries frees.
        demand = KVDemand(held + held // 2, {window: 4}, 0)
        self.assertTrue(allocator.admits(demand, evictable=evictable))
        # Budgets holding nothing back (counted from zero) admit it too.
        self.assertTrue(
            allocator.prealloc_fits(
                demand,
                evictable=evictable,
                full_budget_tokens=0,
                window_budget_tokens={window: 0},
            )
        )


if __name__ == "__main__":
    unittest.main()
