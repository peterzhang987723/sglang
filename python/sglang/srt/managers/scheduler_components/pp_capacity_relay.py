"""Relays the first pipeline stage's capacity decisions to the later stages.

A capacity round is one iteration of a pipeline loop: from receiving the
iteration's requests through scheduling and result processing (whose cache
inserts take state slots and evict) to the iteration's end, where the
requests go on to the next stage. The first stage records every capacity
answer of the round (`StageCapacity`) and sends the record right after the
requests, on the same channel; each later stage receives it right after the
requests, replays it through its own round, and forwards it right after it
forwards the requests. Records and requests therefore stay in order on every
channel, and a later stage starts an iteration only once the first stage has
finished it.

With invariant checks on, the record also carries the round's outcome (the
batch launched, the queues, the decode token ratio and the evictions) and the
counts the first stage priced each stage at; each later stage checks both
against its own at the end of the same round.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.mem_cache.kv.alloc.invariants import STAGE_CAPACITY_AGREEMENT
from sglang.srt.mem_cache.kv.alloc.stage_capacity import (
    StageCapacityDivergence,
    StageCapacityRecord,
)
from sglang.srt.utils.invariants import expect_host, resolve_level


class PPCapacityRelay:
    def __init__(
        self,
        *,
        allocator,
        is_last_stage: bool,
        send: Callable[[List[Any]], List[Any]],
        recv: Callable[[], List[Any]],
        commit: Callable[[List[Any]], None],
    ):
        self.allocator = allocator
        self.capacity = allocator.stage_capacity
        self.is_last_stage = is_last_stage
        self._send = send
        self._recv = recv
        self._commit = commit
        self._send_work: List[Any] = []
        self._outgoing: Optional[StageCapacityRecord] = None
        self._incoming: Optional[StageCapacityRecord] = None
        self._checks = False
        self._tree_cache = None

    def round_begin(self, tree_cache) -> None:
        """Open a round once its requests are received: record on the first
        stage, else receive the first stage's record and replay it."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self._checks = resolve_level() >= InvariantCheckLevel.WARN
        self._tree_cache = None
        if self._checks and isinstance(tree_cache, UnifiedRadixCache):
            self._tree_cache = tree_cache
            tree_cache.eviction_log = []
        capacity = self.capacity
        if capacity.decides:
            capacity.begin()
            return
        received = self._recv()
        if len(received) != 1 or not isinstance(received[0], StageCapacityRecord):
            raise StageCapacityDivergence(
                f"stage {capacity.stage} expected the first stage's capacity "
                f"record after the round's requests, got {received!r}"
            )
        record = received[0]
        capacity.begin(record.decisions)
        self._incoming = record
        if not self.is_last_stage:
            self._outgoing = record

    def round_end(self, outcome: Callable[[], Any]) -> None:
        """Close the round at the iteration's end, before the requests go on.
        ``outcome`` describes the round's result on this stage."""
        capacity = self.capacity
        decisions = capacity.end()
        evicted = ()
        if self._tree_cache is not None:
            evicted = tuple(self._tree_cache.eviction_log)
            self._tree_cache.eviction_log = None
            self._tree_cache = None
        result = (outcome(), evicted) if self._checks else None
        if capacity.decides:
            if not self.is_last_stage:
                self._outgoing = StageCapacityRecord(
                    decisions=decisions,
                    outcome=result,
                    counts=(
                        self.allocator.simulated_stage_counts()
                        if self._checks
                        else None
                    ),
                )
            return
        record, self._incoming = self._incoming, None
        if not self._checks:
            return
        if record.outcome is not None:
            expect_host(
                STAGE_CAPACITY_AGREEMENT,
                result == record.outcome,
                msg=(
                    f"stage {capacity.stage} scheduled {result}, the first "
                    f"stage {record.outcome}"
                ),
            )
        if record.counts is not None:
            counts = self.allocator.stage_counts()
            expected = record.counts[capacity.stage]
            expect_host(
                STAGE_CAPACITY_AGREEMENT,
                counts == expected,
                msg=(
                    f"stage {capacity.stage} holds (sub-pool, live pages, free "
                    f"ids, units alone) {counts}; the first stage priced it at "
                    f"{expected}"
                ),
            )

    def relay(self) -> None:
        """Send the round's record to the next stage right after the round's
        requests."""
        record, self._outgoing = self._outgoing, None
        if record is None:
            return
        self._commit(self._send_work)
        self._send_work = self._send([record])
