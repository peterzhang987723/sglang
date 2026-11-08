"""Bookkeeping of unified-memory PD transfers: each send or receive is its own
transfer; a page move that must wait for one waits on the host, releasing the
GIL, until every rank's engine is done with that one transfer, so a chunked
request never waits for its own later chunks; a request's transfers end at its
row's release point even when a session keeps the row; and a cleared room
leaves nothing behind in the status path."""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation import utils as utils_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    NetworkTransferWaiter,
    record_network_transfer,
)
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.kv.alloc import band as band_mod
from sglang.srt.mem_cache.kv.alloc import signals as signals_mod
from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer
from sglang.srt.mem_cache.kv.alloc.signals import TransferRelease, signals_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Manager(CommonKVManager):
    """The status path of a connection, without its sockets."""

    def __init__(self):
        self.request_status = {}
        self.deferred_bootstrap = None
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}


class TestSendCompletion(CustomTestCase):
    """What a page move waits for on a prefill node: one send's rows, until the
    rank's transfer worker is done with that send."""

    def _queued_send(self, manager, room):
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=room)
        peer = prefill._send_peer(sender)
        manager.note_send_enqueued(room)
        return peer

    def test_a_send_ends_when_the_worker_finishes_it_not_the_room(self):
        """A chunked request's rows a move needs are read by its earlier
        sends only; waiting for the whole room would wait for chunks the
        scheduler has yet to run. Each send ends when the worker is done with
        it, before the room concludes."""
        manager = _Manager()
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.Transferring)
        first = self._queued_send(manager, 7)
        second = self._queued_send(manager, 7)
        self.assertFalse(prefill._send_done(first))
        manager.note_send_finished(7)
        self.assertTrue(prefill._send_done(first))
        self.assertFalse(prefill._send_done(second))
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)

    def test_a_send_never_queued_waits_for_nothing(self):
        manager = _Manager()
        manager.update_status(8, KVPoll.Bootstrapping)
        skipped = prefill._send_peer(SimpleNamespace(kv_mgr=manager, bootstrap_room=8))
        self.assertTrue(prefill._send_done(skipped), "no worker reads its rows")
        self.assertTrue(prefill._send_done(None))

    def test_a_failed_or_cleared_room_waits_for_the_workers_report(self):
        """REGRESSION: a send of a room that failed, or was cleared, counted
        as done at once, so a page move or a reuse could take its rows while
        a transfer that timed out inside the engine still read them. The send
        waits for the worker's report, which reaches the record it was
        counted in after the room is gone."""
        manager = _Manager()
        manager.update_status(8, KVPoll.Bootstrapping)
        manager.update_status(8, KVPoll.Transferring)
        record = manager.send_record(8)
        queued = self._queued_send(manager, 8)
        manager.update_status(8, KVPoll.Failed)
        self.assertFalse(prefill._send_done(queued), "a failed room proves nothing")
        manager._forget_sends(8)  # the sender's clear()
        manager.request_status.pop(8)
        self.assertFalse(prefill._send_done(queued), "nor does a cleared one")
        manager.note_send_finished(8, record)
        self.assertTrue(prefill._send_done(queued))
        self.assertFalse(manager._send_records, "the report recreates nothing")

    def test_a_send_whose_submitted_transfer_failed_never_ends(self):
        """A transfer that failed or timed out after it was handed to the
        engine may still run, and nothing reports its end: its send is never
        done. The room's other sends end as the worker reports them."""
        manager = _Manager()
        manager.update_status(4, KVPoll.Bootstrapping)
        record = manager.send_record(4)
        first, second, third = (self._queued_send(manager, 4) for _ in range(3))
        manager.note_send_finished(4, record)
        manager.note_send_finished(4, record, settled=False)
        manager.note_send_finished(4, record)
        self.assertEqual(
            [prefill._send_done(p) for p in (first, second, third)],
            [True, False, True],
        )

    def test_the_send_waiter_spans_the_attention_tp_ranks_only(self):
        """REGRESSION: the waiter also MIN-reduced the attention CP group,
        whose ranks hold different rows: a rank whose ledger needed a wait
        entered the collective, and a CP peer whose ledger did not never did,
        so both hung. The attention TP ranks record the same rows and reach
        every wait together; the collective is theirs alone, as decode's."""
        scheduler = SimpleNamespace(
            attn_tp_cpu_group="attn_tp",
            attn_cp_cpu_group="attn_cp",
            feed_watchdog=lambda: None,
        )
        waiter = prefill._send_waiter(scheduler)
        self.assertEqual(waiter._groups, ["attn_tp"])

    def test_a_send_finished_after_its_room_was_cleared_leaves_no_record(self):
        """REGRESSION: the worker reports a send finished after it concluded
        the room, and the scheduler may clear the room in between. The report
        recreated the room's count, which nothing removed again."""
        manager = _Manager()
        manager.update_status(5, KVPoll.Bootstrapping)
        record = manager.send_record(5)
        manager.note_send_enqueued(5)
        manager._forget_sends(5)  # the sender's clear()
        manager.note_send_finished(5, record)
        self.assertFalse(manager._send_records)
        self.assertEqual(manager.send_counts(5), (0, 0))
        self.assertEqual((record.queued, record.finished), (1, 1))

    def test_a_late_report_does_not_finish_a_send_of_the_room_started_anew(self):
        """A send of the room's previous lifecycle that finishes late must not
        count as the new lifecycle's send: the engine still reads that one."""
        manager = _Manager()
        manager.update_status(6, KVPoll.Bootstrapping)
        old = manager.send_record(6)
        manager.note_send_enqueued(6)
        manager.request_status.pop(6)
        manager.update_status(6, KVPoll.Bootstrapping)
        manager.update_status(6, KVPoll.Transferring)
        queued = self._queued_send(manager, 6)
        manager.note_send_finished(6, old)
        self.assertFalse(prefill._send_done(queued))
        self.assertEqual(manager.send_counts(6), (1, 0))

    def test_a_new_room_lifecycle_forgets_the_last_ones_sends(self):
        manager = _Manager()
        manager.update_status(9, KVPoll.Bootstrapping)
        manager.note_send_enqueued(9)
        manager.note_send_finished(9)
        manager.request_status.pop(9)
        manager.update_status(9, KVPoll.Bootstrapping)
        self.assertEqual(manager.send_counts(9), (0, 0))


class TestNetworkTransferWaiter(CustomTestCase):
    def test_the_waiter_releases_the_gil_while_the_engine_reports(self):
        """REGRESSION: a move waited on the GPU for a flag a Python thread
        published; the scheduler then synchronized that stream holding the
        GIL, and the engine's thread could never publish. The waiter waits on
        the host and sleeps between rounds, so a thread that must take the GIL
        to report finishes."""
        done = threading.Event()
        reported = []

        def engine():
            time.sleep(0.02)
            reported.append(True)  # needs the GIL
            done.set()

        waiter = NetworkTransferWaiter(lambda peer: done.is_set(), (None,))
        thread = threading.Thread(target=engine)
        thread.start()
        waiter([SimpleNamespace(peer="send")])
        thread.join()
        self.assertEqual(reported, [True])

    def test_the_waiter_returns_once_every_transfer_is_done(self):
        rounds = defaultdict(int)

        def done(peer):
            rounds[peer] += 1
            return rounds[peer] >= (3 if peer == "slow" else 1)

        NetworkTransferWaiter(done, ())(
            [SimpleNamespace(peer="fast"), SimpleNamespace(peer="slow")]
        )
        self.assertEqual(rounds["slow"], 3)

    def test_each_rank_reduces_one_flag_however_it_lists_the_transfers(self):
        """Ranks may list the same transfers in another order or more than
        once; the collective must not depend on that: one done flag and one
        deadline flag per round."""
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            return flags

        waiter = NetworkTransferWaiter(lambda peer: True, (object(),))
        a, b = SimpleNamespace(peer="a"), SimpleNamespace(peer="b")
        with mock.patch.object(utils_mod, "all_reduce_flags", reduce):
            for listed in ([a, b, a], [b, a]):
                waiter(listed)
        self.assertEqual(reduced, [[1, 1], [1, 1]])


def _agreeing_rank(rank, path, out):
    """One rank of a two-rank group: rank 1's engine finishes later. Each rank
    records how many of its own rounds said done before the waiter returned."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2
    )
    try:
        start = time.monotonic()
        delay = 0.0 if rank == 0 else 0.2
        rounds = []

        def done(peer):
            rounds.append(time.monotonic() - start >= delay)
            return rounds[-1]

        NetworkTransferWaiter(done, (dist.group.WORLD,))([SimpleNamespace(peer=0)])
        out.put((rank, len(rounds), sum(rounds), time.monotonic() - start))
    finally:
        dist.destroy_process_group()


class TestWaiterAgreesAcrossRanks(CustomTestCase):
    def test_every_rank_returns_in_the_same_round(self):
        """A rank whose engine is done keeps waiting until every rank's is:
        the ranks keep one layout, so they must leave a wait together."""
        import multiprocessing as mp
        import os
        import tempfile

        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store")
            procs = [
                ctx.Process(target=_agreeing_rank, args=(rank, path, out))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            results = sorted(out.get(timeout=60) for _ in procs)
            for proc in procs:
                proc.join(timeout=60)
        (_, rounds0, done0, _), (_, rounds1, done1, _) = results
        self.assertEqual(rounds0, rounds1, "the ranks left after different rounds")
        self.assertEqual(done0, rounds0)
        self.assertGreater(rounds0, 1, "rank 0 left before rank 1 was done")
        self.assertEqual(done1, 1, "rank 1 left on its first done round")


class _Clock:
    """A clock that a sleep advances; it stands in for the waiter's `time`."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestWaiterEndsWithTheProtocol(CustomTestCase):
    """REGRESSION: the wait had a deadline of its own, shorter than a
    receive's, and killed the scheduler while a healthy receive was still
    within its waiting timeout. The watchdog is fed while the wait lasts, and
    its own deadline lies past the protocol's: it fails only a transfer whose
    end is never proven (a dead peer's, or one that timed out inside the
    engine), together on every rank and with no heartbeat after."""

    def test_a_receive_finishing_after_180_s_within_a_300_s_timeout(self):
        clock = _Clock()
        start = clock.now
        beats = []
        done = lambda peer: clock.now - start >= 180.0  # noqa: E731
        waiter = NetworkTransferWaiter(
            done,
            (),
            heartbeat=lambda: beats.append(clock.now),
            interval=1.0,
        )
        with (
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(300),
            mock.patch.object(utils_mod, "time", clock),
            self.assertLogs(utils_mod.logger, level="WARNING") as logs,
        ):
            waiter([SimpleNamespace(peer="receiving")])
        self.assertGreaterEqual(clock.now - start, 180.0)
        # Periodic warnings while it waits.
        self.assertGreaterEqual(len(logs.output), 17)
        # The watchdog sees a heartbeat every round, never a silent stretch.
        self.assertEqual(beats[0], start)
        self.assertLessEqual(max(b - a for a, b in zip(beats, beats[1:])), 1.0)
        self.assertGreaterEqual(beats[-1] - start, 179.0)

    def test_the_default_deadline_lies_past_the_protocols(self):
        waiter = NetworkTransferWaiter(lambda peer: True, ())
        self.assertGreater(waiter._deadline, 300 + 300 + 30)

    def test_a_transfer_that_never_ends_fails_at_the_deadline(self):
        clock = _Clock()
        start = clock.now
        beats = []
        waiter = NetworkTransferWaiter(
            lambda peer: False,
            (),
            heartbeat=lambda: beats.append(clock.now),
            interval=1.0,
            deadline=50.0,
        )
        sender = SimpleNamespace(bootstrap_room=7)
        with (
            mock.patch.object(utils_mod, "time", clock),
            self.assertRaisesRegex(RuntimeError, r"deadline.*room 7"),
        ):
            waiter([SimpleNamespace(peer=(sender, 0))])
        self.assertGreaterEqual(clock.now - start, 50.0)
        # No heartbeat once the deadline passed: the watchdog sees the stall.
        self.assertLess(max(beats) - start, 50.0)

    def test_ranks_give_up_in_the_same_round(self):
        """The deadline is reduced with the done flag: a rank still within
        its own deadline raises in the round another rank's passed."""
        clock = _Clock()
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            # The other rank is past its deadline.
            return [flags[0], 0]

        waiter = NetworkTransferWaiter(
            lambda peer: False, ("tp",), interval=1.0, deadline=1e9
        )
        with (
            mock.patch.object(utils_mod, "time", clock),
            mock.patch.object(utils_mod, "all_reduce_flags", reduce),
            self.assertRaises(RuntimeError),
        ):
            waiter([SimpleNamespace(peer="sending")])
        self.assertEqual(reduced, [[0, 1]])

    def test_a_failed_transfer_worker_ends_the_wait(self):
        """REGRESSION: a Mooncake transfer worker that raised (a TCP aux send
        timing out) left its room waiting with a send queued and unfinished,
        so a page move waited for it forever. The worker fails its room and
        reports the send finished; raising before it handed the engine
        anything, nothing it submitted still runs, and the wait returns."""
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = _Manager()
        manager.update_status(9, KVPoll.Bootstrapping)
        manager.update_status(9, KVPoll.WaitingForInput)
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=9)
        peer = prefill._send_peer(sender)
        manager.note_send_enqueued(9)
        self.assertFalse(prefill._send_done(peer))

        class _OneChunk:
            def get(self):
                return SimpleNamespace(room=9)

        # A worker whose send raises (here: whatever it reads first is
        # missing, as a timed-out aux send raises out of the same body).
        worker = SimpleNamespace(
            enable_trace=False,
            poison_deferred_ack_room=lambda room: None,
            record_failure=lambda room, reason: None,
            update_status=manager.update_status,
            note_send_finished=manager.note_send_finished,
            bootstrap_port=0,
        )
        with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
            MooncakeKVManager.transfer_worker(worker, _OneChunk(), None)
        self.assertEqual(manager.request_status[9], KVPoll.Failed)
        self.assertTrue(prefill._send_done(peer))
        NetworkTransferWaiter(prefill._send_done, (), interval=0.0)(
            [SimpleNamespace(peer=peer)]
        )

    def test_the_scheduler_watchdog_counts_heartbeats(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components import invariant_checker

        scheduler = SimpleNamespace(
            forward_ct=7,
            watchdog_heartbeats=0,
            is_initializing=False,
            cur_batch_for_debug=object(),
            dump_requests_before_crash=lambda: None,
        )
        with mock.patch.object(invariant_checker, "WatchdogRaw", SimpleNamespace):
            watchdog = invariant_checker.create_scheduler_watchdog(
                scheduler, watchdog_timeout=300.0
            )
        before = watchdog.get_counter()
        Scheduler.feed_watchdog(scheduler)
        self.assertNotEqual(watchdog.get_counter(), before)


class _HostCopy:
    """A host-copy completion event whose firing the test controls."""

    def __init__(self, fired):
        self.fired = fired

    def query(self):
        return self.fired


class TestRankSkewedWaitAfterCompaction(CustomTestCase):
    """REGRESSION: a network transfer was listed once per fence that held it,
    and the waiter reduced one flag per entry. Fences group the pages a move
    vacates by every reader, host copies included, and a host copy's end is
    seen at different times per rank: one rank fenced two pages apart (one is
    still read by a backup), another together, so a later take over both
    listed the send twice on one rank and once on the other, and the ranks
    entered collectives of different lengths."""

    def _rank(self, backup_done):
        """One rank: a send reads the two pages nearest the frontier and a
        host backup reads one of them; an urgent compaction moves both into
        interior holes, and a take of every page reuses both (the span keeps
        their bytes, holes a take reaches last). Returns what the rank passed
        to the collective."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="swa",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                lazy=True,
                device="cpu",
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        storage = allocator.storage
        full = allocator.full_attn_allocator
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            return flags

        def wait(key, stream=None, *, network_waiter=None):
            # The network half as written; no stream on the CPU to wait an
            # unfinished backup on.
            transfers = tuple(
                s for s in signals_of(key) if isinstance(s, TransferRelease)
            )
            signals_mod.wait_signals(transfers, network_waiter=network_waiter)

        storage.set_network_waiter(
            NetworkTransferWaiter(lambda peer: True, (object(),))
        )
        ids = allocator.alloc(12)
        self.assertIsNotNone(ids)
        phys = full.ids.virtual_to_physical[ids].tolist()
        toward_frontier = sorted(
            range(len(ids)),
            key=lambda i: phys[i],
            reverse=full.grow_direction == "up",
        )
        nearest, interior = toward_frontier[:2], toward_frontier[5:7]
        full.free(ids[interior])
        sent = sorted(phys[i] for i in nearest)
        storage.record_network_transfer(
            FakeReq(rid="r", bootstrap_room=1),
            pages=np.array(sent),
            page_size=1,
            writes=False,
            peer="send",
        )
        full.record_transfer(_HostCopy(backup_done), ids[nearest[:1]], writes=False)
        with (
            mock.patch.object(utils_mod, "all_reduce_flags", reduce),
            mock.patch.object(band_mod, "wait_signals", wait),
        ):
            self.assertEqual(full.flush_for_allocation(), 2)
            self.assertEqual(reduced, [], "the move itself waits for no reader")
            pages = full.acquire(
                full.available_size() // full.page_size, Consumer.FORWARD
            )
        self.assertTrue(set(sent) <= set(pages.tolist()), "the take reuses both")
        return reduced

    def test_ranks_whose_backup_ended_apart_enter_the_same_collective(self):
        pending, ended = self._rank(backup_done=False), self._rank(backup_done=True)
        self.assertEqual(pending, ended)
        self.assertEqual(ended, [[1, 1]])


class _Receiver(CommonKVReceiver):
    """A receiver whose transfer reports ``status``, notifying the abort
    through the real path into a socket that records what it sends."""

    def __init__(self, manager, room, n_peers):
        self.kv_mgr = manager
        self.bootstrap_room = room
        self.status = KVPoll.Transferring
        self.require_staging = False
        self.init_time = None  # as after a send that failed partway
        self.abort_notified = False
        self._abort_generation = None
        self.bootstrap_infos = [{"rank": r} for r in range(n_peers)]
        self.sent = []
        sock = SimpleNamespace(send_multipart=self.sent.append)
        self._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())

    def poll(self):
        return self.status

    def failure_exception(self):
        raise NotImplementedError


class TestReceiveCompletion(CustomTestCase):
    """What a page move waits for on a decode node: a receive, until it
    succeeded, or failed and every prefill rank that could write it
    acknowledged its drain. A timeout proves nothing."""

    ROOM = 3

    def _queue(self, *, deferred=True, timeout=30.0):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = False
        queue.staging_handler = None
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = deferred
        queue.deferred_kv_release_timeout = timeout
        queue._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda x: x[0].req
        )
        queue._undrained_receives = []
        return queue

    def _decode_req(self, *, n_peers=2, deferred=True):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager._deferred_abort_ack_tracker = {}
        manager._deferred_abort_generation = 0
        manager.enable_deferred_decode_kv_release = deferred
        manager.local_ip, manager.rank_port = "10.0.0.1", 6000
        receiver = _Receiver(manager, self.ROOM, n_peers)
        return DecodeRequest(
            req=FakeReq(rid="r", bootstrap_room=self.ROOM), kv_receiver=receiver
        )

    def _ack(self, decode_req, prefill_rank):
        receiver = decode_req.kv_receiver
        receiver.kv_mgr.note_abort_ack(
            self.ROOM, prefill_rank, receiver._abort_generation
        )

    def test_a_receive_ends_at_success_not_while_transferring(self):
        queue = self._queue()
        decode_req = self._decode_req()
        self.assertFalse(queue._receive_done(decode_req))
        decode_req.kv_receiver.status = KVPoll.Success
        self.assertTrue(queue._receive_done(decode_req))
        self.assertTrue(queue._receive_done(SimpleNamespace(kv_receiver=None)))

    def test_a_published_failure_decode_has_not_aborted_waits_for_every_ack(self):
        """REGRESSION: a failure the prefill reported, seen by a page move
        before the queue notified the abort, owed no acknowledgement (the
        count followed the notification), so the move took rows a prefill
        rank could still be writing. A destination sent to the peers owes one
        from every peer: the move notifies the abort and waits for them."""
        queue = self._queue()
        decode_req = self._decode_req(n_peers=2)
        decode_req.kv_receiver.status = KVPoll.Failed
        self.assertFalse(queue._receive_done(decode_req))
        receiver = decode_req.kv_receiver
        self.assertTrue(receiver.abort_notified)
        self.assertEqual(len(receiver.sent), 2, "every peer is told of the abort")
        self._ack(decode_req, 0)
        self.assertFalse(queue._receive_done(decode_req))
        self._ack(decode_req, 1)
        self.assertTrue(queue._receive_done(decode_req))
        self.assertEqual(len(receiver.sent), 2, "the abort is notified once")

    def test_an_unacknowledged_failure_never_ends_at_its_deadline(self):
        """REGRESSION: a failed receive counted as done once its release
        timeout passed, so a page move took rows a prefill whose transfer
        timed out inside its engine could still write. The timeout runs from
        the first sight of the failure, a page move's or the queue's, and a
        later hold keeps it and its own acknowledgement count; past it the
        receive still waits for every acknowledgement."""
        queue = self._queue(timeout=10.0)
        decode_req = self._decode_req(n_peers=2)
        decode_req.kv_receiver.status = KVPoll.Failed
        now = [100.0]
        with mock.patch.object(decode_mod.time, "monotonic", lambda: now[0]):
            self.assertFalse(queue._receive_done(decode_req))
            now[0] = 104.0
            queue._defer_release(decode_req)
            ((_, start, deadline, _, required_acks),) = queue._deferred_releases.items()
            self.assertEqual((start, deadline, required_acks), (100.0, 110.0, 2))
            now[0] = 1e6
            self.assertFalse(queue._receive_done(decode_req))
            self._ack(decode_req, 0)
            self.assertFalse(queue._receive_done(decode_req))
            self._ack(decode_req, 1)
            self.assertTrue(queue._receive_done(decode_req))

    def test_a_failure_never_published_is_done_at_once(self):
        """No peer has the addresses of a destination never sent: nothing to
        wait for, and nobody to notify."""
        queue = self._queue()
        for field, value in (
            ("awaiting_publish", True),
            ("pending_metadata", (None, (), {})),
        ):
            with self.subTest(field=field):
                decode_req = self._decode_req()
                setattr(decode_req, field, value)
                decode_req.kv_receiver.status = KVPoll.Failed
                self.assertTrue(queue._receive_done(decode_req))
                self.assertEqual(decode_req.kv_receiver.sent, [])

    def test_without_deferred_release_a_published_failure_never_ends(self):
        """REGRESSION: with deferred release off, a failed receive counted as
        done at once, though no peer had acknowledged anything. No peer
        acknowledges a drain then, so the receive never ends; unified memory
        refuses that configuration at startup."""
        for queue_flag, manager_flag in ((False, True), (True, False)):
            with self.subTest(queue=queue_flag, manager=manager_flag):
                queue = self._queue(deferred=queue_flag)
                decode_req = self._decode_req(deferred=manager_flag)
                decode_req.kv_receiver.status = KVPoll.Failed
                self.assertFalse(queue._receive_done(decode_req))


class TestTransferBookkeeping(CustomTestCase):
    def test_a_one_slot_state_entry_is_recorded_as_rows(self):
        """REGRESSION: a recurrent state's transfer payload is its one slot, a
        0-d array, and the page-move ledger took its length, so a unified PD
        prefill of a hybrid model raised on its first finished request."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="tri",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                state_layer_ids=(2,),
                lazy=True,
                device="cpu",
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        storage = allocator.storage
        entries = storage.wire_entries()
        state_name = allocator.roles.state
        self.assertIn(state_name, [entry.sub_pool for entry in entries])
        slot = np.array(3, dtype=np.int64)
        self.assertEqual(slot.ndim, 0)
        req = SimpleNamespace(rid="r1", bootstrap_room=2)
        record_network_transfer(
            storage,
            req,
            peer=None,
            writes=False,
            page_indices=np.array([], dtype=np.int32),
            page_size=allocator.page_size,
            state_types=[entry.state_type for entry in entries],
            state_indices=[
                [slot] if entry.sub_pool == state_name else None for entry in entries
            ],
        )
        (use,) = allocator.mamba_allocator.usage.transfers()
        self.assertEqual(np.asarray(use.ids).tolist(), [3])


def _tri_pool():
    setup_allocator_context()
    allocator = build_unified_bundle(
        UnifiedGeometry(
            chain="tri",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            state_layer_ids=(2,),
            lazy=True,
            device="cpu",
        )
    ).token_to_kv_pool_allocator
    return allocator, allocator.storage


def _physical_pages(band, ids) -> list:
    pages = band.ids.virtual_to_physical[ids // band.page_size] * band.page_size
    return (pages // band.pool_page_size).tolist()


def _record(allocator, req, pages, *, writes, peer):
    record_network_transfer(
        allocator.storage,
        req,
        peer=peer,
        writes=writes,
        page_indices=np.array(pages, dtype=np.int32),
        page_size=allocator.page_size,
    )


class _Blocked:
    """Runs ``fn`` on a thread, so a test can see it wait and then end."""

    def __init__(self, fn):
        self.result = []
        self.thread = threading.Thread(
            target=lambda: self.result.append(fn()), daemon=True
        )
        self.thread.start()

    def still_waiting(self, seconds=0.3) -> bool:
        self.thread.join(seconds)
        return self.thread.is_alive()

    def finish(self):
        self.thread.join(30)
        assert not self.thread.is_alive(), "the wait never ended"
        return self.result[0]


class TestFailedTransferRowsWaitForTheirProof(CustomTestCase):
    """A failed transfer's rows are reused, or moved over, only once the
    engine is proven done with them: a failure or a timeout aborts the
    request but proves nothing."""

    ROOM = TestReceiveCompletion.ROOM

    def test_a_failed_send_still_read_keeps_its_old_row_until_the_report(self):
        """REGRESSION: a send of a room that failed, or was cleared, counted
        as done at once, so an allocation's flush reused the row a moved page
        vacated while a transfer that timed out inside the engine still read
        it. The flush waits for the worker's report."""
        allocator, storage = _tri_pool()
        storage.set_network_waiter(
            NetworkTransferWaiter(prefill._send_done, (), interval=0.001)
        )
        owner = allocator.full_attn_allocator
        page = allocator.page_size
        ids = owner.alloc(8 * page)
        # The page nearest the frontier is the first a compaction moves.
        top = _physical_pages(owner, ids[:1])
        manager = _Manager()
        manager.update_status(1, KVPoll.Bootstrapping)
        manager.update_status(1, KVPoll.Transferring)
        record = manager.send_record(1)
        peer = prefill._send_peer(SimpleNamespace(kv_mgr=manager, bootstrap_room=1))
        manager.note_send_enqueued(1)
        _record(allocator, SimpleNamespace(rid="r1"), top, writes=False, peer=peer)
        # The engine's batch timed out and keeps reading; the room fails and
        # its sender is cleared.
        manager.update_status(1, KVPoll.Failed)
        manager._forget_sends(1)
        owner.free(ids[2 * page : 6 * page])  # interior holes
        self.assertGreater(owner.flush_opportunistic(), 0)
        self.assertIn(top[0], owner.usage.pending_pages)
        flush = _Blocked(owner.flush_for_allocation)
        self.assertTrue(flush.still_waiting(), "it reused the row still read")
        self.assertNotIn(top[0], owner._free_phys_pages.tolist())
        manager.note_send_finished(1, record)  # the worker's late report
        flush.finish()
        self.assertNotIn(top[0], owner.usage.pending_pages)

    def test_a_receive_written_past_its_timeout_keeps_its_row_until_drained(self):
        """REGRESSION: a failed receive counted as done once its release
        timeout passed, so an allocation's flush moved its row while a
        prefill whose transfer timed out inside its engine could still write
        it, and the late write landed in the bytes another page then took.
        The flush waits for every prefill's drain acknowledgement; until
        then the row stays the failed request's own, where a late write lands
        harmlessly."""
        allocator, storage = _tri_pool()
        queue = TestReceiveCompletion._queue(self, timeout=10.0)
        decode_req = TestReceiveCompletion._decode_req(self, n_peers=1)
        storage.set_network_waiter(
            NetworkTransferWaiter(queue._receive_done, (), interval=0.001)
        )
        owner = allocator.full_attn_allocator
        page = allocator.page_size
        ids = owner.alloc(8 * page)
        top = _physical_pages(owner, ids[:1])
        _record(allocator, decode_req.req, top, writes=True, peer=decode_req)
        owner.free(ids[2 * page : 6 * page])  # interior holes
        decode_req.kv_receiver.status = KVPoll.Failed
        now = [100.0]
        with mock.patch.object(decode_mod.time, "monotonic", lambda: now[0]):
            self.assertFalse(queue._receive_done(decode_req))
            queue._defer_release(decode_req)
            now[0] = 1e6  # far past the release timeout
            self.assertEqual(owner.flush_opportunistic(), 0)
            flush = _Blocked(owner.flush_for_allocation)
            self.assertTrue(flush.still_waiting(), "it moved the row still written")
            self.assertEqual(_physical_pages(owner, ids[:1]), top)
            TestReceiveCompletion._ack(self, decode_req, 0)
            self.assertGreater(flush.finish(), 0)
        self.assertNotEqual(_physical_pages(owner, ids[:1]), top)


class _Orphan:
    """A request with a row whose transfer of ``pages`` pages is recorded and
    unproven; ``proof`` is what the engine reports."""

    def __init__(self, allocator, *, pages=4, writes=True, rid="r", room=7):
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        self.allocator, self.storage = allocator, allocator.storage
        self.owner = allocator.full_attn_allocator
        self.proof = {"done": False}
        self.req = SimpleNamespace(rid=rid, bootstrap_room=room, kv=ReqKvInfo())
        assert self.storage.acquire([self.req]) is not None
        self.ids = self.owner.alloc(pages * allocator.page_size)
        self.pages = set(_physical_pages(self.owner, self.ids))
        _record(allocator, self.req, sorted(self.pages), writes=writes, peer=self.proof)

    def give_back(self):
        """The transfer failed: the request is given back at once."""
        self.owner.free(self.ids)
        self.storage.release_row(self.req)


def _reaper(storage, *, quarantine_after=120.0):
    return utils_mod.OrphanedTransferReaper(
        storage, lambda peer: peer["done"], (), quarantine_after=quarantine_after
    )


class TestTheLedgerFencesAFailedTransfersPages(CustomTestCase):
    """A failed transfer's request is given back at once; the page ledger
    alone keeps the bytes the transfer may still touch out of reuse until its
    end is proven, and QUARANTINES them past a bound."""

    def test_freed_pages_of_a_gone_receive_wait_for_the_drain(self):
        """REGRESSION: a page freed through the normal free path while its
        receive was unreleased joined the free list as a plain hole (only
        vacated fences ordered a take), so the next take handed it out under
        a late write. The request is gone; its pages are taken last, and only
        after the drain."""
        allocator, storage = _tri_pool()
        orphan = _Orphan(allocator)
        page = allocator.page_size
        storage.set_network_waiter(
            NetworkTransferWaiter(lambda peer: peer["done"], (), interval=0.001)
        )
        orphan.give_back()
        self.assertFalse(storage._network_transfers, "the request is gone")
        owner = orphan.owner
        # Every other page is taken first, without waiting.
        others = owner.alloc(owner.available_size() - len(orphan.pages) * page)
        self.assertTrue(
            orphan.pages.isdisjoint(_physical_pages(owner, others)),
            "a page a late write may reach was handed out",
        )
        (record,) = storage.orphaned_transfers()
        self.assertEqual(record.page_count, len(orphan.pages))
        # A take that needs them waits for the drain, then gets them.
        take = _Blocked(lambda: owner.alloc(page))
        self.assertTrue(take.still_waiting(), "it reused a page still written")
        orphan.proof["done"] = True
        taken = take.finish()
        self.assertTrue(orphan.pages.issuperset(_physical_pages(owner, taken)))

    def test_the_reaper_releases_on_proof_and_quarantines_past_the_bound(self):
        """Unproven past the bound, the pages leave capacity, stay in place
        and are counted in the pool stats, logged once with the room and the
        page count; a late proof gives them back."""
        allocator, storage = _tri_pool()
        orphan = _Orphan(allocator, room=9)
        orphan.give_back()
        owner = orphan.owner
        available = owner.available_size()
        reaper = _reaper(storage, quarantine_after=60.0)
        clock = _Clock()
        with mock.patch.object(utils_mod, "time", clock):
            for o in storage._orphaned:
                o.since = clock.now
            clock.now += 59.0
            self.assertEqual(reaper.resolve(), [])
            self.assertFalse(storage.orphaned_transfers()[0].quarantined)
            clock.now += 1.0
            with self.assertLogs(utils_mod.logger, level="ERROR") as logs:
                reaper.resolve()
            reaper.resolve()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("room 9", logs.output[0])
        self.assertIn(f"{len(orphan.pages)} pages", logs.output[0])
        quarantined = len(orphan.pages) * allocator.page_size
        self.assertEqual(owner.available_size(), available - quarantined)
        usage = {u.name: u for u in storage.usage()}
        self.assertEqual(usage[allocator.roles.owner].quarantined, quarantined)
        self.assertTrue(owner.moves_blocked(), "never moved")
        taken = owner.alloc(owner.available_size())
        self.assertTrue(orphan.pages.isdisjoint(_physical_pages(owner, taken)))
        owner.free(taken)

        orphan.proof["done"] = True
        with self.assertLogs(utils_mod.logger, level="WARNING"):
            self.assertEqual(reaper.resolve(), [orphan.proof])
        self.assertEqual(storage.orphaned_transfers(), [])
        self.assertFalse(owner.moves_blocked())
        owner.flush_opportunistic()
        self.assertEqual(owner.available_size(), available)
        self.assertEqual(
            {u.name: u for u in storage.usage()}[allocator.roles.owner].quarantined, 0
        )

    def test_ranks_quarantine_together(self):
        """One rank past the bound quarantines on every rank, in the round
        that reduces the proofs."""
        allocator, storage = _tri_pool()
        _Orphan(allocator).give_back()

        def reduce(flags, group, op):
            return [flags[0], 0]  # another rank is past the bound

        reaper = utils_mod.OrphanedTransferReaper(
            storage, lambda peer: False, ("tp",), quarantine_after=1e9
        )
        with (
            mock.patch.object(utils_mod, "all_reduce_flags", reduce),
            self.assertLogs(utils_mod.logger, level="ERROR"),
        ):
            reaper.resolve()
        self.assertTrue(storage.orphaned_transfers()[0].quarantined)

    def test_a_pool_clear_hands_out_no_fenced_or_quarantined_page(self):
        """A cache flush clears the pools while a dead peer's transfers are
        unproven: every other page comes back, theirs do not."""
        for quarantine in (False, True):
            with self.subTest(quarantine=quarantine):
                allocator, storage = _tri_pool()
                total = allocator.available_size()
                orphan = _Orphan(allocator)
                owner = orphan.owner
                # Pages the cache keeps, which the clear gives back.
                owner.alloc(4 * allocator.page_size)
                orphan.give_back()
                if quarantine:
                    with self.assertLogs(utils_mod.logger, level="ERROR"):
                        _reaper(storage, quarantine_after=0.0).resolve()
                storage.requests.clear()
                allocator.clear()
                free = owner.available_size()
                taken = owner.alloc(
                    free
                    - (0 if quarantine else len(orphan.pages) * allocator.page_size)
                )
                self.assertTrue(orphan.pages.isdisjoint(_physical_pages(owner, taken)))
                owner.free(taken)
                orphan.proof["done"] = True
                _reaper(storage).resolve()
                owner.flush_opportunistic()
                allocator.clear()
                self.assertEqual(allocator.available_size(), total)

    def test_a_send_row_moved_past_the_frontier_is_quarantined_without_a_crash(
        self,
    ):
        """REGRESSION: an urgent flush moved a row an in-flight send still
        read and retreated the frontier past the bytes it vacated; once the
        send failed unproven, the quarantine fixed those bytes outside the
        span, and the next growth over them raised "a fixed page cannot be
        written". The span keeps fenced bytes inside it, so growth skips
        them and the quarantine holds them in place."""
        allocator, storage = _tri_pool()
        storage.set_network_waiter(lambda transfers: None)
        owner = allocator.full_attn_allocator
        page = allocator.page_size
        proof = {"done": False}
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        req = SimpleNamespace(rid="s", bootstrap_room=11, kv=ReqKvInfo())
        self.assertIsNotNone(storage.acquire([req]))
        ids = owner.alloc(8 * page)
        # The page nearest the frontier is the first a compaction moves.
        top = set(_physical_pages(owner, ids[:page]))
        _record(allocator, req, sorted(top), writes=False, peer=proof)
        owner.free(ids[2 * page : 6 * page])  # interior holes
        self.assertGreater(owner.flush_for_allocation(), 0)
        self.assertNotEqual(set(_physical_pages(owner, ids[:page])), top, "it moved")
        first, end = owner._span_range()
        self.assertTrue(all(first <= p < end for p in top), "kept in the span")
        # The send fails unproven; its request is gone; past the bound.
        owner.free(torch.cat([ids[: 2 * page], ids[6 * page :]]))
        storage.release_row(req)
        with self.assertLogs(utils_mod.logger, level="ERROR"):
            _reaper(storage, quarantine_after=0.0).resolve()
        taken = owner.alloc(owner.available_size())
        self.assertIsNotNone(taken)
        self.assertTrue(top.isdisjoint(_physical_pages(owner, taken)))

    def test_a_pool_clear_without_transfers_resets(self):
        allocator, storage = _tri_pool()
        total = allocator.available_size()
        allocator.full_attn_allocator.alloc(8 * allocator.page_size)
        allocator.clear()
        self.assertEqual(allocator.available_size(), total)


class TestAFailedReceiveIsGivenBackAtOnce(CustomTestCase):
    """On decode a failed receive's request is given back at once; its rows
    stay fenced, and its receiver and metadata slot kept, until every peer
    acknowledges the drain."""

    ROOM = TestReceiveCompletion.ROOM

    def test_the_request_goes_and_its_rows_wait_for_the_acknowledgements(self):
        allocator, storage = _tri_pool()
        queue = TestReceiveCompletion._queue(self)
        decode_req = TestReceiveCompletion._decode_req(self, n_peers=1)
        decode_req.metadata_buffer_index = 5
        decode_req.kv_receiver.status = KVPoll.Failed
        queue.fences_failed_receives = True
        queue.tree_cache = SimpleNamespace(
            req_to_token_pool=storage.requests, token_to_kv_pool_allocator=allocator
        )
        queue.transfer_reaper = utils_mod.OrphanedTransferReaper(
            storage, queue._receive_done, (), quarantine_after=120.0
        )
        owner = allocator.full_attn_allocator
        req = decode_req.req
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        req.kv = ReqKvInfo()
        self.assertIsNotNone(storage.acquire([req]))
        ids = owner.alloc(4 * allocator.page_size)
        pages = set(_physical_pages(owner, ids))
        _record(allocator, req, sorted(pages), writes=True, peer=decode_req)
        released, finished = [], []

        def release(r, *, checkpoint):
            released.append(r)
            owner.free(ids)
            storage.release_row(r)

        queue.scheduler = SimpleNamespace(
            request_retirer=SimpleNamespace(release=release)
        )
        queue._finish_release = lambda d, idx: finished.append((d, idx))
        self.assertFalse(queue._receive_done(decode_req))  # notifies the abort
        queue._release_undrained(decode_req)
        self.assertEqual(released, [req], "the request is gone at once")
        self.assertEqual(len(storage.orphaned_transfers()), 1)
        others = owner.alloc(owner.available_size() - len(pages) * allocator.page_size)
        self.assertTrue(pages.isdisjoint(_physical_pages(owner, others)))
        queue._resolve_undrained_receives()
        self.assertEqual(finished, [])
        TestReceiveCompletion._ack(self, decode_req, 0)
        queue._resolve_undrained_receives()
        self.assertEqual(finished, [(decode_req, 5)])
        self.assertEqual(storage.orphaned_transfers(), [])
        self.assertEqual(queue._undrained_receives, [])


class TestMetadataSlotsHeldByUndrainedReceives(CustomTestCase):
    """A failed receive given back with its drain unproven keeps its metadata
    slot: a late write may reach it. Dead peers can exhaust the slots; new
    receives then wait, and the cause is logged once and counted."""

    def test_exhaustion_is_logged_once_and_counted(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda x: x[0].req
        )
        transfer._undrained_receives = [object(), object()]
        self.assertEqual(transfer.num_pending_deferred_releases(), 2)
        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.transfer_queue = transfer
        prealloc.req_to_metadata_buffer_idx_allocator = SimpleNamespace(size=4)
        with self.assertLogs(decode_mod.logger, level="ERROR") as logs:
            prealloc._report_metadata_exhaustion()
            prealloc._report_metadata_exhaustion()
            decode_mod.logger.error("sentinel")
        self.assertEqual(len(logs.output), 2)
        self.assertIn("2 are held by failed KV transfers", logs.output[0])


class TestMemoryReleaseWaitsForUnprovenTransfers(CustomTestCase):
    def test_memory_is_not_released_under_an_unproven_transfer(self):
        """Released and resumed memory would be handed out under a remote
        engine's late writes."""
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        updater = SimpleNamespace(
            scheduler=SimpleNamespace(
                pause_state=SimpleNamespace(paused=False),
                holds_unreleased_network_transfers=lambda: True,
            ),
            is_fully_idle=lambda **kw: True,
            offload_tags=set(),
        )
        out = SchedulerWeightUpdaterManager.release_memory_occupation(
            updater, ReleaseMemoryOccupationReqInput(tags=None)
        )
        self.assertFalse(out.success)
        self.assertIn("has not proven", out.message)
        self.assertEqual(updater.offload_tags, set())


if __name__ == "__main__":
    unittest.main()
