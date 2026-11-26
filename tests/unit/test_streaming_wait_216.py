"""Design 216 S03: event-driven wait/stream output and database notification paging.

No result polling here: delta packets only ack, unrelated InputRefs are consumed without
a store read, an absolute deadline is read at most once when it expires, and a session
observation never ends because one send finished.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.runtime_client import RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_transport import (
    PROTOCOL_VERSION,
    DatabasePollingTransport,
    MemoryMessageTransport,
    SubscriptionSpec,
    TransportPacket,
)
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeRequestError,
    TargetRef,
    WaitTimeout,
)

from .session_native_fixture import ACTOR, DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NS = "power-loop"


async def _session(port, name):
    return await port.create_session(
        DESCRIPTOR, actor=ACTOR, options={"session_id": name},
    )


async def _accept(port, session, item_id):
    receipt = await port.accept(session, "run", message_id=item_id, actor=ACTOR)
    return receipt.input_ref


async def _finish(port, session, *, final_text="done", node="a"):
    batch = await port.claim(session, node_id=node, boot_id=node)
    return await port.final(batch, final_text=final_text)


def _delta(ref, text="t"):
    return TransportPacket(
        PROTOCOL_VERSION, NS, ref.message_id, TargetRef(ref.store_id, ref.session_id),
        "output", ref, {"kind": "delta", "text": text, "is_think": False},
    )


def _completed(ref, status="completed"):
    return TransportPacket(
        PROTOCOL_VERSION, NS, ref.message_id, TargetRef(ref.store_id, ref.session_id),
        "output", ref, {"kind": "completed", "status": status},
    )


async def _count_reads(monkeypatch, port):
    calls = {"n": 0}
    original = port.read_result_state

    async def counted(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(port, "read_result_state", counted)
    return calls


def _client(store):
    transport = MemoryMessageTransport()
    return RuntimeClient(store=store, transport=transport), transport


# ── absolute deadline / no-deadline cancel ──────────────────────────────────


async def test_wait_deadline_is_absolute_under_continuous_deltas(registry_store, monkeypatch):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "deadline")
    ref = await _accept(client.inputs, session, "deadline-in")
    reads = await _count_reads(monkeypatch, client.inputs)

    async def noise():
        for _ in range(200):
            await transport.publish(_delta(ref))
            await asyncio.sleep(0.01)

    publisher = asyncio.create_task(noise())
    started = time.monotonic()
    try:
        with pytest.raises(WaitTimeout):
            await client.wait(ref, actor=ACTOR, deadline_ms=now_ms() + 300)
        elapsed = time.monotonic() - started
    finally:
        publisher.cancel()
        await asyncio.gather(publisher, return_exceptions=True)
        await client.aclose()

    # Deltas never extend the original absolute deadline.
    assert 0.25 <= elapsed < 1.2, elapsed
    # Initial read + one final read; every delta was ack-only.
    assert reads["n"] == 2
    assert not client._waiters and client._slots == 0


async def test_wait_without_deadline_is_cancellable(registry_store):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "cancel-wait")
    ref = await _accept(client.inputs, session, "cancel-in")
    task = asyncio.create_task(client.wait(ref, actor=ACTOR, deadline_ms=None))
    await asyncio.sleep(0.1)
    for _ in range(5):
        await transport.publish(_delta(ref))
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    assert not client._waiters and client._slots == 0


async def test_client_close_releases_a_no_deadline_waiter(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "close-wait")
    ref = await _accept(client.inputs, session, "close-in")
    task = asyncio.create_task(client.wait(ref, actor=ACTOR, deadline_ms=None))
    await asyncio.sleep(0.1)
    await client.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client._waiters and client._slots == 0


async def test_wait_reads_only_on_related_completion(registry_store, monkeypatch):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "related")
    ref = await _accept(client.inputs, session, "related-in")
    other = InputRef(store.store_id, session.session_id, "other-in")
    reads = await _count_reads(monkeypatch, client.inputs)

    task = asyncio.create_task(client.wait(ref, actor=ACTOR, deadline_ms=None))
    await asyncio.sleep(0.1)
    # Unrelated deltas and completions are consumed without a store read.
    for _ in range(5):
        await transport.publish(_delta(other))
    await transport.publish(_completed(other))
    assert not task.done()
    await _finish(client.inputs, session)
    await transport.publish(_completed(ref))
    result = await task
    await client.aclose()

    assert result.status == "completed" and result.final_text == "done"
    assert reads["n"] == 2
    assert not client._waiters and client._slots == 0


# ── subscribe_output: before/after send, initial state, duplicates ──────────


async def test_subscribe_output_before_send_allows_missing_input(registry_store):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "sub-before")
    missing = InputRef(store.store_id, session.session_id, "not-there")

    with pytest.raises(RuntimeRequestError) as error:
        await client.subscribe_output(missing, actor=ACTOR)
    assert error.value.code == "not_found"
    assert not client._outputs and client._slots == 0

    ref = InputRef(store.store_id, session.session_id, "sub-before-in")
    output = await client.subscribe_output(ref, actor=ACTOR, before_send=True)
    first = await anext(output)
    assert first.kind == "gap"
    await _accept(client.inputs, session, ref.message_id)
    await _finish(client.inputs, session)
    await transport.publish(_completed(ref))
    item = await anext(output)
    assert item.kind == "completed" and item.payload["final_text"] == "done"
    with pytest.raises(StopAsyncIteration):
        await anext(output)
    await client.aclose()


async def test_subscribe_output_reuses_the_authorized_initial_read(registry_store, monkeypatch):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "sub-initial")
    ref = await _accept(client.inputs, session, "sub-initial-in")
    await _finish(client.inputs, session)
    reads = await _count_reads(monkeypatch, client.inputs)

    output = await client.subscribe_output(ref, actor=ACTOR)
    assert reads["n"] == 1  # authorize + input + result, once, before exposure
    item = await anext(output)
    assert item.kind == "completed" and item.payload["status"] == "completed"
    assert reads["n"] == 1  # first __anext__ did not inspect or read again
    with pytest.raises(StopAsyncIteration):
        await anext(output)
    await client.aclose()


async def test_subscribe_output_slow_consumer_duplicates_and_unrelated(registry_store, monkeypatch):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "sub-slow")
    ref = await _accept(client.inputs, session, "sub-slow-in")
    other = InputRef(store.store_id, session.session_id, "sub-slow-other")
    reads = await _count_reads(monkeypatch, client.inputs)

    output = await client.subscribe_output(ref, actor=ACTOR, before_send=True)
    assert (await anext(output)).kind == "gap"
    for index in range(5):
        await transport.publish(_delta(other, "noise"))
        await transport.publish(_delta(ref, f"d{index}"))
    seen = []
    for _ in range(5):
        item = await anext(output)
        seen.append(item.payload["text"])
        await asyncio.sleep(0.01)  # slow consumer
    assert seen == [f"d{index}" for index in range(5)]
    assert reads["n"] == 1

    await _finish(client.inputs, session, final_text="slow done")
    await transport.publish(_completed(ref))
    await transport.publish(_completed(ref))  # duplicate completion
    item = await anext(output)
    assert item.kind == "completed" and item.payload["final_text"] == "slow done"
    assert reads["n"] == 2
    with pytest.raises(StopAsyncIteration):
        await anext(output)
    await client.aclose()


async def test_session_observation_does_not_end_with_one_send(registry_store):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "observe")
    target = TargetRef(store.store_id, session.session_id)
    ref = await _accept(client.inputs, session, "observe-in")

    output = await client.subscribe_output(target, actor=ACTOR)
    assert (await anext(output)).kind == "gap"
    await transport.publish(_delta(ref))
    assert (await anext(output)).kind == "delta"
    await _finish(client.inputs, session)
    await transport.publish(_completed(ref))
    # A completion addressed to one input is not a session terminal; the observer stays.
    pending = asyncio.create_task(anext(output))
    await asyncio.sleep(0.05)
    assert not pending.done()
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await output.aclose()
    await client.aclose()


# ── database notification paging ────────────────────────────────────────────


async def _terminal_input(port, session, item_id):
    await port.accept(session, "run", message_id=item_id, actor=ACTOR)
    await _finish(port, session)


class _ObservedDiscovery:
    def __init__(self, store):
        self.source = RuntimeInputStore(store)
        self.pages = []
        self.first_page = asyncio.Event()
        self.scan_allowed = asyncio.Event()
        self.scan_allowed.set()

    async def poll_notifications(self, binding, *, cursor, limit):
        await self.scan_allowed.wait()
        packets, next_cursor = await self.source.poll_notifications(
            binding, cursor=cursor, limit=limit,
        )
        self.pages.append((binding, [p.input_ref for p in packets]))
        self.first_page.set()
        return packets, next_cursor


@asynccontextmanager
async def _observed_polling(store, *, page_size=100):
    discovery = _ObservedDiscovery(store)
    transport = DatabasePollingTransport(discovery, interval_s=0.02, page_size=page_size)
    client = RuntimeClient(store=store, transport=transport)
    try:
        async with asyncio.timeout(10):
            yield client, transport, discovery
    finally:
        await client.aclose()
        await transport.aclose()


def _assert_polling_released(client, transport, poller):
    assert poller.done() and not transport._subscriptions
    assert client._slots == 0 and not client._waiters and not client._outputs


@pytest.mark.parametrize("history", [0, 301])
async def test_database_wait_discovers_only_current_input(registry_store, monkeypatch, history):
    store, connect = registry_store
    writer = RuntimeInputStore(store)
    target = await _session(writer, "point-wait")
    for index in range(history):
        await _terminal_input(writer, target, f"old-{index}")
    ref = await _accept(writer, target, "current")
    reader = await connect()
    remote = DatabasePollingTransport(writer)
    try:
        async with _observed_polling(reader) as (client, transport, discovery):
            reads = await _count_reads(monkeypatch, client.inputs)
            output = await client.subscribe_output(ref, actor=ACTOR)
            assert output._initial_read.record.row["status"] == "pending"
            assert output._initial_read.result is None and reads["n"] == 1
            await discovery.first_page.wait()
            assert not discovery.pages[0][1]

            final = await _finish(writer, target, final_text="current result")
            await remote.publish(_completed(ref))
            result = await output.wait()
            assert result == final.results[0]
            assert result.input_ref == ref and result.final_text == "current result"
            assert [item for _, page in discovery.pages for item in page] == [ref]
            assert discovery.pages[-1][1] == [ref]
            assert reads["n"] == 2
            _assert_polling_released(client, transport, output.subscription.poller)
    finally:
        await remote.aclose()


@pytest.mark.parametrize("retry", [False, True])
async def test_database_combined_can_finish_before_first_scan(registry_store, monkeypatch, retry):
    store, connect = registry_store
    writer = RuntimeInputStore(store)
    target = await _session(writer, "point-combined")
    if retry:
        await _accept(writer, target, "current")
    reader = await connect()
    async with _observed_polling(reader) as (client, transport, discovery):
        reads = await _count_reads(monkeypatch, client.inputs)
        discovery.scan_allowed.clear()
        output = await client.send_and_subscribe(target, "run", message_id="current", actor=ACTOR)
        assert reads["n"] == int(retry)
        final = await _finish(writer, target, final_text="quick result")
        assert not discovery.pages
        discovery.scan_allowed.set()
        poller = output.subscription.poller
        await poller
        assert [(b.message_id, b.discovery_message_id, page) for b, page in discovery.pages] == [
            (None, "current", [output.ref]),
        ]
        assert output.subscription in transport._subscriptions and not output.closed
        assert await output.wait() == final.results[0]
        assert reads["n"] == 1 + int(retry)
        _assert_polling_released(client, transport, poller)

        again = await client.send_and_subscribe(target, "run", message_id="current", actor=ACTOR)
        assert await again.wait() == final.results[0]
        assert reads["n"] == 2 + int(retry)
        _assert_polling_released(client, transport, again.subscription.poller)


async def test_database_before_send_preserves_live_gap_error_and_delta(registry_store, monkeypatch):
    store, connect = registry_store
    writer = RuntimeInputStore(store)
    another = await _session(writer, "another-session")
    await _terminal_input(writer, another, "same-name")
    target = await _session(writer, "point-before-send")
    ref = InputRef(store.store_id, target.session_id, "same-name")
    reader = await connect()
    async with _observed_polling(reader) as (client, transport, discovery):
        reads = await _count_reads(monkeypatch, client.inputs)
        output = await client.subscribe_output(ref, actor=ACTOR, before_send=True)
        assert output._initial_read.record is None and reads["n"] == 1
        await discovery.first_page.wait()
        assert not discovery.pages[0][1]
        assert (await anext(output)).kind == "gap"
        assert await _accept(writer, target, ref.message_id) == ref
        await transport.publish(_delta(ref, "local"))
        assert (await anext(output)).payload["text"] == "local"
        assert reads["n"] == 1
        for kind in ("gap", "error"):
            await transport.publish(TransportPacket(
                PROTOCOL_VERSION, NS, "session-notice", target, "output", None, {"kind": kind},
            ))
            assert (await anext(output)).kind == kind
        assert reads["n"] == 3
        final = await _finish(writer, target, final_text="this session")
        assert await output.wait() == final.results[0]
        assert [item for _, page in discovery.pages for item in page] == [ref]
        assert reads["n"] == 4  # Initial, two session notices, and durable completion.
        _assert_polling_released(client, transport, output.subscription.poller)


@pytest.mark.parametrize("close_client", [False, True])
async def test_database_wait_cancel_or_close_stops_pending_scan(registry_store, close_client):
    store, _ = registry_store
    async with _observed_polling(store) as (client, transport, discovery):
        target = await _session(client.inputs, "point-cancel-wait")
        ref = await _accept(client.inputs, target, "pending")
        task = asyncio.create_task(client.wait(ref, actor=ACTOR))
        await discovery.first_page.wait()
        [raw] = transport._subscriptions
        assert raw.binding.discovery_message_id == ref.message_id
        assert not discovery.pages[0][1] and not task.done()
        if close_client:
            await client.aclose()
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_polling_released(client, transport, raw.poller)


@pytest.mark.parametrize("cancel", [False, True])
async def test_database_wait_workflow_continuation_addresses_both_inputs(registry_store, monkeypatch, cancel):
    from .test_workflow_sessions_215 import _crash_before_aggregate, _expire

    store, connect = registry_store
    runner, handle, model = await _crash_before_aggregate(store, monkeypatch)
    try:
        monkeypatch.undo()
        await _expire(store, handle.target.session_id)
        recovered = await runner.runtime.worker.inputs.recover(
            handle.target, node_id="recovery", boot_id="recovery", ttl_s=30,
        )
        assert recovered.execution_state == "queued"
        row = await store._db.fetchone(
            f"SELECT item_id FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue' AND continuation_of=?",
            (handle.target.session_id, handle.input_ref.message_id),
        )
        refs = (handle.input_ref, InputRef(store.store_id, handle.target.session_id, row["item_id"]))
        reader = await connect()
        async with _observed_polling(reader) as (client, transport, discovery):
            reads = await _count_reads(monkeypatch, client.inputs)
            outputs = [await client.subscribe_output(ref, actor=runner.actor) for ref in refs]
            assert all(output._initial_read.result is None for output in outputs)
            if cancel:
                assert (await handle.cancel("stop before continuation")).state == "voided"
            else:
                await runner.runtime.worker.recover()
            results = await asyncio.gather(*(output.wait() for output in outputs))
            assert tuple(result.input_ref for result in results) == refs
            assert [result.status for result in results] == ["cancelled" if cancel else "completed"] * 2
            if cancel:
                assert all(result.error == "stop before continuation" for result in results)
            else:
                assert all(result.final_text == "journaled result" for result in results)
                assert results[0].result_seq == results[1].result_seq
            assert model.calls == 1 and reads["n"] == 4
            deliveries = {binding.discovery_message_id: page for binding, page in discovery.pages if page}
            assert deliveries == {ref.message_id: [ref] for ref in refs}
            for output in outputs:
                _assert_polling_released(client, transport, output.subscription.poller)
    finally:
        await runner.aclose()


async def test_raw_database_late_subscription_keeps_session_paging(registry_store):
    store, _ = registry_store
    async with _observed_polling(store, page_size=1) as (client, transport, discovery):
        target = await _session(client.inputs, "raw-late")
        await _terminal_input(client.inputs, target, "old")
        await _terminal_input(client.inputs, target, "new")
        binding = SubscriptionSpec(NS, store.store_id, "output", "client", target.session_id)
        sub = await transport.subscribe(binding)
        assert [(await anext(sub)).packet.message_id for _ in range(2)] == ["old", "new"]
        assert all(b.discovery_message_id is None for b, _ in discovery.pages)
        assert not sub.poller.done()
        await sub.aclose()
        _assert_polling_released(client, transport, sub.poller)


async def test_poll_notifications_filters_session_before_limit(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    other = await _session(port, "page-other")
    own = await _session(port, "page-own")
    for index in range(3):
        await _terminal_input(port, other, f"other-{index}")
    await _terminal_input(port, own, "own-0")

    packets, cursor = await port.poll_notifications(
        SubscriptionSpec(NS, store.store_id, "output", "client", own.session_id), limit=1,
    )
    assert [packet.message_id for packet in packets] == ["own-0"]
    assert cursor is not None


async def test_poll_notifications_same_millisecond_crosses_page(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    session = await _session(port, "page-tie")
    await _terminal_input(port, session, "tie-0")
    await _terminal_input(port, session, "tie-1")
    rows = await store._db.fetchall(
        f"SELECT id FROM {port.q} WHERE session_id=? ORDER BY id", (session.session_id,))
    assert len(rows) == 2
    completed_at = 7_000
    await store._db.execute(
        f"UPDATE {port.q} SET delivered_at=? WHERE session_id=?", (completed_at, session.session_id))

    binding = SubscriptionSpec(NS, store.store_id, "output", "client", session.session_id)
    first, cursor = await port.poll_notifications(binding, limit=1)
    second, cursor = await port.poll_notifications(binding, cursor=cursor, limit=1)
    assert [packet.message_id for packet in first + second] == ["tie-0", "tie-1"]
    assert cursor == (completed_at, int(rows[1]["id"]))


async def test_poll_notifications_later_timestamp_with_smaller_id(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    session = await _session(port, "page-later")
    await _terminal_input(port, session, "later-0")
    await _terminal_input(port, session, "later-1")
    rows = await store._db.fetchall(
        f"SELECT id,item_id FROM {port.q} WHERE session_id=? ORDER BY id", (session.session_id,))
    earlier, later = rows
    await store._db.execute(
        f"UPDATE {port.q} SET delivered_at=? WHERE id=?", (1_000, int(earlier["id"])))
    await store._db.execute(
        f"UPDATE {port.q} SET delivered_at=? WHERE id=?", (5_000, int(later["id"])))

    binding = SubscriptionSpec(NS, store.store_id, "output", "client", session.session_id)
    first, cursor = await port.poll_notifications(binding, limit=1)
    assert [packet.message_id for packet in first] == [earlier["item_id"]]
    assert cursor == (1_000, int(earlier["id"]))
    # A later completed_at with a smaller id must still be returned after that cursor.
    second, _ = await port.poll_notifications(binding, cursor=(4_000, 10**9), limit=10)
    assert [packet.message_id for packet in second] == [later["item_id"]]


async def test_poll_notifications_without_session_advances_real_pages(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    first = await _session(port, "page-all-a")
    second = await _session(port, "page-all-b")
    await _terminal_input(port, first, "all-a")
    await _terminal_input(port, second, "all-b")

    binding = SubscriptionSpec(NS, store.store_id, "output", "client", None)
    seen, cursor = [], None
    while True:
        packets, cursor = await port.poll_notifications(binding, cursor=cursor, limit=1)
        if not packets:
            break
        seen.extend(packet.message_id for packet in packets)
    assert seen == ["all-a", "all-b"]

# ── combined send + subscription ───────────────────────────────────────────


async def test_send_and_subscribe_fresh_has_one_terminal_read(registry_store, monkeypatch):
    from power_loop import AgentLoopConfig
    from tests.runtime_fixture import RunningAgent

    from .test_session_send_215 import EchoModel

    store, _ = registry_store
    agent = await RunningAgent(store, EchoModel(text="combined answer"),
                               config=AgentLoopConfig(max_rounds=2,
                                                      context_budget_tokens=10**9)).start()
    try:
        reads = await _count_reads(monkeypatch, agent.runtime.client.inputs)
        output = await agent.runtime.client.send_and_subscribe(
            agent.target, "hello", message_id="combined-fresh", actor=agent.actor,
        )
        assert reads["n"] == 0
        result = await output.wait(deadline_ms=now_ms() + 5000)
        assert result.status == "completed" and result.final_text == "combined answer"
        assert reads["n"] == 1
        assert agent.runtime.client._slots == 0 and not agent.runtime.client._waiters
    finally:
        await agent.aclose()


async def test_send_and_subscribe_retry_reads_existing_terminal_once(registry_store, monkeypatch):
    from power_loop import AgentLoopConfig
    from tests.runtime_fixture import RunningAgent

    from .test_session_send_215 import EchoModel

    store, _ = registry_store
    agent = await RunningAgent(store, EchoModel(text="retry answer"),
                               config=AgentLoopConfig(max_rounds=2,
                                                      context_budget_tokens=10**9)).start()
    try:
        first = await agent.send("same body", message_id="combined-retry")
        assert first.status == "completed"
        reads = await _count_reads(monkeypatch, agent.runtime.client.inputs)
        output = await agent.runtime.client.send_and_subscribe(
            agent.target, "same body", message_id="combined-retry", actor=agent.actor,
        )
        assert reads["n"] == 1
        result = await output.wait(deadline_ms=now_ms() + 5000)
        assert result.status == "completed" and result.final_text == "retry answer"
        assert reads["n"] == 1
    finally:
        await agent.aclose()


async def test_send_and_subscribe_cancel_wakes_without_initial_read(registry_store, monkeypatch):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "combined-cancel")
    reads = await _count_reads(monkeypatch, client.inputs)
    output = await client.send_and_subscribe(
        session, "run", message_id="combined-cancel-in", actor=ACTOR,
    )
    assert reads["n"] == 0
    task = asyncio.create_task(output.wait(deadline_ms=None))
    await asyncio.sleep(0.05)
    await client.cancel(output.ref, actor=ACTOR, reason="stop combined")
    result = await task
    assert result.status == "cancelled"
    assert reads["n"] == 1
    assert client._slots == 0 and not client._waiters
    await client.aclose()


async def test_target_cancel_notifies_only_cancelled_pending_refs(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "combined-target-cancel")
    first = await client.send_and_subscribe(session, "one", message_id="one", actor=ACTOR)
    second = await client.send_and_subscribe(session, "two", message_id="two", actor=ACTOR)
    tasks = [asyncio.create_task(first.wait(deadline_ms=None)),
             asyncio.create_task(second.wait(deadline_ms=None))]
    await asyncio.sleep(0.05)
    await client.cancel(session, actor=ACTOR, reason="stop all")
    results = await asyncio.gather(*tasks)
    assert [result.status for result in results] == ["cancelled", "cancelled"]
    assert client._slots == 0 and not client._waiters
    await client.aclose()


@pytest.mark.parametrize("deltas", [0, 1, 300])
async def test_send_and_subscribe_deltas_do_not_add_reads(registry_store, monkeypatch, deltas):
    from power_loop import AgentLoopConfig
    from tests.runtime_fixture import RunningAgent

    from .test_session_send_215 import EchoModel

    class GatedModel(EchoModel):
        def __init__(self):
            super().__init__(text="stream answer")
            self.release = asyncio.Event()

        async def complete(self, request, **kwargs):
            await self.release.wait()
            return await super().complete(request, **kwargs)

    store, _ = registry_store
    model = GatedModel()
    agent = await RunningAgent(store, model,
                               config=AgentLoopConfig(max_rounds=2,
                                                      context_budget_tokens=10**9)).start()
    try:
        reads = await _count_reads(monkeypatch, agent.runtime.client.inputs)
        output = await agent.runtime.client.send_and_subscribe(
            agent.target, "hello", message_id="combined-delta", actor=agent.actor,
        )
        assert reads["n"] == 0
        async def collect():
            return [item async for item in output]
        consumer = asyncio.create_task(collect())
        for index in range(deltas):
            await agent.runtime.transport.publish(_delta(output.ref, f"d{index}"))
            await asyncio.sleep(0)
        assert reads["n"] == 0
        model.release.set()
        items = await consumer
        assert [item.kind for item in items] == ["gap"] + ["delta"] * deltas + ["completed"]
        assert items[-1].payload["final_text"] == "stream answer"
        assert reads["n"] == 1
    finally:
        model.release.set()
        await agent.aclose()


async def test_workflow_original_cancel_wakes_original_and_latest(registry_store):
    from power_loop.agent.runtime_types import canonical

    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "workflow-cancel-wake")
    start = await _accept(client.inputs, session, "start")
    batch = await client.inputs.claim(session, node_id="a", boot_id="a")
    await store._db.execute(
        f"UPDATE {client.inputs.q} SET status='waiting_continuation' "
        "WHERE session_id=? AND item_id='start'", (session.session_id,),
    )
    wfc = InputRef(store.store_id, session.session_id, "wfc-1")
    await store._db.execute(
        f"INSERT INTO {client.inputs.q} (session_id,item_id,kind,priority,status,"
        "content_json,meta_json,request_hash,created_at,continuation_of,continuation_index) "
        "VALUES (?,?,'workflow_continue','normal','pending',?,?,?,?,?,?)",
        (session.session_id, "wfc-1", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, "start", 1),
    )
    await client.inputs.final(batch)
    first = await client.subscribe_output(start, actor=ACTOR, before_send=True)
    latest = await client.subscribe_output(wfc, actor=ACTOR, before_send=True)
    tasks = [asyncio.create_task(first.wait(deadline_ms=None)),
             asyncio.create_task(latest.wait(deadline_ms=None))]
    await asyncio.sleep(0.05)
    await client.cancel(start, actor=ACTOR, reason="stop workflow")
    results = await asyncio.gather(*tasks)
    assert [result.status for result in results] == ["cancelled", "cancelled"]
    assert client._slots == 0 and not client._waiters
    await client.aclose()


async def test_send_and_subscribe_ready_failure_releases_slot(registry_store, monkeypatch):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "combo-ready-fail")

    async def fail(_binding):
        raise RuntimeRequestError("not_found")

    monkeypatch.setattr(transport, "subscribe", fail)
    with pytest.raises(RuntimeRequestError):
        await client.send_and_subscribe(session, "run", message_id="ready-fail", actor=ACTOR)
    assert client._slots == 0 and not client._waiters and not client._outputs
    await client.aclose()


async def test_send_and_subscribe_callback_failure_does_not_send(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "combo-callback-fail")

    def fail():
        raise RuntimeError("submitted boom")

    with pytest.raises(RuntimeError):
        await client.send_and_subscribe(
            session, "run", message_id="callback-fail", actor=ACTOR,
            on_submit_start=fail,
        )
    assert client._slots == 0 and not client._waiters and not client._outputs
    assert await client.inputs.input(client.inputs.db, InputRef(
        store.store_id, session.session_id, "callback-fail")) is None
    with pytest.raises(RuntimeRequestError) as error:
        await client.inspect(
            InputRef(store.store_id, session.session_id, "callback-fail"), actor=ACTOR,
        )
    assert error.value.code == "not_found"
    await client.aclose()


async def test_send_and_subscribe_accept_rejection_releases_slot(registry_store):
    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "combo-reject")
    with pytest.raises(RuntimeRequestError) as error:
        await client.send_and_subscribe(
            session, "run", message_id="combine-reject",
            actor=TrustedActor("other", "host", None),
        )
    assert error.value.code == "forbidden"
    assert client._slots == 0 and not client._waiters and not client._outputs
    await client.aclose()


async def test_send_and_subscribe_retry_pending_reads_initial_then_terminal(
    registry_store, monkeypatch
):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "combo-retry-pending")
    receipt = await client.send(session, "same", message_id="retry-pending", actor=ACTOR)
    assert receipt.state == "accepted"
    reads = await _count_reads(monkeypatch, client.inputs)
    output = await client.send_and_subscribe(
        session, "same", message_id="retry-pending", actor=ACTOR,
    )
    assert reads["n"] == 1
    task = asyncio.create_task(output.wait(deadline_ms=None))
    await asyncio.sleep(0.05)
    await _finish(client.inputs, session)
    await transport.publish(_completed(output.ref))
    result = await task
    assert result.status == "completed" and result.final_text == "done"
    assert reads["n"] == 2
    await client.aclose()


async def test_wait_deadline_during_completion_close_returns_result_after_cleanup(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "combo-close-window")
    ref = await _accept(client.inputs, session, "close-window-in")
    await _finish(client.inputs, session, final_text="close window")
    output = await client.subscribe_output(ref, actor=ACTOR)
    entered, release = asyncio.Event(), asyncio.Event()
    original = output.subscription.aclose

    async def blocked():
        entered.set()
        await release.wait()
        await original()

    output.subscription.aclose = blocked
    task = asyncio.create_task(output.wait(deadline_ms=now_ms() + 100))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.15)
    release.set()
    result = await task
    assert result.status == "completed" and result.final_text == "close window"
    assert output.closed and output.subscription.closed
    assert client._slots == 0 and not client._outputs
    await client.aclose()


async def test_subtree_cancel_notifies_linked_child_not_independent(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    parent = await client.create_session(DESCRIPTOR, actor=ACTOR, options={"session_id": "subtree-parent"})
    linked = await client.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent, options={"session_id": "subtree-linked"},
    )
    independent = await client.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent, cancel_propagation="independent",
        options={"session_id": "subtree-independent"},
    )
    linked_receipt = await client.send(linked, "linked", message_id="linked-in", actor=ACTOR)
    independent_receipt = await client.send(independent, "independent",
                                            message_id="independent-in", actor=ACTOR)
    linked_output = await client.subscribe_output(
        linked_receipt.input_ref, actor=ACTOR, before_send=True,
    )
    independent_output = await client.subscribe_output(
        independent_receipt.input_ref, actor=ACTOR, before_send=True,
    )
    linked_task = asyncio.create_task(linked_output.wait(deadline_ms=None))
    independent_task = asyncio.create_task(independent_output.wait(deadline_ms=None))
    await asyncio.sleep(0.05)
    await client.cancel(parent, actor=ACTOR, scope="subtree", reason="stop tree")
    result = await linked_task
    assert result.status == "cancelled"
    await asyncio.sleep(0.05)
    assert not independent_task.done()
    await client.cancel(independent, actor=ACTOR, reason="cleanup")
    assert (await independent_task).status == "cancelled"
    assert client._slots == 0 and not client._waiters
    await client.aclose()


# ── R7 lifecycle: preparation, read, cancellation ownership, cleanup ───────


async def test_client_close_cancels_send_and_subscribe_during_accept(registry_store):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "lifecycle-accept")
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.inputs.accept

    async def gated_accept(*args, **kwargs):
        value = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return value

    client.inputs.accept = gated_accept
    task = asyncio.create_task(client.send_and_subscribe(
        session, "payload", message_id="lifecycle-accept-in", actor=ACTOR,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert len(client._waiters) == 1 and client._slots == 1
        await asyncio.wait_for(client.aclose(), 1)
        assert task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._slots == 0 and not client._outputs and not client._waiters
        assert transport._subscriptions == set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        client.inputs.accept = original
        await client.aclose()


async def test_client_close_cancels_retry_initial_read(registry_store, monkeypatch):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-retry")
    await client.send(session, "payload", message_id="lifecycle-retry-in", actor=ACTOR)
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.inputs.read_result_state

    async def gated_read(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(client.inputs, "read_result_state", gated_read)
    task = asyncio.create_task(client.send_and_subscribe(
        session, "payload", message_id="lifecycle-retry-in", actor=ACTOR,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(client.aclose(), 1)
        assert task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._slots == 0 and not client._outputs
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


async def test_client_close_cancels_wait_during_result_read(registry_store, monkeypatch):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-read")
    await _accept(client.inputs, session, "lifecycle-read-in")
    output = await client.send_and_subscribe(
        session, "run", message_id="lifecycle-read-in", actor=ACTOR,
    )
    await _finish(client.inputs, session)
    entered, release = asyncio.Event(), asyncio.Event()
    original = client._durable_result

    async def gated_read(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(client, "_durable_result", gated_read)
    task = asyncio.create_task(output.wait(deadline_ms=None))
    try:
        await client.transport.publish(_completed(output.ref))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(client.aclose(), 1)
        assert task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._slots == 0 and not client._outputs
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


async def test_user_cancel_during_completion_close_is_not_a_result(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-user-cancel")
    ref = await _accept(client.inputs, session, "lifecycle-user-cancel-in")
    await _finish(client.inputs, session, final_text="must not return")
    output = await client.subscribe_output(ref, actor=ACTOR)
    entered, release = asyncio.Event(), asyncio.Event()
    original = output.subscription.aclose

    async def blocked():
        entered.set()
        await release.wait()
        await original()

    output.subscription.aclose = blocked
    task = asyncio.create_task(output.wait(deadline_ms=None))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel("user stopped")
    await asyncio.sleep(0.05)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await output.aclose()
    assert output.closed and output.subscription.closed
    assert client._slots == 0 and not client._outputs
    await client.aclose()


async def test_deadline_during_completion_close_returns_result_after_cleanup(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-deadline")
    ref = await _accept(client.inputs, session, "lifecycle-deadline-in")
    await _finish(client.inputs, session, final_text="deadline result")
    output = await client.subscribe_output(ref, actor=ACTOR)
    entered, release = asyncio.Event(), asyncio.Event()
    original = output.subscription.aclose

    async def blocked():
        entered.set()
        await release.wait()
        await original()

    output.subscription.aclose = blocked
    task = asyncio.create_task(output.wait(deadline_ms=now_ms() + 100))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.15)
    release.set()
    result = await task
    assert result.status == "completed" and result.final_text == "deadline result"
    assert output.closed and output.subscription.closed
    assert client._slots == 0 and not client._outputs
    await client.aclose()


async def test_interrupted_close_can_be_retried(registry_store):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-close-retry")
    ref = await _accept(client.inputs, session, "lifecycle-close-retry-in")
    output = await client.subscribe_output(ref, actor=ACTOR)
    original = output.subscription.aclose
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("close interrupted")
        await original()

    output.subscription.aclose = flaky
    with pytest.raises(RuntimeError):
        await output.aclose()
    assert not output.closed and client._slots == 1 and output in client._outputs
    await output.aclose()
    assert output.closed and client._slots == 0 and output not in client._outputs
    await client.aclose()


async def test_non_deadline_timeout_propagates_without_extra_read(registry_store, monkeypatch):
    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "lifecycle-db-timeout")
    output = await client.send_and_subscribe(
        session, "run", message_id="lifecycle-db-timeout-in", actor=ACTOR,
    )
    calls = {"finish": 0}
    original_finish = output.finish_from_store

    async def finish():
        calls["finish"] += 1
        return await original_finish()

    async def timeout(*args, **kwargs):
        raise TimeoutError("database timeout")

    output.finish_from_store = finish
    monkeypatch.setattr(client, "_durable_result", timeout)
    task = asyncio.create_task(output.wait(deadline_ms=None))
    await asyncio.sleep(0.05)
    await client.transport.publish(_completed(output.ref))
    with pytest.raises(TimeoutError, match="database timeout"):
        await task
    assert calls["finish"] == 0
    await client.aclose()


async def test_target_cancel_wakes_multihop_workflow_chain(registry_store):
    from power_loop.agent.runtime_types import canonical

    store, _ = registry_store
    client, _ = _client(store)
    session = await _session(client.inputs, "target-multihop")
    start = await _accept(client.inputs, session, "start")
    batch = await client.inputs.claim(session, node_id="a", boot_id="a")
    await store._db.execute(
        f"UPDATE {client.inputs.q} SET status='waiting_continuation' "
        "WHERE session_id=? AND item_id='start'", (session.session_id,),
    )
    await store._db.execute(
        f"INSERT INTO {client.inputs.q} (session_id,item_id,kind,priority,status,"
        "content_json,meta_json,request_hash,created_at,delivered_at,delivered_seq,"
        "delivered_send_index,continuation_of,continuation_index) "
        "VALUES (?,?,'workflow_continue','normal','waiting_continuation',?,?,?,?,?,?,?,?,?)",
        (session.session_id, "wfc-1", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, 1, 1, 2, "start", 1),
    )
    await store._db.execute(
        f"INSERT INTO {client.inputs.q} (session_id,item_id,kind,priority,status,"
        "content_json,meta_json,request_hash,created_at,continuation_of,continuation_index) "
        "VALUES (?,?,'workflow_continue','normal','pending',?,?,?,?,?,?)",
        (session.session_id, "wfc-2", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, "wfc-1", 2),
    )
    await client.inputs.final(batch)
    refs = [start, InputRef(store.store_id, session.session_id, "wfc-1"),
            InputRef(store.store_id, session.session_id, "wfc-2")]
    outputs = [await client.subscribe_output(ref, actor=ACTOR, before_send=True)
               for ref in refs]
    tasks = [asyncio.create_task(output.wait(deadline_ms=None)) for output in outputs]
    await asyncio.sleep(0.05)
    await client.cancel(session, actor=ACTOR, reason="stop multihop")
    results = await asyncio.gather(*tasks)
    assert [result.status for result in results] == ["cancelled"] * 3
    assert client._slots == 0 and not client._waiters
    await client.aclose()


# ── R8 resource ownership during preparation cleanup ───────────────────────


async def test_client_close_waits_for_forbidden_subscription_cleanup(registry_store):
    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "r8-forbidden-cleanup")
    entered, release = asyncio.Event(), asyncio.Event()
    subscriptions = []
    original_subscribe = transport.subscribe

    async def gated_subscribe(binding):
        sub = await original_subscribe(binding)
        subscriptions.append(sub)
        original_close = sub.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        sub.aclose = close
        return sub

    transport.subscribe = gated_subscribe
    task = asyncio.create_task(client.subscribe_output(
        session, actor=TrustedActor("wrong", "host", None),
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert client._slots == 1 and len(client._outputs) == 1
        closing = asyncio.create_task(client.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done()
        release.set()
        await closing
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._slots == 0 and not client._outputs and not client._waiters
        assert transport._subscriptions == set()
        assert all(sub.closed for sub in subscriptions)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        transport.subscribe = original_subscribe
        await client.aclose()


async def test_caller_cancel_during_forbidden_cleanup_then_client_close(registry_store):
    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "r8-cancel-cleanup")
    entered, release = asyncio.Event(), asyncio.Event()
    subscriptions = []
    original_subscribe = transport.subscribe

    async def gated_subscribe(binding):
        sub = await original_subscribe(binding)
        subscriptions.append(sub)
        original_close = sub.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        sub.aclose = close
        return sub

    transport.subscribe = gated_subscribe
    task = asyncio.create_task(client.subscribe_output(
        session, actor=TrustedActor("wrong", "host", None),
    ))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel("caller stopped during cleanup")
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        assert isinstance(outcome, asyncio.CancelledError)
        assert client._outputs and client._slots == 1
        release.set()
        await client.aclose()
        assert client._slots == 0 and not client._outputs and not client._waiters
        assert transport._subscriptions == set()
        assert subscriptions and subscriptions[0].closed
        await client.aclose()
        assert client._slots == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        transport.subscribe = original_subscribe
        await client.aclose()


# ── R04: output queue overflow gap is session-scoped ───────────────────────


async def test_output_queue_overflow_from_other_input_wakes_waiter_via_session_gap(
    registry_store, monkeypatch
):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "r04-overflow-wait")
    ref_a = await _accept(client.inputs, session, "input-a")
    ref_b = InputRef(store.store_id, session.session_id, "input-b")
    reads = await _count_reads(monkeypatch, client.inputs)
    try:
        output = await client.subscribe_output(ref_a, actor=ACTOR)
        assert output._initial_read.result is None and reads["n"] == 1

        # Persist A's terminal state and enqueue its completion notification.
        await _finish(client.inputs, session, final_text="answer-A")
        await transport.publish(_completed(ref_a))
        assert output.subscription.queue.qsize() == 1

        # Fill the remaining 255 slots with same-session B packets, then trigger overflow.
        for index in range(255):
            await transport.publish(_delta(ref_b, f"b-{index}"))
        assert output.subscription.queue.full()
        await transport.publish(_delta(ref_b, "b-overflow"))
        assert output.subscription.queue.qsize() == 2

        queued = list(output.subscription.queue._queue)
        assert queued[0].input_ref is None and queued[0].payload == {"kind": "gap"}
        assert queued[1].input_ref == ref_b and queued[1].payload["text"] == "b-overflow"

        # Stop publishing; deadline_ms=None ensures only the overflow gap triggers the DB read.
        result = await asyncio.wait_for(output.wait(deadline_ms=None), timeout=5.0)
        assert result.input_ref == ref_a
        assert result.status == "completed" and result.final_text == "answer-A"
        assert reads["n"] == 2
        assert client._slots == 0 and not client._waiters and not client._outputs
    finally:
        await client.aclose()


async def test_output_queue_overflow_keeps_delta_and_completion_filtered_by_input(
    registry_store, monkeypatch
):
    store, _ = registry_store
    client, transport = _client(store)
    session = await _session(client.inputs, "r04-overflow-filter")
    ref_a = await _accept(client.inputs, session, "input-a")
    ref_b = InputRef(store.store_id, session.session_id, "input-b")
    reads = await _count_reads(monkeypatch, client.inputs)
    try:
        output = await client.subscribe_output(ref_a, actor=ACTOR)
        assert (await anext(output)).kind == "gap"
        assert reads["n"] == 1

        # While A is still pending, B overflows the queue with deltas + completion.
        for index in range(257):
            await transport.publish(_delta(ref_b, f"noise-{index}"))
        await transport.publish(_completed(ref_b))
        gap_item = await asyncio.wait_for(anext(output), timeout=5.0)
        assert gap_item.kind == "gap" and reads["n"] == 2

        # B's trailing delta and completion are filtered out without emitting to A or reading DB.
        await _finish(client.inputs, session, final_text="answer-A-stream")
        await transport.publish(_completed(ref_a))
        for index in range(256):
            await transport.publish(_delta(ref_b, f"after-{index}"))
        done_item = await asyncio.wait_for(anext(output), timeout=5.0)
        assert done_item.kind == "completed"
        assert done_item.payload["final_text"] == "answer-A-stream"
        assert reads["n"] == 3
        with pytest.raises(StopAsyncIteration):
            await anext(output)
    finally:
        await client.aclose()


async def test_database_polling_output_overflow_recovers_after_poller_exit(
    registry_store, monkeypatch
):
    store, connect = registry_store
    writer = RuntimeInputStore(store)
    session = await _session(writer, "r04-db-overflow")
    ref_a = await _accept(writer, session, "input-a")
    ref_b = InputRef(store.store_id, session.session_id, "input-b")
    reader = await connect()
    async with _observed_polling(reader) as (client, transport, discovery):
        reads = await _count_reads(monkeypatch, client.inputs)
        output = await client.subscribe_output(ref_a, actor=ACTOR)
        assert output._initial_read.result is None and reads["n"] == 1
        await discovery.first_page.wait()

        # Poller discovers A's terminal row, enqueues it, and exits (discovery_message_id set).
        await _finish(writer, session, final_text="answer-A-db")
        await asyncio.wait_for(output.subscription.poller, timeout=5.0)
        assert output.subscription.queue.qsize() == 1

        # Local B deltas overflow the queue and drop A's polled completion packet.
        for index in range(256):
            await transport.publish(_delta(ref_b, f"local-b-{index}"))
        assert output.subscription.queue.qsize() == 2

        result = await asyncio.wait_for(output.wait(deadline_ms=None), timeout=5.0)
        assert result.input_ref == ref_a
        assert result.status == "completed" and result.final_text == "answer-A-db"
        assert reads["n"] == 2
        _assert_polling_released(client, transport, output.subscription.poller)
