"""A real session-lease takeover fences old results and pending mutations."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, SessionLeaseLost, StatefulAgentLoop
from power_loop.runtime.store import store as store_module

from .test_job_registry_192 import registry_store
from .test_round_change_signals import Model
from .test_send_sql_profile import Profiled

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("stage", ["before_takeover", "after_takeover"])
async def test_process_crash_during_stale_append_preserves_original_pending_and_new_owner(registry_store, tmp_path, monkeypatch, stage):
    store, connect = registry_store
    sid = await store.create_session()
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "lease-crash.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._lease_write_crash_worker",
        dsn, store.table_prefix, sid, str(evidence), stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (88 if backend == "mysql" else 87), output.decode()
    saved = json.loads(evidence.read_text())
    monkeypatch.setattr(store_module, "_now_ms", lambda: saved["now"])
    cold = await connect(schema="compatible")
    if stage == "before_takeover":
        assert await cold.acquire_session_lease(sid, owner_id="replacement", ttl_ms=60000)
    state = await cold.get_state(sid)
    assert state.next_seq == 2 and state.pending == saved["pending"]
    [original] = await cold.load_all_messages(sid)
    assert original.tool_calls[0]["id"] == "original"
    assert (await cold.session_lease_holder(sid))["owner_id"] == "replacement"


async def test_same_timestamp_renewal_uses_matched_rows_without_a_confirmation_read(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await store.create_session()
    monkeypatch.setattr(store_module, "_now_ms", lambda: 1000)
    assert await store.acquire_session_lease(sid, owner_id="owner", ttl_ms=60000)
    original = store._db
    store._db = counted = Profiled(original)
    try:
        for _ in range(3):
            counted.records.clear()
            assert await store.renew_session_lease(sid, owner_id="owner", ttl_ms=60000)
            assert len(counted.records) == 1 and counted.records[0]["sql"].startswith("UPDATE ")
        counted.records.clear()
        assert not await store.renew_session_lease(sid, owner_id="other", ttl_ms=60000)
        assert len(counted.records) == 1
    finally:
        store._db = original


async def test_new_lease_period_reloads_pending_and_round_changed_by_another_owner(registry_store):
    store, connect = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, compactor=None))
    other = await connect(schema="compatible")
    try:
        async with loop._session_lease(sid):
            await store.append_message(sid, role="user", content="first", round_index=1)
            store.defer_round_index(sid, 5)
        assert await other.acquire_session_lease(sid, owner_id="other", ttl_ms=60000)
        calls = [{"id": "new", "type": "function", "function": {"name": "work", "arguments": "{}"}}]
        await other.append_message(sid, role="assistant", tool_calls=calls, round_index=7, pending_tool_calls=True)
        await other.set_round_index(sid, 7)
        await other.release_session_lease(sid, owner_id="other")
        async with loop._session_lease(sid):
            state = await store.get_state(sid)
            assert state.round_index == 7 and state.pending["tool_call_ids"] == ["new"]
            assert state == await other.get_state(sid)
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("new_owner_writes", [False, True])
async def test_takeover_rejects_old_fast_and_locking_appends_and_pending_clear(registry_store, monkeypatch, new_owner_writes):
    store, connect = registry_store
    now = [1000]
    monkeypatch.setattr(store_module, "_now_ms", lambda: now[0])
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, session_lease_ttl_s=300, compactor=None))
    other = await connect(schema="compatible")
    try:
        async with loop._session_lease(sid):
            calls = [{"id": "original", "type": "function", "function": {"name": "work", "arguments": "{}"}}]
            await store.append_message(sid, role="assistant", tool_calls=calls, pending_tool_calls=True)
            now[0] += 300001
            assert await other.acquire_session_lease(sid, owner_id="replacement", ttl_ms=60000)
            if new_owner_writes:
                await other.append_message(sid, role="tool", tool_call_id="original", content="reconciled", pending=None)
            before = await other.get_state(sid)
            for action in (lambda: store.append_message(sid, role="assistant", content="stale"),
                           lambda: store.set_pending(sid, None), lambda: store.set_round_index(sid, 999)):
                with pytest.raises(SessionLeaseLost):
                    await action()
                assert await other.get_state(sid) == before
            assert all(row.content != "stale" for row in await other.load_all_messages(sid))
        assert (await other.session_lease_holder(sid))["owner_id"] == "replacement"
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_detached_child_cannot_append_after_its_lease_scope_finishes(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, compactor=None))
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="late")

    try:
        async with loop._session_lease(sid):
            child = asyncio.create_task(late())
        release.set()
        with pytest.raises(SessionLeaseLost):
            await child
        assert not await store.load_all_messages(sid)
    finally:
        release.set()
        await loop.aclose(drain_timeout_s=.1)


async def test_old_child_cannot_reuse_the_same_loops_next_lease_or_consume_its_round(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, compactor=None))
    release = asyncio.Event()

    async def late():
        await release.wait()
        await store.append_message(sid, role="assistant", content="stale")

    try:
        async with loop._session_lease(sid):
            child = asyncio.create_task(late())
        async with loop._session_lease(sid):
            store.defer_round_index(sid, 7)
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            await store.append_message(sid, role="assistant", content="current")
            assert (await store.get_state(sid)).round_index == 7
            assert [row.content for row in await store.load_all_messages(sid)] == ["current"]
    finally:
        release.set()
        await loop.aclose(drain_timeout_s=.1)


async def test_writer_waiting_for_a_transaction_cannot_restore_an_expired_turns_round(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, compactor=None))
    entered, release = asyncio.Event(), asyncio.Event()
    transaction = store._db.transaction
    child = None
    delayed = False

    @asynccontextmanager
    async def blocked():
        nonlocal delayed
        if asyncio.current_task() is child and not delayed:
            delayed = True
            entered.set()
            await release.wait()
        async with transaction() as tx:
            yield tx

    async def late():
        await store.append_message(sid, role="assistant", content="stale")

    monkeypatch.setattr(store._db, "transaction", blocked)
    try:
        async with loop._session_lease(sid):
            store.defer_round_index(sid, 5)
            child = asyncio.create_task(late())
            await asyncio.wait_for(entered.wait(), 5)
        async with loop._session_lease(sid):
            store.defer_round_index(sid, 7)
            await store.append_message(sid, role="assistant", content="current")
            release.set()
            with pytest.raises(SessionLeaseLost):
                await child
            assert (await store.get_state(sid)).round_index == 7
            assert [row.content for row in await store.load_all_messages(sid)] == ["current"]
    finally:
        release.set()
        if child is not None:
            await asyncio.gather(child, return_exceptions=True)
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("lost_ack", [False, True])
@pytest.mark.parametrize("operation", ["append", "pending_clear"])
async def test_late_commit_ack_never_overwrites_the_next_turns_pending_or_round(registry_store, monkeypatch, lost_ack, operation):
    store, _ = registry_store
    sid = await store.create_session()
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, compactor=None))
    committed, release = asyncio.Event(), asyncio.Event()
    transaction = store._db.transaction
    child = None
    delayed = False

    @asynccontextmanager
    async def delayed_ack():
        nonlocal delayed
        async with transaction() as tx:
            yield tx
        if asyncio.current_task() is child and not delayed:
            delayed = True
            committed.set()
            await release.wait()
            if lost_ack:
                raise OSError("old_turn_committed_ack_lost")

    def calls(key):
        return [{"id": key, "type": "function", "function": {"name": "work", "arguments": "{}"}}]

    async def original():
        if operation == "pending_clear":
            return await store.set_pending(sid, None)
        return await store.append_message(sid, role="assistant", tool_calls=calls("old"), pending_tool_calls=True)

    monkeypatch.setattr(store._db, "transaction", delayed_ack)
    try:
        async with loop._session_lease(sid):
            if operation == "pending_clear":
                await store.append_message(sid, role="assistant", tool_calls=calls("old"), pending_tool_calls=True)
                await store.append_message(sid, role="tool", tool_call_id="old", content="aborted", pending=None)
            store.defer_round_index(sid, 5)
            child = asyncio.create_task(original())
            await asyncio.wait_for(committed.wait(), 5)
        async with loop._session_lease(sid):
            # The new owner explicitly aborts the untouched original call.
            if operation == "append":
                await store.append_message(sid, role="tool", tool_call_id="old", content="aborted", pending=None)
            store.defer_round_index(sid, 7)
            await store.append_message(sid, role="assistant", tool_calls=calls("new"), pending_tool_calls=True)
            release.set()
            if lost_ack:
                with pytest.raises(OSError, match="old_turn_committed_ack_lost"):
                    await child
            else:
                assert await child == (1 if operation == "append" else None)
            known, pending = store.known_pending(sid)
            assert known and pending["tool_call_ids"] == ["new"]
            state = await store.get_state(sid)
            assert state.next_seq == 4 and state.round_index == 7
            assert state.pending["tool_call_ids"] == ["new"]
    finally:
        release.set()
        if child is not None:
            await asyncio.gather(child, return_exceptions=True)
        await loop.aclose(drain_timeout_s=.1)


async def test_model_reply_after_public_lease_takeover_never_enters_history(registry_store, monkeypatch):
    store, connect = registry_store
    now = [1000]
    monkeypatch.setattr(store_module, "_now_ms", lambda: now[0])
    model = Model()
    loop = StatefulAgentLoop(llm=model, store=store,
        config=AgentLoopConfig(distributed_sessions=True, session_lease_ttl_s=300, compactor=None))
    sid = await loop.new_session()
    other = await connect(schema="compatible")
    sending = asyncio.create_task(loop.send("original", sid))
    try:
        await asyncio.wait_for(model.entered.wait(), 5)
        now[0] += 300001
        assert await other.acquire_session_lease(sid, owner_id="replacement", ttl_ms=60000)
        model.release.set()
        with pytest.raises(SessionLeaseLost):
            await asyncio.wait_for(sending, 10)
        assert len(model.calls) == 1
        assert [row.role for row in await other.load_all_messages(sid)] == ["user"]
        assert (await other.session_lease_holder(sid))["owner_id"] == "replacement"
    finally:
        model.release.set()
        if not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        await loop.aclose(drain_timeout_s=.1)
