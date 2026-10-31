"""A live send reuses its entry facts; cold inspection and later sends do not."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop import LLMResponse, LLMService
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.jobs import TRUST_OWN_WRITES, verify_stored

from .test_job_registry_192 import registry_store
from .test_round_change_signals import fixture_loop

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_committed_appends_and_pending_changes_refresh_owned_state(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    async with store.send_read_scope(sid):
        token = TRUST_OWN_WRITES.set(True)
        try:
            assert (await store.state_snapshot(sid)).next_seq == 1
            calls = [{"id": "call", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]
            await store.append_message(sid, role="assistant", tool_calls=calls, pending_tool_calls=True)
            state = await store.state_snapshot(sid)
            assert state.next_seq == 2 and state.pending["assistant_seq"] == 1
            store.defer_round_index(sid, 4)
            await store.append_message(sid, role="tool", tool_call_id="call", content="done", pending=None)
            state = await store.state_snapshot(sid)
            assert state.next_seq == 3 and state.pending is None and state.round_index == 4
            assert state == await store.get_state(sid)
        finally:
            TRUST_OWN_WRITES.reset(token)


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_failed_append_discards_entry_hint_before_retry_or_inspection(registry_store, monkeypatch, fault):
    store, _ = registry_store
    sid = await store.create_session()
    transaction = store._db.transaction
    @asynccontextmanager
    async def fail():
        async with transaction() as tx:
            yield tx
            if fault == "rollback":
                raise OSError("rolled_back")
        raise OSError("committed_ack_lost")
    async with store.send_read_scope(sid):
        token = TRUST_OWN_WRITES.set(True)
        try:
            assert (await store.state_snapshot(sid)).next_seq == 1
            with monkeypatch.context() as patch:
                patch.setattr(store._db, "transaction", fail)
                with pytest.raises(OSError, match="rolled_back|committed_ack_lost"):
                    await store.append_message(sid, role="user", content="original")
            state = await store.state_snapshot(sid)
            assert state.next_seq == (1 if fault == "rollback" else 2)
            assert state == await store.get_state(sid)
            assert len(await store.load_all_messages(sid)) == int(fault == "ack_loss")
        finally:
            TRUST_OWN_WRITES.reset(token)


async def test_cold_verification_reads_committed_values_even_inside_live_send(registry_store):
    store, connect = registry_store
    sid = await store.create_session(system_prompt="first")
    await store.set_runtime_state(sid, "send_index", 1)
    other = await connect(schema="compatible")
    @verify_stored
    async def inspect():
        return (await store.session_snapshot(sid), await store.state_snapshot(sid),
                await store.runtime_snapshot(sid, "send_index"))
    async with store.send_read_scope(sid):
        token = TRUST_OWN_WRITES.set(True)
        try:
            assert (await store.session_snapshot(sid)).system_prompt == "first"
            assert (await store.state_snapshot(sid)).next_seq == 1
            assert await store.runtime_snapshot(sid, "send_index") == 1
            await other.update_session_prompt(sid, "second")
            await other.append_message(sid, role="user", content="committed elsewhere")
            await other.set_runtime_state(sid, "send_index", 2)
            session, state, index = await inspect()
            assert session.system_prompt == "second" and state.next_seq == 2 and index == 2
        finally:
            TRUST_OWN_WRITES.reset(token)


class Model(LLMService):
    def __init__(self):
        self.seen = []
        self.gate = None
        self.spawn = None
        self.child = None

    async def complete(self, request, **kwargs):
        self.seen.append((get_session_id(), request.system_prompt.split("\n\n", 1)[0]))
        if len(self.seen) == 1 and self.spawn is not None:
            self.child = asyncio.create_task(self.spawn())
        if self.gate is not None:
            if len(self.seen) == 2:
                self.gate.set()
            await self.gate.wait()
        return LLMResponse(raw_text="done")

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def test_next_send_reloads_changed_session_and_advances_original_index(registry_store):
    store, connect = registry_store
    model = Model()
    runtime, loop = fixture_loop(store, model)
    sid = await store.create_session(system_prompt="first")
    try:
        assert (await loop.send("one", sid)).status == "completed"
        other = await connect(schema="compatible")
        await other.update_session_prompt(sid, "second")
        assert (await loop.send("two", sid)).status == "completed"
        assert model.seen == [(sid, "first"), (sid, "second")]
        rows = await store.load_all_messages(sid)
        assert [row.send_index for row in rows] == [1, 1, 2, 2]
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


async def test_concurrent_sessions_never_share_entry_reads(registry_store):
    store, _ = registry_store
    model = Model()
    model.gate = asyncio.Event()
    runtime, loop = fixture_loop(store, model)
    first, second = await store.create_session(system_prompt="alpha"), await store.create_session(system_prompt="beta")
    try:
        results = await asyncio.wait_for(asyncio.gather(loop.send("first", first), loop.send("second", second)), 10)
        assert all(result.status == "completed" for result in results)
        assert sorted(model.seen) == sorted([(first, "alpha"), (second, "beta")])
        for sid in (first, second):
            rows = await store.load_all_messages(sid)
            assert len(rows) == 2 and all(row.send_index == 1 for row in rows)
            assert (await store.get_state(sid)).next_seq == 3
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


async def test_inherited_background_context_cannot_reuse_a_finished_sends_snapshot(registry_store):
    store, connect = registry_store
    model = Model()
    runtime, loop = fixture_loop(store, model)
    sid = await store.create_session(system_prompt="first")
    release = asyncio.Event()
    async def later():
        await release.wait()
        return await loop.send("second", sid)
    model.spawn = later
    try:
        assert (await loop.send("first", sid)).status == "completed"
        other = await connect(schema="compatible")
        await other.update_session_prompt(sid, "second")
        release.set()
        assert (await asyncio.wait_for(model.child, 10)).status == "completed"
        assert model.seen == [(sid, "first"), (sid, "second")]
        assert [row.send_index for row in await store.load_all_messages(sid)] == [1, 1, 2, 2]
    finally:
        release.set()
        if model.child is not None and not model.child.done():
            model.child.cancel()
            await asyncio.gather(model.child, return_exceptions=True)
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
