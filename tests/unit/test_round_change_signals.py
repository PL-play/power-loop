"""Committed changes survive crashes and the last model round cannot hide input."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop, ToolDefinition
from power_loop.agent.sink import SQLiteSink
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.runtime_state import BackgroundRuntimeProjector
from power_loop.runtime.store import store as store_module
from power_loop.runtime.store.change_signals import COLUMNS
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from ._change_signal_crash_worker import write
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def versions(store, sid):
    row = await store._db.fetchone(f"SELECT {','.join(COLUMNS)} FROM {store.t.session_state} WHERE session_id=?", (sid,))
    return tuple(row[column] for column in COLUMNS)


async def payload(store, sid, kind):
    if kind == "inbox":
        return await store.inbox_items(sid)
    if kind == "background":
        return await store.list_unseen_background_updates(sid)
    return await store.get_runtime_state(sid, "todo")


@pytest.mark.parametrize("kind", ["inbox", "background", "runtime"])
@pytest.mark.parametrize("stage", ["after_signal", "before_commit"])
async def test_process_death_rolls_back_signal_and_payload_together(registry_store, tmp_path, kind, stage):
    store, connect = registry_store
    sid = await store.create_session()
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._change_signal_crash_worker",
        dsn, store.table_prefix, sid, kind, stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (91 if stage == "after_signal" else 92), output.decode()
    cold = await connect(schema="compatible")
    assert await versions(cold, sid) == (0, 0, 0)
    assert not await payload(cold, sid, kind)
    await write(cold, sid, kind)
    expected = tuple(int(column == kind + "_version") for column in COLUMNS)
    assert await versions(cold, sid) == expected
    assert await payload(cold, sid, kind)


@pytest.mark.parametrize("kind", ["inbox", "background", "runtime"])
@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_signal_and_payload_share_failed_or_unknown_commit(registry_store, monkeypatch, kind, fault):
    store, connect = registry_store
    sid = await store.create_session()
    transaction = store._db.transaction

    @asynccontextmanager
    async def fail():
        async with transaction() as tx:
            yield tx
            if fault == "rollback":
                raise OSError("before_commit")
        raise OSError("committed_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(store._db, "transaction", fail)
        with pytest.raises(OSError, match="before_commit|committed_ack_lost"):
            await write(store, sid, kind)
    cold = await connect(schema="compatible")
    expected = tuple(int(fault == "ack_loss" and column == kind + "_version") for column in COLUMNS)
    assert await versions(cold, sid) == expected
    assert bool(await payload(cold, sid, kind)) == (fault == "ack_loss")


class Model(LLMService):
    def __init__(self):
        self.calls = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.tool = False
        self.pass_turn = False

    async def complete(self, request, **kwargs):
        self.calls.append(str(request.messages))
        if len(self.calls) == 1:
            self.entered.set()
            await self.release.wait()
            if self.pass_turn:
                return LLMResponse(raw_text="", tool_calls=[{"id": "pass", "type": "function",
                    "function": {"name": "pass_turn", "arguments": '{"reason":"wait"}'}}])
            if self.tool:
                return LLMResponse(raw_text="", tool_calls=[{"id": "change", "type": "function",
                    "function": {"name": "change", "arguments": "{}"}}])
        return LLMResponse(raw_text="first" if len(self.calls) == 1 else "updated")

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


def fixture_loop(store, model, registry=None, projectors=None):
    runtime = TaskRuntime(JobManager(store), tenant_id="change-test", principal_id="host",
                          executor_id="executor", boot_id="boot", boot_contract_hash="c" * 64)
    config = {"runtime_projectors": projectors} if projectors is not None else {}
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry or ToolRegistry(),
        config=AgentLoopConfig(max_rounds=4, compactor=None, delegation_policy="single_level", **config))
    return runtime, loop


@pytest.mark.parametrize("kind", ["inbox", "background", "background_unacked"])
@pytest.mark.parametrize("stage", ["during_model", "after_final_append"])
async def test_update_during_last_model_round_is_consumed_before_completion(registry_store, monkeypatch, kind, stage):
    store, connect = registry_store
    model = Model()
    projectors = (BackgroundRuntimeProjector(mark_seen=False),) if kind == "background_unacked" else None
    runtime, loop = fixture_loop(store, model, projectors=projectors)
    sid = await loop.new_session()
    other = await connect(schema="compatible")
    written = False
    original = SQLiteSink.on_message_appended
    async def append(self, message, **kwargs):
        nonlocal written
        await original(self, message, **kwargs)
        if (stage == "after_final_append" and not written and self.session_id == sid
                and message.get("role") == "assistant" and not message.get("tool_calls")):
            written = True
            await write(other, sid, "background" if kind.startswith("background") else kind)
    monkeypatch.setattr(SQLiteSink, "on_message_appended", append)
    sending = asyncio.create_task(loop.send("start", sid))
    try:
        await asyncio.wait_for(model.entered.wait(), 5)
        if stage == "during_model":
            await write(other, sid, "background" if kind.startswith("background") else kind)
        model.release.set()
        result = await asyncio.wait_for(sending, 10)
        assert result.status == "completed" and result.final_text == "updated"
        assert len(model.calls) == 2 and "once" in model.calls[1]
        if kind == "inbox":
            [item] = await store.inbox_items(sid)
            assert item["status"] == "delivered"
        elif kind == "background":
            assert not await store.list_unseen_background_updates(sid)
    finally:
        model.release.set()
        if not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


async def test_todo_change_replaces_cached_state_in_next_round(registry_store):
    store, connect = registry_store
    model = Model()
    model.tool = True
    model.release.set()
    sid = await store.create_session()
    await store.set_runtime_state(sid, "todo", {"rendered": "before-change"})
    other = await connect(schema="compatible")
    registry = ToolRegistry()
    async def change():
        await other.set_runtime_state(sid, "todo", {"rendered": "after-change"})
        return "changed"
    registry.register(ToolDefinition("change", "Change todo", input_schema={"type": "object"}, pricing=Free("fixture")), change)
    runtime, loop = fixture_loop(store, model, registry)
    try:
        result = await loop.send("start", sid)
        assert result.status == "completed" and len(model.calls) == 2
        assert "before-change" in model.calls[0]
        assert "after-change" in model.calls[1] and "before-change" not in model.calls[1]
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("kind", ["inbox", "background"])
async def test_pass_turn_checks_after_last_append_without_restarting_the_model(registry_store, monkeypatch, kind):
    store, connect = registry_store
    model = Model()
    model.pass_turn = True
    model.release.set()
    runtime, loop = fixture_loop(store, model)
    sid = await loop.new_session()
    other = await connect(schema="compatible")
    original = SQLiteSink.on_message_appended
    async def append(self, message, **kwargs):
        await original(self, message, **kwargs)
        if self.session_id == sid and message.get("role") == "tool" and message.get("tool_call_id") == "pass":
            await write(other, sid, kind)
    monkeypatch.setattr(SQLiteSink, "on_message_appended", append)
    try:
        result = await loop.send("pass after this round", sid)
        assert result.passed and result.new_input_after_pass and len(model.calls) == 1
        if kind == "inbox":
            [row] = await store.inbox_items(sid)
            assert row["status"] == "delivered"
        else:
            assert len(await store.list_unseen_background_updates(sid)) == 1
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


async def test_unchanged_counter_still_reclaims_a_crashed_claim_at_its_deadline(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await store.create_session()
    now = [100000]
    monkeypatch.setattr(store_module, "_now_ms", lambda: now[0])
    await write(store, sid, "inbox")
    changes, token = store.begin_polling(sid)
    try:
        await changes.refresh()
        [first] = await store.inbox_claim(sid)
        await changes.refresh()
        assert await store.inbox_claim(sid) == []
        assert not changes.inbox_due(now[0])
        now[0] += 60001
        assert changes.inbox_due(now[0])
        [second] = await store.inbox_claim(sid)
        assert second["id"] == first["id"] and second["claim_token"] != first["claim_token"]
    finally:
        store.end_polling(token)


async def test_concurrent_writers_advance_counter_without_losing_input(registry_store):
    store, connect = registry_store
    sid = await store.create_session()
    other = await connect(schema="compatible")
    await asyncio.wait_for(asyncio.gather(*(
        (store if number % 2 else other).inbox_put(sid, [{"item_id": str(number), "content": str(number)}])
        for number in range(8))), 10)
    assert await versions(store, sid) == (8, 0, 0)
    assert {row["item_id"] for row in await store.inbox_items(sid)} == {str(number) for number in range(8)}
