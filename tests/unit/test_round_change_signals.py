"""Committed changes survive crashes and the last model round cannot hide input."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop import LLMResponse, LLMService, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.sink import SQLiteSink
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.change_signals import COLUMNS
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

from ._change_signal_crash_worker import prepare_background_batch, serialize_batch, write

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def versions(store, sid):
    row = await store._db.fetchone(f"SELECT {','.join(COLUMNS)} FROM {store.t.sessions} WHERE session_id=?", (sid,))
    return tuple(row[column] for column in COLUMNS)


def expected_versions(kind: str, *, committed: bool) -> tuple[int, int, int]:
    if not committed:
        return (0, 0, 0)
    # Schema 18 routes background child completions via River into the parent's follow_up_queue
    # and bumps parent.inbox_version in the same _final_tx transaction (background_version is unused).
    if kind in ("inbox", "background"):
        return (1, 0, 0)
    return (0, 0, 1)


async def payload(store, sid, kind):
    if kind == "inbox":
        return await store._db.fetchall(
            f"SELECT item_id,kind,content_json FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='user'", (sid,)
        )
    if kind == "background":
        return await store._db.fetchall(
            f"SELECT item_id,kind,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='background_result'",
            (sid,),
        )
    return await store.get_runtime_state(sid, "todo")


@pytest.mark.parametrize("kind", ["inbox", "background", "runtime"])
@pytest.mark.parametrize("stage", ["after_signal", "before_commit"])
async def test_process_death_rolls_back_signal_and_payload_together(registry_store, tmp_path, kind, stage):
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, connect = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    batch = await prepare_background_batch(store, sid) if kind == "background" else None
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    extra_args = [serialize_batch(batch)] if batch is not None else []
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._change_signal_crash_worker",
        dsn, store.table_prefix, sid, kind, stage, *extra_args, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (91 if stage == "after_signal" else 92), output.decode()
    cold = await connect(schema="verify")
    assert await versions(cold, sid) == (0, 0, 0)
    assert not await payload(cold, sid, kind)
    if batch is not None:
        cold_port = RuntimeInputStore(cold)
        assert await cold_port.get_result(batch.input_refs[0], actor=STORE_ACTOR) is None
        child_row = await cold._db.fetchone(
            f"SELECT execution_state FROM {cold.t.sessions} WHERE session_id=?", (batch.target.session_id,)
        )
        assert child_row["execution_state"] == "running"
    await write(cold, sid, kind, batch=batch)
    assert await versions(cold, sid) == expected_versions(kind, committed=True)
    written = await payload(cold, sid, kind)
    assert written
    if batch is not None:
        assert "once" in written[0]["content_json"]
        res = await RuntimeInputStore(cold).get_result(batch.input_refs[0], actor=STORE_ACTOR)
        assert res is not None and res.status == "completed" and res.final_text == "once"


@pytest.mark.parametrize("kind", ["inbox", "background", "runtime"])
@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_signal_and_payload_share_failed_or_unknown_commit(registry_store, monkeypatch, kind, fault):
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import StoreError

    store, connect = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    batch = await prepare_background_batch(store, sid) if kind == "background" else None
    transaction = store._db.transaction
    injected: list[str] = []

    @asynccontextmanager
    async def fail():
        async with transaction() as tx:
            yield tx
            if fault == "rollback":
                injected.append("before_commit")
                raise OSError("before_commit")
        injected.append("committed_ack_lost")
        raise OSError("committed_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(store._db, "transaction", fail)
        if kind == "background" and fault == "ack_loss":
            # RuntimeInputStore.final reconciles post-commit ACK loss via _final_after_unknown.
            reconciled = await write(store, sid, kind, batch=batch)
            assert reconciled is not None and reconciled.results[0].status == "completed"
        else:
            with pytest.raises((OSError, StoreError)):
                await write(store, sid, kind, batch=batch)
    assert injected == ["before_commit" if fault == "rollback" else "committed_ack_lost"]
    cold = await connect(schema="verify")
    committed = fault == "ack_loss"
    assert await versions(cold, sid) == expected_versions(kind, committed=committed)
    written = await payload(cold, sid, kind)
    assert bool(written) == committed
    if batch is not None:
        res = await RuntimeInputStore(cold).get_result(batch.input_refs[0], actor=STORE_ACTOR)
        if committed:
            assert "once" in written[0]["content_json"]
            assert res is not None and res.status == "completed" and res.final_text == "once"
        else:
            assert res is None


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


@pytest.mark.parametrize("stage", ["during_model", "after_final_append"])
async def test_update_during_last_model_round_is_consumed_before_completion(registry_store, monkeypatch, stage):
    store, _ = registry_store
    model = Model()
    runner = await RunningAgent(store, model).start()
    original = SQLiteSink.on_message_appended
    appended = False
    async def inject():
        nonlocal appended
        if not appended:
            appended = True
            return await runner.runtime.client.send(runner.target, "once", actor=runner.actor, message_id="late")
    async def append(self, message, **kwargs):
        await original(self, message, **kwargs)
        if stage == "after_final_append" and message.get("role") == "assistant" and not message.get("tool_calls"):
            await inject()
    monkeypatch.setattr(SQLiteSink, "on_message_appended", append)
    sending = asyncio.create_task(runner.send("start"))
    try:
        await asyncio.wait_for(model.entered.wait(), 10)
        if stage == "during_model":
            await inject()
        model.release.set()
        result = await asyncio.wait_for(sending, 20)
        # Design 217 §4.3: 0-tool final boundary pays 0 SQL; a normal-priority input arriving
        # during the final model round transitions the session to 'queued' in final() and is
        # consumed on the immediately following send without losing the input.
        assert result.status == "completed" and result.final_text == "first"
        late = await runner.runtime.client.wait(
            __import__('power_loop').InputRef(store.store_id, runner.target.session_id, 'late'),
            actor=runner.actor, deadline_ms=now_ms()+5000)
        assert late.status == "completed" and late.final_text == "updated"
        assert late.send_index == result.send_index + 1
        assert len(model.calls) == 2 and "once" in model.calls[1]
    finally:
        model.release.set()
        await runner.aclose()


async def test_todo_change_replaces_cached_state_in_next_round(registry_store):
    store, connect = registry_store
    model = Model()
    model.tool = True
    model.release.set()
    registry = ToolRegistry()
    other = await connect(schema="verify")
    async def change():
        await other.set_runtime_state(runner.target.session_id, "todo", {"rendered": "after-change"})
        return "changed"
    registry.register(ToolDefinition("change", "Change todo", input_schema={"type":"object"}, pricing=Free("fixture")), change)
    runner = await RunningAgent(store, model, registry).start()
    await store.set_runtime_state(runner.target.session_id, "todo", {"rendered": "before-change"})
    try:
        result = await runner.send("start")
        assert result.status == "completed" and len(model.calls) == 2
        assert "before-change" in model.calls[0]
        assert "after-change" in model.calls[1] and "before-change" not in model.calls[1]
    finally:
        await runner.aclose()


async def test_pass_turn_accepts_late_append_without_losing_it(registry_store, monkeypatch):
    from power_loop import create_default_tool_registry
    store, _ = registry_store
    model = Model()
    model.pass_turn = True
    model.release.set()
    runner = await RunningAgent(store, model, create_default_tool_registry(bind=False)).start()
    original = SQLiteSink.on_message_appended
    receipt = None
    async def append(self, message, **kwargs):
        nonlocal receipt
        await original(self, message, **kwargs)
        if message.get("role") == "tool" and message.get("tool_call_id") == "pass":
            receipt = await runner.runtime.client.send(runner.target, "once", message_id="after-pass", actor=runner.actor)
    monkeypatch.setattr(SQLiteSink, "on_message_appended", append)
    try:
        result = await runner.send("pass after this round")
        assert result.status == "completed" and receipt is not None
        late = await runner.runtime.client.wait(receipt.input_ref, actor=runner.actor, deadline_ms=now_ms()+10000)
        await runner.drain()
        assert late.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        assert sum(row.role == "user" and "once" in str(row.content) for row in rows) == 1
    finally:
        await runner.aclose()


async def test_concurrent_writers_advance_counter_without_losing_input(registry_store):
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import TargetRef

    store, connect = registry_store
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    other = await connect(schema="verify")
    ports = (RuntimeInputStore(other), RuntimeInputStore(store))
    target = TargetRef(store.store_id, sid)
    await asyncio.wait_for(asyncio.gather(*(
        ports[number % 2].accept(target, str(number), message_id=str(number), actor=STORE_ACTOR)
        for number in range(8))), 10)
    assert await versions(store, sid) == (8, 0, 0)
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?", (sid,)
    )
    assert {row["item_id"] for row in rows} == {str(number) for number in range(8)}
