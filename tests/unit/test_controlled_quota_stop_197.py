"""Host quota stop completes lifecycle without another paid model request."""

from collections.abc import AsyncIterator

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop._vendor.llm_client.interface import LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.hooks import HookPoint
from power_loop.core.control import ControlledLoopStop
from power_loop.core.hooks import AgentHooks
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


class Scripted(LLMService):
    calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        return LLMResponse(
            raw_text="",
            tool_calls=[{"id": "one", "type": "function", "function": {"name": "paid_work", "arguments": "{}"}}],
        )

    def stream(self, request) -> AsyncIterator[LLMStreamChunk]:
        async def empty():
            if False:
                yield LLMStreamChunk()

        return empty()

    async def close(self):
        pass


async def test_entry_stop_makes_no_model_call_and_finalizes_once():
    hooks = AgentHooks()
    ends = []

    def stop(ctx):
        raise ControlledLoopStop("quota_exhausted", message="stop")

    hooks.register(HookPoint.SESSION_START, stop)
    hooks.register(HookPoint.SESSION_END, lambda ctx: ends.append(ctx.reason))
    llm = Scripted()
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm, store=store, hooks=hooks, config=AgentLoopConfig(system_prompt="test", compactor=None)
    )
    result = await loop.send("go", session_id=await loop.new_session())
    assert result.status == "budget_exceeded" and result.final_text == "stop"
    assert llm.calls == 0 and ends == ["quota_exhausted"]
    await store.close()


async def test_current_round_tool_finishes_then_next_round_stops_without_summary():
    hooks = AgentHooks()
    completed = []
    ended = []

    def check(ctx):
        if completed:
            raise ControlledLoopStop("quota_exhausted", message="out of points")

    hooks.register(HookPoint.ROUND_START, check)
    hooks.register(HookPoint.SESSION_END, lambda ctx: ended.append(ctx.reason))
    reg = ToolRegistry()

    def paid_work():
        completed.append("saved result and fee")
        return "done"

    reg.register(
        ToolDefinition(
            name="paid_work",
            description="test",
            input_schema={"type": "object", "properties": {}},
            pricing=Free("fixture"),
        ),
        paid_work,
    )
    llm = Scripted()
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=reg,
        hooks=hooks,
        config=AgentLoopConfig(system_prompt="test", max_rounds=4, compactor=None),
    )
    sid = await loop.new_session()
    result = await loop.send("go", session_id=sid)
    assert result.status == "budget_exceeded" and result.rounds == 1
    assert llm.calls == 1 and completed == ["saved result and fee"]
    assert ended == ["quota_exhausted"]
    assert any(m.get("role") == "tool" and m.get("content") == "done" for m in await loop.get_messages(sid))
    await store.close()


async def test_host_creation_snapshot_is_durable_and_not_replaced_on_replay(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext

    from power_loop.runtime import invocation
    from power_loop.runtime.jobs import JobActor, JobManager
    from power_loop.runtime.store.factory import open_store

    monkeypatch.setattr(invocation, "_SCOPE_FACTORIES", {})
    snapshot = {"conversation_id": 7, "calls": [{"call_id": "outer", "name": "parent"}]}
    invocation.register_invocation_scope("host.trace", lambda **kw: nullcontext(), exporter=lambda: snapshot)
    path = str(tmp_path / "trace.db")
    store = await open_store(path, schema="tasks", table_prefix="trace197_")
    manager = JobManager(store)
    sid = await store.create_session()
    actor = JobActor("tenant", "owner", sid)
    original = await manager.register_job(actor, kind="send", session_id=sid, system_key="original")
    captured = json.loads(json.dumps(snapshot))
    snapshot["calls"][0]["name"] = "later modification"
    replay = await manager.register_job(actor, kind="send", session_id=sid, system_key="original")
    assert replay.job.job_id == original.job.job_id
    await store.close()
    reopened = await open_store(path, schema="tasks", table_prefix="trace197_")
    event = await reopened._db.fetchone(
        "SELECT payload_json FROM trace197_job_events WHERE job_id=? AND action='create'", (original.job.job_id,)
    )
    payload = json.loads(event["payload_json"])
    assert payload["host_contexts"]["host.trace"] == captured
    assert "host_contexts" not in payload.get("request", {})
    await reopened.close()
