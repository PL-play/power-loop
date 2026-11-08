"""Actual loop adapters, admission failures, receipt identity and cancellation."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    JobRef,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class AdapterModel(LLMService):
    def __init__(self):
        self.root = None
        self.root_calls = 0
        self.child_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.tool_name = "spawn_agent"

    async def complete(self, request, **kwargs):
        if get_session_id() != self.root:
            self.child_calls += 1
            self.entered.set()
            await self.release.wait()
            text, calls = "CHILD-PROOF-42", None
        else:
            self.root_calls += 1
            text, calls = "done", None
            if self.root_calls == 1:
                text = ""
                calls = [{"id": "actual-create", "type": "function", "function": {
                    "name": self.tool_name, "arguments": json.dumps({"task": "Return CHILD-PROOF-42", "tools": [], "max_rounds": 2})}}]
        return LLMResponse(raw_text=text, tool_calls=calls, token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.fixture
async def managed(registry_store, runtime_agent, request):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="isolated-adapter", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=1000)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    model = AdapterModel()
    loop = await runtime_agent(model=model, store=store, task_runtime=runtime, registry=registry,
        capacity=getattr(request, "param", {}).get("capacity", 1),
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, compactor=None))
    sid = loop.target.session_id
    model.root = sid
    try:
        yield store, manager, runtime, loop, model, sid, connect
    finally:
        model.release.set()
        await loop.aclose()


async def jobs(managed):
    _, manager, runtime, _, _, sid, _ = managed
    rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} ORDER BY created_at,job_id")
    return [await manager.get_job(runtime.actor(sid), row["job_id"]) for row in rows]


async def receipt(managed):
    store, _, _, _, _, sid, _ = managed
    return next(json.loads(r.content) for r in await store.load_all_messages(sid) if r.role == "tool")


async def finish(managed):
    _, _, _, loop, model, _, _ = managed
    model.release.set()
    await asyncio.wait_for(loop.drain(), 10)


async def test_actual_create_receipt_child_send_and_result_survive_creator_close_and_reopen(managed):
    store, manager, runtime, loop, model, sid, connect = managed
    result = await loop.send("delegate")
    assert result.status == "completed"
    await asyncio.wait_for(model.entered.wait(), 3)
    accepted = await receipt(managed)
    records = await jobs(managed)
    root = next(j for j in records if j.kind == "send" and j.session_id == sid)
    child = next(j for j in records if j.kind == "subagent")
    child_send = next(j for j in records if j.kind == "send" and j.session_id != sid)
    assert root.status == "model_closed" and child.status == child_send.status == "running"
    assert child.parent_job_id == root.job_id and child.control_owner_job_id is None
    assert child_send.control_owner_job_id == child.job_id and child_send.execution_role == "subagent"
    operation = await manager.get_operation(runtime.actor(sid), accepted["operation_id"])
    call, tool = operation.call_ref, operation.receipt_ref
    assert call["id"] != tool["id"] and call["tool_call_id"] is None and tool["tool_call_id"] == "actual-create"
    assert call["role"] == "assistant" and tool["role"] == "tool"
    assert json.loads((await store.message_by_id(sid, tool["id"])).content) == accepted
    await finish(managed)
    reopened = JobManager(await connect())
    saved = await reopened.get_job(runtime.actor(sid), child.job_id)
    assert saved.result["final_text"] == "CHILD-PROOF-42" and saved.result["usage"]["total_tokens"] == 10
    assert saved.status == "settling" and saved.supervision_status == "closed" and saved.resource_state == "stopped"
    assert (await store.inbox_counts(sid))["pending"] == 0
    await loop.drain()
    assert (await manager.get_job(runtime.actor(sid), child.job_id)).result_status == "handled"
    assert len(await jobs(managed)) == 4 and model.child_calls == 1


async def test_duplicate_actual_call_reuses_admission_without_dispatching_twice(managed, monkeypatch):
    _, _, _, loop, model, sid, _ = managed
    original = delegation.start
    receipts = []

    async def twice(*args, **kwargs):
        receipts.append(await original(*args, **kwargs))
        receipts.append(await original(*args, **kwargs))
        return receipts[0]

    monkeypatch.setattr(delegation, "start", twice)
    assert (await loop.send("delegate")).status == "completed"
    assert receipts[0]["job_id"] == receipts[1]["job_id"]
    assert receipts[0] == receipts[1]
    await finish(managed)
    assert model.child_calls == 1 and len(await jobs(managed)) == 4


@pytest.mark.parametrize("failure", ["registration", "session_binding"])
async def test_required_admission_failure_never_starts_child_provider(managed, monkeypatch, failure):
    store, manager, _, loop, model, sid, _ = managed
    original = manager.register_job

    async def fail_registration(*args, **kwargs):
        if kwargs.get("kind") == "subagent":
            raise JobError("injected_admission_failure")
        return await original(*args, **kwargs)

    original_create = store._create_session_tx
    async def fail_binding(*args, **kwargs):
        if kwargs.get('metadata', {}).get('execution_role') == 'subagent':
            raise JobError("injected_admission_failure")
        return await original_create(*args, **kwargs)

    if failure == 'registration':
        monkeypatch.setattr(manager, 'register_job', fail_registration)
    else:
        monkeypatch.setattr(store, '_create_session_tx', fail_binding)
    assert (await loop.send("delegate")).status == "completed"
    await finish(managed)
    assert model.child_calls == 0
    tool = next(r for r in await store.load_all_messages(sid) if r.role == "tool")
    assert "injected_admission_failure" in tool.content
    child_jobs = [j for j in await jobs(managed) if j.kind == "subagent"]
    assert not child_jobs  # session and child admission roll back in one transaction


async def test_cancel_queued_child_persists_intent_and_result_without_provider_call(managed, monkeypatch):
    store, manager, runtime, agent, model, sid, _ = managed
    acquire = agent.runtime.coordinator.acquire
    async def parent_only(key, **kwargs):
        return await acquire(key, **kwargs) if key[1] == sid else None
    monkeypatch.setattr(agent.runtime.coordinator, 'acquire', parent_only)
    assert (await agent.send('delegate')).status == 'completed'
    accepted = await receipt(managed)
    ref = JobRef(**accepted['job_ref'])
    await agent.runtime.client.cancel(ref, actor=agent.actor, reason='stop queued child')
    result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+10000)
    assert result.status == 'cancelled' and result.job_ref.generation == 0
    assert model.child_calls == 0
    child = await manager.get_job(runtime.actor(sid), accepted['job_id'])
    assert child.cancel_requested_at is not None and child.resource_state == 'stopped'
    await agent.drain()


async def test_same_store_parallel_main_sessions_have_isolated_job_roots_and_message_origins(managed):
    store, manager, runtime, agent, model, sid, _ = managed
    model.root_calls = 10
    model.release.set()
    second = await agent.runtime.open_session(agent.descriptor, actor=agent.actor)
    client = agent.runtime.client
    receipts = await asyncio.gather(
        client.send(agent.target, 'first', message_id='parallel-1', actor=agent.actor),
        client.send(second, 'second', message_id='parallel-2', actor=agent.actor))
    results = await asyncio.gather(*(client.wait(r.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000) for r in receipts))
    assert all(r.status == 'completed' for r in results)
    await agent.drain()
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert len(rows) == 2 and {r['session_id'] for r in rows} == {sid, second.session_id}
    assert all(r['root_job_id'] == r['job_id'] and r['parent_job_id'] is None for r in rows)
    for row in rows:
        unrelated = runtime.actor(second.session_id if row['session_id'] == sid else sid)
        with pytest.raises(JobError, match='denied'):
            await manager.get_job(unrelated, row['job_id'])
    assert await store.list_children(sid) == []


@pytest.mark.parametrize('scope', ['self', 'subtree'])
async def test_explicit_parent_cancel_controls_only_requested_scope(managed, scope):
    store, manager, runtime, agent, model, sid, _ = managed
    parent = await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(), 5)
    accepted = await receipt(managed)
    child_ref = JobRef(**accepted['job_ref'])
    await agent.runtime.client.cancel(parent.job_ref, actor=agent.actor, scope=scope, reason='parent control')
    if scope == 'self':
        current = await agent.runtime.client.inspect(child_ref, actor=agent.actor)
        assert current.state == 'running' and not current.cancel_requested
        model.release.set()
    result = await agent.runtime.client.wait(child_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
    assert result.status == ('completed' if scope == 'self' else 'cancelled')
    if scope == 'self':
        assert result.final_text == 'CHILD-PROOF-42' and result.usage['total_tokens'] == 10
    await agent.drain()
    assert model.child_calls == 1
    assert (await agent.runtime.client.wait(parent.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)) == parent


@pytest.mark.parametrize('case', ['invalid_lease', 'invalid_executor'])
async def test_host_rejects_invalid_execution_configuration(managed, case):
    _, manager, _, _, _, _, _ = managed
    with pytest.raises(ValueError):
        TaskRuntime(manager, tenant_id='test', principal_id='host',
            executor_id='' if case=='invalid_executor' else 'exec', boot_id='boot',
            lease_ms=0 if case=='invalid_lease' else 30000)
