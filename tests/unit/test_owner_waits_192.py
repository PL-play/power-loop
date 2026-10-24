"""Actual original controllers wait after model exit, not inside provider gates."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation, owner_waits
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobError, JobLease, JobManager
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool
from power_loop.workflow.admission import GATES, admit, idle
from power_loop.workflow.engine import WorkflowEngine

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Provider(LLMService):
    def __init__(self, state):
        self.state, self.calls = state, {"main": 0, "child": 0}

    async def complete(self, request, **kwargs):
        state = self.state
        role = "main" if get_session_id() == state.parent else "child"
        self.calls[role] += 1
        calls = None
        text = "Original result: 42."
        if role == "main" and self.calls[role] == 1:
            args = {"name": "original", "system_prompt": "Original finite task", "tools": ["arm_wait"], "task": "Compute 19+23", "max_rounds": 4}
            calls = [("spawn_agent", args)]
        elif role == "main" and state.choice is not None and not state.decided:
            state.decided = True
            calls = [("wait_handoff", {"action": "decide", "event_id": state.due_id, "decision": state.choice})]
        elif role == "child" and self.calls[role] == 1:
            calls = [("arm_wait", {})]
        elif role == "child" and self.calls[role] == 2 and state.ending == "pass_turn":
            calls = [("pass_turn", {"reason": "Wait for the original continuation decision."})]
        if calls:
            calls = [{"id": role + "-" + str(self.calls[role]), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}} for name, args in calls]
        return LLMResponse(raw_text="" if calls else text, tool_calls=calls,
            token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.fixture
async def original(registry_store, request):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="owner-wait", principal_id="host", executor_id="executor", boot_id="boot",
        boot_contract_hash="a" * 64 if getattr(request, "param", None) == "boot" else None)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry)
    state = SimpleNamespace(ending="natural", registrations=[], parent=None, choice=None, decided=False, due_id=None, wait_ms=5000)
    model = Provider(state)

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Review this original task", "Continue original computation"),
            policy=waits.WaitPolicy(now + state.wait_ms))
        state.registrations.append(registration)
        return "Original wait admitted."

    registry.register(ToolDefinition(name="arm_wait", description="Admit an original wait", input_schema={"type": "object", "properties": {}}), arm)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4))
    state.parent = await loop.new_session()
    before = set(delegation._TASKS)
    try:
        yield SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model, state=state, connect=connect)
    finally:
        for registration in state.registrations:
            owner = await manager.get_job(runtime.actor(state.parent), registration.job.control_owner_job_id)
            if owner.outcome is None:
                await manager.request_cancel(runtime.actor(state.parent), owner.job_id, system_key="test-owner-wait-cleanup", reason="test cleanup")
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)


async def entered(c):
    await c.loop.send("Start original finite task", c.state.parent)
    for _ in range(300):
        if c.state.registrations:
            [registration] = c.state.registrations
            owner = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.control_owner_job_id)
            if owner.status == "waiting":
                return registration, owner
            if owner.outcome is not None:
                pytest.fail(f"Original owner ended instead of waiting: {owner}")
        await asyncio.sleep(.01)
    messages = []
    for child in await c.store.list_children(c.state.parent):
        messages.extend({"role": m.role, "content": str(m.content)[:500]} for m in await c.store.load_all_messages(child.session_id))
    pytest.fail(f"Original controller did not enter durable waiting; provider_calls={c.model.calls}; child_messages={messages}")


async def prepare_due(manager, actor, registration):
    saved = await waits.read_wait(manager, actor, registration.job.job_id)
    # Exercise the saved due; avoid relying on scheduler speed to cross it.
    remaining = saved["request"]["request"]["due_at_ms"] - manager.now()
    if remaining > 0:
        await asyncio.sleep(remaining / 1000)
    return await waits.prepare_due(manager, actor, registration.job.job_id)


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_original_child_outlives_closed_model_without_a_provider_gate_and_continues_once(original, ending):
    c = original
    c.state.ending = ending
    registration, owner = await entered(c)
    assert owner.resource_state == "running" and owner.outcome is None
    [model] = await c.manager.db.fetchall(f"SELECT status,resource_state FROM {c.manager.jobs} WHERE control_owner_job_id=? AND kind='send'", (owner.job_id,))
    assert model == {"status": "model_closed", "resource_state": "stopped"}
    before = c.model.calls.copy()
    await asyncio.sleep(.03)
    assert c.model.calls == before and c.model.calls["child"] == 2
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (owner.job_id,))
    restored = JobManager(await c.connect())
    assert (await restored.get_job(c.runtime.actor(c.state.parent), owner.job_id)).status == "waiting"
    [waiting] = await restored.db.fetchall(f"SELECT payload_json FROM {restored.events} WHERE event_type='owner_waiting' AND job_id=?", (owner.job_id,))
    saved = json.loads(waiting["payload_json"])
    assert saved["raw"]["status"] == "completed" and saved["model_exits"]
    with pytest.raises(JobError, match="owner_wait_unresolved"):
        await c.manager.record_execution_result(c.runtime.actor(c.state.parent), JobLease(**saved["owner"]),
            outcome="succeeded", result={"status": "completed"}, resource_state="stopped")
    with pytest.raises(JobError, match="owner_waiting_requires_input"):
        await c.manager.register_job(c.runtime.actor(owner.session_id), kind="send", parent=JobLease(**saved["owner"]),
            session_id=owner.session_id, system_key="no-model-during-wait")
    due = await prepare_due(c.manager, c.runtime.actor(c.state.parent), registration)
    c.state.choice, c.state.due_id = "continue", due["event_id"]
    await notices.enqueue_notification(c.loop, c.runtime.actor(c.state.parent), due["event_id"])
    await c.loop.flush_follow_ups(c.state.parent)
    await effects.apply(c.manager, c.runtime.actor(c.state.parent), due["event_id"])
    tasks = set(delegation._TASKS)
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    owner = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
    assert owner.outcome == "succeeded" and owner.resource_state == "stopped", owner
    assert c.model.calls["child"] == 3 and len(await c.store.list_children(c.state.parent)) == 1
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.job_id)).outcome == "succeeded"
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")) == 1
    await c.loop.flush_follow_ups(c.state.parent)
    rows = await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}")
    assert all(r["supervision_status"] == "closed" for r in rows), rows


async def test_inherited_parent_leaf_context_cannot_release_the_parent_slot():
    gate = asyncio.Semaphore(1)
    token = GATES.set((gate,))
    try:
        async with admit():
            assert gate.locked()
            async def independent_child():
                async with idle():
                    assert gate.locked()
            await asyncio.create_task(independent_child())
            assert gate.locked()
            async with idle():
                assert not gate.locked()
            assert gate.locked()
        assert not gate.locked()
    finally:
        GATES.reset(token)


@pytest.mark.parametrize("after_ready", ["continue", "cancel", "deadline"])
async def test_original_subagent_releases_its_pool_and_reacquires_before_resume(original, monkeypatch, after_ready):
    c = original
    # This case isolates capacity handoff. Wall-clock request interruption has
    # dedicated tests; slow PostgreSQL setup must not consume this test's policy.
    now = c.manager.now()
    monkeypatch.setattr(c.manager, "now", lambda: now)
    if after_ready == "cancel":
        c.runtime.lease_ms = 3000
    gate = asyncio.Semaphore(1)
    start = delegation.start

    async def bounded(*args, **kwargs):
        return await start(*args, **{**kwargs, "admission": gate})

    monkeypatch.setattr(delegation, "start", bounded)
    registration, owner = await entered(c)
    assert not gate.locked()
    requests, second, release = [], asyncio.Event(), asyncio.Event()
    original_complete = c.model.complete
    requested = False
    issued = False

    async def provider(request, **kwargs):
        nonlocal issued
        sid = get_session_id()
        requests.append(sid)
        if sid not in (c.state.parent, owner.session_id):
            second.set()
            await release.wait()
        response = await original_complete(request, **kwargs)
        if sid == c.state.parent and requested and not issued:
            issued = True
            response.raw_text = ""
            response.tool_calls = [{"id": "unrelated-child", "type": "function", "function": {"name": "spawn_agent", "arguments": json.dumps({
                "name": "unrelated", "system_prompt": "Finish unrelated work", "task": "A separate request", "tools": [], "max_rounds": 2})}}]
        return response

    monkeypatch.setattr(c.model, "complete", provider)
    requested = True
    try:
        await c.loop.send("Start another unrelated finite task", c.state.parent)
        await asyncio.wait_for(second.wait(), 3)
        assert gate.locked()
        policy = await waits.read_wait(c.manager, c.runtime.actor(c.state.parent), registration.job.job_id)
        now = policy["request"]["request"]["due_at_ms"]
        due = await prepare_due(c.manager, c.runtime.actor(c.state.parent), registration)
        c.state.choice, c.state.due_id = "continue", due["event_id"]
        await notices.enqueue_notification(c.loop, c.runtime.actor(c.state.parent), due["event_id"])
        await c.loop.flush_follow_ups(c.state.parent)
        await effects.apply(c.manager, c.runtime.actor(c.state.parent), due["event_id"])
        for _ in range(200):
            if await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'"):
                break
            await asyncio.sleep(.01)
        assert gate.locked() and owner.session_id not in requests
        if after_ready == "cancel":
            original_task = delegation._RUNNING[(id(c.store), owner.session_id)].task
            await c.manager.request_cancel(c.runtime.actor(c.state.parent), owner.job_id,
                system_key="stop-during-original-readmission", reason="stop queued continuation")
            await asyncio.wait_for(asyncio.shield(original_task), 5)
            assert gate.locked() and owner.session_id not in requests
        elif after_ready == "deadline":
            saved = await waits.read_wait(c.manager, c.runtime.actor(c.state.parent), registration.job.job_id)
            deadline = saved["request"]["policy"]["deadline_at_ms"]
            monkeypatch.setattr(c.manager, "now", lambda: deadline)
        release.set()
        tasks = set(delegation._TASKS)
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        assert requests.count(owner.session_id) == (1 if after_ready == "continue" else 0)
        result = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
        assert result.outcome == {"continue": "succeeded", "cancel": "cancelled", "deadline": "timed_out"}[after_ready], result
        assert not gate.locked() and len(await c.store.list_children(c.state.parent)) == 2
    finally:
        release.set()


async def test_waiting_workflow_leaf_releases_capacity_for_sibling_and_reacquires_before_its_original_input(managed_workflow, monkeypatch):
    store, manager, task_runtime, loop, model, parent, state, _ = managed_workflow
    task_runtime.lease_ms = 30000
    state["on_complete"] = None
    model.hold = {"AUDIT"}
    state["spec"]["root"] = {"type": "sequence", "steps": [
        {"type": "parallel", "max_concurrency": 1, "concurrency_scope": "leaf", "branches": [leaf("BUILD", tools=["arm_wait"]), leaf("AUDIT")]}, leaf("REVIEW")]}
    register_wait_handoff_tool(loop.tool_registry)
    saved, choice = [], {"event_id": None, "issued": False}
    waiting_committed = asyncio.Event()
    audit_admissions = []
    original_enter = owner_waits._enter
    original_admitted = WorkflowEngine._exec_agent_admitted

    async def entered(*args, **kwargs):
        result = await original_enter(*args, **kwargs)
        if result is not None:
            waiting_committed.set()
        return result

    async def admit_after_original_wait(engine, node, *args, **kwargs):
        # Parallel branches have no start-order guarantee. Stage AUDIT before
        # admission, so this case specifically measures release/reacquisition
        # by a waiting BUILD rather than racing to obtain the initial slot.
        # The actual admission, semaphore and owning executor are unchanged.
        if node.id == "AUDIT":
            audit_admissions.append(node.id)
            await asyncio.wait_for(waiting_committed.wait(), 5)
        return await original_admitted(engine, node, *args, **kwargs)

    monkeypatch.setattr(owner_waits, "_enter", entered)
    monkeypatch.setattr(WorkflowEngine, "_exec_agent_admitted", admit_after_original_wait)

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, task_runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 2000, "Review original BUILD", "Continue original BUILD"),
            policy=waits.WaitPolicy(now + 120000))
        saved.append(registration)
        return "Original node wait admitted."

    loop.tool_registry.register(ToolDefinition(name="arm_wait", description="Admit original node wait", input_schema={"type": "object", "properties": {}}), arm)
    original_complete = model.complete

    async def provider(request, **kwargs):
        response = await original_complete(request, **kwargs)
        sid = get_session_id()
        if sid != parent and "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1:
            response.raw_text = ""
            response.tool_calls = [{"id": "node-wait", "type": "function", "function": {"name": "arm_wait", "arguments": "{}"}}]
        elif sid == parent and choice["event_id"] and not choice["issued"]:
            choice["issued"] = True
            response.raw_text = ""
            response.tool_calls = [{"id": "original-node-decision", "type": "function", "function": {"name": "wait_handoff", "arguments": json.dumps({
                "action": "decide", "event_id": choice["event_id"], "decision": "continue"})}}]
        return response

    monkeypatch.setattr(model, "complete", provider)
    try:
        await loop.send("Run original workflow", parent)
        await asyncio.wait_for(model.entered["AUDIT"].wait(), 5)
        assert audit_admissions == ["AUDIT"] and waiting_committed.is_set()
        if not saved:
            diagnostic = []
            workflow_job = await manager.get_job(task_runtime.actor(parent), state["handles"][0].job_id)
            for child in await store.list_children(workflow_job.session_id):
                diagnostic.extend({"role": m.role, "content": str(m.content)[:500]} for m in await store.load_all_messages(child.session_id))
            pytest.fail(f"Original node wait was not admitted; counts={model.counts}; messages={diagnostic}")
        [registration] = saved
        owner = await manager.get_job(task_runtime.actor(parent), registration.job.control_owner_job_id)
        assert owner.status == "waiting" and owner.outcome is None
        assert model.counts["BUILD"] == 2 and model.counts["REVIEW"] == 0
        assert not state["handles"][0].task.done()
        due = await prepare_due(manager, task_runtime.actor(parent), registration)
        choice["event_id"] = due["event_id"]
        await notices.enqueue_notification(loop, task_runtime.actor(parent), due["event_id"])
        await loop.flush_follow_ups(parent)
        await effects.apply(manager, task_runtime.actor(parent), due["event_id"])
        for _ in range(300):
            if await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='owner_wait_resumed'"):
                break
            await asyncio.sleep(.01)
        assert model.counts["BUILD"] == 2 and model.counts["REVIEW"] == 0
        model.release["AUDIT"].set()
        await asyncio.wait_for(state["handles"][0].task, 8)
        assert model.counts["BUILD"] == 3 and model.counts["AUDIT"] == model.counts["REVIEW"] == 1
        assert (await manager.get_job(task_runtime.actor(parent), owner.job_id)).outcome == "succeeded"
        assert (await manager.get_job(task_runtime.actor(parent), registration.job.job_id)).outcome == "succeeded"
        await loop.flush_follow_ups(parent)
        rows = await manager.db.fetchall(f"SELECT kind,supervision_status FROM {manager.jobs}")
        assert all(r["supervision_status"] == "closed" for r in rows), rows
    finally:
        model.release["AUDIT"].set()


async def test_original_workflow_node_empty_pass_gets_its_answer_without_a_standalone_pool(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    state["on_complete"] = None
    runtime.lease_ms = 30000
    model.hold = set()
    original = model.complete

    async def provider(request, **kwargs):
        result = await original(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1:
            result.raw_text = ""
            result.tool_calls = [{"id": "original-empty-node", "type": "function", "function": {
                "name": "pass_turn", "arguments": '{"reason":"original empty pass"}'}}]
        return result

    monkeypatch.setattr(model, "complete", provider)
    await loop.send("Run original workflow", parent)
    await asyncio.wait_for(state["handles"][0].task, 8)
    job = await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)
    assert job.outcome == "succeeded", job.result
    assert model.counts["BUILD"] == 2 and model.counts["REVIEW"] == 1
    assert not await store.get_runtime_state(parent, "subagent:usage:total")
    await loop.flush_follow_ups(parent)


@pytest.mark.parametrize("ending", ["cancelled", "timed_out"])
async def test_waiting_owner_stops_without_another_model_request_and_retains_usage(original, ending, monkeypatch):
    c = original
    now = [c.manager.now()]
    monkeypatch.setattr(c.manager, "now", lambda: now[0])
    registration, owner = await entered(c)
    before = c.model.calls.copy()
    if ending == "cancelled":
        await c.manager.request_cancel(c.runtime.actor(c.state.parent), owner.job_id,
            system_key="cancel-real-waiting-controller", reason="stop original waiting task")
    else:
        saved = await waits.read_wait(c.manager, c.runtime.actor(c.state.parent), registration.job.job_id)
        now[0] = saved["request"]["policy"]["deadline_at_ms"]
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    current = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
    assert current.outcome == ending and current.resource_state == "stopped", current
    assert current.result["usage"]["total_tokens"] == 20
    assert c.model.calls == before and len(await c.store.list_children(c.state.parent)) == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    if ending == "timed_out":
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_expired'")) == 1
    await c.loop.flush_follow_ups(c.state.parent)
    timer = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.job_id)
    assert timer.outcome != "succeeded"
    rows = await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}")
    assert all(r["supervision_status"] == "closed" for r in rows), rows


@pytest.mark.parametrize("damage", ["waiting_hash", "waiting_role", "waiting_delivery", "missing_receipt", "model_exit", "model_unknown", "tracking"])
async def test_resume_revalidates_original_wait_and_physical_exit_evidence(original, monkeypatch, damage):
    c = original
    ready, release = asyncio.Event(), asyncio.Event()
    step = owner_waits._step

    async def pause_controller(*args):
        ready.set()
        await release.wait()
        return await step(*args)

    monkeypatch.setattr(owner_waits, "_step", pause_controller)
    registration, owner = await entered(c)
    await asyncio.wait_for(ready.wait(), 2)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='owner_waiting'", (owner.job_id,))
    saved = json.loads(event["payload_json"])
    due = await prepare_due(c.manager, c.runtime.actor(c.state.parent), registration)
    c.state.choice, c.state.due_id = "continue", due["event_id"]
    await notices.enqueue_notification(c.loop, c.runtime.actor(c.state.parent), due["event_id"])
    await c.loop.flush_follow_ups(c.state.parent)
    await effects.apply(c.manager, c.runtime.actor(c.state.parent), due["event_id"])
    table, column, key_column, key, value = c.manager.events, "request_hash", "event_id", event["event_id"], "damaged"
    if damage == "waiting_role":
        column, value = "event_type", "owner_wait_resumed"
    elif damage == "waiting_delivery":
        column, value = "delivery_status", "handled"
    elif damage == "missing_receipt":
        column, key, value = "receipt_ref_json", registration.operation.operation_id, None
    elif damage in ("model_exit", "tracking"):
        kind = "model_closed" if damage == "model_exit" else "model_attempt_tracking"
        matches = await c.manager.db.fetchall(f"SELECT e.event_id FROM {c.manager.events} e JOIN {c.manager.jobs} j ON j.job_id=e.job_id "
            "WHERE e.event_type=? AND j.control_owner_job_id=?", (kind, owner.job_id))
        assert matches, kind
        column, key, value = "payload_json", matches[-1]["event_id"], json.dumps({"corrupt": True})
    elif damage == "model_unknown":
        [model] = await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='send' AND control_owner_job_id=?", (owner.job_id,))
        table, column, key_column, key, value = c.manager.jobs, "resource_state", "job_id", model["job_id"], "unknown"
    [original_row] = await c.manager.db.fetchall(f"SELECT {column} FROM {table} WHERE {key_column}=?", (key,))
    try:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {table} SET {column}=? WHERE {key_column}=?", (value, key))
        before = c.model.calls.copy()
        with pytest.raises(JobError):
            await step(c.manager, c.runtime.actor(c.state.parent), JobLease(**saved["owner"]), event["event_id"], saved)
        assert c.model.calls == before
        assert (await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)).status == "waiting"
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed' AND event_id<>?", (event["event_id"],))
    finally:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {table} SET {column}=? WHERE {key_column}=?", (original_row[column], key))
        release.set()
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)).outcome == "succeeded"
