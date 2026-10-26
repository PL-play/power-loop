"""Actual workflow cancellation shares durable control without paid dispatch."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from functools import wraps

import pytest

from power_loop import AgentLoopConfig, LLMResponse, SessionStore, StatefulAgentLoop
from power_loop.core.agent_context import (
    get_session_id,
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime import task_control
from power_loop.runtime.jobs import JobError
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import journal, register_workflow_tools, runner
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool, run_workflow_tool

from .test_job_registry_192 import registry_store
from .test_workflow_facade_192 import SPEC, Model
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def invoke(loop, sid, arguments):
    lt, st = set_current_loop(loop), set_session_id(sid)
    try:
        return json.loads(await run_workflow_tool(arguments))
    finally:
        reset_session_id(st)
        reset_current_loop(lt)


@pytest.mark.parametrize("fault", ["none", "commit_ack", "signal", "same_call_replay"])
async def test_native_status_list_cancel_original_workflow_has_one_intent_and_distinct_receipt(managed_workflow, monkeypatch, fault):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch original", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    original = state["handles"][0]
    register_workflow_tool(loop.tool_registry)
    if fault == "same_call_replay":
        handler = loop.tool_registry.get("workflow").handler
        @wraps(handler)
        async def repeated(**kwargs):
            first = await handler(**kwargs)
            if kwargs.get("action") == "cancel":
                second = await handler(**kwargs)
                assert json.loads(first)["operation_id"] == json.loads(second)["operation_id"]
                return second
            return first
        loop.tool_registry.register(loop.tool_registry.get("workflow").definition, repeated, overwrite=True)
    elif fault == "commit_ack":
        request = task_control.request_stop
        async def lost(*args, **kwargs):
            await request(*args, **kwargs)
            raise RuntimeError("committed stop ACK lost")
        monkeypatch.setattr(task_control, "request_stop", lost)
    elif fault == "signal":
        async def unavailable(*args, **kwargs):
            raise RuntimeError("signal unavailable")
        monkeypatch.setattr(task_control, "signal_local", unavailable)
    actions = [{"action": "status", "run_id": original.run_id}, {"action": "list"},
        {"action": "cancel", "run_id": original.run_id, "reason": "User stopped original workflow", "expected_generation": 1}]
    complete, phase = model.complete, 0
    async def control(request, **kwargs):
        nonlocal phase
        if get_session_id() == sid and phase < len(actions):
            params = actions[phase]
            phase += 1
            return LLMResponse(raw_text="", tool_calls=[{"id": "workflow-control-"+str(phase), "type": "function",
                "function": {"name": "workflow", "arguments": json.dumps(params)}}])
        return await complete(request, **kwargs)
    monkeypatch.setattr(model, "complete", control)
    result = await loop.send("inspect then cancel", sid, max_rounds=5)
    assert result.status == "completed" and phase == 3
    receipts = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
    detail, listing, receipt = [json.loads(m.content) for m in receipts]
    assert detail["node"]["job_id"] == original.job_id and listing["matched_count"] == 1
    assert receipt["run_id"] == original.run_id and receipt["job_id"] == original.job_id
    assert receipt["cancel_requested"] and receipt["control_pending"] and not receipt["resource_stop_confirmed"]
    assert receipt["replayed"] == (fault in ("commit_ack", "same_call_replay"))
    assert receipt["signal_error"] == ("local_signal_requires_reconciliation" if fault == "signal" else None)
    operation = await manager.get_operation(runtime.actor(sid), receipt["operation_id"])
    assert operation.call_ref["id"] != operation.receipt_ref["id"]
    assert operation.receipt_ref["id"] == str(receipts[-1].id)
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1
    model.release["BUILD"].set()
    await asyncio.wait_for(asyncio.gather(original.task, return_exceptions=True), 5)
    current = await manager.get_job(runtime.actor(sid), original.job_id)
    assert current.outcome == "cancelled" and current.supervision_status == "open"
    assert model.counts["REVIEW"] == 0 and not state["notifications"]
    assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1


@pytest.mark.parametrize("mode", ["cross_actor", "wrong_generation", "node_parameter", "no_invocation"])
async def test_invalid_cancel_never_touches_another_task_or_silently_stops_whole_node_scope(managed_workflow, monkeypatch, mode):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch original", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    original = state["handles"][0]
    register_workflow_tool(loop.tool_registry)
    params = {"action": "cancel", "run_id": original.run_id}
    if mode == "cross_actor":
        other = await loop.new_session()
        response = await invoke(loop, other, params)
        assert response["error"]["code"] == "access_denied"
    elif mode == "node_parameter":
        response = await invoke(loop, sid, {**params, "node": "MISSING"})
        assert response["error"]["code"] == "workflow_node_not_found"
    elif mode == "no_invocation":
        response = await invoke(loop, sid, params)
        assert response["error"]["code"] == "persisted_tool_call_required"
    else:
        complete, issued = model.complete, False
        async def wrong(request, **kwargs):
            nonlocal issued
            if get_session_id() == sid and not issued:
                issued = True
                return LLMResponse(raw_text="", tool_calls=[{"id": "wrong-generation", "type": "function",
                    "function": {"name": "workflow", "arguments": json.dumps({**params, "expected_generation": 2})}}])
            return await complete(request, **kwargs)
        monkeypatch.setattr(model, "complete", wrong)
        await loop.send("cancel old generation", sid)
        rows = await manager.store.load_all_messages(sid)
        response = json.loads(next(m.content for m in rows if m.role == "tool" and m.name == "workflow"))
        assert response["error"]["code"] == "stop_generation_changed"
    assert (await manager.get_job(runtime.actor(sid), original.job_id)).cancel_requested_at is None
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")
    assert not original.task.done()


@pytest.mark.parametrize("mode", ["query", "status_list", "cancel", "create", "validate", "spoof", "create_alias", "wrapped_create", "wrapped_validate"])
async def test_expired_execution_can_only_use_actual_library_control_actions(managed_workflow, monkeypatch, mode):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    runtime.lease_ms = 300000
    await loop.send("launch original", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    run_id = state["handles"][0].run_id
    paid = []
    def load(*args):
        paid.append("load")
        return SPEC
    register_workflow_tool(loop.tool_registry, ports=WorkflowToolPorts(load_spec=load))
    name, params = "workflow", {"action": "status", "run_id": run_id}
    if mode == "cancel":
        params = {"action": "cancel", "run_id": run_id}
    elif mode == "status_list":
        params = {"action": "status"}
    elif mode in ("create", "validate"):
        params = {"action": mode, "file": "main.yaml"}
    elif mode == "spoof":
        async def unsafe(**kwargs):
            paid.append("business")
            return "paid"
        loop.tool_registry.register(loop.tool_registry.get(name).definition, unsafe, overwrite=True)
    elif mode == "create_alias":
        register_workflow_tools(loop.tool_registry, ports=WorkflowToolPorts(load_spec=load))
        name, params = "create_workflow", {"spec": SPEC, "action": "status"}
    elif mode in ("wrapped_create", "wrapped_validate"):
        handler = loop.tool_registry.get("workflow").handler
        @wraps(handler)
        async def changes_action(**kwargs):
            # Even a faulty trusted wrapper cannot spend a control-only lease
            # by changing an authorized read into source loading after the gate.
            return await handler(action=mode.removeprefix("wrapped_"), file="main.yaml")
        loop.tool_registry.register(loop.tool_registry.get("workflow").definition, changes_action, overwrite=True)
    complete, clock, issued = model.complete, manager.now, False
    async def expired(request, **kwargs):
        nonlocal issued
        if get_session_id() == sid and not issued:
            issued = True
            manager.now = lambda: clock()+600000
            return LLMResponse(raw_text="", tool_calls=[{"id": "expired-action", "type": "function",
                "function": {"name": name, "arguments": json.dumps(params)}}])
        return await complete(request, **kwargs)
    monkeypatch.setattr(model, "complete", expired)
    rejected = None
    try:
        await loop.send("control original", sid)
    except ValueError as exc:
        rejected = exc
        assert "expired" in str(exc) or "supervision" in str(exc)
    finally:
        manager.now = clock
    assert issued and not paid
    rows = [m for m in await manager.store.load_all_messages(sid) if m.role == "tool" and m.name == name]
    if mode in ("create", "validate", "spoof", "create_alias"):
        # The complete call batch is fenced before dispatch. No callback ran,
        # so a fabricated tool return must not substitute for the pending call.
        assert isinstance(rejected, JobError) and rejected.code == "execution_lease_expired"
        assert not rows
        pending = (await manager.store.get_state(sid)).pending
        assert pending is not None
        raw = await manager.store.load_all_messages(sid)
        assert any(m.role == "assistant" and any(c["id"] == "expired-action" for c in (m.tool_calls or [])) for m in raw)
    else:
        assert rows
    if mode in ("query", "status_list", "cancel"):
        receipt = json.loads(rows[-1].content)
        assert "error" not in receipt
        if mode == "status_list":
            assert receipt["matched_count"] == 1
    current = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert (current.cancel_requested_at is not None) == (mode == "cancel")


@pytest.fixture
async def legacy(tmp_path):
    store = await SessionStore.open(str(tmp_path / "legacy.db"))
    model = Model([])
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=ToolRegistry(),
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3))
    loop.config.stop_policy = replace(loop.config.stop_policy, workflow_s=.02)
    sid = await loop.new_session()
    accepted, notified = [], []
    async def completed(value):
        notified.append(value)
    register_workflow_tool(loop.tool_registry, ports=WorkflowToolPorts(on_accepted=lambda a: accepted.append(a), on_complete=completed))
    lt, st = set_current_loop(loop), set_session_id(sid)
    try:
        receipt = json.loads(await loop.tool_registry.get("workflow").handler(action="create", spec=SPEC))
        await asyncio.wait_for(model.entered.wait(), 3)
        yield store, loop, model, sid, accepted[0].handle, receipt, notified
    finally:
        model.release.set()
        tasks = [a.handle.task for a in accepted] + list(runner._LOCAL_STOP_TASKS)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        reset_session_id(st)
        reset_current_loop(lt)
        await store.close()


async def test_legacy_cancel_returns_before_exit_and_suppresses_callback_without_faking_durable_intent(legacy):
    store, loop, _, sid, handle, accepted, notified = legacy
    first = await invoke(loop, sid, {"action": "cancel", "run_id": accepted["run_id"]})
    second = await invoke(loop, sid, {"action": "cancel", "run_id": accepted["run_id"]})
    assert first["status"] == second["status"] == "accepted"
    assert not first["durable_intent"] and not first["resource_stop_confirmed"]
    assert second["replayed_local_request"] and not handle.task.done()
    await asyncio.wait_for(asyncio.gather(handle._local_stop_task, return_exceptions=True), 3)
    await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), 3)
    assert not notified and (await journal.read(store, sid, accepted["run_id"]))["status"] == "cancelled"


@pytest.mark.parametrize("fault", ["other_session", "other_store", "missing_handle", "generation"])
async def test_legacy_control_checks_ownership_and_keeps_unknown_resources_unknown(legacy, tmp_path, monkeypatch, fault):
    store, loop, model, sid, handle, accepted, _ = legacy
    run_id = accepted["run_id"]
    options = {"action": "cancel", "run_id": run_id}
    if fault == "other_session":
        response = await invoke(loop, await loop.new_session(), options)
        assert response["error"]["code"] == "workflow_run_not_found"
    elif fault == "other_store":
        other = await SessionStore.open(str(tmp_path / "other.db"))
        other_loop = StatefulAgentLoop(llm=model, store=other, config=AgentLoopConfig(delegation_policy="single_level"))
        await other.create_session(session_id=sid)
        await other.set_runtime_state(sid, "workflow:run:"+run_id, await journal.read(store, sid, run_id))
        try:
            response = await invoke(other_loop, sid, options)
            assert response["error"]["code"] == "workflow_control_owner_mismatch"
        finally:
            await other.close()
    elif fault == "missing_handle":
        monkeypatch.delitem(runner._HANDLES, run_id)
        response = await invoke(loop, sid, options)
        assert response["status"] == "control_unavailable" and response["control_pending"]
        assert not response["resource_stop_confirmed"]
        assert (await journal.read(store, sid, run_id))["status"] == "running"
    else:
        response = await invoke(loop, sid, {**options, "expected_generation": 1})
        assert response["error"]["code"] == "legacy_generation_unavailable"
    assert not handle.task.done() and not handle.cancel_token.is_cancelled()


async def test_sdk_legacy_stop_cannot_use_another_parents_workflow_id(legacy):
    _, loop, _, _, handle, accepted, _ = legacy
    other = await loop.new_session()
    result = await loop.stop_job(other, "wf:"+accepted["run_id"])
    assert result["result"] == "unknown"
    assert not handle.task.done() and not handle.cancel_token.is_cancelled()


async def test_status_without_run_id_and_old_alias_detail_list_without_starting_work(legacy):
    _, loop, _, sid, handle, accepted, _ = legacy
    from power_loop.workflow.tool import _handle_workflow_status
    listed = await invoke(loop, sid, {"action": "status", "detail": True})
    assert listed["runs"][0]["run_id"] == accepted["run_id"]
    old = json.loads(await _handle_workflow_status(run_id=None, detail=True))
    assert old["runs"][0]["run_id"] == accepted["run_id"]
    assert not handle.task.done() and not handle.cancel_token.is_cancelled()
