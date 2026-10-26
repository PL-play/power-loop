"""One built-in jobs schema/handler across native runtime and legacy adapters."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import JOBS_DEFINITION, LLMResponse, LLMTokenUsage, create_default_tool_registry, register_jobs_tools
from power_loop.core.agent_context import (
    get_session_id,
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime import delegation, task_control
from power_loop.tools.default_manifest import CORE_TOOL_NAMES, EXPLORE_TOOL_NAMES, FULL_TOOL_NAMES
from power_loop.tools.jobs import run_jobs
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import managed, receipt
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "managed", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def invoke(loop, sid, **kwargs):
    a, b = set_current_loop(loop), set_session_id(sid)
    try:
        return json.loads(await run_jobs(**kwargs))
    finally:
        reset_session_id(b)
        reset_current_loop(a)


async def test_explicit_builtin_registration_does_not_expand_default_presets():
    registry = create_default_tool_registry(include=["jobs"])
    assert registry.get("jobs").definition == JOBS_DEFINITION
    assert all("jobs" not in names for names in (CORE_TOOL_NAMES, EXPLORE_TOOL_NAMES, FULL_TOOL_NAMES))
    second = ToolRegistry()
    register_jobs_tools(second, definition=replace(JOBS_DEFINITION, description="Localized description"))
    assert second.get("jobs").definition.input_schema == JOBS_DEFINITION.input_schema
    with pytest.raises(ValueError):
        register_jobs_tools(second, definition=replace(JOBS_DEFINITION, input_schema={}))
    assert JOBS_DEFINITION.interrupt == "finish"


@pytest.mark.parametrize("fault", ["none", "commit_ack", "signal"])
async def test_actual_native_tools_list_get_stop_bind_receipt_and_cancel_original_child(managed, monkeypatch, fault):
    store, manager, runtime, loop, model, sid, _ = managed
    register_jobs_tools(loop.tool_registry)
    callbacks = []
    start = delegation.start

    async def notified(*args):
        callbacks.append(args)

    async def launch(*args, **kwargs):
        return await start(*args, **kwargs, on_complete=notified)

    monkeypatch.setattr(delegation, "start", launch)
    await loop.send("delegate", sid)
    await asyncio.wait_for(model.entered.wait(), 3)
    # Capture original physical handles before stop can remove the compatibility
    # running entry. Canonical result publication precedes context-stack exit.
    tasks = [r.task for (key, _), r in delegation._RUNNING.items() if key == id(store) and r.parent == sid and r.task]
    assert len(tasks) == 1 and not tasks[0].done()
    accepted = await receipt(managed)
    child_id = accepted["job_id"]
    original = model.complete
    actions = [
        dict(action="list", kinds=["subagent"]),
        dict(action="get", job_id=child_id),
        dict(action="stop", job_id=child_id, reason="User stopped this child", expected_generation=1),
    ]
    phase = 0

    async def control(request, **kwargs):
        nonlocal phase
        if get_session_id() == sid and phase < len(actions):
            params = actions[phase]
            phase += 1
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "jobs-control-" + str(phase),
                        "type": "function",
                        "function": {"name": "jobs", "arguments": json.dumps(params)},
                    }
                ],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", control)
    if fault == "commit_ack":
        request = task_control.request_stop

        async def lost(*args, **kwargs):
            await request(*args, **kwargs)
            raise RuntimeError("cancel committed; ACK lost")

        monkeypatch.setattr(task_control, "request_stop", lost)
    elif fault == "signal":

        async def unavailable(*args, **kwargs):
            raise RuntimeError("local signal unavailable")

        monkeypatch.setattr(task_control, "signal_local", unavailable)
    result = await loop.send("inspect and stop", sid, max_rounds=5)
    assert result.status == "completed" and phase == 3
    tools = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "jobs"]
    assert len(tools) == 3
    listed, detail, stopped = [json.loads(m.content) for m in tools]
    assert listed["backend"] == detail["backend"] == stopped["backend"] == "task_registry"
    assert listed["matched_count"] == 1 and detail["node"]["job_id"] == child_id
    assert stopped["cancel_requested"] and stopped["control_pending"] and not stopped["resource_stop_confirmed"]
    assert stopped["replayed"] == (fault == "commit_ack")
    assert stopped["signal_error"] == ("local_signal_requires_reconciliation" if fault == "signal" else None)
    op = await manager.get_operation(runtime.actor(sid), stopped["operation_id"])
    assert op.call_ref["id"] != op.receipt_ref["id"] and op.receipt_ref["id"] == str(tools[-1].id)
    assert (await store.message_by_id(sid, op.receipt_ref["id"])).content == tools[-1].content
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 4)
    child = await manager.get_job(runtime.actor(sid), child_id)
    assert child.outcome == "cancelled" and child.supervision_status == "open", child
    assert model.child_calls == 1 and not runtime._local_executions
    assert callbacks == []
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1
    assert any(x["item_id"] == "subagent:" + accepted["run_id"] for x in await store.inbox_items(sid))


async def test_native_read_aliases_are_authorized_and_stop_requires_actual_invocation(managed):
    _, manager, runtime, loop, model, sid, _ = managed
    await loop.send("delegate", sid)
    accepted = await receipt(managed)
    for alias in (accepted["job_id"], "sub:" + accepted["run_id"], accepted["run_id"], accepted["operation_id"]):
        found = await invoke(loop, sid, action="get", job_id=alias)
        assert found["node"]["job_id"] == accepted["job_id"]
    missing = await invoke(loop, sid, action="stop", job_id=accepted["job_id"])
    assert missing["error"]["code"] == "persisted_tool_call_required"
    other = await loop.new_session()
    assert (await invoke(loop, other, action="get", job_id=accepted["job_id"]))["error"]["code"] == "access_denied"
    assert (await manager.get_job(runtime.actor(sid), accepted["job_id"])).cancel_requested_at is None
    model.release.set()


@pytest.mark.parametrize(
    "params",
    [
        {"action": "get"},
        {"action": "list", "max_chars": True},
        {"action": "list", "max_chars": 32001},
        {"action": "get", "job_id": "missing"},
        {"action": "list", "ids": ["job_x"]},
        {"action": "stop", "ids": ["one", "two"]},
        {"action": "stop", "ids": []},
        {"action": "get", "job_id": "job_x", "root_job_id": "job_y"},
    ],
)
async def test_native_tool_errors_are_structured_and_do_not_write_control(managed, params):
    _, manager, _, loop, _, sid, _ = managed
    response = await invoke(loop, sid, **params)
    assert "error" in response
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")


async def test_legacy_adapter_keeps_honest_capability_limits_and_selective_stop():
    calls = []
    rows = [
        dict(id="wf:r1", kind="workflow", title="research", status="running"),
        dict(id="timer:7", kind="timer", title="reminder", status="armed"),
        dict(id="bg:r1", kind="background", title="other", status="running"),
    ]

    async def list_jobs(sid):
        assert sid == "session"
        return rows

    async def stop_job(sid, jid):
        calls.append((sid, jid))
        return {"id": jid, "result": "stopped"}

    loop = SimpleNamespace(task_runtime=None, list_jobs=list_jobs, stop_job=stop_job)
    listed = await invoke(loop, "session", action="list", status="active", limit=2)
    assert listed["backend"] == "legacy" and not listed["history_available"] and not listed["hierarchy_available"]
    assert listed["truncated"] and len(listed["nodes"]) == 2
    result = await invoke(loop, "session", action="stop", ids=["wf:r1", "7", "r1", "not-mine"])
    assert calls == [("session", "wf:r1"), ("session", "timer:7")]
    assert [x["result"] for x in result["results"]] == ["stopped", "stopped", "unknown", "unknown"]
    assert (await invoke(loop, "session", action="get", job_id="7"))["node"]["id"] == "timer:7"
    assert (await invoke(loop, "session", action="get", job_id="gone"))["error"]["code"] == "legacy_history_unavailable"
    assert (await invoke(loop, "session", action="list", status="terminal"))["error"][
        "code"
    ] == "task_registry_required"


async def test_jobs_outside_session_is_a_structured_error():
    assert json.loads(await run_jobs(action="list"))["error"]["code"] == "active_session_required"


async def test_workflow_node_can_stop_its_own_required_child_without_ancestor_read_permission(
    managed_workflow, monkeypatch
):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    register_jobs_tools(loop.tool_registry)
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent", "jobs"])
    model.spawn_from_build = True
    model.release["BUILD"].set()
    model.hold.add("AUDIT")
    original = model.complete
    stop_sid, issued = None, False

    async def complete(request, **kwargs):
        nonlocal stop_sid, issued
        if "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1 and not issued:
            sid = get_session_id()
            [child] = await manager.db.fetchall(
                f"SELECT job_id FROM {manager.jobs} WHERE kind='subagent' AND created_by_session_id=?", (sid,)
            )
            issued, stop_sid = True, sid
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "node-stop-child",
                        "type": "function",
                        "function": {
                            "name": "jobs",
                            "arguments": json.dumps({"action": "stop", "job_id": child["job_id"]}),
                        },
                    }
                ],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", complete)
    await loop.send("start workflow", parent)
    [handle] = state["handles"]
    await asyncio.wait_for(handle.task, 8)
    assert issued and stop_sid != parent
    [row] = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")
    op = await manager.get_operation(runtime.actor(parent), row["event_id"])
    assert op.call_ref["session_id"] == stop_sid and op.call_ref["id"] != op.receipt_ref["id"]
    assert op.result["response"]["cancel_requested"] and op.result["response"]["resource_stop_confirmed"] is False
    assert (await manager.get_job(runtime.actor(parent), op.job_id)).outcome == "cancelled"
    [source] = await manager.db.fetchall(
        f"SELECT execution_role FROM {manager.jobs} WHERE job_id=?", (op.result["request"]["caller"]["job_id"],)
    )
    assert source["execution_role"] == "workflow_node"
    assert not runtime._local_executions


async def test_cancel_after_result_commit_suppresses_external_callback_without_erasing_owner_result(
    managed, monkeypatch
):
    store, manager, runtime, loop, model, sid, _ = managed
    saved_result, callbacks = [], []
    save, start = delegation._save_result, delegation.start

    async def notified(*args):
        callbacks.append(args)

    async def launch(*args, **kwargs):
        return await start(*args, **kwargs, on_complete=notified)

    async def stopped(*args, **kwargs):
        result = await save(*args, **kwargs)
        binding = args[3].task_binding
        saved_result.append(await manager.get_result_revision(runtime.actor(sid), binding.lease.job_id, 1))
        await manager.request_cancel(
            runtime.actor(sid), binding.lease.job_id, system_key="stop-after-result", reason="late user stop"
        )
        return result

    monkeypatch.setattr(delegation, "start", launch)
    monkeypatch.setattr(delegation, "_save_result", stopped)
    await loop.send("delegate", sid)
    accepted = await receipt(managed)
    model.release.set()
    work = [r.task for (key, _), r in delegation._RUNNING.items() if key == id(store) and r.task]
    if work:
        await asyncio.wait_for(asyncio.gather(*work, return_exceptions=False), 4)
    assert callbacks == [] and len(saved_result) == 1
    assert await manager.get_result_revision(runtime.actor(sid), accepted["job_id"], 1) == saved_result[0]
    assert any(x["item_id"] == "subagent:" + accepted["run_id"] for x in await store.inbox_items(sid))


@pytest.mark.parametrize("action", ["list", "get", "stop"])
@pytest.mark.parametrize("exception", [RuntimeError, ValueError])
async def test_backend_failure_is_explicit_and_does_not_expose_control_secrets(action, exception):
    async def unavailable(*args, **kwargs):
        raise exception("PRIVATE-CONNECTION-SECRET")

    loop = SimpleNamespace(task_runtime=None, list_jobs=unavailable)
    result = await invoke(loop, "session", action=action, **({"ids": ["wf:x"]} if action == "stop" else {}))
    assert result["error"]["code"] == (
        "stop_requires_reconciliation" if action == "stop" else "jobs_backend_unavailable"
    )
    if action == "stop":
        assert result["control_state"] == "unknown"
    assert "PRIVATE-CONNECTION-SECRET" not in json.dumps(result)


@pytest.mark.parametrize("action", ["list", "get", "stop"])
@pytest.mark.parametrize("spoof", [False, True])
async def test_native_backend_valueerror_is_private_even_with_a_code_attribute(managed, monkeypatch, action, spoof):
    from power_loop.runtime import task_query

    _, manager, runtime, loop, model, sid, _ = managed
    # Error privacy, not heartbeat expiry: allow real SQL admission under load.
    runtime.lease_ms = 30000
    await loop.send("delegate", sid)
    await asyncio.wait_for(model.entered.wait(), 3)
    accepted = await receipt(managed)

    class BackendValueError(ValueError):
        code = "PRIVATE-BACKEND-CODE"

    async def unavailable(*args, **kwargs):
        raise (BackendValueError if spoof else ValueError)("PRIVATE-CONNECTION-SECRET")

    if action == "list":
        monkeypatch.setattr(task_query, "list_jobs", unavailable)
    elif action == "get":
        monkeypatch.setattr(task_query, "get_job", unavailable)
    else:
        monkeypatch.setattr(task_control, "stop_from_current_call", unavailable)
    result = await invoke(loop, sid, action=action, **({"job_id": accepted["job_id"]} if action != "list" else {}))
    assert result["error"]["code"] == (
        "stop_requires_reconciliation" if action == "stop" else "jobs_backend_unavailable"
    )
    assert "PRIVATE" not in json.dumps(result)
    if action == "stop":
        assert result["control_state"] == "unknown"
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")


@pytest.mark.parametrize("trusted", [True, False])
async def test_expired_model_send_can_cancel_but_spoofed_jobs_name_cannot_dispatch_business_work(
    managed, monkeypatch, trusted
):
    _, manager, runtime, loop, model, sid, _ = managed
    # Isolate dispatch admission from the independent heartbeat cancellation
    # timer; expiration below still exceeds the full actual source lease.
    runtime.lease_ms = 300000
    await loop.send("delegate", sid)
    accepted = await receipt(managed)
    writes = []
    if trusted:
        register_jobs_tools(loop.tool_registry)
    else:

        async def unsafe(**kwargs):
            writes.append(kwargs)
            return "business side effect"

        loop.tool_registry.register(JOBS_DEFINITION, unsafe)
    original, clock = model.complete, manager.now
    requested = False

    async def expired(request, **kwargs):
        nonlocal requested
        if get_session_id() == sid and not requested:
            requested = True
            manager.now = lambda: clock() + 600000
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "expired-control",
                        "type": "function",
                        "function": {
                            "name": "jobs",
                            "arguments": json.dumps({"action": "stop", "job_id": accepted["job_id"]}),
                        },
                    }
                ],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", expired)
    try:
        await loop.send("stop the original child", sid)
    except ValueError as exc:
        assert "expired" in str(exc) or "supervision" in str(exc)
    finally:
        manager.now = clock
    assert requested and writes == []
    child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
    assert (child.cancel_requested_at is not None) == trusted
    stops = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")
    assert len(stops) == int(trusted)
    if trusted:
        op = await manager.get_operation(runtime.actor(sid), stops[0]["event_id"])
        assert op.call_ref["id"] != op.receipt_ref["id"]


@pytest.mark.parametrize("name", ["jobs", "inspect_tasks", "send_message"])
async def test_native_jobs_alias_uses_registered_contract_not_catalog_name(managed, monkeypatch, name):
    from power_loop import ToolDefinition
    from power_loop.runtime.tool_contracts import policy
    _, _, _, loop, model, sid, _ = managed
    register_jobs_tools(loop.tool_registry, definition=replace(JOBS_DEFINITION, name=name))
    calls = []
    async def complete(request, **kwargs):
        calls.append(request)
        tools = [{"id": "original-aliased-jobs", "type": "function", "function": {
            "name": name, "arguments": '{"action":"list"}'}}] if len(calls) == 1 else None
        return LLMResponse(raw_text="" if tools else "inspected", tool_calls=tools,
            token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
    monkeypatch.setattr(model, "complete", complete)
    result = await loop.send("Inspect my task tree", sid)
    assert result.status == "completed" and len(calls) == 2
    replies = [row for row in await loop.store.load_all_messages(sid) if row.role == "tool"]
    assert len(replies) == 1 and replies[0].tool_call_id == "original-aliased-jobs"
    payload = json.loads(replies[0].content)
    assert not payload.get("error")
    assert policy(loop.tool_registry.get(name), {"action": "list"}).control_only
    forged = ToolRegistry()
    forged.register(ToolDefinition(name, "ordinary callable"), lambda **kwargs: "ordinary")
    assert not policy(forged.get(name), {"action": "list"}).native_admitted
