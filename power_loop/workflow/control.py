"""Model node steering resolves saved task bindings, never session metadata.

Read authority selects the exact candidate only. The existing input/continuation
adapter still owns atomic admission, dispatch and the physical session.
"""
from __future__ import annotations

import json

from power_loop.runtime.jobs import JobError

_MAX_BINDINGS = 5000


async def select_native_target(runtime, sid, workflow, options):
    """Resolve one saved node, or explicitly the whole workflow for cancel."""
    manager, actor = runtime.manager, runtime.actor(sid)
    has_name, has_job = "node" in options, "node_job_id" in options
    if has_name and has_job:
        raise JobError("ambiguous_workflow_node_selector")
    if not has_name and not has_job:
        return workflow.job_id
    if has_job:
        value = options["node_job_id"]
        if not isinstance(value, str) or not 0 < len(value) <= 64:
            raise JobError("invalid_workflow_node_job")
        node = await manager.get_job(actor, value, include_result=False)
        if node.kind != "workflow_node" or node.parent_job_id != workflow.job_id:
            raise JobError("workflow_node_outside_run")
        return node.job_id
    value = options["node"]
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise JobError("invalid_workflow_node")
    rows = await manager.db.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j "
        f"JOIN {manager.events} e ON e.job_id=j.job_id AND e.event_type='operation' AND e.action='create' "
        "WHERE j.parent_job_id=? AND j.kind='workflow_node' AND j.title=? ORDER BY j.job_id LIMIT 101",
        (workflow.job_id, value[:200]))
    if len(rows) > 100:
        raise JobError("workflow_node_scope_too_large")
    matches = []
    for row in rows:
        await manager.get_job(actor, row["job_id"], include_result=False)
        try:
            request = json.loads(row["payload_json"])["request"]
            if request["node"]["id"] == value:
                matches.append(row["job_id"])
        except (ValueError, TypeError, KeyError) as exc:
            raise JobError("workflow_node_binding_corrupt") from exc
    if not matches:
        raise JobError("workflow_node_not_found")
    if len(matches) != 1:
        raise JobError("workflow_node_ambiguous_choose_job")
    return matches[0]


async def _sid_target(runtime, actor, workflow, sid):
    manager = runtime.manager
    rows = await manager.db.fetchall(f"SELECT j.job_id,j.generation,e.payload_json FROM {manager.jobs} j "
        f"JOIN {manager.events} e ON e.job_id=j.job_id AND e.event_type='node_session' "
        "WHERE j.parent_job_id=? AND j.kind='workflow_node' ORDER BY j.job_id,e.event_id LIMIT ?",
        (workflow.job_id, _MAX_BINDINGS + 1))
    if len(rows) > _MAX_BINDINGS:
        raise JobError("workflow_node_scope_too_large")
    matches = set()
    for row in rows:
        try:
            data = json.loads(row["payload_json"])
            if not isinstance(data, dict):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise JobError("workflow_node_binding_corrupt") from exc
        if data.get("session_id") == sid and data.get("generation", 1) == row["generation"]:
            matches.add(row["job_id"])
    if len(matches) != 1:
        raise JobError("workflow_node_not_found" if not matches else "workflow_node_ambiguous_choose_job")
    return await manager.get_job(actor, matches.pop(), include_result=False)


async def _selected(runtime, actor, workflow, options):
    if "node" in options or "node_job_id" in options:
        target = await select_native_target(runtime, actor.session_id, workflow, options)
        return await runtime.manager.get_job(actor, target, include_result=False)
    if "sid" not in options:
        raise JobError("workflow_node_required")
    return await _sid_target(runtime, actor, workflow, options["sid"])


def _receipt(run_id, node_id, sid, value):
    return {"version": 1, "run_id": run_id, "node_id": node_id, "session_id": sid, **value}


async def nudge(loop, sid, workflow, options):
    from power_loop.runtime.invocation import current_invocation
    from power_loop.runtime.nudges import receipt
    from power_loop.runtime.task_runtime import _SEND

    from . import runner

    message = options.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if "sid" in options and (not isinstance(options["sid"], str) or not 0 < len(options["sid"]) <= 255):
        raise JobError("invalid_task_session_id")
    if "node" in options and "node_job_id" in options:
        raise JobError("ambiguous_workflow_node_selector")
    if not {"node", "node_job_id", "sid"}.intersection(options):
        raise JobError("workflow_node_required")
    runtime = loop.task_runtime
    source = _SEND.get()
    if source is None or source.runtime is not runtime or source.actor.session_id != sid:
        raise JobError("task_send_required")
    seq, call_id = current_invocation(sid)
    if seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    manager, actor = runtime.manager, source.actor
    ref = await manager.store.message_ref(sid, seq)
    old = await manager.operation_for_call(actor, ref, call_id)
    if old is not None:
        request = old.result.get("request", {})
        data = request.get("node", {}) if isinstance(request, dict) else {}
        if (not isinstance(data, dict) or old.action != "nudge" or old.call_ref != ref.to_dict() or
                data.get("workflow_run_id") != options["run_id"] or request.get("message") != message or
                "node" in options and data.get("node_id") != options["node"] or
                "node_job_id" in options and old.job_id != options["node_job_id"] or
                "sid" in options and request.get("session_id") != options["sid"]):
            raise JobError("operation_conflict")
        target = await manager.get_job(actor, old.job_id, include_result=False)
        if target.kind != "workflow_node" or target.parent_job_id != workflow.job_id:
            raise JobError("workflow_node_outside_run")
        return _receipt(options["run_id"], data["node_id"], request["session_id"], receipt(old, True))
    target = await _selected(runtime, actor, workflow, options)
    from power_loop.runtime.task_sessions import list_sessions
    saved = await list_sessions(manager, actor, target.job_id)
    if saved["job_generation"] != target.generation:
        raise JobError("workflow_node_generation_changed")
    candidates = [b for b in saved["bindings"] if b["generation"] == target.generation and
        ("sid" not in options or b["session_id"] == options["sid"])]
    if not candidates:
        raise JobError("workflow_node_session_not_found")
    latest = max(b["attempt"] for b in candidates)
    candidates = [b for b in candidates if b["attempt"] == latest]
    if len(candidates) != 1:
        raise JobError("workflow_node_session_ambiguous_choose_sid")
    selected = candidates[0]
    if not selected["available"]:
        raise JobError("workflow_node_session_deleted")
    value = await runner.steer_node(options["run_id"], selected["node_id"], selected["session_id"], message,
        expected_job=(target.job_id, target.generation))
    return _receipt(options["run_id"], selected["node_id"], selected["session_id"], value)
