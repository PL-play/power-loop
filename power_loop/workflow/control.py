"""Model node input resolves saved task bindings, never session metadata.

Read authority selects the exact candidate only. The existing input/continuation
adapter still owns atomic admission, dispatch and the physical session.
"""

from __future__ import annotations

from power_loop.runtime.jobs import JobError


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
    rows = await manager.db.fetchall(
        f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j "
        f"JOIN {manager.events} e ON e.job_id=j.job_id AND e.event_type='operation' AND e.action='create' "
        "WHERE j.parent_job_id=? AND j.kind='workflow_node' AND j.title=? ORDER BY j.job_id LIMIT 101",
        (workflow.job_id, value[:200]),
    )
    if len(rows) > 100:
        raise JobError("workflow_node_scope_too_large")
    matches = [row["job_id"] for row in rows]
    if not matches:
        raise JobError("workflow_node_not_found")
    if len(matches) != 1:
        raise JobError("workflow_node_ambiguous_choose_job")
    return matches[0]
