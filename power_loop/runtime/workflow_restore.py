"""Consistent recovery plans. Reading a plan never admits or dispatches work."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

from .jobs import JobError, _hash
from .result_revisions import result_id


def selection(spec, node_id):
    from power_loop.workflow.continuation import plan_continuation, walk
    from power_loop.workflow.engine import WorkflowRunError
    from power_loop.workflow.spec import AgentNode, ParallelNode, SequenceNode
    leaves = {}
    for path, node in walk(spec.root):
        if type(node) in (ParallelNode, SequenceNode):
            continue
        if (type(node) is not AgentNode or node.fallback is not None or node.output_file is not None or
                "@@FILEREF:" in node.input):
            raise JobError("workflow_restore_scope_unsupported")
        leaves[node.id] = (path, node)
    try:
        return leaves, plan_continuation(spec, node_id)
    except WorkflowRunError as exc:
        raise JobError("workflow_restore_target_mismatch") from exc


async def _stopped_tree(manager, tx, row, seen=None, depth=0):
    """Traverse immutable admitted members, not only current parent pointers."""
    from .execution_membership import read, require_current, scope
    from .node_inputs import descriptor
    from .processing import require_event_handled, require_handled
    from .workflow_controller import proof
    seen = set() if seen is None else seen
    if depth >= 64:
        raise JobError("job_depth_exceeded")
    key = (row["job_id"], row["generation"])
    if key in seen or len(seen) >= 10000:
        raise JobError("invalid_execution_membership")
    seen.add(key)
    if row["resource_state"] != "stopped" or row["intake"] != "closed" or row["cancel_requested_at"] is not None:
        raise JobError("workflow_restore_execution_unsettled")
    membership = await read(manager, tx, row, row["generation"])
    if not membership["sealed"] or _hash(membership["scope"]["lease"]) != _hash(proof(row)):
        raise JobError("workflow_restore_execution_unsettled")
    await require_current(manager, tx, row)
    pending = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (row["job_id"],))
    for event in pending:
        payload = json.loads(event["payload_json"])
        if row["kind"] == "workflow" and payload.get("mode") == "workflow_continue":
            from .workflow_restore_execution import require_settled
            await require_settled(manager, tx, row, event)
            continue
        request = payload["request"]
        sid = request.get("session_id") or row["session_id"]
        if row["kind"] == "workflow_node":
            generation = request["target"]["generation"]
            if type(generation) is not int or not 1 <= generation <= row["generation"]:
                raise JobError("processing_input_mismatch")
            data = await descriptor(manager, tx, {**row, "generation": generation}, sid, historical=True)
            if _hash(data) != _hash(request.get("node")):
                raise JobError("processing_input_mismatch")
        await require_event_handled(manager, tx, event, session_id=sid, owner_job_id=row["job_id"])
    for edge in membership["members"]:
        lease = edge["payload"]["member"]
        child = await manager._row(tx, lease["job_id"])
        anchor = await scope(manager, tx, lease["job_id"], lease["generation"])
        if (child is None or _hash(proof(child)) != _hash(lease) or
                _hash(anchor["owner"]) != _hash(proof(row)) or
                (child["tenant_id"], child["principal_id"]) != (row["tenant_id"], row["principal_id"])):
            raise JobError("workflow_restore_execution_unsettled")
        if child["kind"] == "subagent":
            await require_handled(manager, tx, child)
        await _stopped_tree(manager, tx, child, seen, depth+1)
    # An admitted but unclaimed child has no execution_member yet. It still
    # blocks recovery; immutable membership must not hide pending work.
    rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE control_owner_job_id=?", (row["job_id"],))
    for child in rows:
        if child["resource_state"] != "stopped":
            raise JobError("workflow_restore_execution_unsettled")
    return membership


async def _prepare(manager, tx, actor, row, *, generation, node_id, message):
    """Caller holds control lock; admission must recompute and compare this plan."""
    from power_loop.runtime.spec import AgentSpec
    from power_loop.workflow.spec import WorkflowSpec, _node_to_dict

    from .execution_membership import require_current
    from .node_inputs import descriptor, is_closed
    from .workflow_controller import creation, proof
    from .workflow_controller import read as controller
    from .workflow_profile import read, require_native
    from .workflow_results import require_closed

    if type(generation) is not int or not 1 <= generation < 2**63-1:
        raise JobError("invalid_result_generation")
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if (row["kind"] != "workflow" or row["execution_role"] != "workflow" or row["created_by_session_id"] != actor.session_id or
            row["control_owner_job_id"] is not None):
        raise JobError("workflow_restore_target_mismatch")
    if row["generation"] != generation:
        raise JobError("resume_generation_changed")
    if (row["status"] != "settling" or row["outcome"] != "succeeded" or row["result_status"] != "handled" or
            row["supervision_status"] != "open"):
        raise JobError("workflow_restore_target_unsettled")
    await manager._supervision_barriers(tx, row)
    capabilities = await read(manager, tx, row)
    if capabilities["version"] == 1:
        require_native(capabilities)
    # A bound profile may be previewed without invoking host code. Admission
    # separately requires prepared original resources; the plan grants no lease.
    await require_current(manager, tx, row)
    membership = await _stopped_tree(manager, tx, row)
    delivery = await require_closed(manager, tx, actor, row["job_id"], generation, consumed=True)
    control = await controller(manager, tx, row, generation)
    if _hash(control["workflow"]) != _hash(proof(row)) or control["session_id"] != row["session_id"]:
        raise JobError("workflow_controller_mismatch")
    operation_id, creation_hash = await creation(manager, tx, row)
    event = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (operation_id,))
    request = json.loads(event["payload_json"])["request"]
    spec = WorkflowSpec.from_json(request["spec"])
    leaves, dependency = selection(spec, node_id)
    result = delivery["result"]
    if set(result["results"]) != set(leaves):
        raise JobError("workflow_restore_result_mismatch")
    nodes = {}
    references = [edge["payload"]["member"] for edge in membership["members"]]
    replayed = {}
    if generation > 1:
        from .workflow_resume import execution
        _, prior = await execution(manager, tx, row)
        replayed = {nid: prior["plan"]["nodes"][nid] for nid in prior["plan"]["replay"]}
        references.extend(n["lease"] for n in replayed.values())
    for lease in references:
        child = await manager._row(tx, lease["job_id"])
        if child["kind"] != "workflow_node":
            raise JobError("workflow_restore_node_mismatch")
        if generation == 1 and child["created_by_session_id"] != control["session_id"]:
            raise JobError("workflow_restore_node_mismatch")
        if generation > 1:
            frozen_replay = next((n for n in replayed.values() if n["lease"]["job_id"] == child["job_id"]), None)
            if frozen_replay is not None:
                if _hash(proof(child)) != _hash(frozen_replay["lease"]):
                    raise JobError("workflow_restore_node_mismatch")
                await _stopped_tree(manager, tx, child)
            else:
                grant = await get_authority_for_restore(manager, tx, child)
                if _hash(grant["workflow"]) != _hash(proof(row)):
                    raise JobError("workflow_restore_node_mismatch")
        canonical = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result' AND job_id=?",
            (result_id(child["job_id"], child["generation"]), child["job_id"]))
        payload = json.loads(canonical["payload_json"]) if canonical else {}
        raw = payload.get("result", {})
        nid, sid = raw.get("node_id"), raw.get("session_id")
        if (nid not in leaves or nid in nodes or not isinstance(sid, str) or child["outcome"] != "succeeded" or
                raw.get("status") != "completed" or _hash(raw) != _hash(result["results"][nid]) or
                _hash(raw) != _hash(json.loads(child["result_json"]))):
            raise JobError("workflow_restore_result_mismatch")
        [node_creation] = await tx.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (child["job_id"],))
        if _hash(json.loads(node_creation["payload_json"])["request"]["node"]) != _hash(_node_to_dict(leaves[nid][1])):
            raise JobError("workflow_restore_node_mismatch")
        binding = await descriptor(manager, tx, child, sid, historical=True)
        if binding["node_id"] != nid or binding["workflow_run_id"] != control["run_id"] or not await is_closed(manager, tx, child, sid):
            raise JobError("node_input_binding_mismatch")
        saved = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (sid, "workflow_execution_contract"))
        contract = json.loads(saved["value_json"]) if saved else None
        if not isinstance(contract, dict) or type(contract.get("version")) is not int or contract["version"] != 1:
            raise JobError("resume_contract_mismatch")
        actual_spec = AgentSpec.from_dict(contract["spec"])
        node = leaves[nid][1]
        tools = node.spec.tools
        if request["allowed_tools"] is not None:
            tools = sorted(request["allowed_tools"]) if tools is None else [t for t in tools if t in request["allowed_tools"]]
        expected_spec = asdict(replace(node.spec, tools=tools, lifecycle="linked", output_schema=node.output_schema))
        if _hash({k: v for k, v in asdict(actual_spec).items() if k != "metadata"}) != _hash({k: v for k, v in expected_spec.items() if k != "metadata"}):
            raise JobError("resume_contract_mismatch")
        physical = await tx.fetchone(f"SELECT metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
        meta = json.loads(physical["metadata_json"] or "{}")
        expected_meta = {**node.spec.metadata, "workflow_run_id": control["run_id"], "workflow_node_id": nid,
            "idempotency_key": control["run_id"]+":"+nid, "workflow_ancestry": meta.get("workflow_ancestry"),
            "task_owner_job_id": child["job_id"], "task_node_route": binding["route"], "attempt": binding["attempt"]}
        if _hash(actual_spec.metadata) != _hash(expected_meta) or contract.get("delegation_policy") not in ("legacy", "single_level"):
            raise JobError("resume_contract_mismatch")
        if not isinstance(contract.get("tools"), list) or any(not isinstance(v, str) for v in contract["tools"]):
            raise JobError("resume_contract_mismatch")
        if tools is not None and not set(contract["tools"]).issubset(tools):
            raise JobError("resume_contract_mismatch")
        state = await tx.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (sid,))
        if state is None or state["pending_json"]:
            raise JobError("resume_tools_pending")
        if await tx.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND status IN ('pending','claimed')", (sid,)):
            raise JobError("resume_input_pending")
        nodes[nid] = {"lease": lease, "binding": binding, "binding_hash": _hash(binding), "contract": contract,
            "contract_hash": _hash(contract), "result_event_id": canonical["event_id"], "result_revision": canonical["job_version"],
            "result_hash": _hash(payload), "result": raw, "path": list(leaves[nid][0])}
    if set(nodes) != set(leaves):
        raise JobError("workflow_restore_node_mismatch")
    return {"version": 1, "workflow": proof(row), "creation_operation_id": operation_id, "creation_hash": creation_hash,
        "run_id": control["run_id"], "controller": control, "profile": capabilities, "spec": request["spec"],
        "allowed_tools": request["allowed_tools"], "delivery_hash": _hash(delivery), "result": delivery["envelope"],
        "target": node_id, "message": message, "dependency": dependency.to_dict(), "nodes": nodes,
        "replay": sorted(set(nodes) - dependency.invalidated)}


async def preview(manager, actor, job_id, **kwargs):
    async with manager._write() as tx:
        return await prepare(manager, tx, actor, await manager._authorize(tx, actor, job_id), **kwargs)


async def prepare(manager, tx, actor, row, **kwargs):
    from power_loop.runtime.spec import AgentSpecError
    from power_loop.workflow.spec import WorkflowSpecError
    try:
        return await _prepare(manager, tx, actor, row, **kwargs)
    except (TypeError, ValueError, KeyError, AttributeError, AgentSpecError, WorkflowSpecError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_restore_evidence_mismatch") from exc


async def get_authority_for_restore(manager, tx, child):
    from .workflow_restore_execution import authority, get, identity
    data = await get(manager, tx, identity(manager, "node", child["job_id"], child["generation"]), child["job_id"], "workflow_node_authority")
    return (await authority(manager, tx, child, data["session_id"], historical=True))[0]
