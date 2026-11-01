"""Prepare original leaf proof before separately fenced continuation writes.

Only the live original workflow may execute a resumed leaf. Recovery completes
its source permission and target admission without reconstructing that engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import control_requests as requests
from .finalization_plan import PlanChanged
from .jobs import JobError, JobLease, _hash, _json, _operation, verify_stored
from .managed_effects import _ref
from .result_revisions import execution_id, result_id

ACTION = "resume_node"
TARGET_BARRIERS = frozenset({
    "stale_execution", "stale_parent_execution", "cancel_requested", "unresolved_prior_attempt",
    "job_intake_closed", "execution_lease_expired", "execution_not_running",
    "owner_completion_deadline_exceeded", "owner_wait_deadline_exceeded",
    "resume_generation_changed", "resume_target_unsettled", "resume_tools_pending", "resume_input_pending",
    "executor_not_accepting", "executor_registration_required",
})


@dataclass(frozen=True)
class PriorTree:
    root_id: str
    generation: int
    count: int
    version_sum: int


def _intent(command, ref, call_id, executor_required):
    return {"job_id": command["target"]["job_id"], "caller": command["caller"],
        "call_ref": ref.to_dict(), "tool_call_id": call_id, "input": command, "executor_required": executor_required}


async def _source(manager, reader, actor, request):
    source = await manager._fence(reader, actor, JobLease(**request["caller"]))
    create = await reader.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (source["job_id"],))
    index = json.loads(create["payload_json"]).get("request", {}).get("send_index") if create else None
    if (source["kind"] != "send" or source["session_id"] != actor.session_id or source["execution_role"] != "main"
            or type(index) is not int or index < 1 or index != _ref(request["call_ref"]).send_index):
        raise JobError("resume_caller_mismatch")
    return source


async def _mutable(manager, reader, actor, request):
    command = request["input"]
    workflow = await manager._fence(reader, actor, JobLease(**command["workflow"]))
    row = await manager._authorize(reader, actor, request["job_id"])
    if (workflow["kind"] != "workflow" or workflow["created_by_session_id"] != actor.session_id
            or row["kind"] != "workflow_node" or row["execution_role"] != "workflow_node"
            or row["parent_job_id"] != workflow["job_id"] or row["control_owner_job_id"] != workflow["job_id"]
            or row["created_by_session_id"] != workflow["session_id"]):
        raise JobError("resume_target_mismatch")
    await manager._supervision_barriers(reader, row)
    await manager._control_owners(reader, row)
    if row["generation"] != command["previous_generation"]:
        raise JobError("resume_generation_changed")
    if row["outcome"] != "succeeded" or row["resource_state"] != "stopped" or row["intake"] != "closed":
        raise JobError("resume_target_unsettled")
    sid = command["session_id"]
    state = await reader.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (sid,))
    if state is not None and state["pending_json"]:
        raise JobError("resume_tools_pending")
    if await reader.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND status IN ('pending','claimed')", (sid,)):
        raise JobError("resume_input_pending")
    from .executor_registry import ExecutorIdentity, require_accepting
    target = command["target"]
    executor = await require_accepting(manager, reader, ExecutorIdentity(actor.tenant_id, actor.principal_id,
        target["executor_id"], target["boot_id"]), required=row["executor_key"] is not None
        or workflow["executor_key"] is not None or request["executor_required"])
    return row, executor


async def _immutable(manager, reader, actor, request, row):
    """Read the actual controller, physical input and consumed result originals."""
    command, node = request["input"], request["input"]["node"]
    sid = command["session_id"]
    from .workflow_controller import for_node
    await for_node(manager, reader, row)
    from .node_inputs import descriptor, is_closed
    previous = {key: value for key, value in node.items() if key != "generation"}
    if command["previous_generation"] > 1:
        previous["generation"] = command["previous_generation"]
    if _hash(await descriptor(manager, reader, row, sid)) != _hash(previous) or not await is_closed(manager, reader, row, sid):
        raise JobError("node_input_binding_mismatch")
    canonical = await reader.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND job_id=? AND event_type='result'",
        (result_id(row["job_id"], row["generation"]), row["job_id"]))
    if (canonical is None or _hash(json.loads(canonical["payload_json"])["result"]) != command["previous_result_hash"]
            or _hash(json.loads(row["result_json"])) != command["previous_result_hash"]):
        raise JobError("node_resume_result_mismatch")
    result = json.loads(canonical["payload_json"])["result"]
    workflow_op = await reader.fetchone(f"SELECT event_id,payload_json FROM {manager.events} WHERE job_id=? AND action='create'",
        (command["workflow"]["job_id"],))
    from power_loop.workflow import journal
    from power_loop.workflow.spec import WorkflowSpec

    from .node_resume import selection
    if workflow_op is None or workflow_op["event_id"][3:] != node["workflow_run_id"]:
        raise JobError("workflow_run_mismatch")
    frozen = json.loads(workflow_op["payload_json"])["request"]["spec"]
    _, _, expected = selection(WorkflowSpec.from_json(frozen), node["node_id"])
    if _hash(expected.to_dict()) != _hash(command["plan"]):
        raise JobError("node_resume_plan_mismatch")
    journal_row = await reader.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (actor.session_id, journal.run_key(node["workflow_run_id"])))
    saved = json.loads(journal_row["value_json"]) if journal_row else {}
    steps = [step for step in saved.get("steps", []) if step.get("node_id") == node["node_id"]]
    fields = ("node_id", "status", "session_id", "text", "payload", "usage", "error")
    if (saved.get("status") != "running" or len(steps) != 1
            or _hash({key: steps[0].get(key) for key in fields}) != _hash({key: result.get(key) for key in fields})):
        raise JobError("node_resume_journal_mismatch")
    contract = await reader.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key='workflow_execution_contract'", (sid,))
    value = json.loads(contract["value_json"]) if contract else None
    if not isinstance(value, dict) or value.get("version") != 1 or _hash(value) != command["contract_hash"]:
        raise JobError("resume_contract_mismatch")


async def _prepare(manager, reader, actor, request):
    row, _ = await _mutable(manager, reader, actor, request)
    from .finalization import prepare_reopen, require_node_consumed, require_resumable
    await require_resumable(manager, reader, row)
    await _immutable(manager, reader, actor, request, row)
    await manager._require_scope_ready(reader, row)
    from .execution_membership import require_current
    await require_current(manager, reader, row)
    await require_node_consumed(manager, reader, row)
    reopening = await prepare_reopen(manager, reader, row)
    root = row["root_job_id"]
    summary = await reader.fetchone(f"SELECT COUNT(*) AS job_count,COALESCE(SUM(version),0) AS version_sum,"
        f"MAX(CASE WHEN job_id=? THEN generation END) AS root_generation FROM {manager.jobs} WHERE root_job_id=?", (root, root))
    return row, PriorTree(root, summary["root_generation"], summary["job_count"], summary["version_sum"]), reopening


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    command = request["input"]
    if (_intent(command, _ref(request["call_ref"]), request["tool_call_id"], request["executor_required"]) != request
            or command["caller"] != request["caller"] or command["target"]["generation"] != command["previous_generation"]+1
            or command["node"]["generation"] != command["target"]["generation"]):
        raise JobError("resume_node_request_corrupt")
    accepted = await requests.stored(manager, reader, actor, event, request, action=ACTION, completion=completion)
    if (accepted.data["target"]["generation"] != command["previous_generation"]
            or accepted.data["source"]["execution_role"] != "main"
            or request["executor_required"] != (accepted.source["executor_key"] is not None)):
        raise JobError("resume_node_request_corrupt")
    return accepted


def _receipt(manager, event, request, request_ref):
    command = request["input"]
    expected = {"job_id": request["job_id"], "session_id": command["session_id"], "mode": "continue", "status": "accepted",
        "generation": command["target"]["generation"], "result_event_id": result_id(request["job_id"], command["target"]["generation"]),
        "request": command, "run_id": command["node"]["workflow_run_id"], "request_ref": request_ref}
    if (event["event_type"] != "operation" or event["action"] != "nudge" or event["job_id"] != request["job_id"]
            or event["request_hash"] != _hash([command, request["call_ref"]]) or json.loads(event["payload_json"]) != expected
            or event["caller_session_id"] != request["call_ref"]["session_id"] or event["tool_call_id"] != request["tool_call_id"]
            or json.loads(event["call_ref_json"]) != request["call_ref"]):
        raise JobError("resume_node_receipt_mismatch")
    return _operation(event)


async def _publish(manager, actor, accepted, prepared):
    request, command = accepted.data["request"], accepted.data["request"]["input"]
    ref = _ref(request["call_ref"])
    key = manager._origin_key(actor, "nudge", ref, request["tool_call_id"], None)
    async with manager._write() as tx:
        await manager.store.signal_changes(tx, command["session_id"], "inbox")
        old = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?",
            (key, requests.rejection_id(accepted.event["event_id"])))
        if old:
            return requests.read_outcome(manager, accepted, old[0]), old[0]
        current = await manager._authorize(tx, actor, request["job_id"])
        try:
            row, executor = await _mutable(manager, tx, actor, request)
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return None, await requests.reject(manager, tx, actor, accepted.event, accepted.data, current, exc.code)
        if prepared is None:
            raise PlanChanged()
        _, tree, reopening = prepared
        target = JobLease(**command["target"])
        version, now = row["version"]+1, manager.now()
        payload = {"job_id": row["job_id"], "session_id": command["session_id"], "run_id": command["node"]["workflow_run_id"],
            "mode": "continue", "status": "accepted", "generation": target.generation,
            "result_event_id": result_id(row["job_id"], target.generation), "request": command,
            "request_ref": requests.reference(accepted.event)}
        operation = await manager._insert_operation(tx, job_id=row["job_id"], version=version, actor=actor, action="nudge",
            key=key, fingerprint=_hash([command, request["call_ref"]]), call_ref=ref, tool_call_id=request["tool_call_id"], payload=payload, tree=tree)
        if operation is None:
            raise PlanChanged()
        if await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,executor_key=?,lease_until_ms=?,"
                "status='running',supervision_status='open',intake='open',resource_state='running',result_status='none',outcome=NULL,result_json=NULL,"
                "version=version+1,updated_at=? WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                (target.generation, target.executor_id, target.boot_id, executor["executor_key"] if executor else None,
                 now+command["lease_ms"], now, row["job_id"], row["root_job_id"], row["generation"], row["version"]), obligations=True) != 1:
            raise PlanChanged()
        await tx.update_event(row["job_id"], f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (operation.operation_id,), notify=True)
        for event_id, kind, data in ((execution_id(row["job_id"], target.generation), "execution",
                {"operation_id": operation.operation_id, "run_id": command["node"]["workflow_run_id"], "target": command["target"], "workflow": command["workflow"]}),
                (manager._node_session_event_id(row["job_id"], command["session_id"], target.generation), "node_session", command["node"])):
            await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'payload_json', 'delivery_status', 'created_at'),
                (event_id, row["job_id"], kind, version, _json(data), "not_applicable", now))
        columns = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        if not await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, columns),
                (command["session_id"], "nudge:"+operation.operation_id, "relay", "steer", "pending", command["message"], now)):
            raise JobError("nudge_inbox_conflict")
        from .execution_membership import start
        from .finalization import publish_reopen
        await start(manager, tx, await manager._row(tx, row["job_id"]))
        await publish_reopen(manager, tx, row, command["target"], operation.operation_id, reopening)
        outcome = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
    return _operation(outcome), outcome


async def _finish(manager, actor, accepted, outcome, prepared=None):
    requests.remember_event(manager, accepted.event)
    while outcome is None:
        if prepared is None:
            async with manager._read() as reader:
                try:
                    prepared = await _prepare(manager, reader, actor, accepted.data["request"])
                except JobError as exc:
                    if exc.code not in TARGET_BARRIERS:
                        raise
        try:
            _, outcome = await _publish(manager, actor, accepted, prepared)
        except PlanChanged:
            prepared = None
    operation = requests.read_outcome(manager, accepted, outcome)
    await requests.complete(manager, actor, accepted, outcome)
    if operation is None:
        raise JobError(json.loads(outcome["payload_json"])["error"])
    return operation


async def admit(manager, actor, command, ref, call_id):
    key = manager._origin_key(actor, "nudge", ref, call_id, None)
    prepared = None
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION, key),))
        if original is not None:
            request = json.loads(original["payload_json"])["request"]
            if request["input"] != command or request["call_ref"] != ref.to_dict() or request["tool_call_id"] != call_id:
                raise JobError("operation_conflict")
        else:
            source = await manager._authorize(reader, actor, command["caller"]["job_id"])
            manager._check_proof(source, JobLease(**command["caller"]))
            request = _intent(command, ref, call_id, source["executor_key"] is not None)
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
        if accepted is None:
            await manager._validate_call(reader, actor, ref, call_id)
            for action in ("create", "steer"):
                if await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?",
                        (manager._origin_key(actor, action, ref, call_id, None),)):
                    raise JobError("operation_conflict")
            prepared = await _prepare(manager, reader, actor, request)
    repeated = accepted is not None
    if accepted is None:
        async def authorize(tx):
            return await _source(manager, tx, actor, request)
        row, _, _ = prepared
        accepted, repeated = await requests.accept(manager, actor, request,
            {"job_id": row["job_id"], "root_job_id": row["root_job_id"], "generation": row["generation"]},
            action=ACTION, authorize=authorize)
    return await _finish(manager, actor, accepted, outcome, prepared), repeated


@verify_stored
async def recover_call(manager, actor, ref, call_id, *, session_id, message, node_key, expected_job=None):
    key = manager._origin_key(actor, "nudge", ref, call_id, None)
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION, key),))
        if original is None:
            return None
        request = json.loads(original["payload_json"])["request"]
        command, node = request["input"], request["input"]["node"]
        if (command["session_id"] != session_id or command["message"] != message
                or (node["workflow_run_id"], node["node_id"]) != node_key
                or expected_job is not None and (request["job_id"], command["previous_generation"]) != expected_job):
            raise JobError("operation_conflict")
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
    return await _finish(manager, actor, accepted, outcome)


@verify_stored
async def reconcile_request(manager, actor, request_id):
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_id,))
        if original is None:
            raise JobError("resume_node_request_original_missing")
        request = json.loads(original["payload_json"])["request"]
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
    return await _finish(manager, actor, accepted, outcome)
