"""Freeze historical workflow proof before independently fenced tree writes."""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import control_requests as requests
from .finalization_plan import PlanChanged
from .jobs import JobError, JobLease, _hash, _json, _operation, verify_stored
from .managed_effects import _ref
from .result_revisions import execution_id, result_id

ACTION = "restore"
CLIENT_FIELDS = ("caller", "target", "previous_generation", "node_id", "message", "expected_plan_hash", "lease_ms", "requested_rounds")
TARGET_BARRIERS = frozenset({"cancel_requested", "unresolved_prior_attempt", "resume_generation_changed",
    "workflow_restore_target_unsettled", "resume_tools_pending", "resume_input_pending",
    "executor_not_accepting", "executor_registration_required"})


@dataclass(frozen=True)
class PriorTree:
    root_id: str
    generation: int
    count: int
    version_sum: int


@dataclass(frozen=True)
class PreparedAdmission:
    row: dict
    plan: dict
    tree: PriorTree
    reopening: list
    journal: dict


def _client(command):
    return {key: command[key] for key in CLIENT_FIELDS}


def _intent(command, ref, call_id, executor_required):
    return {"job_id": command["target"]["job_id"], "caller": command["caller"],
        "call_ref": ref.to_dict(), "tool_call_id": call_id, "input": command, "executor_required": executor_required}


async def _mutable(manager, reader, actor, command, executor_required):
    row = await manager._authorize(reader, actor, command["target"]["job_id"])
    if (row["kind"] != "workflow" or row["execution_role"] != "workflow" or row["created_by_session_id"] != actor.session_id
            or row["control_owner_job_id"] is not None):
        raise JobError("workflow_restore_target_mismatch")
    await manager._supervision_barriers(reader, row)
    if row["generation"] != command["previous_generation"]:
        raise JobError("resume_generation_changed")
    if (row["outcome"] != "succeeded" or row["resource_state"] != "stopped" or row["intake"] != "closed"
            or row["result_status"] != "handled"):
        raise JobError("workflow_restore_target_unsettled")
    from .executor_registry import ExecutorIdentity, require_accepting
    target = command["target"]
    executor = await require_accepting(manager, reader, ExecutorIdentity(actor.tenant_id, actor.principal_id,
        target["executor_id"], target["boot_id"]), required=row["executor_key"] is not None or executor_required)
    return row, executor


async def _originals(manager, reader, actor, client, executor_required):
    row, _ = await _mutable(manager, reader, actor, client, executor_required)
    from .workflow_restore import prepare
    plan = await prepare(manager, reader, actor, row, generation=client["previous_generation"],
        node_id=client["node_id"], message=client["message"])
    if client["expected_plan_hash"] is not None and client["expected_plan_hash"] != _hash(plan):
        raise JobError("workflow_restore_plan_mismatch")
    if client["requested_rounds"] is not None and client["requested_rounds"] > int(plan["nodes"][client["node_id"]]["contract"]["spec"]["max_rounds"]):
        raise JobError("continuation_rounds_exceed_original_contract")
    affected = sorted(set(plan["nodes"]) & set(plan["dependency"]["invalidated"]))
    for node_id in affected:
        generation = plan["nodes"][node_id]["lease"]["generation"]
        if type(generation) is not int or not 1 <= generation < 2**63-1:
            raise JobError("invalid_execution_generation")
    from .execution_membership import usage
    prior = await usage(manager, reader, row, client["previous_generation"])
    if client["previous_generation"] > 1:
        from .workflow_resume import execution
        _, previous = await execution(manager, reader, row)
        if not isinstance(previous.get("prior_usage"), dict):
            raise JobError("workflow_restore_usage_unknown")
        prior = {key: prior.get(key, 0)+previous["prior_usage"].get(key, 0) for key in prior.keys() | previous["prior_usage"].keys()}
    return row, plan, affected, prior


async def _journal(manager, reader, actor, command, plan):
    from power_loop.workflow import journal
    saved = await reader.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (actor.session_id, journal.run_key(plan["run_id"])))
    data = json.loads(saved["value_json"])
    attempts = data.get("attempts", 1)
    if type(attempts) is not int or not 1 <= attempts < 2**63-1:
        raise JobError("workflow_restore_journal_mismatch")
    data.update(task_generation=command["target"]["generation"], status="running", result=None, finished_at_ms=None,
        error=None, woke=False, live={}, attempts=attempts+1, spec=plan["spec"], allowed_tools=plan["allowed_tools"])
    data["steps"], data["live_sessions"] = [], {}
    for node_id, node in sorted(plan["nodes"].items(), key=lambda item: item[1]["path"]):
        step = dict(node["result"])
        data["live_sessions"][node_id] = node["binding"]["session_id"]
        if node_id in command["affected"]:
            step.update(status="pending", text="", payload=None, usage={}, error="invalidated by workflow continuation")
        data["steps"].append(step)
    return data


async def _prepare(manager, reader, actor, request, *, original=None):
    command = request["input"]
    row, plan, affected, prior = original or await _originals(manager, reader, actor, _client(command), request["executor_required"])
    if (command["plan_hash"] != _hash(plan) or command["creation_operation_id"] != plan["creation_operation_id"]
            or command["affected"] != affected or command["prior_usage"] != prior
            or command["input_sessions"] != sorted({node["binding"]["session_id"] for node in plan["nodes"].values()})):
        raise JobError("workflow_restore_plan_mismatch")
    if await reader.fetchone(f"SELECT session_id FROM {manager.store.t.sessions} WHERE session_id=?", (command["driver_sid"],)):
        raise JobError("workflow_restore_driver_conflict")
    from .workflow_restore_resources import validate_evidence
    validate_evidence(plan["profile"], command.get("resource_evidence"))
    from .finalization import prepare_reopen
    reopening = await prepare_reopen(manager, reader, row)
    journal = await _journal(manager, reader, actor, command, plan)
    root = row["root_job_id"]
    summary = await reader.fetchone(f"SELECT COUNT(*) AS job_count,COALESCE(SUM(version),0) AS version_sum,"
        f"MAX(CASE WHEN job_id=? THEN generation END) AS root_generation FROM {manager.jobs} WHERE root_job_id=?", (root, root))
    return PreparedAdmission(row, plan, PriorTree(root, summary["root_generation"], summary["job_count"], summary["version_sum"]), reopening, journal)


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    command = request["input"]
    key = manager._origin_key(actor, "nudge", _ref(request["call_ref"]), request["tool_call_id"], None)
    driver = "wf_driver_"+_hash([manager.store.store_id, key, command["target"]])[:48]
    if (_intent(command, _ref(request["call_ref"]), request["tool_call_id"], request["executor_required"]) != request
            or command["caller"] != request["caller"] or command["target"]["generation"] != command["previous_generation"]+1
            or command["driver_sid"] != driver):
        raise JobError("restore_request_corrupt")
    accepted = await requests.stored(manager, reader, actor, event, request, action=ACTION, completion=completion)
    if (accepted.data["target"]["generation"] != command["previous_generation"]
            or accepted.data["source"]["execution_role"] != "main"
            or request["executor_required"] != (accepted.source["executor_key"] is not None)):
        raise JobError("restore_request_corrupt")
    return accepted


def _receipt(manager, event, request, request_ref):
    command = request["input"]
    value = json.loads(event["payload_json"])
    actual = value["request"]
    expected = {"job_id": request["job_id"], "run_id": actual["plan"]["run_id"], "mode": "workflow_continue", "status": "accepted",
        "generation": command["target"]["generation"], "result_event_id": result_id(request["job_id"], command["target"]["generation"]),
        "request": {**command, "plan": actual["plan"]}, "execution_id": event["event_id"][3:], "request_ref": request_ref}
    if (event["event_type"] != "operation" or event["action"] != "nudge" or event["job_id"] != request["job_id"]
            or _hash(actual["plan"]) != command["plan_hash"] or value != expected
            or event["request_hash"] != _hash([_client(command), request["call_ref"]])
            or event["caller_session_id"] != request["call_ref"]["session_id"] or event["tool_call_id"] != request["tool_call_id"]
            or json.loads(event["call_ref_json"]) != request["call_ref"]):
        raise JobError("workflow_restore_admission_mismatch")
    return _operation(event)


async def _inboxes(manager, reader, sessions):
    for sid in sessions:
        state = await reader.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (sid,))
        if state is None or state["pending_json"]:
            raise JobError("resume_tools_pending")
        if await reader.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND status IN ('pending','claimed')", (sid,)):
            raise JobError("resume_input_pending")


async def _publish(manager, actor, accepted, prepared):
    request, command = accepted.data["request"], accepted.data["request"]["input"]
    ref = _ref(request["call_ref"])
    key = manager._origin_key(actor, "nudge", ref, request["tool_call_id"], None)
    # The plan digest is permission, not another copy of the workflow ledger.
    async with manager.tree_write(accepted.data['target']['job_id']) as tx:
        await manager.store.signal_changes(tx, actor.session_id, "runtime")
        for sid in command["input_sessions"]:
            await manager.store.signal_changes(tx, sid, "inbox")
        old = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?",
            (key, requests.rejection_id(accepted.event["event_id"])))
        if old:
            return requests.read_outcome(manager, accepted, old[0]), old[0]
        current = await manager._authorize(tx, actor, request["job_id"])
        try:
            row, executor = await _mutable(manager, tx, actor, command, request["executor_required"])
            await _inboxes(manager, tx, command["input_sessions"])
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return None, await requests.reject(manager, tx, actor, accepted.event, accepted.data, current, exc.code)
        if prepared is None:
            raise PlanChanged()
        target, version, now = JobLease(**command["target"]), row["version"]+1, manager.now()
        full = {**command, "plan": prepared.plan}
        payload = {"job_id": row["job_id"], "run_id": prepared.plan["run_id"], "mode": "workflow_continue", "status": "accepted",
            "generation": target.generation, "result_event_id": result_id(row["job_id"], target.generation),
            "request": full, "request_ref": requests.reference(accepted.event)}
        operation = await manager._insert_operation(tx, job_id=row["job_id"], version=version, actor=actor, action="nudge",
            key=key, fingerprint=_hash([_client(command), request["call_ref"]]), call_ref=ref, tool_call_id=request["tool_call_id"],
            payload=payload, tree=prepared.tree)
        if operation is None:
            raise PlanChanged()
        fields = dict(generation=target.generation, executor_id=target.executor_id, boot_id=target.boot_id,
            executor_key=executor["executor_key"] if executor else None, lease_until_ms=now+command["lease_ms"], session_id=None,
            status="running", supervision_status="open", intake="open", resource_state="running", result_status="none",
            outcome=None, result_json=None, version=version, updated_at=now)
        if await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET "+','.join(field+'=?' for field in fields)
                +" WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                (*fields.values(), row["job_id"], row["root_job_id"], row["generation"], row["version"]), obligations=True) != 1:
            raise PlanChanged()
        updated = {**row, **fields}
        tx.remember_row(manager.jobs, 'job_id', row['job_id'], updated)
        payload["execution_id"] = operation.operation_id[3:]
        await tx.update_event(row["job_id"], f"UPDATE {manager.events} SET payload_json=?,delivery_status='pending_delivery' WHERE event_id=?",
            (_json(payload), operation.operation_id), notify=True)
        evidence = {"version": 1, "operation_id": operation.operation_id, "request_hash": _hash(full),
            "target": command["target"], "run_id": prepared.plan["run_id"], "driver_sid": command["driver_sid"]}
        await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'request_hash', 'payload_json', 'delivery_status', 'created_at'),
            (execution_id(row["job_id"], target.generation), row["job_id"], "execution", version, _hash(evidence), _json(evidence), "not_applicable", now))
        from .execution_membership import start
        await start(manager, tx, updated)
        from .workflow_resume import reservation_id
        for node_id in command["affected"]:
            previous = prepared.plan["nodes"][node_id]
            reserved = {"version": 1, "workflow": command["target"], "operation_id": operation.operation_id, "node_id": node_id,
                "previous": previous["lease"], "generation": previous["lease"]["generation"]+1, "source_hash": _hash(previous)}
            await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'request_hash', 'payload_json', 'delivery_status', 'created_at'),
                (reservation_id(manager, row["job_id"], target.generation, node_id), row["job_id"], "workflow_node_reservation", version,
                 _hash(reserved), _json(reserved), "not_applicable", now))
        from .finalization import publish_reopen
        await publish_reopen(manager, tx, row, command["target"], operation.operation_id, prepared.reopening)
        from power_loop.workflow import journal
        cached = {**prepared.journal, "updated_at_ms": now}
        await tx.execute(f"UPDATE {manager.store.t.session_runtime_state} SET value_json=?,updated_at=? WHERE session_id=? AND state_key=?",
            (_json(cached), now, actor.session_id, journal.run_key(prepared.plan["run_id"])))
        outcome = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
    return _operation(outcome), outcome


@verify_stored
async def _retained_execution(manager, actor, accepted, outcome):
    """Cold receipt replay requires its original atomic target execution too."""
    if outcome['event_type']=='task_control_request':
        return
    from .workflow_resume import execution
    async with manager._read() as reader:
        row = await manager._authorize(reader, actor, accepted.data['target']['job_id'])
        await execution(manager, reader, row, accepted.data['request']['input']['target']['generation'],
            expected_operation=outcome['event_id'])


async def _finish(manager, actor, accepted, outcome, prepared=None):
    requests.remember_event(manager, accepted.event)
    if outcome is not None:
        await _retained_execution(manager, actor, accepted, outcome)
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


async def admit(manager, actor, client, ref, call_id, *, resources):
    key = manager._origin_key(actor, "nudge", ref, call_id, None)
    prepared = None
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION, key),))
        if original is not None:
            request = json.loads(original["payload_json"])["request"]
            if _client(request["input"]) != client or request["call_ref"] != ref.to_dict() or request["tool_call_id"] != call_id:
                raise JobError("operation_conflict")
        else:
            source = await manager._authorize(reader, actor, client["caller"]["job_id"])
            manager._check_proof(source, JobLease(**client["caller"]))
            request = _intent(client, ref, call_id, source["executor_key"] is not None)
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
        if accepted is None:
            await manager._validate_call(reader, actor, ref, call_id)
            for action in ("create", "steer"):
                if await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?",
                        (manager._origin_key(actor, action, ref, call_id, None),)):
                    raise JobError("operation_conflict")
            frozen = await _originals(manager, reader, actor, client, request["executor_required"])
            row, plan, affected, prior = frozen
            command = {**client, "plan_hash": _hash(plan), "driver_sid": "wf_driver_"+_hash([manager.store.store_id, key, client["target"]])[:48],
                "creation_operation_id": plan["creation_operation_id"], "affected": affected, "prior_usage": prior,
                "input_sessions": sorted({node["binding"]["session_id"] for node in plan["nodes"].values()})}
            from .workflow_restore_resources import admission
            resource = admission(resources, manager, actor, plan["profile"], prior_usage=prior)
            if resource is not None:
                command["resource_evidence"] = resource
            request = _intent(command, ref, call_id, request["executor_required"])
            prepared = await _prepare(manager, reader, actor, request, original=frozen)
    repeated = accepted is not None
    if accepted is None:
        async def authorize(tx):
            from .workflow_resume import _source
            return await _source(manager, tx, actor, JobLease(**client["caller"]), ref)
        row = prepared.row
        accepted, repeated = await requests.accept(manager, actor, request,
            {"job_id": row["job_id"], "root_job_id": row["root_job_id"], "generation": row["generation"]},
            action=ACTION, authorize=authorize)
    return await _finish(manager, actor, accepted, outcome, prepared), repeated


@verify_stored
async def recover_call(manager, actor, ref, call_id, *, job_id, node_id, message, rounds):
    key = manager._origin_key(actor, "nudge", ref, call_id, None)
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION, key),))
        if original is None:
            return None
        request = json.loads(original["payload_json"])["request"]
        command = request["input"]
        if (request["job_id"] != job_id or command["node_id"] != node_id or command["message"] != message or command["requested_rounds"] != rounds):
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
            raise JobError("restore_request_original_missing")
        request = json.loads(original["payload_json"])["request"]
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
    return await _finish(manager, actor, accepted, outcome)
