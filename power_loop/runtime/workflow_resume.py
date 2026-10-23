"""Atomic restored-execution admission. Dispatch claims do not run an engine.

The native adapter must retain owned work before admission, then claim dispatch
and bind the reserved driver before any node/model execution. Until that adapter
is wired, legacy resume/force and ordinary later-generation binds stay closed.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _json, _operation
from .message_ref import MessageRef, integer_id
from .result_revisions import execution_id, result_id


def reservation_id(manager, job_id, generation, node_id):
    return "wf_res_" + _hash([manager.store.store_id, job_id, generation, node_id])[:48]


def dispatch_id(operation_id):
    return "wf_disp_" + _hash(operation_id)[:48]


def _ref(data):
    if type(data.get("version")) is not int or data["version"] != 1 or data.get("id_status") != "available":
        raise JobError("invalid_message_ref")
    return MessageRef(store_id=data["store_id"], session_id=data["session_id"], id=integer_id(data["id"]),
        seq=integer_id(data["seq"]), role=data["role"], tool_call_id=data.get("tool_call_id"),
        send_index=int(data["send_index"]) if data.get("send_index") is not None else None,
        round_index=data.get("round_index"))


async def _source(manager, tx, actor, caller, ref):
    row = await manager._fence(tx, actor, caller)
    origin = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (caller.job_id,))
    index = json.loads(origin["payload_json"]).get("request", {}).get("send_index") if origin else None
    if (row["kind"] != "send" or row["session_id"] != actor.session_id or row["execution_role"] != "main" or
            type(index) is not int or index < 1 or index != ref.send_index):
        raise JobError("resume_caller_mismatch")


async def admit(manager, actor, job_id, *, caller, call_ref, tool_call_id, generation, node_id, message,
                executor_id, boot_id, plan_hash=None, lease_ms=30000, rounds=None, resources=None):
    """All new authority is reserved in one transaction, not physical execution."""
    if type(generation) is not int or not 1 <= generation < 2**63-1:
        raise JobError("invalid_result_generation")
    if not isinstance(node_id, str) or not node_id or len(node_id) > 255:
        raise JobError("workflow_restore_target_mismatch")
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
        raise JobError("invalid_executor")
    if type(lease_ms) is not int or not 1 <= lease_ms <= 300000:
        raise JobError("invalid_lease_duration")
    if plan_hash is not None and (not isinstance(plan_hash, str) or len(plan_hash) != 64 or any(c not in "0123456789abcdef" for c in plan_hash)):
        raise JobError("workflow_restore_plan_mismatch")
    if rounds is not None and (type(rounds) is not int or rounds < 1):
        raise JobError("invalid_continuation_rounds")
    key = manager._origin_key(actor, "nudge", call_ref, tool_call_id, None)
    target = JobLease(job_id, executor_id, boot_id, generation+1)
    intent = {"caller": asdict(caller), "target": asdict(target), "previous_generation": generation,
        "node_id": node_id, "message": message, "expected_plan_hash": plan_hash, "lease_ms": lease_ms, "requested_rounds": rounds}
    fingerprint = _hash([intent, call_ref.to_dict()])
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old is not None:
            await manager._authorize(tx, actor, old["job_id"])
            if old["request_hash"] != fingerprint or old["job_id"] != job_id:
                raise JobError("operation_conflict")
            await execution(manager, tx, await manager._row(tx, job_id), target.generation, expected_operation=old["event_id"])
            return _operation(old), True
        await manager._validate_call(tx, actor, call_ref, tool_call_id)
        for action in ("create", "steer"):
            if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?",
                    (manager._origin_key(actor, action, call_ref, tool_call_id, None),)):
                raise JobError("operation_conflict")
        await _source(manager, tx, actor, caller, call_ref)
        row = await manager._authorize(tx, actor, job_id)
        from .workflow_restore import prepare
        plan = await prepare(manager, tx, actor, row, generation=generation, node_id=node_id, message=message)
        if plan_hash is not None and plan_hash != _hash(plan):
            raise JobError("workflow_restore_plan_mismatch")
        if rounds is not None and rounds > int(plan["nodes"][node_id]["contract"]["spec"]["max_rounds"]):
            raise JobError("continuation_rounds_exceed_original_contract")
        driver_sid = "wf_driver_" + _hash([manager.store.store_id, key, asdict(target)])[:48]
        if await tx.fetchone(f"SELECT session_id FROM {manager.store.t.sessions} WHERE session_id=?", (driver_sid,)):
            raise JobError("workflow_restore_driver_conflict")
        affected = sorted(set(plan["nodes"]) & set(plan["dependency"]["invalidated"]))
        for nid in affected:
            if type(plan["nodes"][nid]["lease"]["generation"]) is not int or not 1 <= plan["nodes"][nid]["lease"]["generation"] < 2**63-1:
                raise JobError("invalid_execution_generation")
        request = {**intent, "plan": plan, "plan_hash": _hash(plan), "driver_sid": driver_sid,
            "creation_operation_id": plan["creation_operation_id"], "affected": affected}
        from .execution_membership import usage
        prior_usage = await usage(manager, tx, row, generation)
        if generation > 1:
            _, previous = await execution(manager, tx, row)
            if not isinstance(previous.get("prior_usage"), dict):
                raise JobError("workflow_restore_usage_unknown")
            prior_usage = {k: prior_usage.get(k, 0)+previous["prior_usage"].get(k, 0) for k in prior_usage.keys() | previous["prior_usage"].keys()}
        request["prior_usage"] = prior_usage
        from .workflow_restore_resources import admission
        resource_evidence = admission(resources, manager, actor, plan["profile"], prior_usage=prior_usage)
        if resource_evidence is not None:
            request["resource_evidence"] = resource_evidence
        now = manager.now()
        updated = await tx.execute(f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,lease_until_ms=?,session_id=NULL,"
            "status='running',supervision_status='open',intake='open',resource_state='running',result_status='none',outcome=NULL,result_json=NULL,"
            "version=version+1,updated_at=? WHERE job_id=?",
            (target.generation, executor_id, boot_id, now+lease_ms, now, job_id))
        from .workflow_controller import proof
        current = await manager._row(tx, job_id)
        if updated != 1 or _hash(proof(current)) != _hash(asdict(target)) or current["session_id"] is not None or current["result_json"] is not None:
            raise JobError("workflow_restore_admission_mismatch")
        payload = {"job_id": job_id, "run_id": plan["run_id"], "mode": "workflow_continue", "status": "accepted",
            "generation": target.generation, "result_event_id": result_id(job_id, target.generation), "request": request}
        op = await manager._insert_operation(tx, job_id=job_id, version=row["version"]+1, actor=actor, action="nudge",
            key=key, fingerprint=fingerprint, call_ref=call_ref, tool_call_id=tool_call_id, payload=payload)
        payload["execution_id"] = op.operation_id[3:]
        if await tx.execute(f"UPDATE {manager.events} SET payload_json=?,delivery_status='pending_delivery' WHERE event_id=?", (_json(payload), op.operation_id)) != 1:
            raise JobError("workflow_restore_admission_mismatch")
        evidence = {"version": 1, "operation_id": op.operation_id, "request_hash": _hash(request),
            "target": asdict(target), "run_id": plan["run_id"], "driver_sid": driver_sid}
        if await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (execution_id(job_id, target.generation), job_id, "execution", row["version"]+1, _hash(evidence), _json(evidence), "not_applicable", now)) != 1:
            raise JobError("workflow_restore_admission_mismatch")
        from .execution_membership import scope, start
        await start(manager, tx, await manager._row(tx, job_id))
        if _hash((await scope(manager, tx, job_id, target.generation))["lease"]) != _hash(asdict(target)):
            raise JobError("workflow_restore_admission_mismatch")
        for nid in affected:
            reserved = {"version": 1, "workflow": asdict(target), "operation_id": op.operation_id, "node_id": nid,
                "previous": plan["nodes"][nid]["lease"], "generation": plan["nodes"][nid]["lease"]["generation"]+1,
                "source_hash": _hash(plan["nodes"][nid])}
            rid = reservation_id(manager, job_id, target.generation, nid)
            if await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (rid, job_id, "workflow_node_reservation", row["version"]+1, _hash(reserved), _json(reserved), "not_applicable", now)) != 1:
                raise JobError("workflow_restore_admission_mismatch")
            stored = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (rid,))
            if stored is None or _hash(json.loads(stored["payload_json"])) != _hash(reserved):
                raise JobError("workflow_restore_admission_mismatch")
        from .finalization import reopen
        await reopen(manager, tx, row, asdict(target), op.operation_id)
        await _reset_journal(manager, tx, actor, request)
        await execution(manager, tx, current, expected_operation=op.operation_id)
        return _operation(await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (op.operation_id,))), False


async def _reset_journal(manager, tx, actor, request):
    from power_loop.workflow import journal
    key = journal.run_key(request["plan"]["run_id"])
    saved = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?", (actor.session_id, key))
    j = json.loads(saved["value_json"])
    attempts = j.get("attempts", 1)
    if type(attempts) is not int or not 1 <= attempts < 2**63-1:
        raise JobError("workflow_restore_journal_mismatch")
    j.update(task_generation=request["target"]["generation"], status="running", result=None, finished_at_ms=None,
        error=None, woke=False, live={}, updated_at_ms=manager.now(), attempts=attempts+1,
        spec=request["plan"]["spec"], allowed_tools=request["plan"]["allowed_tools"])
    # The mutable compatibility cache is not a source for preserved outputs.
    # Canonical node evidence supplies both replayed and invalidated views.
    j["steps"] = []
    j["live_sessions"] = {}
    for nid, node in sorted(request["plan"]["nodes"].items(), key=lambda item: item[1]["path"]):
        step = dict(node["result"])
        j["live_sessions"][nid] = node["binding"]["session_id"]
        if nid in request["affected"]:
            step.update(status="pending", text="", payload=None, usage={}, error="invalidated by workflow continuation")
        j["steps"].append(step)
    if await tx.execute(f"UPDATE {manager.store.t.session_runtime_state} SET value_json=?,updated_at=? WHERE session_id=? AND state_key=?",
            (_json(j), manager.now(), actor.session_id, key)) != 1:
        raise JobError("workflow_restore_journal_mismatch")
    actual = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?", (actor.session_id, key))
    if actual is None or _hash(json.loads(actual["value_json"])) != _hash(j):
        raise JobError("workflow_restore_journal_mismatch")


async def execution(manager, tx, row, generation=None, *, expected_operation=None):
    """Read immutable admission evidence. Historical reads do not grant a lease."""
    generation = row["generation"] if generation is None else generation
    if row["kind"] != "workflow" or row["execution_role"] != "workflow":
        raise JobError("workflow_restore_execution_mismatch")
    record = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (execution_id(row["job_id"], generation),))
    try:
        data = json.loads(record["payload_json"]) if record else {}
        if (record is None or record["event_type"] != "execution" or record["job_id"] != row["job_id"] or
                record["request_hash"] != _hash(data) or type(data.get("version")) is not int or data["version"] != 1 or
                record["delivery_status"] != "not_applicable" or data["target"]["generation"] != generation):
            raise JobError("workflow_restore_execution_mismatch")
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data["operation_id"],))
        payload = json.loads(op["payload_json"]) if op else {}
        request = payload["request"]
        from .execution_membership import _lease
        _lease(request["caller"])
        _lease(request["target"])
        _lease(request["plan"]["workflow"])
        intent = {k: request[k] for k in ("caller", "target", "previous_generation", "node_id", "message", "expected_plan_hash", "lease_ms", "requested_rounds")}
        if (op["event_type"] != "operation" or op["action"] != "nudge" or op["job_id"] != row["job_id"] or
                (expected_operation is not None and op["event_id"] != expected_operation) or
                op["request_hash"] != _hash([intent, json.loads(op["call_ref_json"])]) or
                payload.get("mode") != "workflow_continue" or type(payload["generation"]) is not int or payload["generation"] != generation or
                type(request["previous_generation"]) is not int or request["previous_generation"]+1 != generation or
                request["plan"]["workflow"]["generation"] != request["previous_generation"] or
                request["target"]["job_id"] != row["job_id"] or record["job_version"] != op["job_version"] or
                _hash(request) != data["request_hash"] or _hash(request["plan"]) != request["plan_hash"] or
                request["driver_sid"] != data["driver_sid"] or _hash(request["target"]) != _hash(data["target"]) or
                payload["run_id"] != data["run_id"] or data["run_id"] != request["plan"]["run_id"] or
                payload["execution_id"] != op["event_id"][3:]):
            raise JobError("workflow_restore_execution_mismatch")
        from .workflow_restore_resources import validate_evidence
        validate_evidence(request["plan"]["profile"], request.get("resource_evidence"))
        return op, request
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_restore_execution_mismatch") from exc


async def claim_dispatch(manager, actor, operation_id, *, caller, token, resources=None):
    if not isinstance(token, str) or not 0 < len(token) <= 64 or not isinstance(caller, JobLease):
        raise JobError("invalid_dispatch_token")
    async with manager._write() as tx:
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation_id,))
        if op is None or op["caller_session_id"] != actor.session_id:
            raise JobError("resume_dispatch_mismatch")
        row = await manager._authorize(tx, actor, op["job_id"])
        saved, request = await execution(manager, tx, row, expected_operation=operation_id)
        if request["caller"] != asdict(caller):
            raise JobError("resume_dispatch_mismatch")
        key = dispatch_id(operation_id)
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
        if old:
            payload = json.loads(old["payload_json"])
            expected = {"version": 1, "operation_id": operation_id, "target": request["target"],
                "driver_sid": request["driver_sid"], "token": payload.get("token")}
            if (old["event_type"] != "workflow_dispatch" or old["job_id"] != row["job_id"] or old["delivery_status"] != "claimed" or
                    old["request_hash"] != _hash(payload) or _hash(expected) != _hash(payload) or
                    not isinstance(payload.get("token"), str) or not 0 < len(payload["token"]) <= 64):
                raise JobError("workflow_restore_dispatch_mismatch")
            return payload.get("token") == token
        ref = _ref(json.loads(saved["call_ref_json"]))
        await manager._validate_call(tx, actor, ref, saved["tool_call_id"])
        await _source(manager, tx, actor, caller, ref)
        await manager._fence(tx, actor, JobLease(**request["target"]))
        from .workflow_restore_resources import admission
        admission(resources, manager, actor, request["plan"]["profile"],
            prior_usage=request["prior_usage"], previous=request.get("resource_evidence"))
        evidence = {"version": 1, "operation_id": operation_id, "target": request["target"], "driver_sid": request["driver_sid"], "token": token}
        if await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, row["job_id"], "workflow_dispatch", row["version"], _hash(evidence), _json(evidence), "claimed", manager.now())) != 1:
            raise JobError("workflow_restore_dispatch_mismatch")
        saved = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (key,))
        if saved is None or _hash(json.loads(saved["payload_json"])) != _hash(evidence):
            raise JobError("workflow_restore_dispatch_mismatch")
        return True
