"""Original-decision system effects; enqueueing and physical completion differ.

The host calls apply after the original model decision's raw receipt commits.
No old main lease is reused, no model is dispatched, and no replacement task
is created. Continuation uses the existing original-owner nudge input protocol.
"""

from __future__ import annotations

from . import timer_waits as waits
from . import wait_decisions as decisions
from . import wait_notifications as notices
from .jobs import JobActor, JobError, JobLease, _hash, _json


def effect_id(manager, decision_id):
    return "wait_effect_" + _hash([manager.store.store_id, decision_id])[:48]


async def _decision(manager, tx, actor, event_id, *, _notification=None):
    bundle = _notification if _notification is not None else await notices.notification(manager, tx, actor, event_id)
    job, due, saved, _ = bundle
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (notices.decision_id(manager, event_id),))
    if row is None:
        raise JobError("wait_decision_required")
    data = waits._object(row["payload_json"])
    op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data.get("operation_id"),))
    if op is None or not isinstance(data.get("request"), dict):
        raise JobError("wait_decision_evidence_missing")
    accepted = await decisions._accepted(manager, tx, actor, event_id, data["request"], op, _notification=bundle)
    # A worker must not turn a decision with an uncommitted/missing tool receipt
    # into an input. The completed main model send is not its execution lease.
    if accepted.receipt_ref is None:
        raise JobError("wait_decision_receipt_required")
    call, receipt = waits._ref(accepted.call_ref), waits._ref(accepted.receipt_ref)
    await manager._validate_call(tx, actor, call, op["tool_call_id"])
    await manager._validate_ref(tx, actor, receipt, role="tool")
    if (receipt.seq <= call.seq or receipt.tool_call_id != op["tool_call_id"]
            or receipt.send_index != call.send_index or receipt.round_index != call.round_index):
        raise JobError("wait_decision_receipt_mismatch")
    return job, due, saved, row, data


def _binding(manager, decision, data):
    return {"event_id": data["event_id"], "decision_id": decision["event_id"],
        "decision_hash": decision["request_hash"], "effect_id": effect_id(manager, decision["event_id"])}


def _input(saved, binding):
    request = {"target": saved["binding"]["owner"], "session_id": saved["binding"]["session_id"],
        "message": saved["request"]["request"]["continuation"], "handoff": binding}
    if saved["binding"]["route"]["kind"] == "workflow_node":
        request["node"] = saved["binding"]["route"]["node"]
    return request


async def _effect(manager, tx, actor, job, saved, decision, data, *, _review=True):
    key = effect_id(manager, decision["event_id"])
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if row is None:
        if decision["delivery_status"] != "pending_delivery":
            raise JobError("wait_effect_evidence_missing")
        return None
    payload = waits._object(row["payload_json"])
    choice = data["request"]["decision"]
    expected = {"version": 1, "binding": _binding(manager, decision, data), "wait_job_id": job["job_id"],
        "owner": saved["binding"]["owner"], "decision": choice, "operation_id": payload.get("operation_id"),
        "review_at_ms": data["request"]["next_review_at_ms"]}
    if (row["event_type"] != "wait_effect" or row["job_id"] != job["job_id"]
            or row["caller_session_id"] != actor.session_id or row["request_hash"] != _hash(expected)
            or _hash(payload) != _hash(expected) or row["delivery_status"] not in ("delivered", "handled")
            or decision["delivery_status"] != row["delivery_status"]):
        raise JobError("wait_effect_evidence_corrupt")
    if choice == "defer":
        if payload["operation_id"] is not None:
            raise JobError("wait_effect_evidence_corrupt")
        if _review and row["delivery_status"] == "handled":
            from .wait_reviews import require_link
            await require_link(manager, tx, job, saved, decision, row)
    else:
        action = "nudge" if choice == "continue" else "stop"
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (payload["operation_id"],))
        expected_key = manager._origin_key(actor, action, None, None, "wait-effect:" + decision["event_id"])
        if (op is None or op["event_type"] != "operation" or op["action"] != action
                or op["job_id"] != saved["binding"]["owner"]["job_id"] or op["caller_session_id"] != actor.session_id
                or op["operation_key"] != expected_key or op["call_ref_json"] is not None
                or op["tool_call_id"] is not None or op["receipt_ref_json"] is not None):
            raise JobError("wait_effect_operation_mismatch")
        original = waits._object(op["payload_json"])
        request = (_input(saved, payload["binding"]) if choice == "continue" else
            {"target": saved["binding"]["owner"], "reason": data["request"]["reason"], "handoff": payload["binding"]})
        if (op["request_hash"] != _hash(request) or _hash(original.get("request")) != _hash(request)
                or set(original) != {"version", "request"} or type(original["version"]) is not int or original["version"] != 1):
            raise JobError("wait_effect_operation_mismatch")
        if choice == "continue":
            inbox = await tx.fetchone(f"SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
                (request["session_id"], "nudge:" + op["event_id"]))
            if (inbox is None or inbox["kind"] != "relay" or inbox["mode"] != "steer"
                    or inbox["content"] != request["message"] or inbox["status"] not in ("pending", "claimed", "delivered")
                    or op["delivery_status"] not in ("pending_delivery", "handled")):
                raise JobError("wait_effect_input_mismatch")
    return row


def _reply(effect):
    payload = waits._object(effect["payload_json"])
    return {"version": 1, "effect_id": effect["event_id"], "event_id": payload["binding"]["event_id"],
        "wait_job_id": payload["wait_job_id"], "decision": payload["decision"],
        "status": "review_scheduled" if payload["decision"] == "defer" else "input_enqueued" if payload["decision"] == "continue" else "stop_requested",
        "operation_id": payload["operation_id"], "next_review_at_ms": payload["review_at_ms"],
        "effect_status": effect["delivery_status"], "execution_authority": False, "physical_stop_confirmed": False}


async def apply(manager, actor, event_id):
    """Apply one original decision atomically, without model dispatch or new work."""
    async with manager._write() as tx:
        job, _, saved, decision, data = await _decision(manager, tx, actor, event_id)
        old = await _effect(manager, tx, actor, job, saved, decision, data)
        if old is not None and not (data["request"]["decision"] == "defer" and old["delivery_status"] == "delivered"):
            if old["delivery_status"] == "handled" and data["request"]["decision"] == "continue":
                await _resolved(manager, tx, job, old)
            elif old["delivery_status"] == "handled" and data["request"]["decision"] == "cancel":
                from .wait_terminal import _terminated, admission
                terminal, creation, original = await admission(manager, tx, actor, job["job_id"])
                await _terminated(manager, tx, actor, terminal, creation, original)
            return _reply(old)
        owner = await manager._authorize(tx, actor, saved["binding"]["owner"]["job_id"])
        manager._check_proof(owner, JobLease(**saved["binding"]["owner"]))
        choice = data["request"]["decision"]
        if choice != "cancel":
            await waits._prepare_due(manager, tx, actor, job["job_id"])
            if manager.now() >= saved["request"]["policy"]["deadline_at_ms"]:
                raise JobError("wait_deadline_exceeded")
        elif (job["generation"] != 0 or job["resource_state"] != "not_started" or job["outcome"] is not None):
            raise JobError("wait_state_changed")
        binding = _binding(manager, decision, data)
        if old is not None:
            from .wait_reviews import schedule
            await schedule(manager, tx, actor, job, saved, decision, old)
            return _reply({**old, "delivery_status": "handled"})
        op = None
        if choice != "defer":
            action = "nudge" if choice == "continue" else "stop"
            request = (_input(saved, binding) if choice == "continue" else
                {"target": saved["binding"]["owner"], "reason": data["request"]["reason"], "handoff": binding})
            if choice == "cancel" and owner["cancel_requested_at"] is None:
                status = owner["status"] if owner["supervision_status"] == "closed" else (
                    "cancelled" if owner["resource_state"] == "not_started" else "cancelling")
                await tx.execute(f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,intake='closed',"
                    "version=version+1,updated_at=? WHERE job_id=?", (status, manager.now(), request["reason"], manager.now(), owner["job_id"]))
            else:
                await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), owner["job_id"]))
            owner = await manager._row(tx, owner["job_id"])
            op = await manager._insert_operation(tx, job_id=owner["job_id"], version=owner["version"], actor=actor,
                action=action, key=manager._origin_key(actor, action, None, None, "wait-effect:" + decision["event_id"]),
                fingerprint=_hash(request), payload={"version": 1, "request": request})
            if choice == "continue":
                await tx.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (op.operation_id,))
                columns = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
                inserted = await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, columns),
                    (request["session_id"], "nudge:" + op.operation_id, "relay", "steer", "pending", request["message"], manager.now()))
                if not inserted:
                    raise JobError("wait_effect_input_conflict")
        payload = {"version": 1, "binding": binding, "wait_job_id": job["job_id"], "owner": saved["binding"]["owner"],
            "decision": choice, "operation_id": op.operation_id if op else None, "review_at_ms": data["request"]["next_review_at_ms"]}
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,caller_session_id,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (binding["effect_id"], job["job_id"], "wait_effect", job["version"]+1, actor.session_id, _hash(payload), _json(payload), "delivered", manager.now()))
        await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), job["job_id"]))
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='delivered' WHERE event_id=?", (decision["event_id"],))
        if choice == "defer":
            from .wait_reviews import schedule
            job = await manager._row(tx, job["job_id"])
            fresh = await _effect(manager, tx, actor, job, saved, {**decision, "delivery_status": "delivered"}, data)
            await schedule(manager, tx, actor, job, saved, {**decision, "delivery_status": "delivered"}, fresh)
            decision = {**decision, "delivery_status": "handled"}
        job = await manager._row(tx, job["job_id"])
        effect = await _effect(manager, tx, actor, job, saved, {**decision, "delivery_status": "handled" if choice == "defer" else "delivered"}, data)
        return _reply(effect)


async def validate_input(manager, tx, actor, event):
    """System handoffs must retain the original accepted decision and effect."""
    request = waits._object(event["payload_json"]).get("request", {})
    if event["call_ref_json"] is not None and "handoff" not in request:
        return
    if "handoff" not in request:
        from .workflow_restore_execution import validate_system_input
        await validate_system_input(manager, tx, actor, event)
        return
    binding = request.get("handoff")
    if not isinstance(binding, dict) or not isinstance(binding.get("event_id"), str):
        raise JobError("wait_effect_input_evidence_missing")
    # The system operation's caller is the original main; this is a read of
    # immutable provenance, not a lease delegated to the child.
    main = JobActor(actor.tenant_id, actor.principal_id, event["caller_session_id"])
    job, _, saved, decision, data = await _decision(manager, tx, main, binding["event_id"])
    effect = await _effect(manager, tx, main, job, saved, decision, data)
    if (effect is None or data["request"]["decision"] != "continue"
            or waits._object(effect["payload_json"])["operation_id"] != event["event_id"]
            or _hash(binding) != _hash(_binding(manager, decision, data))):
        raise JobError("wait_effect_input_mismatch")


async def settle_owner(manager, actor, owner_lease):
    """Resolve only waits whose original continuation was actually consumed."""
    async with manager._write() as tx:
        owner = await manager._authorize(tx, actor, owner_lease.job_id)
        manager._check_proof(owner, owner_lease)
        await settle_owner_tx(manager, tx, actor, owner)


async def settle_owner_tx(manager, tx, actor, owner):
    """Same original-owner result transaction; never dispatches or signals work."""
    rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE control_owner_job_id=? AND control_owner_generation=? AND kind='timer'",
        (owner["job_id"], owner["generation"]))
    for row in rows:
        creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create' AND event_type='operation'", (row["job_id"],))
        initial = waits._object(creation["payload_json"]) if creation else {}
        if creation is not None and "wait_binding" not in initial and "managed_wait" not in initial.get("request", {}):
            continue  # Legacy timers retain their own adapter and evidence.
        timer, creation, saved = await waits._read(manager, tx, actor, row["job_id"])
        main = JobActor(actor.tenant_id, actor.principal_id, saved["binding"]["main_session_id"])
        from .wait_reviews import history
        due_id = (await history(manager, tx, actor, timer, creation, saved))["event_id"]
        marker = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (notices.decision_id(manager, due_id),))
        if marker is None:
            continue
        job, _, saved, decision, data = await _decision(manager, tx, main, due_id)
        effect = await _effect(manager, tx, main, job, saved, decision, data)
        if effect is None or data["request"]["decision"] != "continue":
            continue
        if effect["delivery_status"] == "handled":
            await _resolved(manager, tx, job, effect)
            continue
        op_id = waits._object(effect["payload_json"])["operation_id"]
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (op_id,))
        from .processing import require_event_handled
        await validate_input(manager, tx, actor, op)
        await require_event_handled(manager, tx, op, session_id=saved["binding"]["session_id"], owner_job_id=owner["job_id"])
        from .finalization import _processing_references
        proofs = await _processing_references(manager, tx, op)
        if (job["generation"] != 0 or job["resource_state"] != "not_started" or job["outcome"] is not None
                or job["intake"] != "open" or job["cancel_requested_at"] is not None):
            raise JobError("wait_state_changed")
        payload = {"version": 1, "effect_id": effect["event_id"], "effect_hash": effect["request_hash"], "input_id": op_id,
            "input_hash": _hash(waits._object(op["payload_json"])), "proofs": proofs}
        key = "wait_resolved_" + _hash([manager.store.store_id, effect["event_id"]])[:48]
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (key, job["job_id"], "wait_resolved", job["version"]+1, _hash(payload), _json(payload), "not_applicable", manager.now()))
        result = {"status": "succeeded", "wait_resolved_event_id": key}
        canonical = {"outcome": "succeeded", "result": result, "resource_state": "stopped", "unstarted": True,
            "logical_wait": True, "resolution_hash": _hash(payload)}
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("result_" + job["job_id"], job["job_id"], "result", job["version"]+1, _hash(canonical), _json(canonical), "handled", manager.now()))
        await tx.execute(f"UPDATE {manager.jobs} SET status='settling',outcome='succeeded',resource_state='stopped',intake='closed',"
            "result_status='handled',result_json=?,version=version+1,updated_at=? WHERE job_id=?", (_json(result), manager.now(), job["job_id"]))
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id IN (?,?)", (effect["event_id"], decision["event_id"]))


async def _resolved(manager, tx, job, effect):
    key = "wait_resolved_" + _hash([manager.store.store_id, effect["event_id"]])[:48]
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if row is None:
        raise JobError("wait_resolution_evidence_missing")
    data = waits._object(row["payload_json"])
    payload = waits._object(effect["payload_json"])
    op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (payload["operation_id"],))
    from .finalization import _processing_references, _reference
    proofs = await _processing_references(manager, tx, op) if op is not None else None
    expected = {"version": 1, "effect_id": effect["event_id"], "effect_hash": effect["request_hash"],
        "input_id": op["event_id"] if op else None, "input_hash": _hash(waits._object(op["payload_json"])) if op else None, "proofs": proofs}
    result = {"status": "succeeded", "wait_resolved_event_id": key}
    canonical = {"outcome": "succeeded", "result": result, "resource_state": "stopped", "unstarted": True,
        "logical_wait": True, "resolution_hash": _hash(expected)}
    result_event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", ("result_" + job["job_id"],))
    if (row["event_type"] != "wait_resolved" or row["job_id"] != job["job_id"] or row["request_hash"] != _hash(data)
            or _hash(data) != _hash(expected) or proofs is None
            or row["delivery_status"] != "not_applicable" or data.get("effect_id") != effect["event_id"]
            or data.get("effect_hash") != effect["request_hash"] or job["generation"] != 0
            or job["outcome"] != "succeeded" or job["resource_state"] != "stopped" or job["intake"] != "closed"
            or _hash(waits._object(job["result_json"])) != _hash(result)
            or result_event is None or result_event["event_type"] != "result" or result_event["job_id"] != job["job_id"]
            or result_event["delivery_status"] != "handled" or result_event["request_hash"] != _hash(canonical)
            or _hash(waits._object(result_event["payload_json"])) != _hash(canonical)):
        raise JobError("wait_resolution_evidence_corrupt")
    from .processing import require_event_handled
    request = waits._object(op["payload_json"])["request"]
    await require_event_handled(manager, tx, op, session_id=request["session_id"], owner_job_id=payload["owner"]["job_id"])
    for proof in data.get("proofs", []):
        if _hash(await _reference(manager, tx, proof["event_id"])) != _hash(proof):
            raise JobError("wait_resolution_evidence_corrupt")
    return row
