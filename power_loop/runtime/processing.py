"""Durable result observations and decisions for the experimental task runtime.

A request snapshot is prepared evidence, never an acknowledgement. Only a
persisted assistant response and a fenced, clean send boundary can consume it.
No platform imports, model text heuristics, or side-effect replay live here.
"""
from __future__ import annotations

import json

from .message_ref import integer_id
from .node_inputs import input_target
from .result_revisions import delivery_id, result_id


def _fingerprint(caller, snapshot_id, response_ref, decision, reason):
    from .jobs import _hash
    return _hash([snapshot_id, caller.job_id, caller.executor_id, caller.boot_id,
                  caller.generation, response_ref.to_dict(), decision, reason])


async def recall(manager, actor, caller, *, snapshot_id, response_ref, decision="consumed", reason):
    """Read a committed decision after a lost ACK; never executes or acks work."""
    from .jobs import JobError, _hash
    send = await manager._authorize(manager.db, actor, caller.job_id)
    manager._check_proof(send, caller)
    if send["kind"] != "send" or send["session_id"] != actor.session_id:
        raise JobError("processing_caller_mismatch")
    request = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_request'",
                                       (snapshot_id,))
    if request is None or request["job_id"] != caller.job_id:
        raise JobError("processing_snapshot_mismatch")
    if request["delivery_status"] != "handled":
        return None
    fingerprint = _fingerprint(caller, snapshot_id, response_ref, decision, reason)
    snapshot = json.loads(request["payload_json"])
    ids = ["decision_" + _hash([e["event_id"], e["revision"], e["result_hash"]])[:48] for e in snapshot["events"]]
    for event_id in ids:
        old = await manager.db.fetchone(f"SELECT request_hash FROM {manager.events} WHERE event_id=? AND event_type='processing_decision'", (event_id,))
        if old is None or old["request_hash"] != fingerprint:
            raise JobError("processing_decision_conflict")
    return ids


async def prepare(manager, actor, caller, *, round_index, send_index, messages):
    from .jobs import JobError, _hash, _json

    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise JobError("invalid_processing_round")
    # Freeze before the first await. The caller supplies only actual request
    # rows, not every row present somewhere in the durable transcript.
    request_rows = json.loads(_json(messages))
    visible = {integer_id(m["seq"], "seq"): m for m in request_rows}
    async with manager._write() as tx:
        send = await manager._fence(tx, actor, caller)
        if send["kind"] != "send" or send["session_id"] != actor.session_id:
            raise JobError("processing_caller_mismatch")
        create = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
        if (create is None or isinstance(send_index, bool) or not isinstance(send_index, int) or
                json.loads(create["payload_json"]).get("request", {}).get("send_index") != send_index):
            raise JobError("processing_send_mismatch")
        from .wait_notifications import observe_request
        await observe_request(manager, tx, actor, caller, source=send, round_index=round_index,
                              send_index=send_index, messages=request_rows)
        results = await tx.fetchall(f"SELECT e.* FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
            "WHERE e.event_type='result' AND j.created_by_session_id=? AND j.tenant_id=? AND j.principal_id=? "
            "AND j.kind IN ('subagent','workflow') AND j.result_status<>'handled' AND e.delivery_status<>'handled' ORDER BY e.event_id",
            (actor.session_id, actor.tenant_id, actor.principal_id))
        from .managed_delivery import candidates
        results.extend(await candidates(manager, tx, actor))
        from .abort_outcomes import candidates as aborted_candidates
        results.extend(await aborted_candidates(manager, tx, actor))
        from .controller_recovery import candidates as controller_candidates
        results.extend(await controller_candidates(manager, tx, actor))
        results = list({event['event_id']: event for event in results}.values())
        inputs = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' "
            "AND action='nudge' AND delivery_status<>'handled' ORDER BY event_id", (send["control_owner_job_id"],))
        observed = []
        for event in [*results, *inputs]:
            if event["event_type"] == "operation":
                from .wait_effects import validate_input
                await validate_input(manager, tx, actor, event)
            job = (await input_target(manager, tx, actor, send, event) if event["event_type"] == "operation"
                   else await manager._authorize(tx, actor, event["job_id"]))
            if event["event_type"] == "operation":
                request = json.loads(event["payload_json"])["request"]
                if (job["kind"] != "workflow_node" and job["session_id"] != actor.session_id) or request["session_id"] != actor.session_id:
                    continue
                item_id = "nudge:" + event["event_id"]
            else:
                if event["event_id"] != result_id(job["job_id"], job["generation"]):
                    continue
                item_id = await delivery_id(manager, tx, job) if job["kind"] == "subagent" else event["event_id"]
            if event['event_type'] == 'result' and job['kind'] in ('subagent', 'workflow_node', 'workflow'):
                from .controller_recovery import observation as controller_observation
                from .controller_recovery import read as controller_stop
                if await controller_stop(manager, tx, job) is not None:
                    actual = await controller_observation(manager, tx, actor, job, visible)
                    if actual is not None:
                        observed.append({'event_id': event['event_id'], 'job_id': event['job_id'],
                            'revision': int(event['job_version']), 'result_hash': _hash(json.loads(event['payload_json'])), **actual})
                    continue
            if event['event_type'] == 'result' and job['kind'] == 'send':
                from .abort_outcomes import observation
                actual = await observation(manager, tx, actor, job, event, visible)
                if actual is not None:
                    observed.append({'event_id': event['event_id'], 'job_id': event['job_id'],
                        'revision': int(event['job_version']), 'result_hash': _hash(json.loads(event['payload_json'])), **actual})
                continue
            if event['event_type'] == 'result' and job['kind'] == 'background_tool':
                from .managed_delivery import observation

                actual = await observation(manager, tx, actor, job, event, visible)
                if actual is not None:
                    observed.append({"event_id": event["event_id"], "job_id": event["job_id"],
                        "revision": int(event["job_version"]), "result_hash": _hash(json.loads(event["payload_json"])),
                        **actual})
                continue
            inbox = await tx.fetchone(f"SELECT status,delivered_seq FROM {manager.store.t.follow_up_queue} "
                                      "WHERE session_id=? AND item_id=?", (actor.session_id, item_id))
            if inbox is None or inbox["status"] != "delivered" or inbox["delivered_seq"] not in visible:
                continue
            seq = int(inbox["delivered_seq"])
            row = await tx.fetchone(f"SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND seq=?",
                                   (actor.session_id, seq))
            # Hooks may remove, copy, or rewrite rows. Equality with the real
            # inbox row is required; merely appearing later in history is not.
            message = visible[seq]
            if row is None or row["role"] != message.get("role") or row["content"] != message.get("content"):
                continue
            observed.append({"event_id": event["event_id"], "job_id": event["job_id"],
                "revision": int(event["job_version"]), "result_hash": _hash(json.loads(event["payload_json"])),
                "delivered_seq": seq, **({"event_type": "operation"} if event["event_type"] == "operation" else {})})
        if not observed:
            return None
        payload = {"handler_job_id": caller.job_id, "generation": caller.generation,
                   "session_id": actor.session_id, "send_index": send_index, "round_index": round_index, "events": observed}
        event_id = "request_" + _hash([caller.job_id, caller.generation, round_index])[:48]
        old = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (event_id,))
        if old:
            if old["payload_json"] != _json(payload):
                raise JobError("processing_snapshot_conflict")
            return event_id
        await tx.execute(f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,caller_session_id,payload_json,delivery_status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)", (event_id, caller.job_id, "processing_request", send["version"],
                actor.session_id, _json(payload), "prepared", manager.now()))
        return event_id


async def commit(manager, actor, caller, *, snapshot_id, response_ref, decision="consumed", reason):
    from .jobs import JobError, _hash, _json

    if decision not in ("consumed", "rejected", "escalated"):
        raise JobError("invalid_processing_decision")
    if not isinstance(reason, str) or not reason or len(reason) > 1000:
        raise JobError("invalid_processing_reason")
    response = response_ref.to_dict()
    fingerprint = _fingerprint(caller, snapshot_id, response_ref, decision, reason)
    async with manager._write() as tx:
        send = await manager._authorize(tx, actor, caller.job_id)
        manager._check_proof(send, caller)
        if send["kind"] != "send" or send["session_id"] != actor.session_id:
            raise JobError("processing_caller_mismatch")
        request = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_request'",
                                   (snapshot_id,))
        if request is None or request["job_id"] != caller.job_id:
            raise JobError("processing_snapshot_mismatch")
        snapshot = json.loads(request["payload_json"])
        if snapshot["generation"] != caller.generation or snapshot["session_id"] != actor.session_id:
            raise JobError("processing_snapshot_mismatch")
        decision_ids = ["decision_" + _hash([e["event_id"], e["revision"], e["result_hash"]])[:48]
                        for e in snapshot["events"]]
        # A committed replay may be recalled after the handler closes, expires,
        # or is cancelled. It cannot change its response, decision, or reason.
        if request["delivery_status"] == "handled":
            for event_id in decision_ids:
                old = await tx.fetchone(f"SELECT request_hash FROM {manager.events} WHERE event_id=?", (event_id,))
                if old is None or old["request_hash"] != fingerprint:
                    raise JobError("processing_decision_conflict")
            return decision_ids
        await manager._fence(tx, actor, caller)
        state = await tx.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?",
                                  (actor.session_id,))
        if state is not None and state["pending_json"]:
            raise JobError("processing_tools_pending")
        if response_ref.store_id != manager.store.store_id or response_ref.session_id != actor.session_id or response_ref.id is None:
            raise JobError("processing_response_mismatch")
        message = await tx.fetchone(f"SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND id=?",
                                   (actor.session_id, response_ref.id))
        if (message is None or message["role"] != "assistant" or response_ref.role != "assistant" or
                int(message["seq"]) != response_ref.seq or message["round_index"] != snapshot["round_index"] or
                message["round_index"] != response_ref.round_index or message["send_index"] != response_ref.send_index):
            raise JobError("processing_response_mismatch")
        send_index = snapshot.get("send_index")
        if send_index is None or message["send_index"] != send_index:
            raise JobError("processing_response_mismatch")
        assistants = await tx.fetchall(f"SELECT seq,tool_calls_json FROM {manager.store.t.messages} "
            "WHERE session_id=? AND send_index=? AND role='assistant' ORDER BY seq", (actor.session_id, send_index))
        if not assistants or assistants[-1]["seq"] != response_ref.seq:
            raise JobError("processing_response_mismatch")
        for index, assistant in enumerate(assistants):
            for call in json.loads(assistant["tool_calls_json"] or "[]"):
                receipts = await tx.fetchall(f"SELECT seq FROM {manager.store.t.messages} WHERE session_id=? "
                    "AND send_index=? AND role='tool' AND tool_call_id=? AND seq>?", (actor.session_id, send_index,
                        call.get("id"), assistant["seq"]))
                ceiling = assistants[index + 1]["seq"] if index + 1 < len(assistants) else None
                if not any(ceiling is None or row["seq"] < ceiling for row in receipts):
                    raise JobError("processing_tools_pending")
        # Persist links to work already accepted by this handler. A raw tool
        # response alone cannot hide a missing receipt on an async operation.
        operations = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='operation' "
            "AND caller_session_id=? AND call_ref_json IS NOT NULL", (actor.session_id,))
        operation_ids, follow_up_jobs = [], []
        for operation in operations:
            origin = json.loads(operation["call_ref_json"])
            if origin.get("send_index") != str(send_index):
                continue
            if operation["receipt_ref_json"] is None:
                raise JobError("processing_tools_pending")
            receipt = json.loads(operation["receipt_ref_json"])
            actual = await tx.fetchone(f"SELECT seq,tool_call_id,send_index,role FROM {manager.store.t.messages} WHERE session_id=? AND id=?",
                (actor.session_id, integer_id(receipt.get("id"))))
            if (receipt.get("store_id") != manager.store.store_id or receipt.get("session_id") != actor.session_id or
                    receipt.get("role") != "tool" or receipt.get("tool_call_id") != operation["tool_call_id"] or
                    receipt.get("send_index") != str(send_index) or
                    actual is None or actual["role"] != "tool" or actual["send_index"] != send_index or
                    actual["seq"] != integer_id(receipt.get("seq"), "seq") or
                    actual["tool_call_id"] != operation["tool_call_id"]):
                raise JobError("processing_tools_pending")
            operation_ids.append(operation["event_id"])
            if operation["action"] == "create":
                follow_up_jobs.append(operation["job_id"])
        for event, decision_id in zip(snapshot["events"], decision_ids, strict=True):
            event_type = event.get("event_type", "result")
            current = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type=?", (event["event_id"], event_type))
            is_input = event_type == "operation"
            target = (await input_target(manager, tx, actor, send, current) if is_input and current is not None
                      else await manager._authorize(tx, actor, event["job_id"]))
            if is_input:
                request = json.loads(current["payload_json"])["request"] if current else {}
                if (current is None or current["action"] != "nudge" or target["job_id"] != send["control_owner_job_id"] or
                        (target["kind"] != "workflow_node" and target["session_id"] != actor.session_id) or request.get("session_id") != actor.session_id or
                        request.get("target") != {key: target[key] for key in ("job_id", "executor_id", "boot_id", "generation")}):
                    raise JobError("processing_input_mismatch")
            controller_review = not is_input and 'controller_result_delivery' in event
            if ((not is_input and ((not controller_review and target["kind"] not in ("background_tool","send") and target["created_by_session_id"] != actor.session_id) or
                    current is not None and current["event_id"] != result_id(target["job_id"], target["generation"]))) or
                    current is None or current["job_id"] != target["job_id"] or
                    int(current["job_version"]) != event["revision"] or _hash(json.loads(current["payload_json"])) != event["result_hash"]):
                raise JobError("processing_revision_changed")
            if controller_review:
                from .controller_recovery import verify_observation as verify_controller
                from .managed_delivery import _main
                main = await _main(manager, tx, target)
                if (send['execution_role'] != 'main' or send['control_owner_job_id'] is not None
                        or send['session_id'] != main['session_id']):
                    raise JobError('controller_recovery_requires_original_main')
                await verify_controller(manager, tx, actor, target, current, event)
            if not is_input and target['kind'] == 'send':
                from .abort_outcomes import verify_observation
                from .original_owner_review import require_handler
                await require_handler(manager,tx,target,send)
                await verify_observation(manager, tx, actor, target, current, event)
            if not is_input and target['kind'] == 'background_tool':
                from .managed_delivery import verify_observation

                if send['execution_role'] != 'main':
                    raise JobError('managed_delivery_requires_original_main')
                await verify_observation(manager, tx, actor, target, current, event)
            if response_ref.seq <= event["delivered_seq"]:
                raise JobError("processing_response_mismatch")
            old = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (decision_id,))
            if old is not None or (not is_input and target["result_status"] == "handled"):
                raise JobError("processing_decision_conflict")
            payload = {"event": event, "snapshot_id": snapshot_id, "handler_job_id": caller.job_id,
                       "generation": caller.generation, "response_ref": response, "decision": decision, "reason": reason,
                       "operation_ids": sorted(operation_ids), "follow_up_job_ids": sorted(follow_up_jobs)}
            await tx.execute(f"INSERT INTO {manager.events} "
                "(event_id,job_id,event_type,job_version,caller_session_id,request_hash,payload_json,delivery_status,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (decision_id, target["job_id"], "processing_decision", target["version"]+1,
                    actor.session_id, fingerprint, _json(payload), "committed", manager.now()))
            negative_review = not is_input and target['kind'] == 'background_tool' and target['generation'] == 0
            if not negative_review:
                await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (event["event_id"],))
            if not is_input and not negative_review:
                await tx.execute(f"UPDATE {manager.jobs} SET result_status='handled',version=version+1,updated_at=? WHERE job_id=?",
                                 (manager.now(), target["job_id"]))
            else:
                await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?",
                                 (manager.now(), target["job_id"]))
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (snapshot_id,))
        return decision_ids


async def require_handled(manager, tx, child):
    from .jobs import JobError

    if child['kind'] in ('subagent', 'workflow_node', 'workflow'):
        from .controller_recovery import handled_references, read
        if await read(manager, tx, child) is not None:
            await handled_references(manager, tx, child)
            return

    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(child["job_id"], child["generation"]),))
    if result is None or result["delivery_status"] != "handled" or child["result_status"] != "handled":
        raise JobError("required_result_unhandled")
    await require_event_handled(manager, tx, result, session_id=child["created_by_session_id"], owner_job_id=child["control_owner_job_id"])


async def require_inputs_handled(manager, tx, job):
    from .jobs import JobError, _hash
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (job["job_id"],))
    for event in events:
        try:
            if job["kind"] == "workflow" and json.loads(event["payload_json"]).get("mode") == "workflow_continue":
                from .workflow_restore_execution import require_settled
                await require_settled(manager, tx, job, event)
                continue
            sid = job["session_id"]
            if job["kind"] == "workflow_node":
                from .node_inputs import descriptor
                request = json.loads(event["payload_json"])["request"]
                sid = request["session_id"]
                generation = request["target"]["generation"]
                if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation <= job["generation"]:
                    raise JobError("processing_input_mismatch")
                if _hash(request.get("node")) != _hash(await descriptor(manager, tx, {**job, "generation": generation}, sid,
                        historical=generation < job["generation"] or job["resource_state"] == "stopped")):
                    raise JobError("processing_input_mismatch")
            await require_event_handled(manager, tx, event, session_id=sid, owner_job_id=job["job_id"])
        except JobError as exc:
            raise JobError("required_input_unhandled") from exc


async def require_event_handled(manager, tx, result, *, session_id, owner_job_id, allow_suppressed=False):
    from .jobs import JobError, _hash
    if type(allow_suppressed) is not bool or result["delivery_status"] not in (("handled", "suppressed") if allow_suppressed else ("handled",)):
        raise JobError("required_result_unhandled")
    if result.get("action") == "nudge":
        from .input_dispositions import read_tx
        disposed = await read_tx(manager, tx, result)
        if disposed is not None:
            data = json.loads(disposed["payload_json"])
            if data["session_id"] != session_id or data["owner"]["job_id"] != owner_job_id:
                raise JobError("required_result_unhandled")
            return
        from .jobs import JobActor
        from .wait_effects import validate_input
        row = await manager._row(tx, owner_job_id)
        await validate_input(manager, tx, JobActor(row["tenant_id"], row["principal_id"], session_id), result)
    event = {"event_id": result["event_id"], "job_id": result["job_id"], "revision": int(result["job_version"]),
             "result_hash": _hash(json.loads(result["payload_json"]))}
    event_id = "decision_" + _hash([event["event_id"], event["revision"], event["result_hash"]])[:48]
    decision = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_decision'", (event_id,))
    if (decision is None or decision["delivery_status"] != "committed" or decision["job_id"] != result["job_id"] or
            decision["caller_session_id"] != session_id):
        raise JobError("required_result_unhandled")
    payload = json.loads(decision["payload_json"])
    handler = await manager._row(tx, payload["handler_job_id"])
    if (handler is None or handler["kind"] != "send" or handler["session_id"] != session_id or
            handler["control_owner_job_id"] != owner_job_id or handler["generation"] != payload["generation"]):
        raise JobError("required_result_unhandled")
    if result.get("action") == "nudge":
        target = json.loads(result["payload_json"])["request"]["target"]
        proof = {"job_id": handler["control_owner_job_id"], "generation": handler["control_owner_generation"],
            "executor_id": handler["control_owner_executor_id"], "boot_id": handler["control_owner_boot_id"]}
        if _hash(proof) != _hash(target):
            raise JobError("required_result_unhandled")
    closed = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='model_closed'",
                              ("model_closed_" + payload["handler_job_id"],))
    evidence = json.loads(closed["payload_json"]).get("evidence", {}) if closed else {}
    if (closed is None or json.loads(closed["payload_json"]).get("generation") != payload["generation"] or
            evidence.get("exit") != "completed" or evidence.get("new_input_after_pass") or
            evidence.get("last_assistant_ref") != payload["response_ref"] or
            event_id not in evidence.get("processing_decisions", [])):
        raise JobError("required_result_unhandled")
