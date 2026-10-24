"""Negative terminal responsibility, distinct from applying the main's intent.

Only a proven stopped original owner and a terminated logical wait can retire
an unexecuted decision. No execution lease, provider call or money mutation.
"""
from __future__ import annotations

import json

from . import timer_waits as waits
from .jobs import JobActor, JobError, _hash, _json
from .model_exit import content_reference


def identity(manager, decision_id):
    return "wait_disposed_" + _hash([manager.store.store_id, decision_id])[:48]


async def _intent(manager, tx, actor, job, saved, state):
    from . import wait_effects as effects
    from . import wait_notifications as notices

    if state["event"] is None:
        return None
    main = JobActor(actor.tenant_id, actor.principal_id, saved["binding"]["main_session_id"])
    key = notices.decision_id(manager, state["event_id"])
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if row is None:
        return None
    bundle = await notices.notification(manager, tx, main, state["event_id"], _due=(job, state["event"], saved))
    _, _, _, decision, data = await effects._decision(manager, tx, main, state["event_id"], _notification=bundle)
    operation = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data["operation_id"],))
    raw_effect = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (effects.effect_id(manager, decision["event_id"]),))
    effect = None
    if raw_effect is not None:
        effect = await effects._effect(manager, tx, main, job, saved, decision, data, _review=False)
    return main, decision, data, operation, effect


async def _expected(manager, tx, actor, job, creation, saved, state, intent):
    from .wait_terminal import _terminated

    main, decision, data, operation, effect = intent
    if effect is not None and (data["request"]["decision"] != "defer"
            or waits._object(effect["payload_json"])["operation_id"] is not None):
        raise JobError("wait_disposition_effect_already_applied")
    ended = await _terminated(manager, tx, actor, job, creation, saved, _state=state)
    waits._number(ended["created_at"], "termination_time")
    observed = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data["observation"]["observation_id"],))
    call = await manager._validate_ref(tx, main, waits._ref(waits._object(operation["call_ref_json"])), role="assistant")
    receipt = await manager._validate_ref(tx, main, waits._ref(waits._object(operation["receipt_ref_json"])), role="tool")
    try:
        calls = [c for c in json.loads(call["tool_calls_json"] or "[]") if c.get("id") == operation["tool_call_id"]]
        [original] = calls
        arguments = waits._object(original["function"]["arguments"])
        answer = waits._object(receipt["content"])
        response = waits._object(operation["payload_json"])["response"]
        expected_answer = {**response, "operation_id": operation["event_id"], "replayed": answer.get("replayed")}
        if (original["function"]["name"] != "wait_handoff" or arguments.get("action") != "decide"
                or set(arguments) - {"action", "event_id", "decision", "reason", "next_review_at_ms"}
                or arguments.get("event_id") != data["event_id"] or arguments.get("decision") != data["request"]["decision"]
                or arguments.get("reason", "Reviewed original deferred request.") != data["request"]["reason"]
                or arguments.get("next_review_at_ms") != data["request"]["next_review_at_ms"]
                or type(answer.get("replayed")) is not bool or _hash(answer) != _hash(expected_answer)):
            raise ValueError()
    except (TypeError, KeyError, ValueError) as exc:
        raise JobError("wait_disposition_raw_evidence_corrupt") from exc
    columns = ("id", "session_id", "seq", "role", "name", "content", "tool_calls_json", "tool_call_id", "round_index", "send_index")
    # _decision validated the actual observation and original tool receipt.
    return {"version": 1, "disposition": "owner_terminated_without_effect", "owner": saved["binding"]["owner"],
        "wait_job_id": job["job_id"], "main_session_id": main.session_id,
        "creation": content_reference(creation), "due": content_reference(state["event"]),
        "decision": content_reference(decision), "decision_kind": data["request"]["decision"],
        "operation": content_reference(operation), "call_ref": waits._object(operation["call_ref_json"]),
        "receipt_ref": waits._object(operation["receipt_ref_json"]), "observation": content_reference(observed),
        "raw_call_hash": _hash({k: call[k] for k in columns}), "raw_receipt_hash": _hash({k: receipt[k] for k in columns}),
        "effect": content_reference(effect) if effect else None,
        "prior_delivery_status": "delivered" if effect else "pending_delivery", "termination": content_reference(ended),
        "terminated_at_ms": ended["created_at"]}


async def read_tx(manager, tx, actor, job, creation, saved, state, *, required=False):
    """Validate historical negative responsibility with an already read chain."""
    from . import wait_notifications as notices

    key = identity(manager, notices.decision_id(manager, state["event_id"]))
    dispositions = await tx.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='wait_disposition' LIMIT 2", (job["job_id"],))
    if any(item["event_id"] != key for item in dispositions) or len(dispositions) > 1:
        raise JobError("wait_disposition_branch_corrupt")
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if row is None and not required:
        return None
    intent = await _intent(manager, tx, actor, job, saved, state)
    if intent is None:
        if row is not None:
            raise JobError("wait_disposition_evidence_corrupt")
        return None
    main, decision, data, _, effect = intent
    if row is None:
        if effect is not None and data["request"]["decision"] == "cancel" and effect["delivery_status"] == "handled":
            return None  # The original cancel effect has its terminal proof.
        if effect is not None and data["request"]["decision"] == "continue":
            raise JobError("wait_input_unsettled")
        raise JobError("wait_decision_unsettled")
    expected = await _expected(manager, tx, actor, job, creation, saved, state, intent)
    if (row["event_type"] != "wait_disposition" or row["job_id"] != job["job_id"]
            or row["caller_session_id"] != main.session_id or row["delivery_status"] != "not_applicable"
            or row["request_hash"] != _hash(expected) or _hash(waits._object(row["payload_json"])) != _hash(expected)
            or decision["delivery_status"] != "handled" or effect is not None and effect["delivery_status"] != "handled"
            or type(row["job_version"]) is not int or not 0 < row["job_version"] <= job["version"]
            or type(row["created_at"]) is not int or row["created_at"] < expected["terminated_at_ms"]):
        raise JobError("wait_disposition_evidence_corrupt")
    return row


async def reconcile_tx(manager, tx, actor, job, creation, saved):
    """Called after the original logical termination, in that same transaction."""
    from .wait_reviews import history

    state = await history(manager, tx, actor, job, creation, saved)
    old = await read_tx(manager, tx, actor, job, creation, saved, state)
    if old is not None:
        return old
    intent = await _intent(manager, tx, actor, job, saved, state)
    if intent is None:
        return None
    main, decision, data, _, effect = intent
    if effect is not None and data["request"]["decision"] != "defer":
        return None  # Applied input/cancel has its own independent protocol.
    if decision["delivery_status"] != ("delivered" if effect else "pending_delivery"):
        raise JobError("wait_disposition_evidence_missing")
    if job["supervision_status"] != "open":
        raise JobError("wait_disposition_after_close")
    payload = await _expected(manager, tx, actor, job, creation, saved, state, intent)
    key = identity(manager, decision["event_id"])
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,caller_session_id,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (key, job["job_id"], "wait_disposition", job["version"]+1, main.session_id, _hash(payload), _json(payload), "not_applicable", manager.now()))
    await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (decision["event_id"],))
    if effect:
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (effect["event_id"],))
    await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), job["job_id"]))
    current = await manager._row(tx, job["job_id"])
    return await read_tx(manager, tx, actor, current, creation, saved, state, required=True)


def reply(row):
    payload = waits._object(row["payload_json"])
    return {"version": 2, "status": "owner_terminated_without_effect", "disposition_id": row["event_id"],
        "event_id": payload["due"]["event_id"], "wait_job_id": row["job_id"], "decision": payload["decision_kind"],
        "original_owner": {"job_id": payload["owner"]["job_id"], "generation": str(payload["owner"]["generation"])},
        "disposition_status": "handled", "continuation_dispatched": False, "execution_authority": False,
        "billing_settled": False}


async def for_event(manager, tx, actor, job, saved, decision, event_id):
    """Read one exact original disposition before any attempt at execution."""
    if not await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (identity(manager, decision["event_id"]),)):
        return None
    from .wait_reviews import history
    _, creation, _ = await waits._read(manager, tx, actor, job["job_id"])
    state = await history(manager, tx, actor, job, creation, saved)
    if state["event_id"] != event_id:
        raise JobError("wait_disposition_evidence_corrupt")
    return await read_tx(manager, tx, actor, job, creation, saved, state, required=True)


def event_ids(row):
    payload = waits._object(row["payload_json"])
    return [row["event_id"], *[payload[k]["event_id"] for k in
        ("creation", "due", "decision", "operation", "observation", "termination")],
        *([payload["effect"]["event_id"]] if payload["effect"] else [])]
