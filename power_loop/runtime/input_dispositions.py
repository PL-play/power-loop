"""Negative task-input responsibility, independent of normal model consumption.

Requires the original stopped execution tree. Never dispatches, edits transcript
history, deletes dedupe keys, or settles payment.
"""
from __future__ import annotations

from . import timer_waits as waits
from .execution_membership import _lease, scope
from .jobs import JobActor, JobError, _hash, _json
from .message_ref import MessageRef
from .model_exit import content_reference, historical_row

_RAW = ("id", "session_id", "seq", "role", "name", "content", "tool_calls_json",
    "tool_call_id", "round_index", "send_index", "meta_json")
_QUEUE = ("id", "session_id", "item_id", "kind", "mode", "status", "content",
    "meta_json", "created_at", "delivered_at", "delivered_seq", "delivered_send_index")


def identity(manager, operation_id):
    return "input_disposed_" + _hash([manager.store.store_id, operation_id])[:48]


async def _original(manager, tx, event):
    from .owner_waits import _input
    from .wait_terminal import negative_owner_exit

    data = waits._object(event["payload_json"])
    request = data.get("request", {})
    if not isinstance(request, dict):
        raise JobError("input_disposition_origin_corrupt")
    target = _lease(request.get("target"))
    row = await manager._row(tx, target["job_id"])
    if (row is None or row["kind"] not in ("subagent", "workflow_node")
            or event["job_id"] != target["job_id"] or event["action"] != "nudge"
            or event["event_type"] != "operation" or not isinstance(request.get("session_id"), str)):
        raise JobError("input_disposition_origin_corrupt")
    sid = request["session_id"]
    actor = JobActor(row["tenant_id"], row["principal_id"], sid)
    outcome, exit_refs = await negative_owner_exit(manager, tx, actor, target, row["kind"])
    original_scope = await scope(manager, tx, row["job_id"], target["generation"])
    original = historical_row(row, original_scope)
    if row["kind"] == "subagent":
        if original["session_id"] != sid:
            raise JobError("input_disposition_route_corrupt")
    else:
        from .node_inputs import descriptor
        if _hash(request.get("node")) != _hash(await descriptor(manager, tx, original, sid, historical=True)):
            raise JobError("input_disposition_route_corrupt")
    # Only this path, after actual negative exit validation, may inspect void
    # provenance without requiring a disposition which we are about to create.
    await _input(manager, tx, actor, original, sid, event, _retired=True)
    raw = {}
    for name, role in (("call", "assistant"), ("receipt", "tool")):
        value = event[name + "_ref_json"]
        if value is None:
            raw[name] = None
        else:
            caller = JobActor(row["tenant_id"], row["principal_id"], event["caller_session_id"])
            message = await manager._validate_ref(tx, caller, waits._ref(waits._object(value)), role=role)
            raw[name] = {"ref": waits._object(value), "raw_hash": _hash({k: message[k] for k in _RAW})}
    return row, request, outcome, exit_refs, raw


async def _expected(manager, tx, event):
    row, request, outcome, exit_refs, raw = await _original(manager, tx, event)
    queue = await tx.fetchone(f"SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (request["session_id"], "nudge:" + event["event_id"]))
    if queue is None or queue["status"] not in ("void", "delivered") or queue["claim_token"] is not None:
        raise JobError("input_disposition_queue_unsettled")
    delivered = None
    if queue["status"] == "delivered":
        message = await tx.fetchone(f"SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND seq=?",
            (request["session_id"], queue["delivered_seq"]))
        meta = waits._object(message["meta_json"]) if message else {}
        inbox_meta = meta.get("inbox")
        if (message is None or message["role"] != "user" or queue["delivered_at"] is None
                or message["send_index"] != queue["delivered_send_index"]
                or not isinstance(inbox_meta, dict) or not isinstance(inbox_meta.get("item_ids"), list)
                or queue["item_id"] not in inbox_meta["item_ids"]):
            raise JobError("input_disposition_message_corrupt")
        ref = MessageRef(manager.store.store_id, request["session_id"],
            message["id"] if manager.store.has_message_ids else None, message["seq"], message["role"],
            message["tool_call_id"], message["send_index"], message["round_index"])
        delivered = {"ref": ref.to_dict(), "raw_hash": _hash({k: message[k] for k in _RAW})}
    elif any(queue[k] is not None for k in ("delivered_at", "delivered_seq", "delivered_send_index")):
        raise JobError("input_disposition_queue_corrupt")
    return row, {"version": 1,
        "disposition": "delivered_unprocessed" if delivered else "withdrawn_before_delivery",
        "operation": content_reference(event), "owner": request["target"], "session_id": request["session_id"],
        "owner_outcome": outcome, "owner_exit": exit_refs, "raw_origin": raw,
        "queue": {k: queue[k] for k in _QUEUE}, "delivered_message": delivered}


async def read_tx(manager, tx, event, *, required=False):
    key = identity(manager, event["event_id"])
    proof = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if proof is None:
        if required:
            raise JobError("input_disposition_evidence_missing")
        return None
    row, expected = await _expected(manager, tx, event)
    forks = await tx.fetchall(f"SELECT event_id,payload_json FROM {manager.events} WHERE job_id=? AND event_type='input_disposition'", (row["job_id"],))
    for fork in forks:
        operation = waits._object(fork["payload_json"]).get("operation")
        original_id = operation.get("event_id") if isinstance(operation, dict) else None
        if not isinstance(original_id, str) or identity(manager, original_id) != fork["event_id"]:
            raise JobError("input_disposition_branch_corrupt")
    if (proof["event_type"] != "input_disposition" or proof["job_id"] != row["job_id"]
            or proof["caller_session_id"] != event["caller_session_id"] or proof["delivery_status"] != "not_applicable"
            or event["delivery_status"] != "handled" or proof["request_hash"] != _hash(expected)
            or _hash(waits._object(proof["payload_json"])) != _hash(expected)
            or type(proof["job_version"]) is not int or not event["job_version"] < proof["job_version"] <= row["version"]
            or type(proof["created_at"]) is not int):
        raise JobError("input_disposition_evidence_corrupt")
    return proof


async def reconcile_tx(manager, tx, actor, event):
    """One original input, under the caller's existing control transaction."""
    owner = await manager._authorize(tx, actor, event["job_id"])
    old = await read_tx(manager, tx, event)
    if old is not None:
        return old
    if event["delivery_status"] == "handled":
        from .processing import require_event_handled
        request = waits._object(event["payload_json"])["request"]
        await require_event_handled(manager, tx, event, session_id=request["session_id"], owner_job_id=owner["job_id"])
        return None  # A normal consumption proof remains normal consumption.
    if event["delivery_status"] != "pending_delivery" or owner["supervision_status"] != "open":
        raise JobError("input_disposition_after_close")
    _, request, _, _, _ = await _original(manager, tx, event)
    if owner["generation"] != request["target"]["generation"]:
        raise JobError("input_disposition_historical_read_only")
    decision = "decision_" + _hash([event["event_id"], event["job_version"], _hash(waits._object(event["payload_json"]))])[:48]
    if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (decision,)):
        raise JobError("input_disposition_processing_conflict")
    await tx.execute(f"UPDATE {manager.store.t.follow_up_queue} SET status='void',claim_token=NULL "
        "WHERE session_id=? AND item_id=? AND status IN ('pending','claimed')",
        (request["session_id"], "nudge:" + event["event_id"]))
    _, payload = await _expected(manager, tx, event)
    key = identity(manager, event["event_id"])
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,caller_session_id,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (key, owner["job_id"], "input_disposition", owner["version"]+1, event["caller_session_id"], _hash(payload), _json(payload), "not_applicable", manager.now()))
    await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (event["event_id"],))
    await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), owner["job_id"]))
    updated = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event["event_id"],))
    return await read_tx(manager, tx, updated, required=True)


async def reconcile_owner_tx(manager, tx, actor, owner, generation):
    if owner["kind"] not in ("subagent", "workflow_node"):
        return
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge' ORDER BY event_id", (owner["job_id"],))
    for event in events:
        request = waits._object(event["payload_json"]).get("request")
        if not isinstance(request, dict):
            raise JobError("input_disposition_origin_corrupt")
        target = request.get("target", {})
        if not isinstance(target, dict):
            raise JobError("input_disposition_origin_corrupt")
        if type(target.get("generation")) is not int:
            raise JobError("input_disposition_origin_corrupt")
        if target["generation"] == generation:
            await reconcile_tx(manager, tx, actor, event)


def event_ids(proof):
    payload = waits._object(proof["payload_json"])
    return [proof["event_id"], payload["operation"]["event_id"],
        *[ref["event_id"] for ref in payload["owner_exit"]]]


def summary(proof):
    """Public responsibility status; never expose execution leases or raw text."""
    data = waits._object(proof["payload_json"])
    return {"event_id": proof["event_id"], "disposition": data["disposition"],
        "owner_outcome": data["owner_outcome"], "generation": str(data["owner"]["generation"]),
        "handled_as": "negative_terminal_disposition", "execution_authority": False,
        "billing_settled": False}
