"""Fenced main-model decisions; accepted intent is not a continuation effect."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from . import timer_waits as waits
from . import wait_notifications as notices
from .control_requests import reference
from .execution_membership import _lease
from .finalization_plan import PlanChanged
from .jobs import JobError, JobLease, _hash, _json, _operation
from .message_ref import MessageRef


@dataclass
class PreparedSources:
    """One operation's verified immutable sources, never execution authority."""
    grants: dict
    receipts: set
    publishing: bool = False


_SOURCE_PROOFS = ContextVar('prepared_wait_sources',default=None)


@contextmanager
def prepared_sources():
    proofs = PreparedSources({},set())
    token = _SOURCE_PROOFS.set(proofs)
    try:
        yield proofs
    finally:
        _SOURCE_PROOFS.reset(token)


def _intent(event_id, caller, call_ref, tool_call_id, decision, reason, next_review_at_ms):
    notices._event_id(event_id)
    if not isinstance(caller, JobLease) or not isinstance(call_ref, MessageRef):
        raise JobError("persisted_tool_call_required")
    _lease(asdict(caller))
    waits._text(tool_call_id, "tool_call_id", 255)
    if not isinstance(decision, str) or decision not in ("continue", "defer", "cancel"):
        raise JobError("invalid_wait_decision")
    waits._text(reason, "decision_reason", 1000)
    if decision == "defer":
        waits._number(next_review_at_ms, "next_review_at_ms")
    elif next_review_at_ms is not None:
        raise JobError("unexpected_wait_review_time")
    return {"event_id": event_id, "caller": asdict(caller), "call_ref": call_ref.to_dict(),
        "tool_call_id": tool_call_id, "decision": decision, "reason": reason, "next_review_at_ms": next_review_at_ms}


async def _accepted(manager, tx, actor, event_id, request, operation, *, _notification=None):
    """Read exact accepted evidence, even after the caller closes or expires."""
    job, due, saved, receipt = _notification if _notification is not None else await notices.notification(manager, tx, actor, event_id)
    try:
        reconstructed = _intent(event_id, JobLease(**request["caller"]), waits._ref(request["call_ref"]),
            request["tool_call_id"], request["decision"], request["reason"], request["next_review_at_ms"])
        if _hash(reconstructed) != _hash(request):
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError("wait_decision_evidence_corrupt") from exc
    from .task_query import _reference
    call = _reference(manager, {"created_by_session_id": actor.session_id}, operation["call_ref_json"], "assistant")
    data = waits._object(operation["payload_json"])
    from . import wait_decision_admission as admission
    request_ref = data.get('request_ref')
    if not isinstance(request_ref, dict):
        raise JobError('wait_decision_evidence_corrupt')
    proofs = _SOURCE_PROOFS.get()
    key = manager.store.store_id, request_ref.get('event_id')
    known = proofs.grants.get(key) if proofs is not None else None
    if known is None:
        if proofs is not None and proofs.publishing:
            # Another decision/review appeared since readonly preparation.
            # Abort this tree write and prepare that new original outside it.
            raise PlanChanged()
        grant = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_ref.get('event_id'),))
        if grant is None:
            raise JobError('wait_decision_evidence_missing')
        original = waits._object(grant['payload_json'])['request']
        await admission._stored_request(manager, tx, actor, grant, original, _notification=(job, due, saved, receipt))
        if proofs is not None:
            proofs.grants[key] = grant,original
    else:
        grant,original = known
    if admission.client(original)!=request or request_ref!=reference(grant):
        raise JobError('wait_decision_evidence_corrupt')
    canonical_id = notices.decision_id(manager, event_id)
    expected_response = {"status": "accepted", "event_id": event_id, "decision_event_id": canonical_id,
        "decision": request["decision"], "effect_status": "pending", "execution_authority": False}
    if (operation["event_type"] != "operation" or operation["action"] != "wait_decision"
            or operation["job_id"] != due["job_id"] or operation["caller_session_id"] != actor.session_id
            or operation["tool_call_id"] != request["tool_call_id"] or operation["request_hash"] != _hash(request)
            or _hash(call) != _hash(request["call_ref"])
            or _hash(data) != _hash({"version": 1, "request": request, "response": expected_response, 'request_ref':request_ref})):
        raise JobError("wait_decision_evidence_corrupt")
    canonical = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (canonical_id,))
    if canonical is None:
        raise JobError("wait_decision_evidence_missing")
    payload = waits._object(canonical["payload_json"])
    keys = {"version", "event_id", "due_hash", "operation_id", "request", "observation", "original_owner", "continuation_hash", 'request_ref'}
    if (canonical["event_type"] != "wait_decision" or canonical["job_id"] != due["job_id"]
            or set(payload) != keys or type(payload["version"]) is not int or payload["version"] != 1
            or canonical["request_hash"] != _hash(payload) or payload.get("operation_id") != operation["event_id"]
            or payload.get("due_hash") != due["request_hash"] or payload.get("event_id") != event_id
            or _hash(payload.get("request")) != _hash(request)
            or _hash(payload["original_owner"]) != _hash(saved["binding"]["owner"])
            or payload["continuation_hash"] != _hash(saved["request"]["request"]["continuation"])
            or payload['request_ref'] != request_ref or payload['observation'] != original['proof']['observation']
            or canonical["delivery_status"] not in ("pending_delivery", "delivered", "handled")
            or due["delivery_status"] != "handled"):
        raise JobError("wait_decision_evidence_corrupt")
    # The source permission above validates the actual observation and exact
    # returned model attempts once; the target original must reference them.
    return _operation(operation)


async def recall_decision(manager, actor, *, event_id, caller, call_ref, tool_call_id,
                          decision, reason="Reviewed original deferred request.", next_review_at_ms=None):
    request = _intent(event_id, caller, call_ref, tool_call_id, decision, reason, next_review_at_ms)
    key = manager._origin_key(actor, "wait_decision", call_ref, tool_call_id, None)
    async with manager._read() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old is None:
            return None
        if old["request_hash"] != _hash(request):
            raise JobError("operation_conflict")
        return await _accepted(manager, tx, actor, event_id, request, old)


async def _validate_observation(manager, tx, actor, caller, call_ref, due, receipt, event):
    """Apply identical semantic checks on first acceptance and historical recall."""
    data = waits._object(event["payload_json"])
    fields = {"version", "due_event_id", "due_hash", "notification_id", "caller", "round_index", "send_index", "message_ref", "message_hash"}
    if (set(data) != fields or type(data["version"]) is not int or data["version"] != 1
            or event["event_type"] != "wait_observed" or event["job_id"] != caller.job_id
            or event["caller_session_id"] != actor.session_id
            or event["request_hash"] != _hash(data) or event["delivery_status"] != "prepared"
            or data["due_event_id"] != due["event_id"] or data["due_hash"] != due["request_hash"]
            or data["notification_id"] != receipt["item_id"] or _hash(data["caller"]) != _hash(asdict(caller))
            or type(data["round_index"]) is not int or data["round_index"] != call_ref.round_index
            or type(data["send_index"]) is not int or data["send_index"] != call_ref.send_index):
        raise JobError("wait_observation_mismatch")
    ref = waits._ref(data["message_ref"])
    row = await manager._validate_ref(tx, actor, ref, role="user")
    if (ref.seq >= call_ref.seq or data["message_hash"] != _hash(row["content"])
            or notices.render_message(receipt["public"]) not in row["content"]):
        raise JobError("wait_observation_mismatch")


async def _observation(manager, tx, actor, caller, call_ref, due, receipt):
    if type(call_ref.round_index) is not int or call_ref.round_index < 0:
        raise JobError("wait_observation_required")
    key = notices.observation_id(manager, caller, call_ref.round_index, receipt["item_id"])
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if event is None:
        raise JobError("wait_observation_required")
    await _validate_observation(manager, tx, actor, caller, call_ref, due, receipt, event)
    return {"observation_id": key, "observation_hash": event["request_hash"]}


async def record_decision(manager, actor, *, event_id, caller, call_ref, tool_call_id,
                          decision, reason="Reviewed original deferred request.", next_review_at_ms=None):
    request = _intent(event_id, caller, call_ref, tool_call_id, decision, reason, next_review_at_ms)
    from .wait_decision_admission import record
    return await record(manager, actor, request)


async def inspect(manager, actor, event_id, *, field="summary", char_offset=0, max_chars=8000):
    if (field not in ("summary", "note", "continuation") or type(char_offset) is not int or char_offset < 0
            or type(max_chars) is not int or not 256 <= max_chars <= 16000):
        raise JobError("invalid_wait_inspection")
    async with manager._read() as tx:
        job, due, saved, receipt = await notices.notification(manager, tx, actor, event_id)
        if field == "summary":
            if char_offset:
                raise JobError("invalid_wait_inspection")
            decision = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (notices.decision_id(manager, event_id),))
            if decision is not None:
                saved_decision = waits._object(decision["payload_json"])
                op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (saved_decision.get("operation_id"),))
                if op is None or not isinstance(saved_decision.get("request"), dict):
                    raise JobError("wait_decision_evidence_missing")
                await _accepted(manager, tx, actor, event_id, saved_decision["request"], op)
            elif due["delivery_status"] == "handled":
                raise JobError("wait_decision_evidence_missing")
            summary = {"version": 1, "event_id": event_id, "wait_job_id": job["job_id"],
                "observed_at_ms": str(manager.now()), "notification": receipt["public"],
                "decision_status": "recorded" if decision else "pending", "effect_status": decision["delivery_status"] if decision else None,
                "execution_authority": False}
            if decision is not None:
                from .wait_dispositions import for_event, reply
                disposed = await for_event(manager, tx, actor, job, saved, decision, event_id)
                if disposed is not None:
                    summary.update(effect_status="not_applied", disposition=reply(disposed))
            if len(_json(summary)) > max_chars:
                raise JobError("wait_inspection_response_too_large")
            return summary
        body = saved["request"]["request"][field]
        if char_offset > len(body):
            raise JobError("invalid_wait_inspection")
        end = min(len(body), char_offset + max_chars)
        return {"version": 1, "event_id": due["event_id"], "field": field, "text": body[char_offset:end],
            "char_offset": char_offset, "next_char_offset": end if end < len(body) else None,
            "total_chars": len(body), "execution_authority": False}
