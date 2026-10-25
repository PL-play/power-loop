"""Original managed-send return values, without permission to replay work.

The receipt shares the model-exit transaction. Historical sends without it
remain explicitly unavailable; assistant text alone cannot reconstruct a send.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, _hash, _json
from .task_history import _snapshot

KEY = "stateful_result_v1"
MAX_BYTES = 1024 * 1024
_TEXT = ("session_id", "status", "final_text")
_COUNTS = ("rounds", "tool_calls")
_FLAGS = ("passed", "new_input_after_pass")
_LISTS = ("pending_tool_calls", "pending_interactions", "texts")
_FIELDS = {*_TEXT, *_COUNTS, *_FLAGS, *_LISTS, "usage", "pass_reason"}


def _validate(value):
    if (
        type(value) is not dict or set(value) != _FIELDS
        or any(type(value[k]) is not str for k in _TEXT)
        or not value["session_id"] or not value["status"]
        or any(type(value[k]) is not int or not 0 <= value[k] < 2**63 for k in _COUNTS)
        or any(type(value[k]) is not bool for k in _FLAGS)
        or any(type(value[k]) is not list for k in _LISTS)
        or any(type(item) is not dict for k in _LISTS for item in value[k])
        or type(value["usage"]) is not dict
        or any(type(k) is not str or type(v) is not int or not 0 <= v < 2**63
               for k, v in value["usage"].items())
        or value["pass_reason"] is not None and type(value["pass_reason"]) is not str
    ):
        raise JobError("send_receipt_invalid")
    try:
        encoded = _json(value)
        if len(encoded.encode("utf-8")) > MAX_BYTES:
            raise JobError("send_receipt_too_large")
        return json.loads(encoded)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("send_receipt_invalid") from None


def capture(result):
    """Library-owned result only; frozen before releasing model execution."""
    from power_loop.agent.stateful_loop import StatefulResult

    if type(result) is not StatefulResult:
        raise JobError("send_receipt_result_required")
    value = _validate(asdict(result))
    return {"version": 1, "result": value, "result_hash": _hash(value)}


async def read(manager, actor, job_id, *, expected_host_context=None, max_bytes=MAX_BYTES * 2):
    """Read an exact original completed send, including its raw message refs.

    Does not acquire a new lease, revive a send, close descendants or settle
    budgets. The host must still reconcile its own effects and continuation.
    """
    if (type(job_id) is not str or not 0 < len(job_id) <= 64
            or type(max_bytes) is not int or not 1024 <= max_bytes <= MAX_BYTES * 2):
        raise JobError("invalid_send_receipt_query")
    from . import model_exit, root_context

    expected = None
    if expected_host_context is not None:
        if type(expected_host_context) is not dict or set(expected_host_context) != {"namespace", "payload"}:
            raise JobError("invalid_send_receipt_context")
        expected = root_context._document(**expected_host_context)

    async with _snapshot(manager) as tx:
        row = await manager._authorize(tx, actor, job_id)
        if row["kind"] != "send" or row["session_id"] != actor.session_id:
            raise JobError("send_receipt_scope_mismatch")
        if row["resource_state"] == "unknown":
            raise JobError("model_exit_unconfirmed")
        # SQL byte preflight prevents loading an oversized exit before checking
        # it. Length on PostgreSQL/SQLite must count UTF-8 bytes, not characters.
        length = "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
        size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?",
            ("model_closed_" + job_id,))
        if size is None:
            raise JobError("send_receipt_exit_unavailable")
        if size["n"] > max_bytes:
            raise JobError("send_receipt_too_large")
        event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?",
            ("model_closed_" + job_id,))
        try:
            payload = json.loads(event["payload_json"])
            evidence = payload["evidence"]
            receipt = evidence[KEY]
            value = _validate(receipt["result"])
            if (set(receipt) != {"version", "result", "result_hash"}
                    or type(receipt["version"]) is not int or receipt["version"] != 1
                    or receipt["result_hash"] != _hash(value)
                    or value["session_id"] != row["session_id"]
                    or value["status"] != evidence["exit"]
                    or value["usage"] != evidence["usage"]
                    or value["rounds"] != evidence["rounds"]
                    or value["new_input_after_pass"] != evidence["new_input_after_pass"]
                    or event["job_id"] != job_id or event["event_type"] != "model_closed"
                    or event["delivery_status"] != "not_applicable"
                    or type(payload["generation"]) is not int or payload["generation"] != row["generation"]):
                raise ValueError()
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            if isinstance(exc, JobError):
                raise
            raise JobError("send_receipt_evidence_unavailable_or_corrupt") from None
        await model_exit.require_stopped(manager, tx, row, payload)
        if value["status"] != "completed" or value["pending_tool_calls"] or value["pending_interactions"]:
            raise JobError("send_receipt_requires_continuation")
        provenance = None
        if expected is not None:
            provenance = await root_context._read(manager, tx, actor, job_id, namespace=expected["namespace"])
            if any(provenance[k] != expected[k] for k in expected):
                raise JobError("send_receipt_context_mismatch")
        last = evidence.get("last_assistant_ref")
        if type(last) is not dict:
            raise JobError("send_receipt_response_required")
        from .message_ref import MessageRef, integer_id
        try:
            raw = await tx.fetchone(f"SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND seq=?",
                (actor.session_id, integer_id(last["seq"], "seq")))
            if raw is None:
                raise ValueError()
            actual = MessageRef(manager.store.store_id, actor.session_id, int(raw["id"]), int(raw["seq"]),
                raw["role"], raw["tool_call_id"], raw["send_index"], raw["round_index"])
            if (actual.to_dict() != last or actual.role != "assistant"
                    or actual.send_index != evidence["send_index"]
                    or evidence["session_id"] != actor.session_id):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise JobError("send_receipt_response_mismatch") from None
        return {"version": 1, "store_id": manager.store.store_id, "job_id": job_id,
            "generation": payload["generation"], "model_exit_event_id": event["event_id"],
            "model_exit_payload_hash": _hash(payload), "result": value,
            "last_assistant_ref": last, "host_context": provenance,
            "execution_authority": False, "resource_stop_confirmed": False, "billing_settled": False}
