"""Check stopped local work and reference task event content."""
from __future__ import annotations

import json

from .jobs import JobError, _hash


def content_reference(event):
    """Business ACK may change later; the original event content cannot."""
    return {k: event[k] for k in ("event_id", "job_id", "event_type", "job_version", "request_hash")} | {
        "payload_hash": _hash(json.loads(event["payload_json"]))}


async def require_stopped(manager, tx, row, model):
    if (model.get("resource_state") != "stopped" or model.get("generation") != row["generation"]
            or row["kind"] != "send"):
        raise JobError("model_exit_unconfirmed")
    from .model_attempts import require_quiescent
    return await require_quiescent(manager, tx, row)


def historical_row(row, scope):
    from .execution_membership import _lease
    lease = _lease(scope["lease"])
    owner = scope["owner"]
    return {**row, **lease, "kind": scope["kind"], "session_id": scope["session_id"], "intake": "closed",
        "control_owner_job_id": owner["job_id"] if owner else None}
