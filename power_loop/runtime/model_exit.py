"""Freeze actual model-attempt identities at local model exit.

This is execution evidence, not payment settlement or permission to retry an
unknown request. Legacy exits without tracking do not gain negative closure.
"""
from __future__ import annotations

import json

from .jobs import JobError, _hash

SNAPSHOT = "model_attempt_snapshot"


def content_reference(event):
    """Business ACK may change later; the original event content cannot."""
    return {k: event[k] for k in ("event_id", "job_id", "event_type", "job_version", "request_hash")} | {
        "payload_hash": _hash(json.loads(event["payload_json"]))}


async def capture(manager, tx, row):
    captured = await _capture(manager, tx, row)
    if captured[0] is not None:
        from .execution_membership import _remember_after_commit
        _remember_after_commit(manager, tx, "_model_exit_captures", (row["job_id"], row["generation"]), captured)
    return captured[0]


async def _capture(manager, tx, row):
    from . import model_attempts as attempts
    from .workflow_controller import proof

    scope = proof(row)
    projected = _own_projection(manager, row)
    if projected is not None and projected.get(attempts.event_id(manager, "tracking", scope)) is not None:
        return _from_projection(manager, row, scope, projected)
    tracking = await attempts._read(manager, tx, "tracking", scope, required=False)
    if tracking is None:
        creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (row["job_id"],))
        phases = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type LIKE 'model_attempt_%' LIMIT 1", (row["job_id"],))
        if phases or creation and json.loads(creation["payload_json"]).get("request", {}).get("model_attempt_protocol") is not None:
            raise JobError("model_attempt_evidence_missing")
        return None, None
    expected = {"version": 1, "scope": scope, "session_id": row["session_id"]}
    if _hash(tracking[1]) != _hash(expected):
        raise JobError("model_attempt_evidence_corrupt")
    phases = await attempts._all(manager, tx, row["job_id"])
    events = []
    for phase in ("prepared", "dispatched", "finished"):
        for _identity, (event, data) in sorted(phases[phase].items()):
            if _hash(data["scope"]) != _hash(scope):
                raise JobError("model_attempt_evidence_corrupt")
            events.append(content_reference(event))
    return {"version": 1, "scope": scope, "tracking": content_reference(tracking[0]), "phases": events}, (tracking, phases)


def _own_projection(manager, row):
    """Attempt rows this live process projected for its own first generation."""
    from .execution_membership import _trusted
    if row["generation"] != 1 or _trusted(manager, "_first_claims", row["job_id"]) is None:
        return None
    return _trusted(manager, "_projected_attempts", row["job_id"])


def _from_projection(manager, row, scope, projected):
    from . import model_attempts as attempts
    tracking = projected[attempts.event_id(manager, "tracking", scope)]
    phases = {"prepared": {}, "dispatched": {}, "finished": {}}
    for event in sorted(projected.values(), key=lambda event: event["event_id"]):
        phase = event["event_type"].removeprefix("model_attempt_")
        if phase in phases:
            data = json.loads(event["payload_json"])
            phases[phase][data["attempt_id"]] = (event, data)
    events = [content_reference(event) for phase in ("prepared", "dispatched", "finished")
              for _identity, (event, _data) in sorted(phases[phase].items())]
    pair = (tracking, json.loads(tracking["payload_json"]))
    return ({"version": 1, "scope": scope, "tracking": content_reference(tracking), "phases": events},
            (pair, phases))


async def require_stopped(manager, tx, row, model, *, known=None):
    """``known`` receives the loaded attempt rows by event_id for reference building."""
    if (model.get("resource_state") != "stopped" or type(model.get("generation")) is not int
            or model["generation"] != row["generation"] or row["kind"] != "send"):
        raise JobError("model_exit_unconfirmed")
    saved = model.get("evidence", {}).get(SNAPSHOT)
    if not isinstance(saved, dict) or type(saved.get("version")) is not int or saved["version"] != 1:
        raise JobError("model_exit_snapshot_required")
    from .execution_membership import _trusted
    # A live send's own close captured these attempts after its final flush.
    current, loaded = (_trusted(manager, "_model_exit_captures", (row["job_id"], row["generation"]))
                       or await _capture(manager, tx, row))
    if current is None or _hash(current) != _hash(saved):
        raise JobError("model_exit_snapshot_mismatch")
    from .pending_native import validate
    await validate(manager, tx, row, model)
    # A late fee observation is independent. These checks only rule out an
    # unfinished/unknown provider request, retaining original unstarted proof.
    from .model_attempts import require_quiescent
    if known is not None:
        known[loaded[0][0]["event_id"]] = loaded[0][0]
        known.update({event["event_id"]: event for phase in loaded[1].values() for event, _ in phase.values()})
    return await require_quiescent(manager, tx, row, include_budget=False, loaded=loaded)


def historical_row(row, scope):
    from .execution_membership import _lease
    lease = _lease(scope["lease"])
    owner = scope["owner"]
    return {**row, **lease, "kind": scope["kind"], "session_id": scope["session_id"], "intake": "closed",
        "control_owner_job_id": owner["job_id"] if owner else None,
        "control_owner_generation": owner["generation"] if owner else None,
        "control_owner_executor_id": owner["executor_id"] if owner else None,
        "control_owner_boot_id": owner["boot_id"] if owner else None}
