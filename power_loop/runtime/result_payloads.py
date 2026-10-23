"""Canonical effective results retain the exact original submission for replay."""
from __future__ import annotations

from .jobs import JobError, _hash


def canonical(requested, outcome, result):
    if outcome == requested["outcome"] and _hash(result) == _hash(requested["result"]):
        return requested
    return {**requested, "outcome": outcome, "result": result, "requested": requested}


def require_replay(stored, requested):
    fields = {"outcome", "result", "resource_state"}
    if (not isinstance(requested, dict) or not fields <= set(requested)
            or set(requested) - fields - {"generation", "unstarted"}
            or requested["outcome"] not in ("succeeded", "failed", "cancelled", "timed_out", "budget_exceeded", "interrupted")
            or not isinstance(requested["result"], dict) or requested["resource_state"] not in ("stopped", "unknown")
            or "unstarted" in requested and requested["unstarted"] is not True
            or "generation" in requested and (type(requested["generation"]) is not int or not 1 < requested["generation"] < 2**63)):
        raise JobError("result_conflict")
    # Preserve historical v1 payloads exactly; no implicit event rewrite.
    if _hash(stored) == _hash(requested):
        return
    if not isinstance(stored, dict) or _hash(stored.get("requested")) != _hash(requested):
        raise JobError("result_conflict")
    outcome = stored.get("outcome")
    if outcome not in (requested["outcome"], "cancelled"):
        raise JobError("result_conflict")
    effective = dict(requested["result"])
    if requested.get("unstarted") is True:
        effective["status"] = outcome
    elif outcome == "cancelled" and "status" in effective and effective["status"] != "cancelled":
        effective.update(execution_status=effective["status"], status="cancelled")
    if _hash(stored) != _hash(canonical(requested, outcome, effective)):
        raise JobError("result_conflict")
