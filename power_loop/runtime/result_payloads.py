"""Stable result payload comparison retained only for unmigrated declarations."""
from __future__ import annotations

from power_loop.agent.runtime_types import RuntimeRequestError, sha256_canonical


def canonical(requested, outcome, result):
    if outcome == requested["outcome"] and sha256_canonical(result) == sha256_canonical(requested["result"]):
        return requested
    return {**requested, "outcome": outcome, "result": result, "requested": requested}


def require_replay(stored, requested):
    if sha256_canonical(stored) == sha256_canonical(requested):
        return
    if sha256_canonical(stored.get("requested")) != sha256_canonical(requested):
        raise RuntimeRequestError("request_conflict", "result payload changed")
    outcome = stored.get("outcome")
    effective = dict(requested["result"])
    if requested.get("unstarted") is True:
        effective["status"] = outcome
    elif outcome == "cancelled" and effective.get("status") not in (None, "cancelled"):
        effective.update(execution_status=effective["status"], status="cancelled")
    if sha256_canonical(stored) != sha256_canonical(canonical(requested, outcome, effective)):
        raise RuntimeRequestError("request_conflict", "result payload changed")
