"""Pending tool-call facts; recovery is owned by the session storage boundary."""
from __future__ import annotations

import json

from power_loop.agent.runtime_types import RuntimeRequestError, sha256_canonical

ERROR = "pending_call_error"
SNAPSHOT = "pending_call_snapshot"


def identity(manager, source):
    return "pending_" + sha256_canonical([
        manager.store.store_id, source["session_id"], source["job_id"], source["generation"],
    ])[:48]


async def _object(manager, tx, table, column, where, parameters, limit):
    length = "length(CAST(" + column + " AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(" + column + ")"
    row = await tx.fetchone(
        f"SELECT CASE WHEN {length}<={limit} THEN {column} ELSE NULL END AS value FROM {table} WHERE {where}",
        parameters,
    )
    if row is None or row["value"] is None:
        return None
    return json.loads(row["value"])


async def current(manager, tx, sid):
    row = await tx.fetchone(
        f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (sid,),
    )
    return json.loads(row["pending_json"]) if row and row["pending_json"] else None


async def capture(manager, tx, source):
    return await current(manager, tx, source["session_id"])


async def model(manager, tx, source):
    row = await tx.fetchone(
        f"SELECT * FROM {manager.jobs} WHERE job_id=? AND generation=?",
        (source["job_id"], source["generation"]),
    )
    if row is None:
        raise RuntimeRequestError("not_found")
    return {"row": dict(row), "evidence": json.loads(row["evidence_json"] or "{}")}


async def require_disposition(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job pending disposition is removed")


async def retire_pending_runtime(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job pending retirement is removed")


async def recover_runtime(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job pending recovery is removed")


async def recover_tree(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job tree recovery is removed")


__all__ = [
    "ERROR", "SNAPSHOT", "capture", "current", "identity", "model",
    "recover_runtime", "recover_tree", "require_disposition", "retire_pending_runtime",
]
