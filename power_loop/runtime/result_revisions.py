"""Immutable per-execution results for the experimental task runtime."""
from __future__ import annotations

import json

from .jobs import JobError


def result_id(job_id, generation):
    return "result_" + job_id + (f"_g{generation}" if generation > 1 else "")


def execution_id(job_id, generation):
    return f"exec_{job_id}_g{generation}"


async def delivery_id(manager, reader, job, generation=None):
    import hashlib
    generation = job["generation"] if generation is None else generation
    return hashlib.sha256(("result:" + result_id(job["job_id"], generation)).encode()).hexdigest()


async def get(manager, actor, job_id, generation):
    if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation <= 9223372036854775807:
        raise JobError("invalid_result_generation")
    await manager._authorize(manager.db, actor, job_id)
    event = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'",
        (result_id(job_id, generation),))
    if event is None:
        raise JobError("result_revision_missing")
    payload = json.loads(event["payload_json"])
    if payload.get("generation", 0 if payload.get("unstarted") else 1) != generation:
        raise JobError("result_revision_missing")
    return {"event_id": event["event_id"], "generation": generation, "revision": int(event["job_version"]),
            "delivery_status": event["delivery_status"], "payload": payload}
