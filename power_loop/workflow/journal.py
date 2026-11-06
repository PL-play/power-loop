"""Frozen run definition and small node references in the existing runtime store."""

from __future__ import annotations

import json

from power_loop.agent.runtime_types import canonical

JOURNAL_PREFIX = "workflow:run:"
ROUTE_KEY = "\x00workflow_route"


def run_key(run_id):
    return JOURNAL_PREFIX + run_id


def node_key(identity):
    return "workflow:node:" + identity


async def put(store, tx, sid, key, value):
    from power_loop.agent.runtime_store import now_ms

    await tx.execute(
        store._db.dialect.upsert(
            store.t.session_runtime_state, ("session_id", "state_key"), ("value_json", "updated_at")
        ),
        (sid, key, canonical(value), now_ms()),
    )


async def get(store, reader, sid, key):
    row = await reader.fetchone(
        f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?", (sid, key)
    )
    return json.loads(row["value_json"]) if row else None


async def read(store, parent_sid, run_id):
    from power_loop.runtime.jobs import JobActor, JobManager

    manager = JobManager(store)
    row = await manager.db.fetchone(
        f"SELECT j.* FROM {manager.jobs} j JOIN {manager.events} e "
        "ON e.job_id=j.job_id WHERE e.event_id=? AND e.action='create' AND j.kind='workflow'",
        ("op_" + run_id,),
    )
    if row is None:
        return None
    await manager._authorize(manager.db, JobActor(row["tenant_id"], row["principal_id"], parent_sid), row["job_id"])
    value = await get(store, manager.db, row["session_id"], run_key(run_id))
    nodes = await manager.db.fetchall(
        f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? "
        "AND state_key LIKE 'workflow:node:%' ORDER BY state_key LIMIT 10001",
        (row["session_id"],),
    )
    if len(nodes) > 10000:
        from power_loop.agent.runtime_types import CapacityError

        raise CapacityError("workflow node query limit exceeded")
    return {
        **value,
        "nodes": [json.loads(node["value_json"]) for node in nodes],
        "task_generation": row["generation"],
        "status": row["status"],
        "result": json.loads(row["result_json"] or "null"),
    }


async def list_run_ids(store, parent_sid):
    from power_loop.runtime.jobs import JobManager

    manager = JobManager(store)
    rows = await manager.db.fetchall(
        f"SELECT e.event_id FROM {manager.jobs} j JOIN {manager.events} e ON e.job_id=j.job_id "
        "WHERE j.created_by_session_id=? AND j.kind='workflow' AND e.action='create' ORDER BY j.created_at DESC LIMIT 100",
        (parent_sid,),
    )
    return [row["event_id"][3:] for row in rows]
