"""Workflow graph state stored beside its driver session.

The journal contains only TargetRef/InputRef coordinates, request hashes, ordinary
result payloads and graph progress. Node results remain ordinary driver inbox
inputs; this module never creates a second result ledger.
"""

from __future__ import annotations

import json

from power_loop.agent.runtime_types import canonical, sha256_canonical

JOURNAL_PREFIX = "workflow:run:"
NODE_PREFIX = "workflow:node:"
ROUTE_KEY = "\x00workflow_route"
NODE_LIMIT = 10000


def run_key(run_id):
    return JOURNAL_PREFIX + run_id


def node_key(driver, run_id, node_path, iteration_path, attempt, continuation):
    """Include driver/run coordinates as well as the logical node execution."""
    identity = sha256_canonical([
        "workflow_journal_node", driver.store_id, driver.session_id, run_id,
        node_path, iteration_path, attempt, continuation,
    ])
    return NODE_PREFIX + identity


async def put(store, tx, sid, key, value):
    from power_loop.agent.runtime_store import now_ms

    await tx.execute(
        store._db.dialect.upsert(
            store.t.session_runtime_state, ("session_id", "state_key"),
            ("value_json", "updated_at"),
        ),
        (sid, key, canonical(value), now_ms()),
    )


async def get(store, reader, sid, key):
    row = await reader.fetchone(
        f"SELECT value_json FROM {store.t.session_runtime_state} "
        "WHERE session_id=? AND state_key=?",
        (sid, key),
    )
    return json.loads(row["value_json"]) if row else None


async def nodes(store, reader, sid, *, limit=None):
    if limit is None:
        limit = NODE_LIMIT
    rows = await reader.fetchall(
        f"SELECT state_key,value_json FROM {store.t.session_runtime_state} "
        f"WHERE session_id=? AND state_key LIKE '{NODE_PREFIX}%' "
        "ORDER BY state_key LIMIT ?",
        (sid, limit + 1),
    )
    if len(rows) > limit:
        from power_loop.agent.runtime_types import CapacityError

        raise CapacityError()
    return [json.loads(row["value_json"]) for row in rows]


async def read(store, parent_sid, run_id):
    row = await store._db.fetchone(
        f"SELECT session_id FROM {store.t.sessions} "
        "WHERE parent_session_id=? AND session_kind='workflow_driver' "
        "AND start_input_message_id=?",
        (parent_sid, run_id),
    )
    if row is None:
        return None
    value = await get(store, store._db, row["session_id"], run_key(run_id))
    if value is None:
        return None
    return {**value, "nodes": await nodes(store, store._db, row["session_id"])}


async def list_run_ids(store, parent_sid):
    rows = await store._db.fetchall(
        f"SELECT start_input_message_id FROM {store.t.sessions} "
        "WHERE parent_session_id=? AND session_kind='workflow_driver' "
        "ORDER BY created_at DESC,session_id DESC LIMIT 100",
        (parent_sid,),
    )
    return [row["start_input_message_id"] for row in rows]
