"""Graph definition, driver job, session and start are admitted together."""

import pytest

from power_loop import AgentLoopConfig
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_driver_admission_rollback_has_no_orphan_job_or_input(registry_store, monkeypatch):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    original = journal.put

    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("journal write failed")

    try:
        monkeypatch.setattr(journal, "put", fail)
        with pytest.raises(RuntimeError, match="journal write failed"):
            await runner.runtime.client.workflow(
                runner.target, {"name": "atomic", "root": leaf("A")}, message_id="run", actor=runner.actor
            )
        assert not await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs")
        assert not await store._db.fetchall(f"SELECT item_id FROM {store.t.follow_up_queue}")
        sessions = await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}")
        assert [s["session_id"] for s in sessions] == [runner.target.session_id]
    finally:
        await runner.aclose()


async def test_final_job_and_inbox_schema(registry_store):
    store, _ = registry_store
    db = store._db

    async def columns(table):
        if db.dialect.name == "sqlite":
            return {r["name"]: bool(r["notnull"]) for r in await db.fetchall(f"PRAGMA table_info({table})")}
        scope = "current_schema()" if db.dialect.name == "postgres" else "DATABASE()"
        rows = await db.fetchall(
            f"SELECT column_name AS name,is_nullable AS nullable FROM information_schema.columns WHERE table_schema={scope} AND table_name=?",
            (table,),
        )
        return {r["name"]: r["nullable"] == "NO" for r in rows}

    jobs = await columns(store.table_prefix + "jobs")
    assert jobs["session_id"]
    assert not {
        "parent_generation",
        "parent_executor_id",
        "parent_boot_id",
        "control_owner_generation",
        "control_owner_executor_id",
        "control_owner_boot_id",
        "lease_until_ms",
    }.intersection(jobs)
    state = await columns(store.t.session_state)
    assert {
        "owner_epoch",
        "owner_token",
        "owner_node_id",
        "owner_boot_id",
        "owner_lease_until_ms",
        "lifecycle",
    } <= state.keys()
    inbox = await columns(store.t.follow_up_queue)
    assert inbox["meta_json"]
    assert {
        "target_job_id",
        "expected_job_generation",
        "expires_at",
        "claim_owner_epoch",
        "delivered_job_id",
        "delivered_job_generation",
    } <= inbox.keys()
