"""Driver admission and graph-state writes use session-native atomic boundaries."""
import pytest

from power_loop import AgentLoopConfig
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_driver_admission_rollback_has_no_orphan_session_or_input(registry_store, monkeypatch):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    original = journal.put

    async def fail(store, tx, sid, key, value):
        await original(store, tx, sid, key, value)
        raise RuntimeError("journal write failed")

    try:
        monkeypatch.setattr(journal, "put", fail)
        with pytest.raises(RuntimeError, match="journal write failed"):
            await runner.runtime.client.workflow(
                runner.target, {"name": "atomic", "root": leaf("A")}, message_id="run", actor=runner.actor,
            )
        sessions = await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}")
        assert [row["session_id"] for row in sessions] == [runner.target.session_id]
        assert not await store._db.fetchall(f"SELECT item_id FROM {store.t.follow_up_queue}")
        assert not await store._db.fetchall(f"SELECT state_key FROM {store.t.session_runtime_state}")
    finally:
        await runner.aclose()


async def test_graph_state_is_separate_from_messages_and_has_no_job_columns(registry_store):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "schema", "root": leaf("A")}, message_id="schema", actor=runner.actor,
        )
        assert (await handle.wait()).status == "completed"
        state = await store._db.fetchall(
            f"SELECT state_key,value_json FROM {store.t.session_runtime_state} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert {tuple(row["state_key"].split(":", 2)[:2]) for row in state} == {
            ("workflow", "run"), ("workflow", "node"),
        }
        assert all("job_id" not in row["value_json"] and "generation" not in row["value_json"] for row in state)
        messages = await store._db.fetchall(
            f"SELECT content,meta_json FROM {store.t.messages} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert all("workflow:node:" not in (row["content"] or "") for row in messages)
        assert all("workflow:run:" not in (row["content"] or "") for row in messages)
    finally:
        await runner.aclose()
