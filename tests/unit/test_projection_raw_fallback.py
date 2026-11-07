"""Current projection loss/version mismatch reads the unchanged original sends."""
import pytest

from power_loop import AgentLoopConfig, LLMResponse, ProjectedRepresentation
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_stateful_loop import _Scripted

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_missing_and_stale_projection_reads_originals(registry_store):
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text=f"answer {i}") for i in range(1, 5)])
    agent = await RunningAgent(store, model, config=AgentLoopConfig(
        representation=ProjectedRepresentation(), context_budget_tokens=10**6)).start()
    sid = agent.target.session_id
    try:
        for i in range(1, 4):
            await agent.send(f"original fact {i}")
        before = await store.load_all_messages(sid)
        await store._db.execute(f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=2", (sid,))
        await store._db.execute(f"UPDATE {store.t.project_messages} SET projector_version=0 WHERE session_id=? AND send_index=3", (sid,))
        assert (await agent.send("continue")).status == "completed"
        assert all(text in str(model.calls[-1]) for text in (
            "original fact 1", "original fact 2", "answer 2", "original fact 3", "answer 3"))
        assert (await store.load_all_messages(sid))[:len(before)] == before
    finally:
        await agent.aclose()
