"""Past history comes from projections only: repair it, or stop before the model."""
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, LLMResponse, ProjectedRepresentation
from tests.runtime_fixture import RunningAgent

from .test_stateful_loop import _Scripted

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@dataclass
class _FailingProjector(ProjectedRepresentation):
    fail_sends: frozenset[int] = frozenset()

    def project_send(self, send_rows, *, send_index, tool_registry):
        if send_index in self.fail_sends:
            raise RuntimeError(f"projection repair failed for send {send_index}")
        return super().project_send(send_rows, send_index=send_index, tool_registry=tool_registry)


async def test_missing_and_stale_projection_are_rebuilt_before_use(registry_store):
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
        # The rebuilt projections (not the raw protocol rows) reach the model.
        assert all(text in str(model.calls[-1]) for text in (
            "original fact 1", "original fact 2", "answer 2", "original fact 3", "answer 3"))
        assert not any("tool_calls" in m for m in model.calls[-1])
        assert (await store.load_all_messages(sid))[:len(before)] == before
        rebuilt = [r for r in await store.load_project_messages(sid) if r.send_index in (2, 3)]
        assert rebuilt and {r.projector_version for r in rebuilt} == {1}
    finally:
        await agent.aclose()


async def test_unrepairable_projection_stops_before_the_model_call(registry_store):
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text="answer 1"), LLMResponse(raw_text="answer 2")])
    projector = _FailingProjector(fail_sends=frozenset({2}))
    agent = await RunningAgent(store, model, config=AgentLoopConfig(
        representation=projector, context_budget_tokens=10**6)).start()
    sid = agent.target.session_id
    try:
        assert (await agent.send("original fact 1")).status == "completed"
        # Send 2 completes but its end-of-send projection write fails (best-effort by design).
        assert (await agent.send("original fact 2")).status == "completed"
        assert not [r for r in await store.load_project_messages(sid) if r.send_index == 2]
        calls = len(model.calls)

        result = await agent.send("third")
        assert result.status == "failed"
        assert result.error_code == "projection_unavailable"
        assert len(model.calls) == calls  # stopped BEFORE the first model call
        # The raw original is still in the audit log; it was not silently fed to the model.
        assert any(m.content == "original fact 2" or "original fact 2" in str(m.content)
                   for m in await store.load_all_messages(sid))
    finally:
        await agent.aclose()

async def test_stale_projection_without_usable_audit_stops_before_the_model(registry_store):
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text="answer 1"), LLMResponse(raw_text="answer 2")])
    agent = await RunningAgent(store, model, config=AgentLoopConfig(
        representation=ProjectedRepresentation(), context_budget_tokens=10**6)).start()
    sid = agent.target.session_id
    try:
        assert (await agent.send("original fact 1")).status == "completed"
        assert (await agent.send("original fact 2")).status == "completed"
        await store._db.execute(
            f"UPDATE {store.t.project_messages} SET projector_version=0 "
            "WHERE session_id=? AND send_index=2", (sid,))
        raw = await store.load_active_messages(sid, send_index=2)
        assert raw and await store.deactivate_messages(sid, [r.seq for r in raw]) == len(raw)
        calls = len(model.calls)
        result = await agent.send("third")
        assert result.status == "failed"
        assert result.error_code == "projection_unavailable"
        assert len(model.calls) == calls  # no model/tool call before the stop
    finally:
        await agent.aclose()


async def test_mixed_current_and_stale_rows_in_one_send_require_repair(registry_store):
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text="answer 1"), LLMResponse(raw_text="answer 2")])
    agent = await RunningAgent(store, model, config=AgentLoopConfig(
        representation=ProjectedRepresentation(), context_budget_tokens=10**6)).start()
    sid = agent.target.session_id
    try:
        assert (await agent.send("original fact 1")).status == "completed"
        assert (await agent.send("original fact 2")).status == "completed"
        # One row is current, its sibling is stale: the send must not be skipped just because
        # "have" contains the current version.
        await store._db.execute(
            f"UPDATE {store.t.project_messages} SET projector_version=0 "
            "WHERE session_id=? AND send_index=2 AND kind='user'", (sid,))
        raw = await store.load_active_messages(sid, send_index=2)
        assert raw and await store.deactivate_messages(sid, [r.seq for r in raw]) == len(raw)
        calls = len(model.calls)
        result = await agent.send("third")
        assert result.status == "failed"
        assert result.error_code == "projection_unavailable"
        assert len(model.calls) == calls
    finally:
        await agent.aclose()
