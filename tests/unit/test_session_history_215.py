"""P1 raw-message coordinates plus P2 in-memory history / projection behavior."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, ToolDefinition
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .session_native_fixture import ACTOR, target
from .test_send_sql_profile import Profiled

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ManyRoundModel(LLMService):
    """N-1 tool rounds then a final answer; records every request."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = 0
        self.requests = []

    async def complete(self, request, **kwargs):
        self.calls += 1
        self.requests.append(request)
        if self.calls < self.rounds:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": f"c{self.calls}", "type": "function",
                "function": {"name": "echo", "arguments": '{"text":"ok"}'}}],
                token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        return LLMResponse(raw_text="done", tool_calls=None,
                           token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


def _echo_registry():
    registry = ToolRegistry()

    async def echo(text):
        return text

    registry.register(ToolDefinition("echo", "echo",
                                     input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                                     pricing=Free("fixture")), echo)
    return registry


async def test_200_rounds_read_history_once_at_entry(registry_store):
    """design 215 §8.1/P2.4: a 200-round send reads raw history only at entry; the loop
    appends to memory + store without re-reading, and every raw row is still written."""
    store, _ = registry_store
    model = ManyRoundModel(200)
    agent = await RunningAgent(store, model, _echo_registry(),
                               config=AgentLoopConfig(max_rounds=250, context_budget_tokens=10**9)).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix)
    await meter.reset()
    profiled = Profiled(original, meter)
    store._db = profiled
    try:
        result = await agent.send("go")
        assert result.status == "completed" and model.calls == 200
        messages_table = store.t.messages
        # The send's only raw user row came from the claim transaction; a first send has no
        # past history to load, and there is NO per-round reload of messages.
        history_selects = [r for r in profiled.records
                           if r["sql"].upper().startswith("SELECT") and messages_table in r["sql"]
                           and not any("get_result" in c or "read_result_state" in c for c in r["chain"])]
        assert history_selects == [], [r["sql"] for r in history_selects]
        # The projection save is the only writer; on a cold session its statistics path needs
        # at most the combined compact/existence read (the window read is skipped for send 1).
        proj_reads = [r for r in profiled.records
                      if r["sql"].upper().startswith("SELECT") and store.t.project_messages in r["sql"]]
        assert len(proj_reads) <= 1, [r["sql"] for r in proj_reads]
        # Every round's raw rows are durably written: 1 user + 200 assistant + 199 tool.
        rows = await store.load_all_messages(agent.target.session_id)
        assert len(rows) == 1 + 200 + 199
        assert sum(r.role == "assistant" for r in rows) == 200
        assert sum(r.role == "tool" for r in rows) == 199
        assert all(r.send_index == 1 for r in rows)
        # The loop's model requests grew from memory (each round sees the prior tool result).
        assert len(model.requests[-1].messages) > len(model.requests[0].messages)
    finally:
        store._db = original
        await meter.close()
        await agent.aclose()


async def test_projection_coverage_and_owner_fence(registry_store):
    """design 215 §8.2: a projection row carries the coverage triple and a stale owner's
    derived write is rejected by the fence."""
    from power_loop.contracts.errors import SessionLeaseLost
    from power_loop.runtime.representation import ProjectedRepresentation
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")

    class Lease:
        epoch = batch.owner_epoch

    projector = ProjectedRepresentation()
    rows = await store.load_active_messages(root.session_id, send_index=batch.send_index)
    projected = projector.project_send(rows, send_index=batch.send_index, tool_registry=None)
    await store.write_send_projection_rows(
        root.session_id, send_index=batch.send_index,
        rows=[(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows],
        source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
        projector_version=projector.version, load_snapshot=False, ownership=Lease(),
    )
    proj = await store.load_project_messages(root.session_id)
    assert proj and all(p.completed_through_send_index == batch.send_index for p in proj)
    assert all(p.projector_version == projector.version for p in proj)
    assert all(p.source_seq_hi == projected.source_seq_hi for p in proj)

    # A stale owner (wrong epoch/token) cannot publish a derived projection row.
    class StaleLease:
        epoch = batch.owner_epoch + 99
        token = "stale"

    with pytest.raises(SessionLeaseLost):
        await store.write_send_projection_rows(
            root.session_id, send_index=batch.send_index,
            rows=[(pr.kind, pr.content, pr.rendered_text) for pr in projected.rows],
            source_seq_lo=projected.source_seq_lo, source_seq_hi=projected.source_seq_hi,
            projector_version=projector.version, load_snapshot=False, ownership=StaleLease(),
        )


async def test_next_seq_fence_and_gaps(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "before", message_id="before", actor=ACTOR)
    first = await port.claim(root, node_id="a", boot_id="a")
    await port.final(first, final_text="before result")
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET next_seq=next_seq+3 WHERE session_id=?",
        (root.session_id,),
    )
    await port.accept(root, "after", message_id="after", actor=ACTOR)
    second = await port.claim(root, node_id="b", boot_id="b")
    assert second.message_seq == 6 and second.send_index == 2
    await port.final(second, final_text="after result")
    rows = (await port.messages(root, actor=ACTOR)).items
    assert [row.seq for row in rows] == [1, 2, 6, 7]
    tail = await port.messages(root, actor=ACTOR, after_seq=2)
    assert [row.seq for row in tail.items] == [6, 7]
    assert (await store.message_ref(root.session_id, 6)).role == "user"
    cold = RuntimeInputStore(await connect(schema="verify"))
    assert (await cold.get_result(second.input_refs[0], actor=ACTOR)).result_seq == 7
