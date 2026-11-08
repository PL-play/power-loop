"""power-loop 3.0 — loop-level coverage of the REAL fold strategies (not the deterministic test
stub), the fold-timeout / soft-fail paths, agentic note-writing, and the custom-fold-under-verbatim
adapter. Closes the review's test gaps #9/#10/#21/#32 + the async-fold-out-of-lock behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.fold import AgenticFold, FoldContext, FoldResult, LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation, VerbatimRepresentation
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_stateful_loop import _echo_registry, _Scripted


def _tc(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


async def _proj_loop(runtime_agent, store, llm, fold_strategy, *, max_tokens=10, target=None):
    return await runtime_agent(target=target,
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, representation=ProjectedRepresentation(),
            fold_strategy=fold_strategy, max_tokens=max_tokens,
        ),
    )


# ── #10: the loop drives the REAL LLMSummaryFold (extra LLM call → the compact summary) ──


@pytest.mark.asyncio
async def test_projection_loop_runs_real_llm_summary_fold(store: SessionStore, runtime_agent) -> None:
    # 3 sends (d1/d2/d3), then the end-of-send fold makes ONE more LLM call → the compact summary.
    llm = _Scripted(responses=[
        LLMResponse(raw_text="d1"), LLMResponse(raw_text="d2"), LLMResponse(raw_text="d3"),
        LLMResponse(raw_text="<summary>FOLD-1</summary>"),  # consumed by LLMSummaryFold._summarize
    ])
    loop = await _proj_loop(runtime_agent, store, llm, LLMSummaryFold(keep_last_sends=2))
    sid = loop.target.session_id
    for i in range(1, 4):
        assert (await loop.send(f"m{i}")).status == "completed"
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 1
    assert compact.content["summary"] == "FOLD-1"  # the REAL LLM fold ran (not a deterministic concat)


# ── #9: an agentic PROJECTION fold's captured note_ops are actually WRITTEN to the store ──


@pytest.mark.asyncio
async def test_projection_loop_agentic_fold_writes_notes(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[
        LLMResponse(raw_text="d1"), LLMResponse(raw_text="d2"), LLMResponse(raw_text="d3"),
        # the fold's agentic loop: round 1 saves a note, round 2 emits the summary
        LLMResponse(raw_text="", tool_calls=[_tc("n1", "note", '{"action": "add", "content": "FACT-X"}')]),
        LLMResponse(raw_text="<summary>folded</summary>"),
    ])
    loop = await _proj_loop(runtime_agent, store, llm, AgenticFold(keep_last_sends=2, max_rounds=3))
    sid = loop.target.session_id
    for i in range(1, 4):
        assert (await loop.send(f"m{i}")).status == "completed"
    assert await store.latest_project_compact(sid) is not None
    notes = await store.list_notes(sid)
    assert any(n.content == "FACT-X" for n in notes)  # the fold's NoteOp was persisted


# ── fold timeout → soft-fail (rows committed, no compact this send) ──


@dataclass
class _SlowFold:
    keep_last_sends: int = 2
    trigger_ratio: float = 0.75
    fold_id: str = "slow"

    async def fold(self, rows, *, context: FoldContext) -> FoldResult | None:
        import asyncio
        await asyncio.sleep(5)  # longer than the configured fold_timeout_s
        return FoldResult(content={"summary": "never"}, folded_to_send=1)


@pytest.mark.asyncio
async def test_projection_fold_timeout_soft_fails(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 4)])
    loop = await runtime_agent(
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, representation=ProjectedRepresentation(),
            fold_strategy=_SlowFold(), max_tokens=10, fold_timeout_s=0.05,
        ),
    )
    sid = loop.target.session_id
    for i in range(1, 4):
        assert (await loop.send(f"m{i}")).status == "completed"
    assert await store.latest_project_compact(sid) is None  # fold timed out → no compact
    proj = {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"}
    assert proj == {1, 2, 3}  # every send's rows still committed


# ── #32: a clean soft-fail (fold returns None) → rows committed, no compact ──


@dataclass
class _DeclineFold:
    keep_last_sends: int = 2
    trigger_ratio: float = 0.75
    fold_id: str = "decline"

    async def fold(self, rows, *, context: FoldContext) -> FoldResult | None:
        return None  # strategy declines (soft-fail)


@pytest.mark.asyncio
async def test_projection_fold_soft_fail_returns_none(store: SessionStore, runtime_agent) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 4)])
    loop = await _proj_loop(runtime_agent, store, llm, _DeclineFold())
    sid = loop.target.session_id
    for i in range(1, 4):
        assert (await loop.send(f"m{i}")).status == "completed"
    assert await store.latest_project_compact(sid) is None
    assert {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"} == {1, 2, 3}


# ── #21: a CUSTOM FoldStrategy under VERBATIM goes through the in-place adapter end-to-end ──


@dataclass
class _CustomVerbatimFold:
    keep_last_sends: int = 1
    trigger_ratio: float = 0.5
    summary_max_tokens: int = 500
    fold_id: str = "custom-verbatim"
    seen: int = 0

    async def fold(self, rows, *, context: FoldContext) -> FoldResult | None:
        # exercised via FoldStrategyCompactor._summarize_async (verbatim in-place path)
        object.__setattr__(self, "seen", self.seen + 1)
        return FoldResult(content={"summary": "CUSTOM-VERBATIM-SUMMARY"}, folded_to_send=1)


@pytest.mark.asyncio
async def test_verbatim_custom_fold_via_adapter_compacts(store: SessionStore, monkeypatch, runtime_agent) -> None:
    # conftest loads .env (CONTEXT_COMPACT_THRESHOLD=50000) which the in-place compactor honors as
    # an absolute override; clear it so the verbatim fold triggers at max_tokens × trigger_ratio.
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    fold = _CustomVerbatimFold()
    llm = _Scripted(responses=[LLMResponse(raw_text=f"r{i}") for i in range(1, 7)])
    loop = await runtime_agent(
        model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, representation=VerbatimRepresentation(),
            fold_strategy=fold, max_tokens=10,  # tiny → in-place compactor triggers
        ),
    )
    sid = loop.target.session_id
    for i in range(1, 6):
        assert (await loop.send(f"m{i}")).status == "completed"
    comps = await store.list_compactions(sid)
    assert comps, "custom fold under verbatim should compact in place via the adapter"
    assert fold.seen >= 1  # the custom strategy's fold() actually ran
    note = next((m for m in await store.load_active_messages(sid) if m.name == "compact_note"), None)
    assert note is not None and "CUSTOM-VERBATIM-SUMMARY" in (note.content or "")


# ── concurrency: two loops sharing the store, REAL (awaiting) fold, optimistic commit ──


@pytest.mark.asyncio
async def test_rebuilt_runtime_real_fold_no_double_compact(store: SessionStore, runtime_agent) -> None:
    # Sequential (deterministic) but using the REAL LLMSummaryFold so the out-of-lock fold +
    # optimistic commit path is exercised: exactly one compact, one project row per send.
    def _script() -> _Scripted:
        return _Scripted(responses=[LLMResponse(raw_text=f"x{i}") for i in range(1, 4)]
                         + [LLMResponse(raw_text="<summary>s</summary>") for _ in range(3)])
    la = await _proj_loop(runtime_agent, store, _script(), LLMSummaryFold(keep_last_sends=2))
    sid = la.target.session_id
    await la.send("m1")
    await la.aclose()
    lb = await _proj_loop(runtime_agent, store, _script(), LLMSummaryFold(keep_last_sends=2), target=la.target)
    await lb.send("m2", message_id="replacement-m2")
    await lb.send("m3", message_id="replacement-m3")
    keys = [(r.send_index, r.kind) for r in await store.load_project_messages(sid)]
    assert len(keys) == len(set(keys))  # no duplicate/clobbered rows
    assert sorted(r.send_index for r in await store.load_project_messages(sid)
                  if r.kind == "project") == [1, 2, 3]


# ── helper: ensure _proj_loop config exposes the new fold_timeout default ──


def test_default_fold_timeout_present() -> None:
    c = AgentLoopConfig()
    assert c.fold_timeout_s == 120.0


# ── store-level: commit_projection_fold optimistic concurrency + supersede-delete ──


@pytest.mark.asyncio
async def test_commit_projection_fold_stale_skip_and_supersede(store: SessionStore, runtime_agent) -> None:
    sid = await store.create_session()
    rows1 = [("user", {"human": ["q1"]}, None), ("project", {"tools": [], "final_text": "a1"}, None)]
    prior, _ = await store.write_send_projection_rows(
        sid, send_index=1, rows=rows1, source_seq_lo=None, source_seq_hi=None, projector_version=1,
    )
    assert prior is None
    # First fold → compact at to_send=1.
    assert await store.commit_projection_fold(
        sid, content={"summary": "c1"}, rendered_text=None, from_send=1, to_send=1,
        projector_version=1, expected_prior_to_send=None,
    ) is True
    assert (await store.latest_project_compact(sid)).compact_to_send == 1

    rows2 = [("user", {"human": ["q2"]}, None), ("project", {"tools": [], "final_text": "a2"}, None)]
    await store.write_send_projection_rows(
        sid, send_index=2, rows=rows2, source_seq_lo=None, source_seq_hi=None, projector_version=1,
    )
    # STALE commit (its snapshot saw NO prior, but the cursor is now 1) → rejected, no clobber.
    assert await store.commit_projection_fold(
        sid, content={"summary": "STALE"}, rendered_text=None, from_send=1, to_send=2,
        projector_version=1, expected_prior_to_send=None,
    ) is False
    assert (await store.latest_project_compact(sid)).compact_to_send == 1  # untouched

    # VALID advancing commit (saw prior cursor=1) → succeeds AND deletes the superseded compact.
    assert await store.commit_projection_fold(
        sid, content={"summary": "c2"}, rendered_text=None, from_send=1, to_send=2,
        projector_version=1, expected_prior_to_send=1,
    ) is True
    compacts = [r for r in await store.load_project_messages(sid) if r.kind == "compact"]
    assert len(compacts) == 1 and compacts[0].compact_to_send == 2  # only the latest remains


async def test_factory_memory_is_available_to_custom_fold_and_next_session(runtime_agent):
    from power_loop import MemorySnapshot

    class Memory:
        def __init__(self):
            self.saved = []

        async def remember(self, *, snapshot, session_id):
            if snapshot.status == 'compaction':
                self.saved.extend(snapshot.messages)

        async def recall(self, **kwargs):
            return [{'content': 'Retained fact: BLUEHERON'}] if self.saved else []

    class CaptureFold(LLMSummaryFold):
        async def fold(self, rows, *, context):
            result = await super().fold(rows, context=context)
            if result:
                await context.memory.remember(snapshot=MemorySnapshot(session_id=context.session_id,
                    messages=context.representation.render(rows), status='compaction'), session_id=context.session_id)
            return result

    memory = Memory()
    first = await runtime_agent(model=_Scripted([LLMResponse(raw_text='ack')]*8), registry=_echo_registry(),
        config=AgentLoopConfig(max_tokens=100, context_budget_tokens=40, memory_budget_tokens=10,
            memory=memory, representation=ProjectedRepresentation(), fold_strategy=CaptureFold(keep_last_sends=1)))
    for _ in range(3):
        assert (await first.send('BLUEHERON '+ 'context '*80)).status == 'completed'
    assert any('BLUEHERON' in str(message) for message in memory.saved)
    from .test_memory_recall_hook import _RecLLM
    model = _RecLLM()
    second = await runtime_agent(model=model, config=AgentLoopConfig(compactor=None, memory=memory))
    assert (await second.send('Recall the project')).status == 'completed'
    assert 'Retained fact: BLUEHERON' in str(model.seen)
