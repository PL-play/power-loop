"""Real-LLM end-to-end for the power-loop 3.0 orthogonal context model: every
Representation × FoldStrategy combo, plus a representation mode switch.

The live provider must ACCEPT the (folded) history each send and keep completing. Asserts the
mode-appropriate compaction artifact:
  * verbatim  → an in-place compaction recorded (compact_note in pl_messages / compactions table)
  * projection → a derived ``compact`` row in pl_project_messages (folded via the strategy's LLM)
AgenticFold additionally MAY persist notes (model-discretionary → checked, not hard-asserted).

Auto-skips without a real-LLM provider env (see tests/conftest.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.fold import AgenticFold, LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation, VerbatimRepresentation
from power_loop.runtime.store.store import SessionStore
from power_loop.tools.default_tools import run_recall_send
from tests.real._llm import make_llm

pytestmark = pytest.mark.asyncio


def _loop(store, representation, fold_strategy, *, max_tokens=40):
    return StatefulAgentLoop(
        llm=make_llm(max_tokens=80, temperature=0.0),
        store=store,
        config=AgentLoopConfig(
            system_prompt="You are terse. Reply with one short sentence.",
            max_rounds=3, max_tokens=max_tokens,
            representation=representation, fold_strategy=fold_strategy,
        ),
    )


async def _drive(loop, sid, n=5):
    for i in range(1, n + 1):
        r = await loop.send(f"Note the word apple{i} and reply OK.", session_id=sid)
        assert r.status == "completed", f"send {i} status={r.status}"


# ── verbatim × {default, agentic}: in-place compaction via the mapped compactor ──


async def test_real_verbatim_llm_summary_fold(monkeypatch) -> None:
    # The in-place compactor honors CONTEXT_COMPACT_THRESHOLD (env) as an absolute override; clear
    # it so the fold triggers at max_tokens × trigger_ratio in this small test.
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    store = await SessionStore.open(":memory:")
    try:
        loop = _loop(store, VerbatimRepresentation(), LLMSummaryFold(keep_last_sends=2))
        sid = await loop.new_session()
        await _drive(loop, sid, 5)
        # Verbatim folds in place: a real LLM-summary compaction must have been recorded.
        comps = await store.list_compactions(sid)
        assert comps, "expected at least one in-place compaction"
        assert any(m.name == "compact_note" for m in await store.load_active_messages(sid))
    finally:
        await store.close()


async def test_real_verbatim_agentic_fold(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    store = await SessionStore.open(":memory:")
    try:
        loop = _loop(store, VerbatimRepresentation(), AgenticFold(keep_last_sends=2, max_rounds=3))
        sid = await loop.new_session()
        await loop.send("Remember: my project is Zephyr; I prefer metric units. Reply OK.",
                        session_id=sid)
        await _drive(loop, sid, 5)
        comps = await store.list_compactions(sid)
        assert comps, "expected at least one in-place (agentic) compaction"
        notes = await store.list_notes(sid)  # model-discretionary
        assert isinstance(notes, list)
    finally:
        await store.close()


# ── projection × {default, agentic}: derived-layer fold via the strategy's LLM call ──


async def test_real_projection_llm_summary_fold() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = _loop(store, ProjectedRepresentation(), LLMSummaryFold(keep_last_sends=2))
        sid = await loop.new_session()
        await _drive(loop, sid, 4)
        compact = await store.latest_project_compact(sid)
        assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 2
        assert (compact.content or {}).get("summary")  # real LLM summary
        # recall re-expands a folded send from the immutable pl_messages audit.
        t_loop = set_current_loop(SimpleNamespace(store=store, config=None))  # type: ignore[arg-type]
        t_sid = set_session_id(sid)
        try:
            recalled = await run_recall_send(1)
        finally:
            reset_session_id(t_sid)
            reset_current_loop(t_loop)
        assert recalled.startswith("send #1") and "apple1" in recalled.lower()
    finally:
        await store.close()


async def test_real_projection_agentic_fold() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = _loop(store, ProjectedRepresentation(),
                     AgenticFold(keep_last_sends=2, summary_max_tokens=160, max_rounds=3))
        sid = await loop.new_session()
        await loop.send("Remember: my project is Zephyr; I prefer metric units. Reply OK.",
                        session_id=sid)
        await _drive(loop, sid, 4)
        compact = await store.latest_project_compact(sid)
        assert compact is not None and (compact.content or {}).get("summary")
        assert isinstance(await store.list_notes(sid), list)
    finally:
        await store.close()


# ── mode switch: verbatim → projection migrates prior history (one-time, never throws) ──


async def test_real_mode_switch_verbatim_to_projection_migrates() -> None:
    store = await SessionStore.open(":memory:")
    try:
        v = _loop(store, VerbatimRepresentation(), LLMSummaryFold(keep_last_sends=4), max_tokens=8000)
        sid = await v.new_session()
        await _drive(v, sid, 2)
        assert await store.load_project_messages(sid) == []  # verbatim wrote no projection rows

        # Re-open the SAME session in projection mode → migration folds prior history once.
        p = _loop(store, ProjectedRepresentation(), LLMSummaryFold(keep_last_sends=2), max_tokens=40)
        r = await p.send("Now reply OK once more.", session_id=sid)
        assert r.status == "completed"
        assert await store.load_project_messages(sid), "prior history migrated into projection table"
        assert (await store.get_session(sid)).metadata.get("projection_migrated") is not None
    finally:
        await store.close()
