"""Regression tests for the 2026-06-22 deep-review Tier-1 fixes (BUG_REVIEW_3.0.md).

Each test FAILS against the pre-fix code and passes after:
  B2  — workflow wake-guard must claim atomically (no journal clobber / no double-wake).
  B3+B5 — resume() must self-heal a crash-left ids-only pending instead of stranding the session.
  B4  — projection migration must not write a compact that COVERS sends it didn't fold (data loss).
  B13 — projection migration must keep the would-be-folded sends (as project rows) on fold soft-fail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective
from power_loop.runtime.history_projector import DefaultDeterministicProjector, IdentityProjector
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.types import ProjectMessageRow
from power_loop.workflow import journal
from power_loop.workflow.runner import _wake_note, make_wake_guard
from tests.unit.test_history_projection_loop import _projector_loop
from tests.unit.test_stateful_loop import _echo_registry, _Scripted

pytestmark = pytest.mark.unit


def _pmr(send_index, kind, content, **kw):
    return ProjectMessageRow(
        session_id="s", send_index=send_index, kind=kind, content=content,
        rendered_text=kw.get("rendered_text"), source_seq_lo=None, source_seq_hi=None,
        compact_from_send=kw.get("compact_from_send"), compact_to_send=kw.get("compact_to_send"),
        projector_version=1, token_estimate=None, created_at=0,
    )


# ── B7 + B10: legacy IdentityProjector must route verbatim + never-fold + render compacts ──


def test_b7_identity_projector_routes_verbatim_never_folds_renders_compact() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AgentLoopConfig(history_projector=IdentityProjector(), max_tokens=50)
    assert cfg.projection_representation is None  # verbatim path (not the projection-fold path)
    assert cfg.resolve_compactor() is None  # never-fold (keep_last_sends==0)
    # render must surface a compact's summary, not silently drop it (the data-loss root cause)
    rendered = IdentityProjector().render([_pmr(5, "compact", {"summary": "OLD"})])
    assert rendered == [{"role": "user", "content": "OLD"}]


def test_b10_keep_zero_projection_projector_folds_aggressively_not_four() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AgentLoopConfig(history_projector=DefaultDeterministicProjector(keep_last_sends=0), max_tokens=50)
    assert cfg.fold_strategy.keep_last_sends == 1  # 0 → clamp to floor 1, NOT silently 4


# ── B9: recall_send #N labels must appear in projected render ──


def test_b9_projected_render_tags_sends_with_index() -> None:
    rep = ProjectedRepresentation()
    rows = [
        _pmr(7, "user", {"human": ["q"]}),
        _pmr(7, "project", {"tools": [], "final_text": "a"}),
        _pmr(3, "compact", {"summary": "folded"}, compact_from_send=1, compact_to_send=3),
    ]
    out = rep.render(rows)
    assert out[0]["content"].startswith("[#7]")  # user send tagged
    assert out[1]["content"].startswith("#7 ")  # assistant turn tagged → recall_send(send_index=7)
    assert "recall_send" in out[2]["content"] and "#1" in out[2]["content"]  # compact shows its range


@dataclass
class _DeclineFold:
    """A fold strategy that always soft-fails (returns None) — to exercise the migration's
    soft-fail branch deterministically without an LLM."""

    keep_last_sends: int = 1
    trigger_ratio: float = 0.75
    fold_id: str = "decline"

    async def fold(self, rows, *, context):
        return None


# ── B2: wake-guard atomic claim ───────────────────────────────────────────────


async def test_wake_guard_claim_does_not_clobber_concurrent_journal_write() -> None:
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session(system_prompt="p")
        run_id = "run-W"
        await journal.seed(store, parent, run_id, "wf", spec={})
        guard = make_wake_guard(store)
        ctx = TimerFireCtx(session_id=parent, note=_wake_note(run_id, "completed"))
        # The wake-guard's claim and a journal step land concurrently on the SAME run key.
        await asyncio.gather(
            guard(ctx),
            journal.record_step(store, parent, run_id, node_id="leaf", status="completed", text="o"),
        )
        rec = await journal.read(store, parent, run_id)
        assert rec.get("woke") is True  # wake claimed
        assert [s["node_id"] for s in rec["steps"]] == ["leaf"]  # step NOT clobbered by the guard


async def test_wake_guard_double_fire_delivers_once() -> None:
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session(system_prompt="p")
        run_id = "run-D"
        await journal.seed(store, parent, run_id, "wf", spec={})
        guard = make_wake_guard(store)
        c1 = TimerFireCtx(session_id=parent, note=_wake_note(run_id, "completed"))
        c2 = TimerFireCtx(session_id=parent, note=_wake_note(run_id, "completed"))
        await asyncio.gather(guard(c1), guard(c2))
        # exactly one CONTINUE (first delivery), the other SKIP — never two wakes.
        skips = [c for c in (c1, c2) if c.directive == HookDirective.SKIP]
        assert len(skips) == 1
        assert (await journal.read(store, parent, run_id)).get("woke") is True


# ── B3 + B5: resume() self-heals a crash-left ids-only pending ─────────────────


async def test_resume_self_heals_ids_only_pending() -> None:
    async with await SessionStore.open(":memory:") as store:
        loop = StatefulAgentLoop(
            llm=_Scripted([LLMResponse(raw_text="resumed-final")]), store=store,
            tool_registry=_echo_registry(),
            config=AgentLoopConfig(system_prompt="S", max_rounds=3),
        )
        sid = await loop.new_session()
        # A crash mid-abort can persist a pending carrying only tool_call_ids (tool_calls dropped).
        await store.set_pending(
            sid, {"round_index": 0, "assistant_seq": None, "tool_call_ids": ["b"], "tool_calls": []}
        )
        await loop.resume(sid)
        assert await loop.get_pending(sid) is None  # pending cleared, NOT stranded
        rows = await store.load_active_messages(sid)
        assert any(r.role == "tool" and r.tool_call_id == "b" for r in rows)  # protocol resolved


# ── B13: migration soft-fail keeps the would-be-folded sends ──────────────────


async def test_migration_softfail_keeps_fold_sends_as_project_rows() -> None:
    async with await SessionStore.open(":memory:") as store:
        base = StatefulAgentLoop(
            llm=_Scripted([LLMResponse(raw_text="a1"), LLMResponse(raw_text="a2")]), store=store,
            tool_registry=_echo_registry(),
            config=AgentLoopConfig(system_prompt="S", max_rounds=2),
        )
        sid = await base.new_session()
        await base.send("m1", session_id=sid)  # send 1
        await base.send("m2", session_id=sid)  # send 2
        proj = _projector_loop(
            store, _Scripted([LLMResponse(raw_text="a3")]), DefaultDeterministicProjector(),
            fold_strategy=_DeclineFold(keep_last_sends=1),
        )
        await proj.send("m3", session_id=sid)  # migrating send; keep=1 → fold span = [1], soft-fails
        proj_sends = {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"}
        assert {1, 2}.issubset(proj_sends)  # send 1 (the fold span) preserved, not dropped


# ── Test-gap (test-audit): the fold GATE — a foldable span UNDER budget must NOT fold ──
# The existing fold tests all use max_tokens=10 (threshold ~0) so the fold ALWAYS fires; none assert
# the token gate actually gates. A regression that ignored the threshold (always-fold) would slip past.


async def test_fold_gate_does_not_fold_below_token_budget() -> None:
    async with await SessionStore.open(":memory:") as store:
        # keep_last_sends=2 with 4 short sends → a foldable span [1,2] EXISTS, but the rendered prefix
        # is far below max_tokens(8000) × trigger_ratio → nothing should fold.
        proj = _projector_loop(
            store, _Scripted([LLMResponse(raw_text=f"d{i}") for i in range(1, 5)]),
            DefaultDeterministicProjector(keep_last_sends=2), max_tokens=8000,
        )
        sid = await proj.new_session()
        for i in range(1, 5):
            assert (await proj.send(f"m{i}", session_id=sid)).status == "completed"
        compacts = [r for r in await store.load_project_messages(sid) if r.kind == "compact"]
        assert compacts == [], "fold fired below the token budget (the trigger gate is broken)"


# ── B4: migration soft-fail with a pre-existing compact_note must not over-claim ──


async def test_migration_softfail_with_note_does_not_overclaim_range() -> None:
    async with await SessionStore.open(":memory:") as store:
        base = StatefulAgentLoop(
            llm=_Scripted([LLMResponse(raw_text=f"a{i}") for i in range(1, 4)]), store=store,
            tool_registry=_echo_registry(),
            config=AgentLoopConfig(system_prompt="S", max_rounds=2),
        )
        sid = await base.new_session()
        for m in ("m1", "m2", "m3"):
            await base.send(m, session_id=sid)  # sends 1,2,3
        # Simulate the in-place compactor folding send 1 into a compact_note.
        s1 = [r.seq for r in await store.load_active_messages(sid) if r.send_index == 1]
        await store.record_compaction(
            sid, from_seq=min(s1), to_seq=max(s1), note_content="OLD-SUMMARY",
            before_tokens=None, after_tokens=None, round_index=0, fold_seqs=s1, order_key=0,
        )
        proj = _projector_loop(
            store, _Scripted([LLMResponse(raw_text="a4")]), DefaultDeterministicProjector(),
            fold_strategy=_DeclineFold(keep_last_sends=1),
        )
        await proj.send("m4", session_id=sid)  # migrating send 4; fold span = [2], soft-fails
        compact = await store.latest_project_compact(sid)
        assert compact is not None
        # the note-only compact must claim NO real send range (else the reader excludes send 2 forever)
        assert compact.compact_to_send == 0
        proj_sends = {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"}
        assert 2 in proj_sends  # the un-merged fold send is preserved as a project row
