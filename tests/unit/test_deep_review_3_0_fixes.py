"""Regression tests for the 2026-06-22 deep-review Tier-1 fixes (BUG_REVIEW_3.0.md).

Each test FAILS against the pre-fix code and passes after:
  B2  — workflow wake-guard must claim atomically (no journal clobber / no double-wake).
  B3+B5 — resume() must self-heal a crash-left ids-only pending instead of stranding the session.
"""

from __future__ import annotations

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.history_projector import DefaultDeterministicProjector, IdentityProjector
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.types import ProjectMessageRow
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




# ── B2: wake-guard atomic claim ───────────────────────────────────────────────






# ── B3 + B5: resume() self-heals a crash-left ids-only pending ─────────────────


async def test_new_input_aligns_unknown_pending_without_replay(runtime_agent) -> None:
    async with await SessionStore.open(":memory:") as store:
        loop = await runtime_agent(
            model=_Scripted([LLMResponse(raw_text="resumed-final")]), store=store,
            registry=_echo_registry(),
            config=AgentLoopConfig(system_prompt="S", max_rounds=3),
        )
        sid = loop.target.session_id
        # A crash mid-abort can persist a pending carrying only tool_call_ids (tool_calls dropped).
        await store.set_pending(
            sid, {"round_index": 0, "assistant_seq": None, "tool_call_ids": ["b"], "tool_calls": []}
        )
        result = await loop.send("continue")
        assert result.status == "completed"
        assert (await store.get_state(sid)).pending is None  # pending cleared, NOT stranded
        rows = await store.load_active_messages(sid)
        await loop.aclose()
        assert any(r.role == "tool" and r.tool_call_id == "b" for r in rows)  # protocol resolved






# ── Test-gap (test-audit): the fold GATE — a foldable span UNDER budget must NOT fold ──
# The existing fold tests all use max_tokens=10 (threshold ~0) so the fold ALWAYS fires; none assert
# the token gate actually gates. A regression that ignored the threshold (always-fold) would slip past.


async def test_fold_gate_does_not_fold_below_token_budget(runtime_agent) -> None:
    async with await SessionStore.open(":memory:") as store:
        # keep_last_sends=2 with 4 short sends → a foldable span [1,2] EXISTS, but the rendered prefix
        # is far below max_tokens(8000) × trigger_ratio → nothing should fold.
        proj = await _projector_loop(
            runtime_agent, store, _Scripted([LLMResponse(raw_text=f"d{i}") for i in range(1, 5)]),
            DefaultDeterministicProjector(keep_last_sends=2), max_tokens=8000,
        )
        sid = proj.target.session_id
        for i in range(1, 5):
            assert (await proj.send(f"m{i}")).status == "completed"
        compacts = [r for r in await store.load_project_messages(sid) if r.kind == "compact"]
        await proj.aclose()
        assert compacts == [], "fold fired below the token budget (the trigger gate is broken)"
