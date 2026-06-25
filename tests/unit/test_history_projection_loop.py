"""Step 3 integration: the projection layer wired into StatefulAgentLoop._run_loop.

Covers the reader (history = rendered projections of finished sends + the in-flight send
verbatim), the writer (project at end-of-send into pl_project_messages), and the must-fix
behaviors: child-session exclusion (by parent_session_id), reason-gate (waiting_for_input
defers), memory-recall composition, and that the default path (no projector) is unchanged.
Reuses the scripted-LLM helpers from test_stateful_loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import create_default_tool_registry
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.fold import FoldResult
from power_loop.runtime.history_projector import (
    DefaultDeterministicProjector,
    IdentityProjector,
)
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.types import SessionKind, SubagentLifecycle
from tests.unit.test_stateful_loop import _echo_registry, _Scripted, _tool_resp


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@dataclass
class _DeterministicFold:
    """Test-only FoldStrategy (no LLM): delegates to the deterministic projector's ``compact()``
    so the loop's fold WIRING (span selection, keep_last_sends, roll-forward, migration) is
    assertable deterministically. The real LLMSummaryFold/AgenticFold are covered in tests/real."""

    keep_last_sends: int = 4
    trigger_ratio: float = 0.75
    fold_id: str = "test_deterministic"

    async def fold(self, rows, *, context):
        folded = [r.send_index for r in rows if r.kind in ("user", "project")]
        if not folded:
            return None
        rep = context.representation
        pc = rep.compact(rows) if hasattr(rep, "compact") else None  # raises propagate → degrade
        if pc is None:
            return None
        return FoldResult(content=pc.content, rendered_text=pc.rendered_text, folded_to_send=max(folded))


def _two_send_script() -> _Scripted:
    # send 1: echo tool then "done1"; send 2: "done2"
    return _Scripted(responses=[
        _tool_resp("c1", "echo", '{"text": "a"}'),
        LLMResponse(raw_text="done1"),
        LLMResponse(raw_text="done2"),
    ])


def _projector_loop(
    store: SessionStore, llm: _Scripted, projector, *, max_tokens: int = 8000, migrate: bool = True,
    fold_strategy=None,
) -> StatefulAgentLoop:
    # 3.0 API: representation = the (projection) projector; fold = a deterministic test fold seeded
    # with the projector's keep/trigger so fold mechanics match the old deterministic behavior.
    fs = fold_strategy or _DeterministicFold(
        keep_last_sends=int(getattr(projector, "keep_last_sends", 4)),
        trigger_ratio=float(getattr(projector, "trigger_ratio", 0.75) or 0.75),
    )
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4, representation=projector, fold_strategy=fs,
            max_tokens=max_tokens, migrate_history_on_switch=migrate,
        ),
    )


@pytest.mark.asyncio
async def test_projection_written_and_fed_to_next_send(store: SessionStore) -> None:
    llm = _two_send_script()
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()

    await loop.send("first", session_id=sid)
    # End-of-send 1 wrote user+project projection rows…
    proj = await store.load_project_messages(sid)
    assert {(r.send_index, r.kind) for r in proj} == {(1, "user"), (1, "project")}
    # …and pl_messages is untouched (the immutable audit still holds send 1's structured rows).
    assert any(m.tool_calls for m in await store.load_active_messages(sid))

    n = len(llm.calls)
    await loop.send("second", session_id=sid)
    req = llm.calls[n]  # send 2's first LLM request (history + runtime)
    # The historical prefix is PLAIN TEXT — no tool-call protocol leaks into send 2's context.
    assert all("tool_calls" not in m and "tool_call_id" not in m for m in req)
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined   # send 1 rendered (human + final_text)
    assert "second" in joined                          # send 2's own user (in-flight, verbatim)


# ── H1 (BUG_REVIEW_3.4): out-of-band tool rows must carry the in-flight send_index ──
# submit_input / resume(_execute_pending) / abort_pending append tool rows out-of-band of the
# pipeline. In projection mode the reader partitions active rows by send_index; a NULL-send_index
# tool row falls into the legacy prefix and renders BEFORE its own assistant tool_call, so
# align_tool_calls drops it as an orphan — silently losing the answer. These three tests assert the
# rows are stamped with the active send's index (so they stay paired in current_rows) AND that the
# answer actually reaches the resumed LLM request.


def _projection_loop_with_tools(store, llm, tool_registry, *, max_rounds=3):
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=tool_registry,
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=max_rounds,
            representation=DefaultDeterministicProjector(), max_tokens=8000,
        ),
    )


@pytest.mark.asyncio
async def test_projection_submit_input_preserves_answer(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("tc-input", "request_user_input",
                   '{"kind":"confirm","prompt":"Approve?","options":[{"id":"yes","label":"Yes"}]}'),
        LLMResponse(raw_text="Approved."),
    ])
    loop = _projection_loop_with_tools(
        store, llm, create_default_tool_registry(include=["request_user_input"]))
    sid = await loop.new_session()

    waiting = await loop.send("needs approval", session_id=sid)
    assert waiting.status == "waiting_for_input"
    iid = waiting.pending_interactions[0]["interaction_id"]

    n = len(llm.calls)
    resumed = await loop.submit_input(sid, iid, {"choice": "yes"})
    assert resumed.status == "completed"
    assert resumed.final_text == "Approved."

    # The answer reaches the resumed LLM request as a paired tool result (would be DROPPED pre-fix).
    req = llm.calls[n]
    tool_msgs = [m for m in req if m.get("role") == "tool" and m.get("tool_call_id") == "tc-input"]
    assert tool_msgs, "submitted answer dropped from the projected context"
    assert "choice" in str(tool_msgs[0].get("content"))
    # The assistant tool_call that owns it is also present and precedes the result (no orphan).
    roles = [m.get("role") for m in req]
    assert roles.index("assistant") < roles.index("tool")

    # Store-level proof: the out-of-band tool row carries the active send's index, not NULL.
    tool_rows = [r for r in await store.load_active_messages(sid) if r.role == "tool"]
    assert tool_rows and all(r.send_index == 1 for r in tool_rows)


@pytest.mark.asyncio
async def test_projection_resume_preserves_replayed_tool_result(store: SessionStore) -> None:
    # Seed a realistic in-flight send (send_index=1) with an unresolved echo tool_call, as a crash
    # mid-tool would leave it, then resume() → _execute_pending replays the tool.
    llm = _Scripted(responses=[LLMResponse(raw_text="after resume")])
    loop = _projection_loop_with_tools(store, llm, _echo_registry())
    sid = await store.create_session()
    await store.mutate_runtime_state(sid, "send_index", lambda v: 1, default=0)
    await store.append_message(sid, role="user", content="kick off", send_index=1)
    asst_seq = await store.append_message(
        sid, role="assistant", send_index=1,
        tool_calls=[{"id": "tc-y", "function": {"name": "echo", "arguments": '{"text":"resumed"}'}}],
        round_index=0,
    )
    await store.set_pending(sid, {
        "assistant_seq": asst_seq, "round_index": 0, "tool_call_ids": ["tc-y"],
        "tool_calls": [{"id": "tc-y", "function": {"name": "echo", "arguments": '{"text":"resumed"}'}}],
    })

    r = await loop.resume(sid)
    assert r.status == "completed"
    assert r.final_text == "after resume"

    # The replayed result reaches the LLM (paired), not dropped as a legacy-prefix orphan.
    req = llm.calls[0]
    tool_msgs = [m for m in req if m.get("role") == "tool" and m.get("tool_call_id") == "tc-y"]
    assert tool_msgs and tool_msgs[0].get("content") == "resumed"
    tool_rows = [r for r in await store.load_active_messages(sid) if r.role == "tool"]
    assert tool_rows and all(row.send_index == 1 for row in tool_rows)


@pytest.mark.asyncio
async def test_projection_abort_pending_stamps_send_index(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="recovered")])
    loop = _projection_loop_with_tools(store, llm, _echo_registry())
    sid = await store.create_session()
    await store.mutate_runtime_state(sid, "send_index", lambda v: 1, default=0)
    await store.append_message(sid, role="user", content="kick off", send_index=1)
    seq = await store.append_message(
        sid, role="assistant", send_index=1,
        tool_calls=[{"id": "tc-x", "function": {"name": "echo", "arguments": "{}"}}],
        round_index=0,
    )
    await store.set_pending(sid, {
        "assistant_seq": seq, "round_index": 0, "tool_call_ids": ["tc-x"],
        "tool_calls": [{"id": "tc-x", "function": {"name": "echo", "arguments": "{}"}}],
    })

    aborted = await loop.abort_pending(sid, reason="user_cancelled")
    assert aborted == 1
    # The <aborted> row is stamped with the pending send's index (paired, not orphaned in projection).
    tool_rows = [r for r in await store.load_active_messages(sid) if r.role == "tool"]
    assert tool_rows and all(r.send_index == 1 for r in tool_rows)
    assert "<aborted: user_cancelled>" in str(tool_rows[0].content)


@pytest.mark.asyncio
async def test_default_mode_writes_no_projections(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="hi")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )  # no projector → default (verbatim + compactor); zero behavior change
    sid = await loop.new_session()
    await loop.send("hi", session_id=sid)
    assert await store.load_project_messages(sid) == []


@pytest.mark.asyncio
async def test_identity_projector_matches_verbatim(store: SessionStore) -> None:
    base_llm, ident_llm = _two_send_script(), _two_send_script()
    base = StatefulAgentLoop(
        llm=base_llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
    )
    ident = _projector_loop(store, ident_llm, IdentityProjector())
    sb, si = await base.new_session(), await ident.new_session()

    await base.send("first", session_id=sb)
    await ident.send("first", session_id=si)
    nb, ni = len(base_llm.calls), len(ident_llm.calls)
    await base.send("second", session_id=sb)
    await ident.send("second", session_id=si)
    # IdentityProjector reproduces verbatim history → send 2's request is byte-identical.
    assert ident_llm.calls[ni] == base_llm.calls[nb]


@pytest.mark.asyncio
async def test_child_session_not_projected(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="child done")])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    parent = await store.create_session()
    child = await store.create_session(
        parent_session_id=parent, kind=SessionKind.SUBAGENT,
        lifecycle=SubagentLifecycle.LINKED,
    )
    await loop.send("go", session_id=child)
    # The writer skips child sub-agent sessions (by parent_session_id, not scope).
    assert await store.load_project_messages(child) == []


@pytest.mark.asyncio
async def test_waiting_for_input_defers_projection(store: SessionStore) -> None:
    llm = _Scripted(responses=[_tool_resp("c1", "request_user_input", '{"prompt": "ok?"}')])
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        tool_registry=create_default_tool_registry(include=["request_user_input"]),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=3, compactor=None,
            history_projector=DefaultDeterministicProjector(),
        ),
    )
    sid = await loop.new_session()
    r = await loop.send("need approval", session_id=sid)
    assert r.status == "waiting_for_input"
    # Send is mid-flight (resumes under the same send_index) → projection deferred, not written.
    assert await store.load_project_messages(sid) == []


@pytest.mark.asyncio
async def test_projection_compaction_keeps_last_n(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)])
    # tiny max_tokens so the token-driven fold trigger fires on these short sends
    loop = _projector_loop(store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = await loop.new_session()
    for i in range(1, 4):  # 3 sends → send 1 folded (keep last 2)
        await loop.send(f"m{i}", session_id=sid)
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 1
    assert "d1" in compact.content["summary"] and "m1" in compact.content["summary"]
    # folded user/project rows REMAIN (append-only; recoverable)
    rows = await store.load_project_messages(sid)
    assert any(r.send_index == 1 and r.kind == "project" for r in rows)
    # the next send's reader window = compact + sends 2,3
    after = await store.load_project_messages(sid, after_send_index=compact.compact_to_send)
    assert sorted({r.send_index for r in after if r.kind == "project"}) == [2, 3]


@pytest.mark.asyncio
async def test_projection_compaction_rolls_prior_forward(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 7)])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = await loop.new_session()
    for i in range(1, 5):  # 4 sends → compact now spans 1-2, nothing lost
        await loop.send(f"m{i}", session_id=sid)
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 2
    assert "d1" in compact.content["summary"] and "d2" in compact.content["summary"]


class _CountingMem:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"content": "a persistent note"}]

    async def remember(self, *, snapshot, session_id):
        return None


def _mem_projector_loop(store, *, memory, memory_budget_tokens, max_tokens):
    return StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)]),
        store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4,
            representation=DefaultDeterministicProjector(keep_last_sends=2),
            fold_strategy=_DeterministicFold(keep_last_sends=2, trigger_ratio=0.75),
            max_tokens=max_tokens, memory=memory, memory_budget_tokens=memory_budget_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_memory_headroom_lowers_projection_fold_trigger(store: SessionStore) -> None:
    """P0: the projection fold trigger uses config.effective_context_budget(), so
    configuring memory (which reserves memory_budget_tokens) LOWERS the threshold
    and makes folding fire earlier on identical sends. Guards stateful_loop.py
    against a regression that reverts the trigger to raw max_tokens."""
    # No memory: threshold = 200 * 0.75 = 150 → 3 tiny sends stay under → NO fold.
    loop_a = _mem_projector_loop(store, memory=None, memory_budget_tokens=0, max_tokens=200)
    sid_a = await loop_a.new_session()
    for i in range(1, 4):
        await loop_a.send(f"m{i}", session_id=sid_a)
    assert await store.latest_project_compact(sid_a) is None  # no headroom reserved → no fold

    # Memory + large budget: effective = max(1, 200-190) = 10 → threshold ≈ 7 →
    # the SAME 3 sends now exceed it → fold fires. (memory is injected at the LLM
    # tail, not the projection snapshot, so only the THRESHOLD changes.)
    store_b = await SessionStore.open(":memory:")
    try:
        loop_b = _mem_projector_loop(store_b, memory=_CountingMem(), memory_budget_tokens=190, max_tokens=200)
        sid_b = await loop_b.new_session()
        for i in range(1, 4):
            await loop_b.send(f"m{i}", session_id=sid_b)
        assert await store_b.latest_project_compact(sid_b) is not None  # headroom → folds earlier
    finally:
        await store_b.close()


@pytest.mark.asyncio
async def test_identity_projector_never_compacts(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)])
    loop = _projector_loop(store, llm, IdentityProjector())
    sid = await loop.new_session()
    for i in range(1, 5):
        await loop.send(f"m{i}", session_id=sid)
    assert await store.latest_project_compact(sid) is None  # keep_last_sends=0 → never folds


@pytest.mark.asyncio
async def test_legacy_null_send_index_rows_rendered_verbatim(store: SessionStore) -> None:
    # Rows predating the projection era carry send_index=NULL (pre-v2 migration / export-import).
    # Enabling projection must NOT silently drop them — they render VERBATIM and temporally BEFORE
    # the projected/in-flight sends (regression for the NULL-send_index silent-drop bug).
    llm = _Scripted(responses=[LLMResponse(raw_text="reply1"), LLMResponse(raw_text="reply2")])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()
    await store.append_message(sid, role="user", content="LEGACY-USER")      # send_index defaults NULL
    await store.append_message(sid, role="assistant", content="LEGACY-REPLY")

    await loop.send("first", session_id=sid)
    n = len(llm.calls)
    await loop.send("second", session_id=sid)
    contents = [str(m.get("content", "")) for m in llm.calls[n]]
    joined = "\n".join(contents)
    assert "LEGACY-USER" in joined and "LEGACY-REPLY" in joined  # not dropped
    assert "first" in joined and "reply1" in joined and "second" in joined  # projected + in-flight
    # temporal order: legacy verbatim < send-1 projection < send-2 in-flight user
    i_legacy = next(i for i, c in enumerate(contents) if "LEGACY-USER" in c)
    i_first = next(i for i, c in enumerate(contents) if "first" in c)
    i_second = next(i for i, c in enumerate(contents) if "second" in c)
    assert i_legacy < i_first < i_second


@pytest.mark.asyncio
async def test_legacy_null_rows_not_re_dropped_across_sends(store: SessionStore) -> None:
    # Across MULTIPLE projected sends the legacy rows must persist in context every send (the bug
    # dropped them the moment current_send_index advanced past 1).
    llm = _Scripted(responses=[LLMResponse(raw_text=f"r{i}") for i in range(1, 5)])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()
    await store.append_message(sid, role="user", content="LEGACY-X")
    for i in range(1, 4):
        n = len(llm.calls)
        await loop.send(f"m{i}", session_id=sid)
        assert "LEGACY-X" in "\n".join(str(m.get("content", "")) for m in llm.calls[n])


@pytest.mark.asyncio
async def test_two_loops_sharing_store_project_consistently(store: SessionStore) -> None:
    # The locked write path keeps ONE coherent projection when two loops share a store on the same
    # session: every send projected exactly once, no duplicate (send_index,kind) rows, one compact
    # (regression for #5/#8 — sequential here so it stays deterministic).
    p = DefaultDeterministicProjector(keep_last_sends=2)
    llm_a = _Scripted(responses=[LLMResponse(raw_text=f"a{i}") for i in range(1, 4)])
    llm_b = _Scripted(responses=[LLMResponse(raw_text=f"b{i}") for i in range(1, 4)])
    loop_a = _projector_loop(store, llm_a, p, max_tokens=10)
    loop_b = StatefulAgentLoop(
        llm=llm_b, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4, representation=p,
                               fold_strategy=_DeterministicFold(keep_last_sends=2), max_tokens=10),
    )
    sid = await loop_a.new_session()
    await loop_a.send("m1", session_id=sid)
    await loop_b.send("m2", session_id=sid)  # different loop, same shared store + session
    await loop_a.send("m3", session_id=sid)
    await loop_b.send("m4", session_id=sid)

    rows = await store.load_project_messages(sid)
    assert sorted(r.send_index for r in rows if r.kind == "project") == [1, 2, 3, 4]  # once each
    keys = [(r.send_index, r.kind) for r in rows]
    assert len(keys) == len(set(keys))  # no duplicate/clobbered (send_index, kind) rows
    assert await store.latest_project_compact(sid) is not None  # folded into one coherent compact


def test_config_legacy_kwargs_map_onto_axes() -> None:
    # 3.0: the old projector⊻compactor exclusion is GONE. Legacy history_projector/compactor map
    # onto representation × fold_strategy (deprecated, warned). A legacy projector → the projection
    # representation; a legacy compactor=None (no in-place fold) is preserved exactly.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AgentLoopConfig(history_projector=DefaultDeterministicProjector(), compactor=None)
    assert cfg.projection_representation is not None  # projector → projection representation
    assert cfg.resolve_compactor() is None            # compactor=None preserved (no in-place fold)


def test_config_max_tokens_revalidates_on_reassignment() -> None:
    # max_tokens drives the fold trigger, so a non-positive reassignment is rejected + rolled back.
    cfg = AgentLoopConfig()
    assert cfg.representation is not None and cfg.fold_strategy is not None
    with pytest.raises(ValueError):
        cfg.max_tokens = 0
    assert cfg.max_tokens != 0  # rolled back to the prior valid value
    cfg.max_tokens = 4242
    assert cfg.max_tokens == 4242


@pytest.mark.asyncio
async def test_resume_before_any_send_degrades_gracefully(store: SessionStore) -> None:
    # resume() on a projector-enabled session that never completed a send() has no allocated
    # send_index. New behavior: NEVER throw — degrade to verbatim rendering and run best-effort.
    llm = _Scripted(responses=[LLMResponse(raw_text="x")])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()
    r = await loop.resume(sid)  # must NOT raise
    assert r.status == "completed"
    assert await store.load_project_messages(sid) == []  # degraded → not projected


@pytest.mark.asyncio
async def test_projection_compact_larger_fold_span(store: SessionStore) -> None:
    # Over many sends the compact rolls prior folds forward and never loses content: every send is
    # either inside the compact range or still has a live project row.
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 11)])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector(keep_last_sends=2), max_tokens=10)
    sid = await loop.new_session()
    for i in range(1, 8):  # 7 sends
        await loop.send(f"m{i}", session_id=sid)
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.compact_from_send == 1
    assert "d1" in compact.content["summary"] and "d2" in compact.content["summary"]  # prior preserved
    live = {r.send_index for r in await store.load_project_messages(sid)
            if r.kind == "project" and r.send_index > compact.compact_to_send}
    covered = set(range(1, compact.compact_to_send + 1)) | live
    assert covered.issuperset(set(range(1, 8)))  # nothing lost


@dataclass
class _StubMemory:
    text: str

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"content": self.text}]

    async def remember(self, *, snapshot, session_id):
        return None


@pytest.mark.asyncio
async def test_projection_composes_with_memory_recall(store: SessionStore) -> None:
    llm = _two_send_script()
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4, compactor=None,
            history_projector=DefaultDeterministicProjector(),
            memory=_StubMemory(text="MEMORY: user prefers terse replies"),
        ),
    )
    sid = await loop.new_session()
    await loop.send("first", session_id=sid)
    n = len(llm.calls)
    r = await loop.send("second", session_id=sid)
    assert r.status == "completed"  # no crash: recall front-insert + projected history coexist
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[n])
    assert "MEMORY: user prefers terse replies" in joined   # recalled memory present
    assert "done1" in joined and "second" in joined          # projected past + current send


# ── pre-release hardening: reader verbatim fallback + fold-failure degradation + config ──


@pytest.mark.asyncio
async def test_missing_projection_row_falls_back_to_verbatim(store: SessionStore) -> None:
    # A best-effort end-of-send projection write can fail/crash, leaving a past send with rows in
    # pl_messages but none in pl_project_messages. The reader must render that send VERBATIM from
    # the audit log, not silently drop it from context forever.
    llm = _two_send_script()  # send 1 has a tool call → verbatim carries tool_calls
    # migrate=False so we exercise the per-send VERBATIM FALLBACK path specifically (with migration
    # on, an empty projection table would instead trigger the one-time migration).
    loop = _projector_loop(store, llm, DefaultDeterministicProjector(), migrate=False)
    sid = await loop.new_session()
    await loop.send("first", session_id=sid)
    assert await store.load_project_messages(sid)  # send 1 was projected…
    # …simulate the projection write having failed for send 1: drop its derived rows.
    await store._db.execute(
        f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=?", (sid, 1)
    )
    assert not await store.load_project_messages(sid)

    n = len(llm.calls)
    await loop.send("second", session_id=sid)
    req = llm.calls[n]
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined  # send 1 recovered (NOT dropped)
    # recovered VERBATIM (the audit rows incl. tool protocol), proving the fallback fired —
    # the projection path emits plain text only.
    assert any("tool_calls" in m for m in req)


@pytest.mark.asyncio
async def test_stale_projector_version_falls_back_to_verbatim(store: SessionStore) -> None:
    # Bumping the projector version between runs must not silently mis-render rows written by the
    # old version: a past send whose projection rows carry a different version renders verbatim.
    loop1 = _projector_loop(store, _two_send_script(), DefaultDeterministicProjector(version=1))
    sid = await loop1.new_session()
    await loop1.send("first", session_id=sid)
    assert {r.projector_version for r in await store.load_project_messages(sid)} == {1}

    # A new run with a version-2 projector on the SAME session.
    llm2 = _Scripted(responses=[LLMResponse(raw_text="done2")])
    loop2 = _projector_loop(store, llm2, DefaultDeterministicProjector(version=2))
    await loop2.send("second", session_id=sid)
    req = llm2.calls[0]
    joined = "\n".join(str(m.get("content", "")) for m in req)
    assert "first" in joined and "done1" in joined  # send 1 still present…
    assert any("tool_calls" in m for m in req)  # …rendered verbatim (version mismatch → fallback)


class _BoomCompactProjector(DefaultDeterministicProjector):
    def compact(self, rows):  # type: ignore[override]
        raise RuntimeError("boom in compact")


@pytest.mark.asyncio
async def test_compact_failure_degrades_to_no_fold(store: SessionStore) -> None:
    # A projector whose compact() raises must NOT abort the whole locked write: the per-send rows
    # still commit (so nothing is lost), only the fold is skipped.
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 4)])
    # keep_last_sends=1 + max_tokens=1 (threshold 0) → the fold triggers every send → compact() raises.
    loop = _projector_loop(store, llm, _BoomCompactProjector(keep_last_sends=1), max_tokens=1)
    sid = await loop.new_session()
    statuses = [(await loop.send(f"m{i}", session_id=sid)).status for i in range(1, 4)]

    assert statuses == ["completed", "completed", "completed"]  # no crash despite compact() raising
    assert await store.latest_project_compact(sid) is None  # fold skipped (degraded)
    proj_sends = {r.send_index for r in await store.load_project_messages(sid) if r.kind == "project"}
    assert proj_sends == {1, 2, 3}  # every send's rows committed (not lost with the failed fold)


@pytest.mark.asyncio
async def test_switch_default_to_projection_migrates_prior_history(store: SessionStore) -> None:
    # A session created WITHOUT a projector allocates send_index but writes NO projection rows.
    # Re-opening it WITH a projector (migration on by default) MIGRATES the prior history into the
    # projection table once, so the prior send is rendered as PROJECTION (plain text), not verbatim.
    base = StatefulAgentLoop(
        llm=_two_send_script(), store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),  # default: no projector
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)
    assert await store.load_project_messages(sid) == []  # default mode wrote zero projection rows

    proj_llm = _Scripted(responses=[LLMResponse(raw_text="done2")])
    proj = _projector_loop(store, proj_llm, DefaultDeterministicProjector())  # switch (migrate on)
    await proj.send("second", session_id=sid)
    joined = "\n".join(str(m.get("content", "")) for m in proj_llm.calls[0])
    assert "first" in joined and "done1" in joined  # prior send present…
    assert not any("tool_calls" in m for m in proj_llm.calls[0])  # …as PROJECTION (migrated, plain text)
    assert {r.send_index for r in await store.load_project_messages(sid)} == {1, 2}  # 1 migrated, 2 forward
    assert (await store.get_session(sid)).metadata.get("projection_migrated") is not None


@pytest.mark.asyncio
async def test_switch_default_to_projection_verbatim_when_migration_off(store: SessionStore) -> None:
    # With migration disabled, the prior send is rendered VERBATIM from pl_messages (the fallback),
    # and only new sends project forward.
    base = StatefulAgentLoop(
        llm=_two_send_script(), store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)

    proj_llm = _Scripted(responses=[LLMResponse(raw_text="done2")])
    proj = _projector_loop(store, proj_llm, DefaultDeterministicProjector(), migrate=False)
    await proj.send("second", session_id=sid)
    joined = "\n".join(str(m.get("content", "")) for m in proj_llm.calls[0])
    assert "first" in joined and "done1" in joined
    assert any("tool_calls" in m for m in proj_llm.calls[0])  # verbatim (not migrated)
    assert {r.send_index for r in await store.load_project_messages(sid)} == {2}  # only send 2 projected


@pytest.mark.asyncio
async def test_compactor_history_migrates_seeds_compact_from_note(store: SessionStore) -> None:
    # default → projection when compaction HAD fired (compact_note rows). Migration (on by default)
    # SEEDS a projection compact from the in-place note summary, so the session becomes
    # projection-native and the note content still reaches the prompt — never throws.
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()
    await store.append_message(sid, role="system", content="folded summary", name="compact_note")
    r = await loop.send("hi", session_id=sid)  # must NOT raise
    assert r.status == "completed"
    proj = await store.load_project_messages(sid)
    assert any(
        p.kind == "compact" and "folded summary" in (p.content or {}).get("summary", "") for p in proj
    )  # the note seeded a projection compact
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[0])
    assert "folded summary" in joined  # note content reached the prompt (via the projection compact)
    assert (await store.get_session(sid)).metadata.get("projection_migrated") is not None


@pytest.mark.asyncio
async def test_compactor_history_degrades_when_migration_off(store: SessionStore) -> None:
    # With migration disabled, a compact_note session can't be partitioned → degrade to verbatim
    # (default-style) rendering, render the note, and skip projecting this send. Never throws.
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector(), migrate=False)
    sid = await loop.new_session()
    await store.append_message(sid, role="system", content="folded summary", name="compact_note")
    r = await loop.send("hi", session_id=sid)
    assert r.status == "completed"
    assert await store.load_project_messages(sid) == []  # degraded → not projected
    joined = "\n".join(str(m.get("content", "")) for m in llm.calls[0])
    assert "folded summary" in joined  # note reached the prompt verbatim


@pytest.mark.asyncio
async def test_migration_folds_old_keeps_recent(store: SessionStore) -> None:
    # default session with several sends → switch to projection with keep_last_sends=1: migration
    # folds the OLD sends into one compact and keeps the most-recent one as project rows.
    base = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 4)]),
        store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = await base.new_session()
    for i in range(1, 4):
        await base.send(f"m{i}", session_id=sid)  # sends 1,2,3 in default mode (no projection rows)

    proj = _projector_loop(store, _Scripted(responses=[LLMResponse(raw_text="d4")]),
                           DefaultDeterministicProjector(keep_last_sends=1))
    await proj.send("m4", session_id=sid)  # send 4 → migration folds 1,2 into a compact, keeps 3

    proj_rows = await store.load_project_messages(sid)
    compacts = [p for p in proj_rows if p.kind == "compact"]
    assert len(compacts) == 1 and compacts[0].compact_from_send == 1 and compacts[0].compact_to_send == 2
    live = {p.send_index for p in proj_rows if p.kind in ("user", "project")}
    assert live == {3, 4}  # send 3 kept (migrated), send 4 projected forward


@pytest.mark.asyncio
async def test_migration_runs_once(store: SessionStore) -> None:
    base = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text="a")]), store=store,
        tool_registry=_echo_registry(), config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)
    proj = _projector_loop(store, _Scripted(responses=[LLMResponse(raw_text="b"), LLMResponse(raw_text="c")]),
                           DefaultDeterministicProjector())
    await proj.send("second", session_id=sid)  # migrates send 1
    flag = (await store.get_session(sid)).metadata.get("projection_migrated")
    assert flag is not None
    rows_after = sorted((p.send_index, p.kind) for p in await store.load_project_messages(sid))
    await proj.send("third", session_id=sid)  # must NOT re-migrate
    assert (await store.get_session(sid)).metadata.get("projection_migrated") == flag  # unchanged
    # send 3 projected forward; sends 1,2 unchanged (no duplicate/re-fold of the migrated prefix)
    rows_now = sorted((p.send_index, p.kind) for p in await store.load_project_messages(sid))
    assert set(rows_after).issubset(set(rows_now)) and (3, "project") in rows_now


@pytest.mark.asyncio
async def test_switch_projection_to_default_is_safe(store: SessionStore) -> None:
    # Inverse switch (projection → default): projection NEVER marks pl_messages rows inactive, so
    # reopening in default mode sees the FULL verbatim history (nothing lost); the now-stale
    # projection rows are simply ignored.
    proj = _projector_loop(store, _two_send_script(), DefaultDeterministicProjector())
    sid = await proj.new_session()
    await proj.send("first", session_id=sid)
    assert await store.load_project_messages(sid)  # projection rows exist

    base_llm = _Scripted(responses=[LLMResponse(raw_text="done2")])
    base = StatefulAgentLoop(
        llm=base_llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),  # default: no projector
    )
    r = await base.send("second", session_id=sid)
    assert r.status == "completed"  # no crash
    joined = "\n".join(str(m.get("content", "")) for m in base_llm.calls[0])
    assert "first" in joined and "done1" in joined  # full verbatim history (nothing lost)


@pytest.mark.asyncio
async def test_orphan_tool_row_sanitized_from_prompt(store: SessionStore) -> None:
    # A corrupt orphan tool-result row in pl_messages must NOT reach the provider (it would 400 and
    # brick the session forever). The always-on backstop drops it from the assembled prompt; with
    # repair off (default) the audit row itself is left intact.
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=2),  # default mode
    )
    sid = await loop.new_session()
    await store.append_message(sid, role="tool", tool_call_id="ghost", content="orphan result")
    r = await loop.send("hi", session_id=sid)
    assert r.status == "completed"  # not bricked
    assert not any(m.get("tool_call_id") == "ghost" for m in llm.calls[0])  # orphan never sent
    assert any(m.tool_call_id == "ghost" for m in await store.load_all_messages(sid))  # audit intact


@pytest.mark.asyncio
async def test_repair_corrupt_history_deactivates_orphan_row(store: SessionStore) -> None:
    # With repair_corrupt_history=True the dropped orphan row is durably deactivated (state=DROPPED):
    # gone from the active history, still present in the full audit (repaired, not deleted).
    llm = _Scripted(responses=[LLMResponse(raw_text="ok")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=2, repair_corrupt_history=True),
    )
    sid = await loop.new_session()
    await store.append_message(sid, role="tool", tool_call_id="ghost", content="orphan")
    await loop.send("hi", session_id=sid)
    assert not any(m.tool_call_id == "ghost" for m in await store.load_active_messages(sid))  # deactivated
    assert any(m.tool_call_id == "ghost" for m in await store.load_all_messages(sid))  # audit kept


@pytest.mark.asyncio
async def test_history_mode_recorded_and_switch_warns(store: SessionStore, caplog) -> None:
    # The ORIGINAL mode is recorded once; a later switch logs a warning (never raises) and does not
    # overwrite the original — so it stays available for inspection/comparison.
    import logging

    base = StatefulAgentLoop(
        llm=_Scripted(responses=[LLMResponse(raw_text="a")]), store=store,
        tool_registry=_echo_registry(), config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)
    meta = (await store.get_session(sid)).metadata
    assert meta["history_mode"] == "default"  # original recorded in SESSION METADATA
    assert meta["projector_version"] is None  # default mode → no projector config

    proj = _projector_loop(store, _Scripted(responses=[LLMResponse(raw_text="b")]),
                           DefaultDeterministicProjector())
    with caplog.at_level(logging.WARNING):
        await proj.send("second", session_id=sid)
    assert "originally" in caplog.text and "default" in caplog.text  # switch warned
    assert (await store.get_session(sid)).metadata["history_mode"] == "default"  # NOT overwritten


@pytest.mark.asyncio
async def test_projection_mode_records_projector_config_in_metadata(store: SessionStore) -> None:
    # Projection sessions stamp the full projector config in metadata (for inspection + comparison).
    proj = _projector_loop(store, _two_send_script(),
                           DefaultDeterministicProjector(version=3, trigger_ratio=0.5, keep_last_sends=2))
    sid = await proj.new_session()
    await proj.send("first", session_id=sid)
    meta = (await store.get_session(sid)).metadata
    assert meta["history_mode"] == "projection"
    assert meta["projector_version"] == 3
    assert meta["projector_trigger_ratio"] == 0.5
    assert meta["projector_keep_last_sends"] == 2


@pytest.mark.asyncio
async def test_migration_failure_falls_back_to_verbatim(store: SessionStore, monkeypatch) -> None:
    # If the one-time migration raises, the run must NOT crash: it falls back to the per-send
    # verbatim rendering, leaves the marker unset (retry-able), and still projects new sends.
    base = StatefulAgentLoop(
        llm=_two_send_script(), store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)

    proj_llm = _Scripted(responses=[LLMResponse(raw_text="done2")])
    proj = _projector_loop(store, proj_llm, DefaultDeterministicProjector())

    async def _boom(*a, **k):
        raise RuntimeError("migration boom")

    monkeypatch.setattr(proj, "_migrate_prior_history_to_projection", _boom)
    r = await proj.send("second", session_id=sid)
    assert r.status == "completed"  # migration failure did not crash the send
    joined = "\n".join(str(m.get("content", "")) for m in proj_llm.calls[0])
    assert "first" in joined and any("tool_calls" in m for m in proj_llm.calls[0])  # verbatim fallback
    assert (await store.get_session(sid)).metadata.get("projection_migrated") is None  # not marked
    assert 2 in {r.send_index for r in await store.load_project_messages(sid)}  # send 2 still projects


@pytest.mark.asyncio
async def test_migration_skipped_when_projection_rows_already_exist(store: SessionStore) -> None:
    # If the projection table is already non-empty (a prior migration/projection, marker lost to a
    # crash), the reader must NOT re-migrate — it marks resolved and leaves the existing rows alone.
    base = StatefulAgentLoop(
        llm=_two_send_script(), store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
    )
    sid = await base.new_session()
    await base.send("first", session_id=sid)
    # Simulate a pre-existing projection row for the prior send (no marker set).
    await store.upsert_project_message(
        sid, send_index=1, kind="project", content={"tools": [], "final_text": "x"}, projector_version=1
    )
    proj = _projector_loop(store, _Scripted(responses=[LLMResponse(raw_text="done2")]),
                           DefaultDeterministicProjector())
    await proj.send("second", session_id=sid)
    assert (await store.get_session(sid)).metadata.get("projection_migrated") == 0  # resolved, not migrated
    assert not any(p.kind == "compact" for p in await store.load_project_messages(sid))  # no migration fold


def test_config_rejects_nonpositive_max_tokens_with_projector() -> None:
    for bad in (0, -10):
        with pytest.raises(ValueError, match="max_tokens"):
            AgentLoopConfig(
                history_projector=DefaultDeterministicProjector(), compactor=None, max_tokens=bad
            )
    # post-construction reassignment is re-validated AND rolled back on failure
    cfg = AgentLoopConfig(
        history_projector=DefaultDeterministicProjector(), compactor=None, max_tokens=8000
    )
    with pytest.raises(ValueError, match="max_tokens"):
        cfg.max_tokens = 0
    assert cfg.max_tokens == 8000  # rolled back, config still valid
