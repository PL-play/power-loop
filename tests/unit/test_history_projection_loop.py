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


def _two_send_script() -> _Scripted:
    # send 1: echo tool then "done1"; send 2: "done2"
    return _Scripted(responses=[
        _tool_resp("c1", "echo", '{"text": "a"}'),
        LLMResponse(raw_text="done1"),
        LLMResponse(raw_text="done2"),
    ])


def _projector_loop(
    store: SessionStore, llm: _Scripted, projector, *, max_tokens: int = 8000
) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=4, compactor=None, history_projector=projector,
            max_tokens=max_tokens,
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


@pytest.mark.asyncio
async def test_identity_projector_never_compacts(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 6)])
    loop = _projector_loop(store, llm, IdentityProjector())
    sid = await loop.new_session()
    for i in range(1, 5):
        await loop.send(f"m{i}", session_id=sid)
    assert await store.latest_project_compact(sid) is None  # keep_last_sends=0 → never folds


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
