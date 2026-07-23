"""3.23 — max_context_rows: projection-mode recent-rows context cap.

History used to run from the fold compact to the newest message. With the cap, the assembled
history keeps ≤N rows: the compact (if any) is ALWAYS kept, the in-flight send is kept in full,
and older material drops in whole chunks from the oldest end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.fold import LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_stateful_loop import _echo_registry, _Scripted


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


def _loop(store, llm, *, cap, fold=None):
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, representation=ProjectedRepresentation(),
            max_context_rows=cap,
            # a lazy fold that never triggers on these tiny sends unless stated otherwise
            fold_strategy=fold or LLMSummaryFold(keep_last_sends=2, trigger_ratio=0.99),
            max_tokens=100_000,
        ),
    )


@pytest.mark.asyncio
async def test_cap_drops_oldest_sends_keeps_newest(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 9)])
    loop = _loop(store, llm, cap=5)
    sid = await loop.new_session()
    for i in range(1, 8):
        await loop.send(f"u{i}", session_id=sid)
    # The messages of the 7th send's LLM call: projected sends are 2 rows each (user+project);
    # cap=5 → at most 5 non-current rows + the current user turn.
    msgs = llm.calls[-1]
    non_system = [m for m in msgs if m.get("role") != "system"]
    assert len(non_system) <= 5 + 1
    joined = "\n".join(str(m.get("content")) for m in non_system)
    assert "u6" in joined            # newest past send survives
    assert "u1" not in joined        # oldest send dropped by the cap
    assert "u7" in joined            # current send always present


@pytest.mark.asyncio
async def test_cap_disabled_keeps_everything(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text=f"d{i}") for i in range(1, 9)])
    loop = _loop(store, llm, cap=None)
    sid = await loop.new_session()
    for i in range(1, 8):
        await loop.send(f"u{i}", session_id=sid)
    joined = "\n".join(str(m.get("content")) for m in llm.calls[-1])
    assert "u1" in joined and "u6" in joined


@pytest.mark.asyncio
async def test_cap_never_drops_the_fold_compact(store: SessionStore) -> None:
    # Aggressive fold: after a few sends the older ones live in the compact summary. The cap must
    # keep that compact row even while dropping uncompacted old sends.
    llm = _Scripted(responses=[
        LLMResponse(raw_text="d1"), LLMResponse(raw_text="d2"), LLMResponse(raw_text="d3"),
        LLMResponse(raw_text="<summary>FOLD-KEEP</summary>"),
        LLMResponse(raw_text="d4"), LLMResponse(raw_text="d5"),
        LLMResponse(raw_text="d6"), LLMResponse(raw_text="d7"),
    ])
    loop = _loop(store, llm, cap=3, fold=LLMSummaryFold(keep_last_sends=2, trigger_ratio=0.0001))
    sid = await loop.new_session()
    for i in range(1, 7):
        await loop.send(f"u{i}", session_id=sid)
    # llm.calls interleaves MAIN sends with the fold's own summarizer calls — pick the last MAIN one.
    main_calls = [
        c for c in llm.calls
        if not str((c[0] or {}).get("content", "")).startswith("You are a conversation summarizer")
    ]
    joined = "\n".join(str(m.get("content")) for m in main_calls[-1])
    assert "folded — recall_send" in joined   # the compact row survives the cap
    assert "u1" not in joined                 # while pre-compact content itself is gone from live rows
