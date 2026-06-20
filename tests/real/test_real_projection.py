"""Real-LLM end-to-end for the send-context projection layer (v2).

Runs several real sends with a DefaultDeterministicProjector (compactor disabled): the live
provider must ACCEPT the projected plain-text history on later sends, finished sends get
projected into pl_project_messages, older sends fold into a compact row, and recall_send
re-expands a folded send's original detail (which always remains in pl_messages).

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
from power_loop.runtime.history_projector import DefaultDeterministicProjector
from power_loop.runtime.store.store import SessionStore
from power_loop.tools.default_tools import run_recall_send
from tests.real._llm import make_llm

pytestmark = pytest.mark.asyncio


async def test_real_projection_compaction_and_recall() -> None:
    llm = make_llm(max_tokens=64, temperature=0.0)
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm, store=store,
            config=AgentLoopConfig(
                system_prompt="You are terse. Reply with one short sentence.",
                max_rounds=3, compactor=None,
                # small max_tokens so the token-driven fold (max_tokens × trigger_ratio) fires
                max_tokens=40,
                history_projector=DefaultDeterministicProjector(keep_last_sends=2),
            ),
        )
        sid = await loop.new_session()
        # 4 real sends. Sends 2-4 are assembled with PROJECTED plain-text history (not verbatim);
        # the live provider accepting them (status=completed) is the core validation.
        for i in range(1, 5):
            r = await loop.send(f"Say the word apple{i} back to me.", session_id=sid)
            assert r.status == "completed", f"send {i} status={r.status}"

        # Each finished send was projected (user + project rows).
        proj = await store.load_project_messages(sid)
        assert {r.kind for r in proj} >= {"user", "project"}

        # keep_last_sends=2 over 4 sends → older sends folded into a compact row spanning 1-2.
        compact = await store.latest_project_compact(sid)
        assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 2

        # recall_send re-expands a FOLDED send — its original detail is still in pl_messages.
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
