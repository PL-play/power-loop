"""Real-LLM end-to-end for the send-context projection layer (v2).

Runs several real sends with a DefaultDeterministicProjector: the live
provider must ACCEPT the projected plain-text history on later sends, finished sends get
projected into pl_project_messages, older sends fold into a compact row, and recall_send
re-expands a folded send's original detail (which always remains in pl_messages).

Auto-skips without a real-LLM provider env (see tests/conftest.py).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop.agent.types import AgentLoopConfig
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.fold import _SUMMARY_PROMPT, LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.store import SessionStore
from power_loop.tools.default_tools import run_recall_send
from tests.real._llm import make_llm

pytestmark = pytest.mark.asyncio


def _assert_compact_1_2(compact):
    assert compact is not None and compact.compact_from_send == 1 and compact.compact_to_send == 2
    assert str(compact.content.get("summary") or "").strip()


async def test_real_projection_compaction_and_recall(runtime_agent, monkeypatch) -> None:
    llm = make_llm(max_tokens=2048, temperature=0.0)
    store, loop = None, None
    observed = {"business_calls": 0, "fold_calls": 0, "calls": [],
                "second_business_messages": None, "send_results": []}
    active_send = 0
    original_complete = llm.complete

    async def complete(request, **kwargs):
        fold = bool(request.messages and str(request.messages[0].get("content", "")).startswith(_SUMMARY_PROMPT))
        observed["fold_calls" if fold else "business_calls"] += 1
        if not fold and active_send == 2 and observed["second_business_messages"] is None:
            observed["second_business_messages"] = copy.deepcopy(request.messages)
        response = await original_complete(request, **kwargs)
        usage = response.token_usage
        observed["calls"].append({"kind": "fold" if fold else "business", "send": active_send,
                                  "usage": {name: getattr(usage, name, 0) for name in (
                                      "prompt_tokens", "completion_tokens", "total_tokens")},
                                  "response_nonempty": bool((response.raw_text or "").strip())})
        return response

    monkeypatch.setattr(llm, "complete", complete)
    try:
        store = await SessionStore.open(":memory:")
        projector = ProjectedRepresentation()
        loop = await runtime_agent(
            model=llm, store=store,
            config=AgentLoopConfig(
                system_prompt="You are terse. Reply with one short sentence.",
                max_rounds=3,
                # tiny FOLD budget so the token-driven fold (budget × trigger_ratio) fires; the
                # output cap stays roomy — a reasoning model needs it before its first word.
                max_tokens=2048, context_budget_tokens=40,
                representation=projector,
                fold_strategy=LLMSummaryFold(keep_last_sends=2),
            ),
        )
        sid = loop.target.session_id
        # 4 real sends. Sends 2-4 are assembled with PROJECTED plain-text history (not verbatim);
        # the live provider accepting them (status=completed) is the core validation.
        for i in range(1, 5):
            active_send = i
            r = await loop.send(f"Say the word apple{i} back to me.")
            observed["send_results"].append({"send": i, "status": r.status, "usage": dict(r.usage)})
            assert r.status == "completed", f"send {i} status={r.status}"
            if i == 1:
                first_projection = [row for row in await store.load_project_messages(sid)
                                    if row.send_index == 1 and row.kind in ("user", "project")]
                observed["send1_rendered_history"] = projector.render(first_projection)

        # Each finished send was projected (user + project rows).
        proj = await store.load_project_messages(sid)
        assert {r.kind for r in proj} >= {"user", "project"}

        # keep_last_sends=2 over 4 sends → older sends folded into a compact row spanning 1-2.
        compact = await store.latest_project_compact(sid)
        _assert_compact_1_2(compact)
        observed["compact"] = {"from_send": compact.compact_from_send, "to_send": compact.compact_to_send,
                               "content": compact.content}
        assert observed["fold_calls"] > 0
        expected = observed["send1_rendered_history"]
        second = observed["second_business_messages"]
        assert expected and second
        spans = [index for index in range(len(second) - len(expected))
                 if second[index:index + len(expected)] == expected]
        assert spans, "second business request must include send1's actual rendered projection"
        assert second[-1]["role"] == "user" and "Say the word apple2 back to me." in str(second[-1]["content"])

        # recall_send re-expands a FOLDED send — its original detail is still in pl_messages.
        t_loop = set_current_loop(SimpleNamespace(store=store, config=None))  # type: ignore[arg-type]
        t_sid = set_session_id(sid)
        try:
            recalled = await run_recall_send(1)
        finally:
            reset_session_id(t_sid)
            reset_current_loop(t_loop)
        assert recalled.startswith("send #1") and "apple1" in recalled.lower()
        observed["recall_send1"] = recalled
    finally:
        try:
            if loop is not None:
                await loop.aclose()
            else:
                await llm.close()
        finally:
            if store is not None:
                await store.close()
            if output := os.environ.get("R10_EVIDENCE_DIR"):
                path = Path(output)
                path.mkdir(parents=True, exist_ok=True)
                (path / "real-projection.json").write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n")
