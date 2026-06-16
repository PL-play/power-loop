"""32 · 取回被压缩的细节 / Recall compacted detail with the recall_compacted tool

What you learn / 你将学到
--------------------------
- 压缩把旧消息折叠成 ``compact_note`` 并标 ``compacted_out`` —— 原文**没删**,
  只是不再每轮发给模型。``recall_compacted`` 工具让 agent**按需**把原文捞回来。
  / compaction folds old messages into a ``compact_note`` and marks them
  ``compacted_out`` — the originals are NOT deleted, just not sent each turn. The
  ``recall_compacted`` tool lets the agent pull them back on demand.
- 一个具体编码埋在被折叠的早期消息里;摘要未必逐字保留。模型可以用
  ``recall_compacted(query=...)`` 精确取回。
  / a specific code is buried in folded early turns; the summary may not keep it
  verbatim. The model can retrieve it exactly via ``recall_compacted(query=...)``.

Run / 运行
----------
    python examples/32_recall_compacted.py
"""

from __future__ import annotations

import asyncio
import os
import re

from _helpers import make_llm

from power_loop import (
    AgentEventType,
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor
from power_loop.tools import create_default_tool_registry

# Alphanumeric (no separators) so a model can't render it with a fancy hyphen/space
# and break an exact match; we still compare via a normalized form to be safe.
ACCESS_CODE = "ZX7741QORN"


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


async def _seed_with_buried_code(store: SessionStore, sid: str) -> None:
    """Bury the code as an INCIDENTAL detail mid-filler (not flagged "remember
    this"), then a lot of fat filler so it lands well inside the folded-out range.
    Incidental specifics are exactly what a summary tends to drop — making the
    tool the reliable way to get it back."""
    await store.append_message(
        sid, role="user",
        content=(
            "Here's the full context of the migration: " + ("blah " * 60)
            + f" the old box was decommissioned (its rack serial happened to be {ACCESS_CODE}) "
            + ("and then " * 60) + " — anyway, continue setting things up."
        ),
        round_index=0,
    )
    await store.append_message(sid, role="assistant", content="Understood, continuing the setup. " + ("ok " * 60), round_index=0)
    for i in range(1, 6):
        await store.append_message(sid, role="user", content="(unrelated step) " + ("filler " * 140), round_index=i)
        await store.append_message(sid, role="assistant", content="done " + ("ok " * 140), round_index=i)


async def main() -> dict:
    os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"  # force the early turns to fold

    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        await _seed_with_buried_code(store, sid)

        tools_called: list[str] = []
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=300),
            store=store,
            tool_registry=create_default_tool_registry(include=["recall_compacted"]),
            config=AgentLoopConfig(
                system_prompt=(
                    "Older turns may have been summarized into a compact_note and may omit "
                    "exact values. If you need a precise detail the summary doesn't contain, "
                    "call recall_compacted (optionally with a query) to look it up before "
                    "answering. Answer concisely."
                ),
                max_rounds=4,
                # A deliberately tiny summary budget: the note can't fit the
                # incidental serial, so retrieving it requires the tool.
                compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1, summary_max_tokens=40),
            ),
        )
        loop.event_bus.subscribe(
            AgentEventType.TOOL_CALL_STARTED, lambda e: tools_called.append(e.data.name)
        )

        r = await loop.send(
            "Earlier I mentioned the rack serial of the decommissioned box. What was its exact value?",
            session_id=sid,
        )
        print(f"status   : {r.status}")
        print(f"reply    : {r.final_text}")
        print(f"tools    : {tools_called}")

        # the code is recoverable from the folded rows (the tool's data source)
        folded = [m for m in await store.load_all_messages(sid)
                  if m.state is MessageState.COMPACTED_OUT and ACCESS_CODE in (m.content or "")]
        code_in_folded = bool(folded)
        answer_has_code = _norm(ACCESS_CODE) in _norm(r.final_text)  # robust to bold/hyphen/spacing
        print(f"code buried in folded rows : {code_in_folded}")
        print(f"recall_compacted called    : {'recall_compacted' in tools_called}")
        print(f"answer contains the code   : {answer_has_code}")

        # Deterministic end-to-end check: invoke the tool directly on the REAL
        # compacted session and confirm it recovers the buried code. (Whether the
        # MODEL chose to call it autonomously above is non-deterministic — a soft
        # demo — but the retrieval path itself must always work.)
        from power_loop.core.agent_context import (
            reset_current_loop,
            reset_session_id,
            set_current_loop,
            set_session_id,
        )
        from power_loop.tools.default_tools import run_recall_compacted

        t_loop, t_sid = set_current_loop(loop), set_session_id(sid)
        try:
            tool_out = await run_recall_compacted(query="rack serial")
        finally:
            reset_session_id(t_sid)
            reset_current_loop(t_loop)
        code_retrievable = _norm(ACCESS_CODE) in _norm(tool_out)
        print(f"recall_compacted recovers it : {code_retrievable}")

        return {
            "final_text": r.final_text,
            "code_in_folded": code_in_folded,
            "code_retrievable": code_retrievable,
            "answer_has_code": answer_has_code,
            "recall_called": "recall_compacted" in tools_called,
        }
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
