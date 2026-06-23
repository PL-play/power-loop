"""40 · Send 上下文投影 / Send-context projection (ProjectedRepresentation × fold)

What you learn / 你将学到
--------------------------
- 默认情况下,每次 send 把**整段历史逐字**(含完整工具调用结构和长工具结果)发给模型;
  `pl_messages` 既是审计、又被直接当上下文。/ by default each send feeds the model the
  ENTIRE verbatim history (full tool-call structure + long tool results); pl_messages is
  both the audit log and the context.
- 把 ``representation`` 设为 ``ProjectedRepresentation`` 后,**已结束的 send** 被投影成
  紧凑纯文本,存进派生表 ``pl_project_messages``;下一次 send 的历史 = 这些投影 + **当前 send
  逐字**。``pl_messages`` 永不被改写。/ with ``representation=ProjectedRepresentation()``,
  FINISHED sends are projected to compact plain text into the derived ``pl_project_messages``
  table; the next send's history = those projections + the IN-FLIGHT send verbatim.
  ``pl_messages`` is never rewritten.
- ``keep_last_sends`` 把更早的 send 折成一条 append-only ``compact`` 行;原始明细仍在
  ``pl_messages``,可用 ``recall_send(N)`` 取回。/ ``keep_last_sends`` folds older sends into
  one append-only ``compact`` row; the originals stay in ``pl_messages`` and are recoverable
  via ``recall_send(N)``.

Run / 运行
----------
    python examples/40_send_context_projection.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    LLMSummaryFold,
    ProjectedRepresentation,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.agent.stateful_loop import _row_to_loop_message
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.tools.default_tools import run_recall_send


def _has_tool_protocol(msgs: list[dict]) -> bool:
    """True if any message carries OpenAI tool-call protocol fields (verbatim history does;
    a projected plain-text history must not)."""
    return any("tool_calls" in m or "tool_call_id" in m for m in msgs)


async def main() -> dict:
    store = await SessionStore.open(":memory:")
    try:
        # 3.0: representation (HOW sends are recorded) × fold_strategy (HOW older history compacts)
        # are orthogonal. Projection representation + an LLM-summary fold keeping the last 2 sends.
        representation = ProjectedRepresentation()
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=64),
            store=store,
            config=AgentLoopConfig(
                system_prompt="You are terse. Reply with one short sentence.",
                max_rounds=3,
                # small max_tokens so the token-driven projection fold (threshold =
                # max_tokens × trigger_ratio) demonstrably fires over a few short sends.
                max_tokens=40,
                representation=representation,
                fold_strategy=LLMSummaryFold(keep_last_sends=2),
            ),
        )
        sid = await loop.new_session()

        fruits = ["apple", "banana", "cherry", "date"]
        for i, fruit in enumerate(fruits, start=1):
            r = await loop.send(f"Note the fruit '{fruit}{i}'. Reply OK.", session_id=sid)
            print(f"send {i}: status={r.status}")

        # 1) Every finished send was projected into pl_project_messages (user + project rows).
        proj_rows = await store.load_project_messages(sid)
        kinds = {r.kind for r in proj_rows}

        # 2) keep_last_sends=2 over 4 sends → the oldest sends fold into one compact row.
        compact = await store.latest_project_compact(sid)

        # 3) pl_messages is the IMMUTABLE audit: every original row is still there, untouched.
        full = await store.load_all_messages(sid)

        # 4) Reconstruct what the LLM would see for a 5th send, both ways (deterministic):
        #    DEFAULT = every active pl_messages row verbatim; PROJECTION = compact + recent
        #    projections rendered to plain text. Note the size + structure difference.
        verbatim_ctx = [_row_to_loop_message(m) for m in await store.load_active_messages(sid)]
        cutoff = compact.compact_to_send if compact else None
        prefix_rows = ([compact] if compact else []) + [
            r for r in await store.load_project_messages(sid, after_send_index=cutoff)
        ]
        projected_ctx = representation.render(prefix_rows)

        # 5) recall_send re-expands a folded send's ORIGINAL detail (from pl_messages).
        t_loop, t_sid = set_current_loop(loop), set_session_id(sid)
        try:
            recalled = await run_recall_send(1)
        finally:
            reset_session_id(t_sid)
            reset_current_loop(t_loop)

        print(f"\nprojection rows in pl_project_messages : {len(proj_rows)} ({sorted(kinds)})")
        print(f"compact row covers sends               : {compact.compact_from_send}-{compact.compact_to_send}"
              if compact else "compact row                            : none")
        print(f"pl_messages rows (immutable audit)     : {len(full)}")
        print(f"DEFAULT context (verbatim)  messages   : {len(verbatim_ctx)} "
              f"(tool-protocol present={_has_tool_protocol(verbatim_ctx)})")
        print(f"PROJECTION context          messages   : {len(projected_ctx)} "
              f"(tool-protocol present={_has_tool_protocol(projected_ctx)})")
        print("PROJECTION history (what the model sees for older sends):")
        for m in projected_ctx:
            print(f"    [{m['role']}] {m['content'][:80]}")
        print(f"recall_send(1) recovered send 1        : {'apple1' in recalled.lower()}")

        return {
            "projected_kinds": sorted(kinds),
            "has_compact": compact is not None,
            "compact_from_send": compact.compact_from_send if compact else None,
            "pl_messages_intact": len(full) >= 2 * len(fruits),  # ≥ user+assistant per send
            "projection_is_plain_text": not _has_tool_protocol(projected_ctx),
            "recall_recovers_send1": "apple1" in recalled.lower(),
        }
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
