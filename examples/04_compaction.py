"""04 · 上下文压缩：DefaultCompactor 自动折叠长历史

What you learn
--------------
- ``AgentLoopConfig.compactor`` 接受任何实现 :class:`Compactor` 协议的对象；
  默认 ``DefaultCompactor()`` 已开启
- 触发条件：``estimate_tokens(history) ≥ max_tokens × trigger_ratio``；
  env ``CONTEXT_COMPACT_THRESHOLD`` 可设绝对阈值
- 触发后：被折叠的消息在 store 里标 ``state='compacted_out'``，
  插入一条 ``role=system, name=compact_note`` 摘要，
  ``compactions`` 表新增审计行
- 模型基于 "system + compact_note + 最近一段尾巴" 继续答题

Run
---
    python examples/04_compaction.py
"""

from __future__ import annotations

import asyncio
import os

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor


def _seed_fat_history(store: SessionStore, sid: str, turns: int = 4) -> None:
    """灌入若干轮 user/assistant，让 history 远超阈值。"""
    for i in range(turns):
        store.append_message(sid, role="user", content="filler " + ("u" * 400), round_index=i)
        store.append_message(
            sid, role="assistant",
            content="filler ack " + ("a" * 400),
            round_index=i,
        )


async def main() -> str:
    # 强制低阈值，保证每次都触发
    os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"

    store = SessionStore.open(":memory:")
    try:
        sid = store.create_session(system_prompt="S")
        _seed_fat_history(store, sid, turns=4)

        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "Earlier turns may have been summarized into a compact_note. "
                    "Answer the user's latest question concisely in English."
                ),
                max_rounds=1,
                compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
            ),
        )
        r = await loop.send("Name the largest planet in our solar system.", session_id=sid)
        print(f"status   : {r.status}, rounds: {r.rounds}")
        print(f"reply    : {r.final_text}")

        # ── 查看压缩痕迹 ──
        comps = store.list_compactions(sid)
        all_rows = store.load_all_messages(sid)
        folded = sum(1 for m in all_rows if m.state is MessageState.COMPACTED_OUT)
        notes = [m for m in all_rows if m.name == "compact_note"]
        print(f"compactions recorded : {len(comps)}")
        print(f"messages compacted   : {folded}")
        print(f"compact_note preview : {(notes[0].content or '')[:80]!r}")
        return r.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
