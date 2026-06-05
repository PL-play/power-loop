"""03 · 上下文压缩：DefaultCompactor 自动折叠长历史

What this example shows
-----------------------
- 配置 ``DefaultCompactor`` 触发阈值
- 给 session 预先灌入一段“假历史”，让阈值在下一轮 send 之前就被触发
- 触发后查看 ``SessionStore``：被折叠的消息标记为 ``compacted_out``，
  插入一条 ``role=system, name=compact_note`` 摘要，``compactions`` 表新增审计行
- 模型基于 “system + compact_note + 最近一段尾巴” 继续回答，验证正确性

Key concepts (see README §压缩)
------------------------------
* 触发条件：``estimate_tokens(history) ≥ max_tokens × trigger_ratio``，
  或 env ``CONTEXT_COMPACT_THRESHOLD`` 绝对阈值覆盖。
* 不变量：保留所有 system 行；保留尾部 ``keep_last_n`` 个 user 段；
  绝不切开 ``assistant(tool_calls) ↔ tool`` 原子对。
* 失败软降级：摘要 LLM 抛错 → ``maybe_compact`` 返回 ``None`` → 主循环继续用未压缩 history。

How to run
----------
    python examples/03_compaction.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_llm() -> OpenAICompatibleChatLLMService:
    cfg = OpenAICompatibleChatConfig(
        base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
        api_key=os.environ["OPENAI_COMPAT_API_KEY"],
        model=os.environ["OPENAI_COMPAT_MODEL"],
        max_tokens=512,
        temperature=0,
    )
    return OpenAICompatibleChatLLMService(cfg)


def _seed_fat_history(store: SessionStore, sid: str, *, turns: int = 4) -> None:
    """灌入若干轮 user/assistant，让 history 远超压缩阈值。"""
    for i in range(turns):
        store.append_message(sid, role="user", content="filler " + ("u" * 400), round_index=i)
        store.append_message(
            sid, role="assistant",
            content="filler ack " + ("a" * 400),
            round_index=i,
        )


async def main() -> str:
    # 强制低阈值以保证示例每次都触发压缩。
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
                max_tokens=256,
                temperature=0,
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
        print(f"compact_note inserted: {len(notes)}; first preview = {notes[0].content[:80]!r}")
        return r.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
