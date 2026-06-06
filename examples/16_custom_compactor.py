"""16 · 自定义 Compactor / Custom Compactor: implement your own compaction strategy

## What you'll learn / 你将学到
- 实现 ``Compactor`` 协议
  / implement the ``Compactor`` protocol
  (``async def maybe_compact(messages, *, llm, max_tokens, round_index) -> CompactionPlan | None``)
- 替代默认的 ``DefaultCompactor``，注入自定义压缩逻辑
  / replace the default ``DefaultCompactor`` with custom compaction logic
- 简单的"只保留最后 N 条"压缩器 vs "基于角色过滤"压缩器
  / simple "keep last N only" compactor vs "role-based filtering" compactor

## Prerequisites / 前提
- 需要 ``.env`` 配置 ``POWER_LOOP_*``
  / requires ``.env`` with ``POWER_LOOP_*``

## Run / 运行
    python examples/16_custom_compactor.py

## Key concepts / 关键概念
- **Compactor 协议 / Compactor protocol**: 返回 ``CompactionPlan(fold_start_idx, fold_end_idx, summary_text, ...)`` 或 ``None``。
  / returns ``CompactionPlan(fold_start_idx, fold_end_idx, summary_text, ...)`` or ``None``
- **None = 跳过本轮压缩 / skip this round's compaction**。
- 压缩器在每轮开始前被调用；pipeline 自动处理消息折叠和持久化。
  / compactor is called before each round; pipeline handles message folding and persistence automatically

## Next / 下一步
看看 `17_custom_memory_provider.py` — 自定义跨会话记忆
/ see `17_custom_memory_provider.py` — custom cross-session memory
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.compact import CompactionPlan

# ── 1. Tail-only 压缩器：只保留最后 N 条消息 ────────────────────────────
#    Tail-only compactor: keep only the last N messages


class TailOnlyCompactor:
    """Drop everything except the last ``keep`` messages + a summary note.

    This is the simplest possible compactor — no LLM summary call needed.
    """

    def __init__(self, *, keep: int = 6, trigger_ratio: float = 0.75):
        self.keep = keep
        self.trigger_ratio = trigger_ratio

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index):
        from power_loop.runtime.budget import estimate_tokens

        total = estimate_tokens(messages)
        if total <= max_tokens * self.trigger_ratio:
            return None  # no compaction needed

        n = len(messages)
        if n <= self.keep:
            return None

        fold_start = 0
        # Skip leading system messages
        while fold_start < n and messages[fold_start].get("role") == "system":
            fold_start += 1

        fold_end = n - self.keep - 1
        if fold_end <= fold_start:
            return None

        removed = fold_end - fold_start + 1
        summary = (
            f"[Compacted {removed} earlier messages. "
            f"Last {self.keep} messages preserved.]"
        )

        return CompactionPlan(
            fold_start_idx=fold_start,
            fold_end_idx=fold_end,
            summary_text=summary,
            before_tokens=total,
            after_tokens=total - estimate_tokens(messages[fold_start:fold_end + 1]) + 50,
        )


# ── 2. Run / 运行 ────────────────────────────────────────────────────────


async def main() -> None:
    llm = make_llm(max_tokens=300, temperature=0.0)

    # 造一段足够长的历史，让压缩器触发
    # Build a long enough history to trigger the compactor
    compactor = TailOnlyCompactor(keep=4, trigger_ratio=0.2)  # 低触发，确保触发 / low threshold to ensure trigger

    loop = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(
            system_prompt="Reply with exactly one sentence. Be concise.",
            max_rounds=1,
            compactor=compactor,
        ),
    )

    try:
        # Fill up history with several exchanges
        facts = [
            "My favorite color is blue.",
            "I live in Shanghai.",
            "My pet is a cat named Luna.",
            "I work as a backend engineer.",
            "My hobby is hiking.",
            "I drink coffee every morning.",
        ]
        sid = loop.new_session()
        for fact in facts:
            r = await loop.send(f"Remember: {fact}", session_id=sid)

        # Now ask — the compactor should have triggered, but the last few
        # exchanges (with the most recent facts) should still be in context.
        r = await loop.send(
            "What do I drink every morning and what is my pet's name? One sentence.",
            session_id=sid,
        )
        print(f"Reply: {r.final_text}")
        print(f"Rounds: {r.rounds}")
    finally:
        loop.close()


if __name__ == "__main__":
    asyncio.run(main())
