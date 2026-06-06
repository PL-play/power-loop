"""01 · 多轮对话 / Multi-turn chat: continue a conversation with session_id

What you learn / 你将学到
--------------------------
- ``loop.new_session()`` 返回的 ``session_id`` 是你**唯一**需要保管的东西
  / the ``session_id`` returned by ``new_session()`` is the **only** thing you
  need to keep track of
- 后续每次 ``send(..., session_id=...)`` 都自动加载历史 → 模型看到完整上下文
  / every subsequent ``send(..., session_id=...)`` auto-loads history → model
  sees full context
- ``loop.get_messages(sid)`` 查看持久化的完整 history
  / inspect the persisted full history
- ``loop.close_session(sid)`` 物理删除（含 sessions / messages / 压缩痕迹）
  / physically delete (sessions / messages / compaction records)

Run / 运行
----------
    python examples/01_multi_turn_chat.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop


async def main() -> str:
    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=":memory:",
        config=AgentLoopConfig(
            system_prompt="You are a friendly assistant with perfect memory of this chat.",
            max_rounds=1,
            compactor=None,
        ),
    )

    sid = loop.new_session()

    # 第 1 轮：建立事实 / Turn 1: establish a fact
    r1 = await loop.send("My favorite color is teal. Acknowledge briefly.", session_id=sid)
    print(f"turn 1: {r1.final_text}\n")

    # 第 2 轮：传同一个 session_id，模型应该记得 teal
    # Turn 2: pass the same session_id — model should remember teal
    r2 = await loop.send(
        "What did I just tell you my favorite color was?",
        session_id=sid,
    )
    print(f"turn 2: {r2.final_text}\n")

    # 检查持久化的 history / Check persisted history
    msgs = loop.get_messages(sid)
    print(f"history has {len(msgs)} messages: roles = {[m['role'] for m in msgs]}")

    # 用完即删（生产里通常不删）/ Delete when done (usually not in production)
    deleted = loop.close_session(sid)
    print(f"deleted {deleted} session row(s)")

    return r2.final_text


if __name__ == "__main__":
    asyncio.run(main())
