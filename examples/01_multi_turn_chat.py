"""01 · 多轮对话：用 session_id 续话

What you learn
--------------
- ``loop.new_session()`` 返回的 ``session_id`` 是你**唯一**需要保管的东西
- 后续每次 ``send(..., session_id=...)`` 都自动加载历史 → 模型看到完整上下文
- ``loop.get_messages(sid)`` 查看持久化的完整 history
- ``loop.close_session(sid)`` 物理删除（含 sessions / messages / 压缩痕迹）

Run
---
    python examples/01_multi_turn.py
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

    # 第 1 轮：建立事实
    r1 = await loop.send("My favorite color is teal. Acknowledge briefly.", session_id=sid)
    print(f"turn 1: {r1.final_text}\n")

    # 第 2 轮：传同一个 session_id，模型应该记得 teal
    r2 = await loop.send(
        "What did I just tell you my favorite color was?",
        session_id=sid,
    )
    print(f"turn 2: {r2.final_text}\n")

    # 检查持久化的 history
    msgs = loop.get_messages(sid)
    print(f"history has {len(msgs)} messages: roles = {[m['role'] for m in msgs]}")

    # 用完即删（生产里通常不删）
    deleted = loop.close_session(sid)
    print(f"deleted {deleted} session row(s)")

    return r2.final_text


if __name__ == "__main__":
    asyncio.run(main())
