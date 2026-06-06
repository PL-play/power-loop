"""00 · 最简：发一条、拿一条回复

The shortest possible power-loop program. No system prompt, no tools, no
persistence — just feed the model one string and print the reply.

What you learn
--------------
- ``StatefulAgentLoop(llm=…)`` 是唯一公开入口
- ``loop.new_session()`` 显式创建 session
- ``await loop.send(user_input, session_id=sid)`` 返回 :class:`StatefulResult`
- ``db_path=":memory:"`` → 不落盘的临时 store；生产请传文件路径

Run
---
    python examples/00_minimal.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import StatefulAgentLoop


async def main() -> str:
    loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:")
    sid = loop.new_session()
    result = await loop.send("In one sentence: what is HTTP?", session_id=sid)
    print(result.final_text)
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
