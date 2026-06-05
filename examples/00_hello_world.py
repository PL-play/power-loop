"""00 · 最简：发一条、拿一条回复

The shortest possible power-loop program. No system prompt, no tools, no
persistence — just feed the model one string and print the reply.

What you learn
--------------
- ``StatefulAgentLoop(llm=…)`` 是唯一公开入口
- ``await loop.send(user_input)`` 返回 :class:`StatefulResult`
- 不传 ``session_id`` → 自动创建新 session
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
    result = await loop.send("In one sentence: what is HTTP?")
    print(result.final_text)
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
