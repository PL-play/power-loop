"""03 · 子代理：父 agent 通过 ``spawn_agent`` 工具委托子任务

What you learn
--------------
- ``register_spawn_agent`` 一行注入两个 meta-tool：``spawn_agent`` 和 ``run_agent``
- 父 LLM 自主决定 ``spawn_agent`` → 自动新建子 session，跑独立的小循环
- 子结果作为 ``tool`` 消息回灌父 session
- ``EPHEMERAL`` 生命周期：子 session 完成后从 SessionStore 物理删除（保留失败者供 debug）
- ``store.list_children(parent_sid)`` 查看父下还存活的子 session

Run
---
    python examples/03_subagent.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    ToolRegistry,
    register_spawn_agent,
)


async def main() -> str:
    registry = ToolRegistry()
    register_spawn_agent(registry)        # ← spawn_agent + run_agent 都注册

    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a delegating orchestrator. For any factual "
                    "question, call the `spawn_agent` tool with a clear "
                    "`task` description; do NOT answer from memory. After "
                    "the sub-agent replies, summarize it in one sentence."
                ),
                max_rounds=5,
                compactor=None,
            ),
        )
        sid = loop.new_session()
        result = await loop.send(
            "Delegate this and report back: what is the capital of Japan?",
            session_id=sid,
        )
        print(f"status        : {result.status}, rounds: {result.rounds}")
        print(f"reply         : {result.final_text}")
        # EPHEMERAL：子会话已删，list_children 返回空。
        print(f"surviving subs: {store.list_children(result.session_id)}")
        return result.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
