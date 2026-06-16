"""03 · 子代理委托 / Sub-agent delegation: parent delegates via ``spawn_agent``

What you learn / 你将学到
--------------------------
- ``register_spawn_agent`` 一行注入两个 meta-tool：``spawn_agent`` 和 ``run_agent``
  / injects two meta-tools in one line: ``spawn_agent`` and ``run_agent``
- 父 LLM 自主决定 ``spawn_agent`` → 自动新建子 session，跑独立的小循环
  / parent LLM autonomously calls ``spawn_agent`` → creates child session, runs
  independent sub-loop
- 子结果作为 ``tool`` 消息回灌父 session
  / child result fed back to parent as a ``tool`` message
- ``EPHEMERAL`` 生命周期：子 session 完成后从 SessionStore 物理删除（保留失败者供 debug）
  / ``EPHEMERAL`` lifecycle: child session physically deleted on success (failures
  preserved for debugging)
- ``store.list_children(parent_sid)`` 查看父下还存活的子 session
  / inspect surviving child sessions under a parent

Run / 运行
----------
    python examples/03_subagent_delegation.py
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
    register_spawn_agent(registry)        # ← spawn_agent + run_agent 都注册 / both registered

    store = await SessionStore.open(":memory:")
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
        sid = await loop.new_session()
        result = await loop.send(
            "Delegate this and report back: what is the capital of Japan?",
            session_id=sid,
        )
        print(f"status        : {result.status}, rounds: {result.rounds}")
        print(f"reply         : {result.final_text}")
        # EPHEMERAL：子会话已删，list_children 返回空
        # EPHEMERAL: child session deleted, list_children returns empty
        print(f"surviving subs: {await store.list_children(result.session_id)}")
        return result.final_text
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
