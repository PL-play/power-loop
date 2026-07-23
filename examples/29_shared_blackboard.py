"""29 · Shared blackboard / 共享黑板: two agents coordinate on one scoped board

What you learn / 你将学到
--------------------------
- The **scoped shared blackboard**: a small, structured, mutable space that
  several agents read and write to coordinate — distinct from each agent's
  private message history.
  / 作用域共享黑板：多个 agent 读写的一小块结构化共享状态，与各自私有的消息历史分开。
- Inject a ``Blackboard`` + ``blackboard_id`` per run via
  ``runtime_env_context(RuntimeEnv(blackboard=…, blackboard_id=…))`` — the same
  seam as the sandbox ``ShellBackend``. Agents with the **same id** share a board;
  different ids are isolated.
  / 用 ``runtime_env_context`` 按次注入黑板与 id：相同 id 共享，不同 id 隔离。
- The single ``board`` tool (``action=read|post|update|remove``) is the
  agent-facing API; the author is stamped from session
  metadata (``spec_name``), not supplied by the model.
  / ``board_*`` 工具是 agent 侧 API；作者由会话元数据决定，模型无法伪造。
- The host **projects** the current board into each turn's prompt (the "pull"
  side) so an agent acts on what its peers wrote.
  / 宿主把当前黑板投影进每轮提示，agent 据此行动。

Why it matters / 为什么重要
---------------------------
This is the primitive behind multi-agent coordination: a planner leaves tasks,
a worker claims and completes them, each sees the other's updates — without
dumping one agent's entire transcript into the other's context. DeepTalk uses
exactly this (scope = conversation) for its in-room agent coordination board.

Run / 运行
----------
    python examples/29_shared_blackboard.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    RuntimeEnv,
    SqliteBlackboard,
    StatefulAgentLoop,
    ToolRegistry,
    register_blackboard_tools,
    render_entries,
    runtime_env_context,
)

BOARD_ID = "project-onboarding"

SYSTEM_PROMPT = (
    "You are one of two agents collaborating through a small SHARED BOARD. "
    "The board's current contents are shown at the top of your message. Use the "
    "board tool (action=post/update/read) to coordinate with your "
    "teammate — keep entries short. Do exactly what you're asked, then stop "
    "(don't keep posting)."
)


def _board_registry() -> ToolRegistry:
    reg = ToolRegistry()
    # Vocabulary is the host's policy: tasks/notes with an open→doing→done flow.
    register_blackboard_tools(reg, kinds=("note", "task"), statuses=("open", "doing", "done"))
    return reg


async def main() -> dict:
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=500),
        db_path=":memory:",
        tool_registry=_board_registry(),
        config=AgentLoopConfig(system_prompt=SYSTEM_PROMPT, max_rounds=6, compactor=None),
    )
    # One board, shared by both agents because they use the same blackboard_id.
    # The owned store opens lazily, so resolve it explicitly (loop.store is None until then).
    board = SqliteBlackboard(await loop.ensure_store())

    async def run_agent(spec_name: str, instruction: str) -> str:
        # New session per agent → independent private histories. spec_name becomes
        # the board author. We PROJECT the live board into this turn's prompt.
        sid = await loop.new_session(metadata={"spec_name": spec_name})
        snapshot = render_entries(
            await board.read(BOARD_ID), header="Shared team board:", empty="(the board is empty)"
        )
        with runtime_env_context(RuntimeEnv(blackboard=board, blackboard_id=BOARD_ID)):
            res = await loop.send(f"{snapshot}\n\n{instruction}", session_id=sid)
        return res.final_text

    # 1) The planner posts two tasks onto the shared board.
    await run_agent(
        "planner",
        "You are the planner. Using board action=post (kind=task, status=open), add exactly two "
        "tasks for your teammate: (1) 'write the welcome email', (2) 'set up the calendar "
        "invite'. Post them, then stop.",
    )
    # 2) The worker reads the board, completes the first task, and leaves a note.
    await run_agent(
        "worker",
        "You are the worker. The shared board above lists the planner's open tasks. Use "
        "board action=update to set the 'write the welcome email' task to status=done, then "
        "board action=post a short note (kind=note) saying the welcome email has been sent. Then stop.",
    )

    entries = await board.read(BOARD_ID)
    print("Final shared board:")
    print(render_entries(entries, header="", empty="(empty)"))
    return {
        "authors": sorted({e.author for e in entries if e.author}),
        "statuses": [e.status for e in entries],
        "kinds": [e.kind for e in entries],
        "texts": [e.text for e in entries],
        "n": len(entries),
    }


if __name__ == "__main__":
    asyncio.run(main())
