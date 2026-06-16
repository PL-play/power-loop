"""24 · Agent 自主笔记 / Agent-authored notes: self-managed persistent memory

What you learn / 你将学到
--------------------------
- ``note_add`` / ``note_update`` / ``note_delete`` 默认工具：模型自己决定记什么
  / default tools — the model decides what to remember
- ``SQLiteNoteMemory``：把笔记作为 system 消息在每次 send 注入
  / a MemoryProvider that injects the session's notes every send
- ``NotesPolicy``：容量上限、单条长度、注入预算；默认 **reject**（满了必须先删/合并），
  可选 ``eviction="fifo"`` / capacity bounds; default **reject** when full, optional fifo
- 笔记存在 session store 的 ``notes`` 表里，随 ``close_session`` 级联删除
  / notes live in the session store and cascade-delete with the session

和 MemoryProvider 的关系 / Relationship to MemoryProvider
---------------------------------------------------------
``MemoryProvider`` 是被动记忆（边界自动 recall/remember）；笔记是**自主记忆**——
模型在对话中途用工具增删改，``SQLiteNoteMemory.recall()`` 只负责"让模型每轮都看到
自己的笔记"，``remember()`` 是 no-op（写入早已实时发生）。

Run / 运行
----------
    python examples/24_agent_notes.py        # 需要 .env 配置 POWER_LOOP_*
"""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    NotesPolicy,
    SQLiteNoteMemory,
    StatefulAgentLoop,
    create_default_tool_registry,
)
from power_loop.runtime.store.store import SessionStore


async def main() -> None:
    with TemporaryDirectory() as tmp:
        store = await SessionStore.open(f"{tmp}/sessions.db")
        policy = NotesPolicy(max_notes=20)  # eviction="reject" 是默认

        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,  # 关键：provider 和 loop 共享同一个 store
            config=AgentLoopConfig(
                system_prompt="You are a helpful assistant with persistent notes.",
                memory=SQLiteNoteMemory(store, policy=policy),
                notes_policy=policy,
            ),
            tool_registry=create_default_tool_registry(
                include=["note_add", "note_update", "note_delete"],
                workspace_dir=tmp,
            ),
        )

        sid = await loop.new_session()

        # 第一轮：让模型记点东西 / round 1: ask the model to take a note
        r1 = await loop.send(
            "请记住：我的项目代号是 BLUEBIRD，部署目标是 ap-southeast-1。"
            "用你的笔记工具记下来，然后简单确认。",
            sid,
        )
        print("R1:", r1.final_text)
        for n in await store.list_notes(sid):
            print(f"  note #{n.note_id} pinned={n.pinned}: {n.content}")

        # 第二轮：新的 send 自动注入笔记 / round 2: notes are injected automatically
        r2 = await loop.send("我的项目代号和部署区域是什么？", sid)
        print("R2:", r2.final_text)

        await loop.close_session(sid)  # notes 级联删除 / notes cascade-delete
        await loop.aclose()


if __name__ == "__main__":
    asyncio.run(main())
