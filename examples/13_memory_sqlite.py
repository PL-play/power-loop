"""13 · MemoryProvider：跨 session 的 SQLite 事实记忆 / Cross-session SQLite fact memory

What you learn / 你将学到
--------------------------
- ``MemoryProvider`` Protocol 两个方法：``recall`` / ``remember``。
  / ``MemoryProvider`` Protocol has two methods: ``recall`` / ``remember``
- 怎么把召回的事实注入到 ``role=system, name=memory_*`` 区域，与
  ``compact_note`` 一道躲过 ``DefaultCompactor`` 的折叠（system 区天然保留）。
  / how recalled facts are injected into ``role=system, name=memory_*`` region,
  surviving ``DefaultCompactor`` folds alongside ``compact_note`` (system region
  is naturally preserved)
- ``recall`` / ``remember`` 任一抛错都不会打断主循环 —— 框架只发
  ``MEMORY_FAILED`` 事件，照常返回。
  / errors in ``recall`` / ``remember`` never interrupt the main loop — framework
  only emits ``MEMORY_FAILED`` event and continues normally
- 跨 session：第一个 session 让 Agent 记住一个事实；第二个 session 用同一
  ``MemoryProvider`` 但**新 session_id**，仍然能用上事实。
  / cross-session: first session teaches a fact; second session uses the same
  ``MemoryProvider`` with a **new session_id** and still has access to the fact

本例的事实抽取很粗（让 LLM 在第一轮答案末尾贴一个 ``FACT: …`` 行，
``remember`` 抓出来落库）；真实系统应当走更稳的 schema（M1.3 结构化
输出）或独立 extract 步骤。**库内零实现**正是为了让这种业务策略留在
应用层。
/ This example uses coarse fact extraction (LLM appends ``FACT: …`` lines,
``remember`` extracts them). Real systems should use more stable schemas
(structured output) or separate extraction steps. **Zero in-library implementation**
keeps business logic in the application layer.

Run / 运行
----------
    python examples/13_memory_sqlite.py
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import tempfile
from pathlib import Path

from _helpers import make_llm

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    MemorySnapshot,
    SessionStore,
    StatefulAgentLoop,
)

# ── 1. 一个非常薄的 SQLite 事实库 / A minimal SQLite fact store ──────────


class SqliteFactMemory:
    """Key-value 事实库。``recall`` 拉出全部事实拼成一条 system 消息；
    ``remember`` 从 final_text 里抽 ``FACT: key=value`` 行入库。
    / Key-value fact store. ``recall`` pulls all facts into a system message;
    ``remember`` extracts ``FACT: key=value`` lines from final_text into the DB."""

    _FACT_RE = re.compile(r"FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", re.M)

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                " key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " updated_at REAL DEFAULT (strftime('%s','now')))"
            )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        if not rows:
            return []
        text = "Known facts about the user:\n" + "\n".join(
            f"- {r['key']}: {r['value']}" for r in rows
        )
        # 返回一条；框架会自动 tag 成 role=system, name=memory_*
        # Returns one message; framework auto-tags as role=system, name=memory_*
        return [{"content": text}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        captured = self._FACT_RE.findall(snapshot.final_text or "")
        if not captured:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT INTO facts(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=strftime('%s','now')",
                captured,
            )
            c.commit()


# ── 2. 跑两段对话观察跨 session 的记忆生效 ───────────────────────────────
#    Run two conversations to observe cross-session memory in action


SYSTEM = (
    "You are a concise assistant. After answering, if the user told you "
    "a personal fact (name, favorite number, location, …), append one or "
    "more lines of the form:\n  FACT: key=value\n"
    "where key is a short snake_case identifier. If you already know a "
    "fact from prior memory, use it without asking again."
)


def _print_events(bus: AgentEventBus, label: str) -> list:
    seen: list = []
    interesting = {AgentEventType.MEMORY_RECALLED, AgentEventType.MEMORY_FAILED}
    bus.subscribe(None, lambda e: seen.append(e) if e.type in interesting else None)
    return seen


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="power_loop_p13_") as tmp:
        mem_path = str(Path(tmp) / "memory.db")
        sess_path = str(Path(tmp) / "sessions.db")
        memory = SqliteFactMemory(mem_path)

        # ── Session A：教 Agent 一个事实 / Teach the agent a fact ────────
        print("\n── Session A: teach the agent a fact ──")
        store = SessionStore.open(sess_path)
        try:
            bus_a = AgentEventBus()
            seen_a = _print_events(bus_a, "A")
            loop = StatefulAgentLoop(
                llm=make_llm(max_tokens=200), store=store, event_bus=bus_a,
                config=AgentLoopConfig(
                    system_prompt=SYSTEM, max_rounds=1, compactor=None,
                    memory=memory,
                ),
            )
            sid1 = loop.new_session()
            r1 = await loop.send(
                "My name is Alan and my favorite number is 37. Reply in one sentence.",
                session_id=sid1,
            )
            print(f"[A] reply={r1.final_text}")
            print(f"[A] memory events: {[e.type.value for e in seen_a]}")
        finally:
            store.close()

        # 验：直接看 SQLite，事实已经入库
        # Verify: check SQLite directly — facts are stored
        with sqlite3.connect(mem_path) as c:
            facts = list(c.execute("SELECT key, value FROM facts ORDER BY key"))
        print(f"[memory.db] facts after session A: {facts}")

        # ── Session B：完全新 session，靠 recall 把事实带回来 ──────────
        #    Brand new session — facts come back via recall
        print("\n── Session B: new session — does the agent remember? ──")
        store = SessionStore.open(sess_path)
        try:
            bus_b = AgentEventBus()
            seen_b = _print_events(bus_b, "B")
            loop = StatefulAgentLoop(
                llm=make_llm(max_tokens=200), store=store, event_bus=bus_b,
                config=AgentLoopConfig(
                    system_prompt=SYSTEM, max_rounds=1, compactor=None,
                    memory=memory,
                ),
            )
            sid2 = loop.new_session()
            r2 = await loop.send(
                "What is my name? What is my favorite number? One sentence.",
                session_id=sid2,
            )
            print(f"[B] reply={r2.final_text}")
            print(f"[B] memory events: {[e.type.value for e in seen_b]}")
            assert r2.session_id != r1.session_id, "should be different sessions"
        finally:
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
