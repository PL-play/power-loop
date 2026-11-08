"""13 · User-scoped SQLite memory across two independent Runtime sessions."""
from __future__ import annotations

import asyncio
import re
import sqlite3
import tempfile
import time
from pathlib import Path

from _runtime import demo_runtime

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    MemorySnapshot,
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


class FactModel(LLMService):
    async def complete(self, request, **kwargs):
        if 'Known facts about the user' in str(request.messages):
            assert 'Alan' in str(request.messages) and '37' in str(request.messages)
            return LLMResponse(raw_text='Alan / 37')
        return LLMResponse(raw_text='FACT: name=Alan\nFACT: favorite_number=37')

    async def close(self):
        pass


async def main():
    with tempfile.TemporaryDirectory(prefix='power_loop_memory_test_') as directory:
        memory = SqliteFactMemory(str(Path(directory)/'memory.db'))
        targets = []
        for index, question in enumerate(['Remember my name and number', 'What do you remember?']):
            async with demo_runtime(FactModel(), config=AgentLoopConfig(memory=memory, compactor=None)) as (runtime, loop, target, actor):
                targets.append(target)
                result = await loop.send(target, question, message_id=f'memory-{index}', actor=actor,
                                         deadline_ms=int(time.time()*1000)+10000)
                assert result.status == 'completed'
                print(result.final_text)
        assert targets[0] != targets[1] and result.final_text == 'Alan / 37'


if __name__ == '__main__':
    asyncio.run(main())
