"""19 · 旗舰示例 / Flagship: full-featured chatbot
     (session + tools + hooks + events + memory + compaction)

## What you'll learn / 你将学到
- 一个 ``StatefulAgentLoop`` 同时开启：session 持久化、两个自定义工具、安全门 hook、
  流式事件订阅、跨会话 SQLite 记忆、默认上下文压缩。
  / a single ``StatefulAgentLoop`` simultaneously enables: session persistence,
  two custom tools, safety gate hook, streaming event subscription, cross-session
  SQLite memory, and default context compaction
- 这是 power-loop 的「全功能演示」——适合做集成测试或新人了解全部能力。
  / this is power-loop's "full-feature demo" — ideal for integration tests or
  newcomers wanting to see all capabilities at once

## Prerequisites / 前提
- 需要 ``.env`` 配置 ``POWER_LOOP_*``
  / requires ``.env`` with ``POWER_LOOP_*``

## Run / 运行
    python examples/19_full_chatbot.py

## Expected output / 预期输出
    [Stream] I am the flagship demo of power-loop...
    [Tool] get_weather → "Weather in Tokyo: sunny, 22°C"
    [Memory] Session A ended, facts stored
    [Session B] Agent recalls from memory
    [Done] 3 sessions completed, all features verified

## Key concepts / 关键概念
- **Session persistence / 会话持久化**: 两个 session 共用同一个 ``db_path``，通过 ``session_id`` 延续。
  / multiple sessions share the same ``db_path``, continued via ``session_id``
- **Tools / 工具**: ``get_weather`` + ``calculator`` 两个自定义工具。/ two custom tools
- **Hooks / 钩子**: ``TOOL_BEFORE`` 安全门——拦截危险操作。/ safety gate — blocks dangerous operations
- **Events / 事件**: 订阅 ``STREAM_DELTA`` 做打字机效果，``TOOL_CALL_STARTED`` 追踪工具调用。
  / subscribe to ``STREAM_DELTA`` for typewriter effect, ``TOOL_CALL_STARTED`` to track tool calls
- **Memory / 记忆**: SQLite 事实库，跨 session 召回。/ SQLite fact store, cross-session recall
- **Compaction / 压缩**: 默认开启（``DefaultCompactor``），长会话自动压缩。
  / enabled by default (``DefaultCompactor``), auto-compacts long conversations

## Next / 下一步
这是 examples/ 的最后一站。完整文档见 `docs/en/index.md`。
/ This is the last stop in examples/. Full documentation at `docs/en/index.md`.
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
    AgentHooks,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    MemorySnapshot,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx

# ── 1. Tools / 工具 ──────────────────────────────────────────────────────


def get_weather(city: str) -> str:
    """Mock weather lookup."""
    weathers = {
        "tokyo": "sunny, 22°C",
        "shanghai": "cloudy, 18°C",
        "london": "rainy, 12°C",
    }
    return f"Weather in {city}: {weathers.get(city.lower(), 'unknown')}"


def calculator(expression: str) -> str:
    """Safe eval for simple arithmetic."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return f"Error: unsafe expression: {expression!r}"
    try:
        return str(eval(expression))
    except Exception as exc:
        return f"Error: {exc}"


REGISTRY = ToolRegistry()
REGISTRY.register(
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city. Param: city (string).",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    ),
    get_weather,
)
REGISTRY.register(
    ToolDefinition(
        name="calculator",
        description="Evaluate a math expression. Param: expression (string, e.g. '2+3*4').",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    ),
    calculator,
)


# ── 2. Hooks / 钩子 ──────────────────────────────────────────────────────


HOOKS = AgentHooks()


def safety_gate(ctx: ToolBeforeCtx) -> None:
    """Block any tool call with suspicious input."""
    args_str = str(ctx.tool_args)
    if any(dangerous in args_str.lower() for dangerous in ("rm -rf", "sudo", "delete all")):
        ctx.output = "[blocked by safety gate]"
        ctx.directive = HookDirective.SKIP


HOOKS.register(HookPoint.TOOL_BEFORE, safety_gate)


# ── 3. Memory / 记忆 ─────────────────────────────────────────────────────


class SqliteFactMemory:
    """Simple key-value fact store backed by SQLite."""

    _FACT_RE = re.compile(r"FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", re.M)

    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT)")

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        if not rows:
            return []
        return [{"content": "Known facts:\n" + "\n".join(f"- {r[0]}: {r[1]}" for r in rows)}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        captured = self._FACT_RE.findall(snapshot.final_text or "")
        if captured:
            with sqlite3.connect(self.db_path) as c:
                c.executemany("INSERT OR REPLACE INTO facts VALUES (?, ?)", captured)
                c.commit()


# ── 4. Events / 事件 ─────────────────────────────────────────────────────


def build_event_bus() -> tuple[AgentEventBus, list[str]]:
    """Subscribe to streaming and tool events for demo output."""
    bus = AgentEventBus()
    log: list[str] = []

    def on_stream(event):
        print(event.data.text, end="", flush=True)

    def on_tool(event):
        name = event.data.name
        inp = event.data.tool_input
        log.append(f"[Tool] {name}({inp})")

    bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
    bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_tool)
    return bus, log


# ── 5. System Prompt / 系统提示词 ─────────────────────────────────────────


SYSTEM = (
    "You are a helpful assistant. You have two tools:\n"
    "- get_weather(city) — get current weather\n"
    "- calculator(expression) — evaluate math\n\n"
    "Use them when appropriate. After answering, if the user told you "
    "a personal fact, append one line:\n  FACT: key=value\n"
    "Be concise."
)


# ── 6. Main / 主程序 ─────────────────────────────────────────────────────


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="power_loop_ex19_") as tmp:
        db_path = str(Path(tmp) / "sessions.db")
        mem_path = str(Path(tmp) / "memory.db")
        memory = SqliteFactMemory(mem_path)

        llm = make_llm(max_tokens=300, temperature=0.0)

        # ── Session A: tool calling + memory ─────────────────────────────
        #    工具调用 + 记忆
        print("=== Session A: tool calling + memory ===\n")
        bus_a, tool_log_a = build_event_bus()

        loop = StatefulAgentLoop(
            llm=llm, db_path=db_path, tool_registry=REGISTRY,
            hooks=HOOKS, event_bus=bus_a,
            config=AgentLoopConfig(
                system_prompt=SYSTEM, max_rounds=4, memory=memory,
            ),
        )
        try:
            print("User: What's the weather in Tokyo and what is 15 * 7?")
            sid_a = await loop.new_session(metadata={"label": "session-a"})
            r1 = await loop.send(
                "What's the weather in Tokyo and what is 15 * 7?",
                session_id=sid_a,
            )
            print(f"\n\n[Session A] status={r1.status}, rounds={r1.rounds}")
            print(f"[Session A] tools used: {tool_log_a}")
        finally:
            await loop.aclose()

        # ── Session B: cross-session memory recall ───────────────────────
        #    跨会话记忆召回
        print("\n=== Session B: cross-session memory recall ===\n")
        bus_b, tool_log_b = build_event_bus()

        loop2 = StatefulAgentLoop(
            llm=llm, db_path=db_path, tool_registry=REGISTRY,
            hooks=HOOKS, event_bus=bus_b,
            config=AgentLoopConfig(
                system_prompt=SYSTEM, max_rounds=4, memory=memory,
            ),
        )
        try:
            print("User: My name is Alan. I live in Shanghai. Confirm in one sentence.")
            sid_b = await loop2.new_session(metadata={"label": "session-b"})
            r2 = await loop2.send(
                "My name is Alan. I live in Shanghai. Confirm in one sentence.",
                session_id=sid_b,
            )
            print(f"\n\n[Session B] status={r2.status}, rounds={r2.rounds}")
        finally:
            await loop2.aclose()

        # ── Verify memory persisted / 验证记忆已持久化 ────────────────────
        with sqlite3.connect(mem_path) as c:
            facts = {r[0]: r[1] for r in c.execute("SELECT key, value FROM facts").fetchall()}
        print(f"\n[Memory] facts stored: {facts}")

        # ── Session C: new session, should recall Alan's facts ───────────
        #    新会话，应当召回 Alan 的事实
        print("\n=== Session C: recall from memory ===\n")
        bus_c, _ = build_event_bus()

        loop3 = StatefulAgentLoop(
            llm=llm, db_path=db_path, tool_registry=REGISTRY,
            hooks=HOOKS, event_bus=bus_c,
            config=AgentLoopConfig(
                system_prompt=SYSTEM, max_rounds=2, memory=memory,
            ),
        )
        try:
            print("User: What is my name and where do I live?")
            sid_c = await loop3.new_session(metadata={"label": "session-c"})
            r3 = await loop3.send("What is my name and where do I live?", session_id=sid_c)
            print(f"\n\n[Session C] reply: {r3.final_text}")
            print(f"[Session C] status={r3.status}, rounds={r3.rounds}")
        finally:
            await loop3.aclose()

        print(f"\n[Done] 3 sessions completed. "
              f"Session A={sid_a[-8:]}, B={sid_b[-8:]}, C={sid_c[-8:]}")


if __name__ == "__main__":
    asyncio.run(main())
