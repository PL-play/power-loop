"""19 · Tools, memory, hooks and persisted history on a rebuilt local Runtime.

The fixed demo model avoids remote calls. A deployment resolves an actual
provider and user-scoped memory service in its factory.
"""
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
    AgentHooks,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    LLMResponse,
    LLMService,
    MemorySnapshot,
    SessionStore,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx
from power_loop.runtime.pricing import Free

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
    pricing=Free("local")),
    get_weather,
)
REGISTRY.register(
    ToolDefinition(
        name="calculator",
        description="Evaluate a math expression. Param: expression (string, e.g. '2+3*4').",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    pricing=Free("local")),
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


class ChatModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m.get('role') == 'tool' for m in request.messages):
            return LLMResponse(raw_text='FACT: name=Alan\nAlan: weather is clear.')
        return LLMResponse(raw_text='', tool_calls=[{'id':'weather','type':'function','function':{
            'name':'weather','arguments':'{"city":"Tokyo"}'}}])

    async def close(self):
        pass


async def main():
    with tempfile.TemporaryDirectory(prefix='power_loop_chat_test_') as directory:
        path = str(Path(directory)/'sessions.db')
        memory = SqliteFactMemory(str(Path(directory)/'memory.db'))
        registry = ToolRegistry()
        registry.register(ToolDefinition('weather','Weather lookup', required_params=('city',),
            pricing=Free('local example')), get_weather)
        hooks = AgentHooks()
        hooks.register(HookPoint.TOOL_BEFORE, safety_gate)
        target = None
        for index in range(2):
            store = await SessionStore.open(path)
            bus, events = build_event_bus()
            async with demo_runtime(ChatModel(), registry, store=store, target=target, hooks=hooks, event_bus=bus,
                    config=AgentLoopConfig(memory=memory, max_rounds=3, compactor=None)) as (runtime, loop, target, actor):
                result = await loop.send(target, 'Weather and remember Alan' if index == 0 else 'Continue our conversation',
                    message_id=f'chat-{index}', actor=actor, deadline_ms=int(time.time()*1000)+10000)
                assert result.status == 'completed'
                print(result.final_text, 'events:', len(events))
                rows = await store.load_all_messages(target.session_id)
                assert sum(row.role == 'user' for row in rows) >= index+1


if __name__ == '__main__':
    asyncio.run(main())
