"""17 · 自定义 MemoryProvider：HTTP API 后端跨会话记忆

## What you'll learn
- 实现 ``MemoryProvider`` 协议，用 HTTP API 做后端（模拟真实 DeepTalk 架构）
- ``recall()`` 从远程拉取记忆，``remember()`` 推送到远程
- 软失败：API 不可用时不影响 Agent 回复

## Prerequisites
- 需要 ``.env`` 配置 ``POWER_LOOP_*``
- 不需要真实的 HTTP 服务——本例用 mock 函数模拟

## Run
    python examples/17_custom_memory_provider.py

## Key concepts
- **MemoryProvider**: 两个方法：``recall()`` 在 session 开始时调用，``remember()`` 在结束时调用。
- 失败绝不阻塞用户获取回复——框架发 ``MEMORY_FAILED`` 事件，照常返回。
- 本例模拟「DeepTalk 风格」的 HTTP API 后端——`api/memory/recall` 和 `api/memory/remember`。

## Next
看看 `18_multi_provider.py` — 同时使用三家不同的 LLM provider
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, MemorySnapshot, StatefulAgentLoop

# ── 1. Mock HTTP API ─────────────────────────────────────────────────────


class MockMemoryAPI:
    """模拟后端记忆服务。数据库用内存 dict 代替。"""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}  # user_id → {key: value}

    async def get(self, user_id: str, endpoint: str, payload: dict) -> dict:
        """GET /api/memory/recall"""
        if endpoint == "/api/memory/recall":
            facts = self._store.get(user_id, {})
            if not facts:
                return {"facts": []}
            return {"facts": [{"key": k, "value": v} for k, v in facts.items()]}
        return {}

    async def post(self, user_id: str, endpoint: str, payload: dict) -> dict:
        """POST /api/memory/remember"""
        if endpoint == "/api/memory/remember":
            facts = payload.get("facts", [])
            if user_id not in self._store:
                self._store[user_id] = {}
            for f in facts:
                self._store[user_id][f["key"]] = f["value"]
            return {"status": "ok"}
        return {}


# ── 2. HTTP-backed MemoryProvider ────────────────────────────────────────


class HttpMemoryProvider:
    """MemoryProvider that talks to a remote HTTP API.

    In production, this would use ``httpx`` or ``aiohttp``. Here we mock it.
    """

    def __init__(self, user_id: str, api: MockMemoryAPI):
        self.user_id = user_id
        self.api = api

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        try:
            resp = await self.api.get(self.user_id, "/api/memory/recall", {})
            facts = resp.get("facts", [])
            if not facts:
                return []
            text = "Known facts about the user:\n" + "\n".join(
                f"- {f['key']}: {f['value']}" for f in facts
            )
            return [{"content": text}]
        except Exception:
            return []  # soft-fail

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        try:
            # Extract facts from final_text: look for FACT: key=value lines
            import re
            pattern = re.compile(r"FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", re.M)
            captured = pattern.findall(snapshot.final_text or "")
            if captured:
                facts = [{"key": k, "value": v.strip()} for k, v in captured]
                await self.api.post(self.user_id, "/api/memory/remember", {"facts": facts})
        except Exception:
            pass  # soft-fail


# ── 3. Run ────────────────────────────────────────────────────────────────


SYSTEM = (
    "You are a concise assistant. After answering, if the user told you "
    "a personal fact, append one or more lines of the form:\n"
    "  FACT: key=value\n"
    "where key is a short snake_case identifier."
)


async def main() -> None:
    llm = make_llm(max_tokens=200, temperature=0.0)
    bus = AgentEventBus()
    events: list = []
    bus.subscribe(
        None,
        lambda e: events.append(e.type.value)
        if e.type in (AgentEventType.MEMORY_RECALLED, AgentEventType.MEMORY_FAILED)
        else None,
    )

    api = MockMemoryAPI()
    memory = HttpMemoryProvider(user_id="user_42", api=api)

    # Session A: teach the agent
    loop = StatefulAgentLoop(
        llm=llm, event_bus=bus,
        config=AgentLoopConfig(
            system_prompt=SYSTEM, max_rounds=1, compactor=None, memory=memory,
        ),
    )
    try:
        r1 = await loop.send("My name is Alan and I work at Acme Corp. One sentence reply.")
        print(f"[Session A] reply: {r1.final_text}")
        print(f"[Session A] events: {events}")
    finally:
        loop.close()

    # Verify: facts stored in mock API
    assert "Alan" in str(api._store.get("user_42", {}))

    # Session B: new session, same memory provider → agent remembers
    events.clear()
    bus2 = AgentEventBus()
    bus2.subscribe(
        None,
        lambda e: events.append(e.type.value)
        if e.type in (AgentEventType.MEMORY_RECALLED, AgentEventType.MEMORY_FAILED)
        else None,
    )
    loop2 = StatefulAgentLoop(
        llm=llm, event_bus=bus2,
        config=AgentLoopConfig(
            system_prompt=SYSTEM, max_rounds=1, compactor=None, memory=memory,
        ),
    )
    try:
        r2 = await loop2.send("What is my name and where do I work? One sentence.")
        print(f"[Session B] reply: {r2.final_text}")
        print(f"[Session B] events: {events}")
    finally:
        loop2.close()


if __name__ == "__main__":
    asyncio.run(main())