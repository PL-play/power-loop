"""17 · A host memory adapter over an explicitly supplied API client.

The demo uses an in-memory fake API. A deployment supplies its authenticated
HTTP client in each node's factory; no network fallback or SDK business writes.
"""
from __future__ import annotations

import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService

# ── 1. Mock HTTP API ─────────────────────────────────────────────────────


class MockMemoryAPI:
    """模拟后端记忆服务。数据库用内存 dict 代替。
    / Mock backend memory service. Uses in-memory dict as the database."""

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
        response = await self.api.get(self.user_id, '/api/memory/recall', {})
        facts = response['facts']
        return [{'content':'Known facts: '+str(facts)}] if facts else []

    async def remember(self, *, snapshot, session_id):
        import re
        captured = re.findall(r'FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$', snapshot.final_text or '', re.M)
        if captured:
            await self.api.post(self.user_id, '/api/memory/remember',
                {'facts':[{'key':key,'value':value} for key,value in captured]})


# ── 3. Run / 运行 ────────────────────────────────────────────────────────


SYSTEM = (
    "You are a concise assistant. After answering, if the user told you "
    "a personal fact, append one or more lines of the form:\n"
    "  FACT: key=value\n"
    "where key is a short snake_case identifier."
)


class MemoryModel(LLMService):
    async def complete(self, request, **kwargs):
        if 'Known facts:' in str(request.messages):
            assert 'Alan' in str(request.messages)
            return LLMResponse(raw_text='Alan works at Acme.')
        return LLMResponse(raw_text='FACT: name=Alan\nFACT: employer=Acme')

    async def close(self):
        pass


async def main():
    api = MockMemoryAPI()
    memory = HttpMemoryProvider('example-user', api)
    for index in range(2):
        async with demo_runtime(MemoryModel(), config=AgentLoopConfig(memory=memory, compactor=None)) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Remember' if index == 0 else 'Recall', message_id=f'memory-{index}',
                actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
            print(result.final_text)
    assert result.final_text == 'Alan works at Acme.'
    assert (await HttpMemoryProvider('other-user', api).recall(messages=[],session_id=None)) == []


if __name__ == '__main__':
    asyncio.run(main())
