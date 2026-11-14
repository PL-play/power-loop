import asyncio
from typing import Any

import pytest

from power_loop import LLMRequest, LLMResponse, LLMService, SessionStore


class _Gate(LLMService):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.gates: dict[int, asyncio.Event] = {}
        self.system_prompts = []

    def gate(self, n: int) -> asyncio.Event:
        self.gates[n] = asyncio.Event()
        return self.gates[n]

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        n = len(self.calls)
        self.calls.append([dict(m) for m in request.messages])
        self.system_prompts.append(request.system_prompt)
        if n in self.gates:
            await self.gates[n].wait()
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="done")

    async def close(self) -> None:
        return None

async def _wait_calls(llm: _Gate, n: int) -> None:
    for _ in range(400):
        if len(llm.calls) >= n:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"expected ≥{n} LLM calls, got {len(llm.calls)}")

@pytest.fixture
async def store():
    session_store = await SessionStore.open(":memory:")
    yield session_store
    await session_store.close()
