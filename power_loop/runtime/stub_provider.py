"""A deterministic, offline LLM service for dev and cross-process testing.

Selected via config (``provider="echo"``), so it is reachable purely through
env/config — which matters because a real subprocess worker rebuilds its LLM
from config and **cannot** be handed a Python factory/closure. Tests that spawn
a worker process use ``POWER_LOOP_PROVIDER=echo`` to get a fast, deterministic
agent loop without a real provider or network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
)

__all__ = ["EchoLLMService"]


@dataclass
class EchoLLMService(LLMService):
    """Returns ``reply`` if set, otherwise echoes the last user message.

    Never makes a tool call, so an agent loop with this service completes in one
    round — ideal for exercising orchestration/IPC without LLM nondeterminism.
    """

    reply: str | None = None
    prefix: str = ""

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text=None,
        on_chunk_think=None,
        on_stream_end=None,
    ) -> LLMResponse:
        text = self.reply
        if text is None:
            last = ""
            for m in reversed(getattr(request, "messages", None) or []):
                if m.get("role") == "user":
                    last = str(m.get("content") or "")
                    break
            text = f"{self.prefix}{last}"
        usage = LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(raw_text=text, content_text=text, token_usage=usage)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _one() -> AsyncIterator[LLMStreamChunk]:
            yield LLMStreamChunk(delta_text=(self.reply or ""), is_final=True)

        return _one()

    async def close(self) -> None:
        return None
