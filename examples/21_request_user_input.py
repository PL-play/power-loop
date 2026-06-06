"""21 · Request user input / 可恢复的人类输入

Demonstrates ``request_user_input`` as a special built-in tool:

1. the LLM requests external input;
2. the loop returns ``status="waiting_for_input"`` instead of blocking;
3. the caller submits the collected answer with ``submit_input``;
4. the loop continues from the same persisted session.

Run:

    python examples/21_request_user_input.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry


@dataclass
class ScriptedLLM(LLMService):
    responses: list[LLMResponse]
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    index: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(request.messages))
        response = self.responses[self.index]
        self.index += 1
        return response

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _tool_call_response() -> LLMResponse:
    return LLMResponse(
        raw_text="I need a confirmation before continuing.",
        tool_calls=[
            {
                "id": "tc_confirm_1",
                "type": "function",
                "function": {
                    "name": "request_user_input",
                    "arguments": (
                        '{"kind":"confirm","prompt":"Send this relationship summary?",'
                        '"options":[{"id":"send","label":"Send"},{"id":"revise","label":"Revise"}],'
                        '"metadata":{"surface":"chat_composer"}}'
                    ),
                },
            }
        ],
    )


async def main() -> None:
    llm = ScriptedLLM(
        responses=[
            _tool_call_response(),
            LLMResponse(raw_text="Confirmed. I will send the summary now."),
        ]
    )
    registry = create_default_tool_registry(include=["request_user_input"])

    with tempfile.TemporaryDirectory(prefix="power-loop-input-") as tmp:
        store = SessionStore.open(f"{tmp}/sessions.sqlite3")
        loop = StatefulAgentLoop(
            llm=llm,
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(system_prompt="Ask for confirmation when needed.", max_rounds=3, compactor=None),
        )
        sid = loop.new_session()

        waiting = await loop.send("Draft and send a summary.", session_id=sid)
        interaction = waiting.pending_interactions[0]
        print(f"status: {waiting.status}")
        print(f"prompt: {interaction['prompt']}")
        print(f"options: {[option['id'] for option in interaction['options']]}")

        result = await loop.submit_input(
            sid,
            interaction["interaction_id"],
            {"choice": "send"},
        )
        print(f"status: {result.status}")
        print(f"final: {result.final_text}")
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
