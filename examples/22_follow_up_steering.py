"""22 · Follow-up steering / 运行中追加指引

This example uses a real LLM. It starts a multi-round ``send()`` (tool call +
final answer), injects ``follow_up()`` while the session lock is still held,
and shows how steering text appears in the next LLM round as a wrapped
``<follow_up>`` user message.

Run:

    python examples/22_follow_up_steering.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    FollowUpQueued,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)

DEFAULT_STEERING = (
    "Before you finish, your final answer MUST include the exact word STEERED "
    "in uppercase."
)

SYSTEM_PROMPT = """You are demonstrating in-flight steering with follow_up.

Rules:
- For each user request, call the echo tool exactly once with a short phrase
  related to the topic.
- After the tool result arrives, write one concise final sentence.
- Obey any <follow_up> guidance that appears before your final answer.
"""

ECHO_TOOL = ToolDefinition(
    name="echo",
    description="Repeat the given text verbatim.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    required_params=("text",),
)


def _echo_handler(**kwargs: Any) -> str:
    return str(kwargs.get("text") or "")


async def main(steering: str | None = None) -> dict[str, Any]:
    llm = make_llm(max_tokens=700, temperature=0.0)
    registry = ToolRegistry()
    registry.register(ECHO_TOOL, _echo_handler)
    steering_text = steering or DEFAULT_STEERING

    with tempfile.TemporaryDirectory(prefix="power-loop-follow-up-") as tmp:
        store = SessionStore.open(f"{tmp}/sessions.sqlite3")
        loop = StatefulAgentLoop(
            llm=llm,
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(system_prompt=SYSTEM_PROMPT, max_rounds=4, compactor=None),
        )

        try:
            sid = loop.new_session(metadata={"demo": "follow_up"})
            send_task = asyncio.create_task(
                loop.send(
                    "Write one sentence about patience while waiting for good news.",
                    session_id=sid,
                )
            )

            for _ in range(200):
                if loop._lock_for(sid).locked():
                    break
                await asyncio.sleep(0.005)
            else:
                raise RuntimeError("expected session lock while send() is running")

            queued = await loop.follow_up(steering_text, sid)
            if not isinstance(queued, FollowUpQueued):
                raise RuntimeError(f"expected FollowUpQueued while running, got {queued!r}")

            result = await send_task
            messages = loop.get_messages(sid)
            follow_rows = [
                m
                for m in messages
                if m.get("name") == "follow_up"
                or "<follow_up>" in str(m.get("content") or "")
            ]

            payload = {
                "session_id": sid,
                "status": result.status,
                "final_text": result.final_text,
                "queue_depth": queued.queue_depth,
                "follow_up_message_count": len(follow_rows),
                "steering": steering_text,
            }
            print(_json(payload))
            return payload
        finally:
            await llm.close()
            store.close()


def _json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return repr(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


if __name__ == "__main__":
    asyncio.run(main())
