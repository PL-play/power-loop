"""Advanced runtime 02: a tool uses public runtime context.

Scenario
--------
A support triage tool needs the current session metadata and message count. It
uses `get_tool_runtime_context()` instead of private internals, writes runtime
state, and a custom projector exposes that state to the following LLM round.

Note
----
This example calls your configured real LLM and asks it to invoke a tool.
Make sure `.env` is configured and be mindful of provider usage/cost.

Run:
    python examples/advanced_runtime/02_tool_runtime_context.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    RuntimeProjector,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)


class TicketProjector(RuntimeProjector):
    def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        ticket = store.get_runtime_state(session_id, "ticket", default={}) or {}
        if not ticket:
            return []
        return [{"role": "user", "name": "ticket_state", "content": f"<ticket_state>{ticket}</ticket_state>"}]


def capture_ticket_context() -> str:
    ctx = get_tool_runtime_context(required=True)
    session = ctx.store.get_session(ctx.session_id)
    messages = ctx.loop.get_messages(ctx.session_id)
    ctx.store.set_runtime_state(
        ctx.session_id,
        "ticket",
        {
            "customer": session.metadata["customer"],
            "messages_seen": len(messages),
        },
    )
    return "ticket context captured"


async def main() -> str:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="capture_ticket_context", description="Capture current support ticket context."),
        capture_ticket_context,
    )
    store = SessionStore.open(":memory:")
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=512, temperature=0.0),
        store=store,
        tool_registry=registry,
        config=AgentLoopConfig(
            system_prompt=(
                "You triage support tickets. First call capture_ticket_context. "
                "After the tool returns, use the projected ticket_state and mention the customer name."
            ),
            max_rounds=4,
            compactor=None,
            runtime_projectors=(TicketProjector(),),
        ),
    )
    sid = loop.new_session(metadata={"customer": "Ada"})

    result = await loop.send("Capture this ticket context.", session_id=sid)
    print(result.final_text)
    print("runtime ticket:", store.get_runtime_state(sid, "ticket"))
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
