"""Advanced runtime 04: events observe custom tool lifecycle.

Scenario
--------
An operations console wants an audit stream for tool calls without changing
tool implementations. Subscribe to the event bus and collect lifecycle events.

Note
----
This example calls your configured real LLM and asks it to invoke a tool.
Make sure `.env` is configured and be mindful of provider usage/cost.

Run:
    python examples/advanced_runtime/04_events_observability.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import make_llm

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)


def restart_worker(name: str) -> str:
    return f"worker {name} restarted"


async def main() -> list[str]:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="restart_worker",
            description="Restart a named worker process.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            required_params=("name",),
        ),
        restart_worker,
    )

    audit: list[str] = []
    bus = AgentEventBus()

    def on_event(event: AgentEvent) -> None:
        if event.type in {AgentEventType.TOOL_CALL_STARTED, AgentEventType.TOOL_CALL_COMPLETED}:
            audit.append(f"{event.type.value}:{event.data.name}")

    bus.subscribe(None, on_event)

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=512, temperature=0.0),
        store=SessionStore.open(":memory:"),
        tool_registry=registry,
        event_bus=bus,
        config=AgentLoopConfig(
            system_prompt=(
                "You operate workers. Call restart_worker with name='search-indexer'. "
                "After the tool returns, answer briefly."
            ),
            max_rounds=4,
            compactor=None,
        ),
    )
    sid = loop.new_session()

    await loop.send("Restart the search indexer.", session_id=sid)
    print("\n".join(audit))
    return audit


if __name__ == "__main__":
    asyncio.run(main())
