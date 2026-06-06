"""Advanced runtime 04: events observe custom tool lifecycle.

Scenario
--------
An operations console wants an audit stream for tool calls without changing
tool implementations. Subscribe to the event bus and collect lifecycle events.

Run:
    python examples/advanced_runtime/04_events_observability.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runtime_helpers import ScriptedLLM, tool_response

from llm_client.interface import LLMResponse
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

    llm = ScriptedLLM(
        responses=[
            tool_response("tc-worker", "restart_worker", '{"name":"search-indexer"}'),
            LLMResponse(raw_text="Worker restart complete."),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=SessionStore.open(":memory:"),
        tool_registry=registry,
        event_bus=bus,
        config=AgentLoopConfig(system_prompt="You operate workers.", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()

    await loop.send("Restart the search indexer.", session_id=sid)
    print("\n".join(audit))
    return audit


if __name__ == "__main__":
    asyncio.run(main())
