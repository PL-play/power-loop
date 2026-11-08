"""Advanced runtime 04: events observe custom tool lifecycle.
Uses an offline model and the current Runtime factory; no real deployment occurs.
"""

from __future__ import annotations

import asyncio
import time

from _helpers import demo_runtime

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.runtime.pricing import Free


def restart_worker(name: str) -> str:
    return f"worker {name} restarted"


class ToolModel(LLMService):
    async def complete(self, request, **kwargs):
        results = [m['content'] for m in request.messages if m.get('role') == 'tool']
        if results:
            return LLMResponse(raw_text=results[-1])
        return LLMResponse(raw_text='', tool_calls=[{'id':'operation','type':'function','function':{
            'name':'restart_worker','arguments':'{"name":"search-indexer"}'}}])
    async def close(self):
        pass

async def main() -> list[str]:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            pricing=Free("local simulated action"),
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

    async with demo_runtime(ToolModel(), registry, event_bus=bus) as (runtime, loop, target, actor):
        result = await loop.send(target, 'Run simulated operation', message_id='operation', actor=actor,
            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
    assert audit == ['tool_call_started:restart_worker', 'tool_call_completed:restart_worker']
    print(audit)
    return audit

if __name__ == '__main__':
    asyncio.run(main())
