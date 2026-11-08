"""Advanced runtime 02: a tool uses public runtime context.
Uses an offline model and the current Runtime factory; no real deployment occurs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from _helpers import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    RuntimeProjector,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop.runtime.pricing import Free


class TicketProjector(RuntimeProjector):
    async def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        ticket = await store.get_runtime_state(session_id, "ticket", default={}) or {}
        if not ticket:
            return []
        return [{"role": "user", "name": "ticket_state", "content": f"<ticket_state>{ticket}</ticket_state>"}]


async def capture_ticket_context() -> str:
    ctx = get_tool_runtime_context(required=True)
    session = await ctx.store.get_session(ctx.session_id)
    messages = await ctx.store.load_all_messages(ctx.session_id)
    await ctx.store.set_runtime_state(
        ctx.session_id,
        "ticket",
        {
            "customer": session.metadata["customer"],
            "messages_seen": len(messages),
        },
    )
    return "ticket context captured"


class TicketModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m.get('role') == 'tool' for m in request.messages):
            assert 'Ada' in str(request.messages)
            return LLMResponse(raw_text='Captured Ada ticket')
        return LLMResponse(raw_text='', tool_calls=[{'id':'ticket','type':'function','function':{
            'name':'capture_ticket_context','arguments':'{}'}}])
    async def close(self):
        pass

async def main():
    registry = ToolRegistry()
    registry.register(ToolDefinition(name='capture_ticket_context', description='Capture ticket context', pricing=Free('local')), capture_ticket_context)
    config = AgentLoopConfig(compactor=None, runtime_projectors=(TicketProjector(),))
    async with demo_runtime(TicketModel(), registry, config=config) as (runtime, loop, target, actor):
        await runtime.store.merge_session_metadata(target.session_id, {'customer':'Ada'})
        result = await loop.send(target, 'Capture ticket', message_id='ticket', actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        ticket = await runtime.store.get_runtime_state(target.session_id, 'ticket')
        assert ticket['customer'] == 'Ada' and ticket['messages_seen'] > 0
        print(result.final_text, ticket)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
