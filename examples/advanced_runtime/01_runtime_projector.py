"""Advanced runtime 01: project incident state into the next LLM round.
Uses an offline model and the current Runtime factory; no real deployment occurs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from _helpers import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService, RuntimeProjector


class IncidentProjector(RuntimeProjector):
    async def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        incident = await store.get_runtime_state(session_id, "incident", default={}) or {}
        if not incident:
            return []
        return [
            {
                "role": "user",
                "name": "incident_runtime_state",
                "content": (
                    "<incident_state>\n"
                    f"severity: {incident['severity']}\n"
                    f"summary: {incident['summary']}\n"
                    "</incident_state>"
                ),
            }
        ]


class IncidentModel(LLMService):
    async def complete(self, request, **kwargs):
        assert 'SEV2' in str(request.messages)
        return LLMResponse(raw_text='SEV2: checkout latency above threshold')
    async def close(self):
        pass

async def main():
    config = AgentLoopConfig(compactor=None, runtime_projectors=(IncidentProjector(),))
    async with demo_runtime(IncidentModel(), config=config) as (runtime, loop, target, actor):
        await runtime.store.set_runtime_state(target.session_id, 'incident',
            {'severity':'SEV2','summary':'checkout latency above threshold'})
        result = await loop.send(target, 'Latest incident?', message_id='incident', actor=actor,
            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        assert not any('<incident_state>' in row.content for row in await runtime.store.load_all_messages(target.session_id))
        print(result.final_text)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
