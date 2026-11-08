"""25 · Result usage is per input; stored session statistics accumulate.

Usage events are for observation. A billing host must use the TaskRuntime
billing port and must not treat an event subscriber as its authoritative ledger.
"""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import (
    AgentEventBus,
    AgentEventType,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    create_default_tool_registry,
)


class MeteredModel(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        calls = [{'id':'todo','type':'function','function':{'name':'todo',
            'arguments':'{"items":[{"id":"report","text":"Write report","status":"pending"}]}'}}] if self.calls == 1 else []
        return LLMResponse(raw_text='' if calls else 'Done', tool_calls=calls,
            token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def close(self):
        pass

async def main():
    events = []
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.USAGE_UPDATED, lambda event: events.append(event.payload))
    async with demo_runtime(MeteredModel(), create_default_tool_registry(include=['todo']), event_bus=bus) as (runtime, loop, target, actor):
        results = []
        for index in range(2):
            result = await loop.send(target, 'Update report', message_id=f'usage-{index}', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
            results.append(result)
        assert results[0].usage['total_tokens'] == 30 and results[1].usage['total_tokens'] == 15
        stats = await runtime.store.get_session_stats(target.session_id)
        assert stats.prompt_tokens == 30 and stats.completion_tokens == 15
        print([r.usage for r in results], stats)
    assert len(events) == 3

if __name__ == '__main__':
    asyncio.run(main())
