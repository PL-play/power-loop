"""25 · Result usage is per send; the host sums returned result.usage for totals.

Each ``send`` returns its own ``result.usage`` dict; the combined totals shown in
this example are computed by the host summing the two returned ``result.usage``
records (rather than reading SDK-persisted session statistics).

``USAGE_UPDATED`` events are emitted once per model call for observation only.
Authoritative billing is owned by the host via its ``BillingChargeEvent``
interface and must not treat an event subscriber as its billing ledger.
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
        calls = (
            [{
                'id': 'todo',
                'type': 'function',
                'function': {
                    'name': 'todo',
                    'arguments': '{"items":[{"id":"report","text":"Write report","status":"pending"}]}',
                },
            }]
            if self.calls == 1
            else []
        )
        return LLMResponse(
            raw_text='' if calls else 'Done',
            tool_calls=calls,
            token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def close(self):
        pass


async def main():
    events = []
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.USAGE_UPDATED, lambda event: events.append(event.payload))
    model = MeteredModel()
    async with demo_runtime(
        model,
        create_default_tool_registry(include=['todo']),
        event_bus=bus,
    ) as (runtime, loop, target, actor):
        results = []
        for index in range(2):
            result = await loop.send(
                target,
                'Update report',
                message_id=f'usage-{index}',
                actor=actor,
                deadline_ms=int(time.time() * 1000) + 10000,
            )
            assert result.status == 'completed'
            results.append(result)

        assert model.calls == 3
        assert results[0].usage['total_tokens'] == 30 and results[1].usage['total_tokens'] == 15

        host_totals = {
            key: sum(int(r.usage[key]) for r in results)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
        }
        assert host_totals == {'prompt_tokens': 30, 'completion_tokens': 15, 'total_tokens': 45}
        for index, r in enumerate(results, start=1):
            print(f'send #{index} usage:', dict(r.usage))
        print('host-summed total (from both result.usage):', host_totals)

    assert len(events) == 3
    print(f'observed USAGE_UPDATED events: {len(events)} (observation only; billing uses host BillingChargeEvent)')


if __name__ == '__main__':
    asyncio.run(main())
