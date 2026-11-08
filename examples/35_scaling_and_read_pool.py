"""35 · Explicit worker capacity bounds concurrent model calls.

This local example uses one store, coordinator and transport. Shared workers
require host adapters and homogeneous capability profiles. This is a behavior
example, not a database/transport performance benchmark.
"""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, RuntimeDescriptor


class ConcurrentModel(LLMService):
    def __init__(self):
        self.active = 0
        self.peak = 0
    async def complete(self, request, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.05)
            return LLMResponse(raw_text='Done')
        finally:
            self.active -= 1
    async def close(self):
        pass

async def main():
    model = ConcurrentModel()
    async with demo_runtime(model, capacity=2) as (runtime, loop, target, actor):
        descriptor = RuntimeDescriptor(1, 'agent', 'example@1', None, 'example.tools@1', 'builtin.none@1', 'builtin.none@1', {}, {})
        targets = [target] + [await runtime.open_session(descriptor, actor=actor) for _ in range(3)]
        results = await asyncio.gather(*(loop.send(t, 'Work', message_id=f'work-{i}', actor=actor,
            deadline_ms=int(time.time()*1000)+10000) for i, t in enumerate(targets)))
        assert all(r.status == 'completed' for r in results) and model.peak == 2
        summary = f'{len(results)} sessions completed; peak model concurrency={model.peak}'
        print(summary)
        return summary

if __name__ == '__main__':
    asyncio.run(main())
