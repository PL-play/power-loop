"""Run with: python examples/design208_timers.py (no Redis or model credentials)."""
import asyncio
import time

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    InputRef,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    TimerContext,
    TrustedActor,
)
from power_loop.runtime.store.factory import open_store


class Echo(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Timer received', content_text='Timer received')

    async def close(self):
        pass


async def main():
    store = await open_store(':memory:', schema='tasks')
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(Echo(), None, AgentLoopConfig(max_rounds=1))
    factory.register('echo@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='example', required_profiles=['echo@1'])
    try:
        await runtime.worker.start()
        actor = TrustedActor('example', 'host', None)
        target = await runtime.open_session(RuntimeDescriptor(1, 'agent', 'echo@1', None,
            'builtin.empty@1', 'builtin.none@1', 'builtin.none@1', {}, {}), actor=actor)
        timer = await runtime.timer_backend.create(TimerContext(store.store_id, target.session_id, actor),
            delay_s=.01, note='Check progress')
        async with asyncio.timeout(10):
            while (await store.get_timer(target.session_id,timer.timer_id)).status != 'fired':
                await asyncio.sleep(.1)  # Wait for the Runtime-owned driver to accept it.
        result = await runtime.client.wait(InputRef(store.store_id, target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0'),
            actor=actor, deadline_ms=int(time.time()*1000)+10000)
        print(result.final_text)
    finally:
        await runtime.aclose()


if __name__ == '__main__':
    asyncio.run(main())
