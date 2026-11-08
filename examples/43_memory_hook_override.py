"""43 · Factory-owned hooks: default memory, disabled memory, custom injection."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentHooks, AgentLoopConfig, HookPoint, LLMResponse, LLMService, MemoryRecallHook


class Memory:
    async def recall(self, **kwargs):
        return [{'content':'Project codename: Bluefin'}]
    async def remember(self, **kwargs):
        pass

class SeeingModel(LLMService):
    def __init__(self, expected, absent):
        self.expected, self.absent = expected, absent
    async def complete(self, request, **kwargs):
        text = str(request.messages)
        assert self.expected in text and (not self.absent or self.absent not in text)
        return LLMResponse(raw_text=self.expected)
    async def close(self):
        pass

async def main():
    for mode in ['default', 'disabled', 'custom']:
        hooks = AgentHooks()
        if mode == 'custom':
            def inject(ctx):
                ctx.messages.append({'role':'system','name':'memory_custom','content':'Be playful'})
            hooks.register(HookPoint.LLM_BEFORE, inject, name=MemoryRecallHook.NAME)
        expected = 'Bluefin' if mode == 'default' else 'Be playful' if mode == 'custom' else 'Recall'
        config = AgentLoopConfig(compactor=None, memory=Memory(), builtin_memory_hook=mode != 'disabled')
        async with demo_runtime(SeeingModel(expected, '' if mode == 'default' else 'Bluefin'),
                config=config, hooks=hooks) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Recall', message_id=mode, actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
            print(mode, result.final_text)
    return 'Memory default, disable and override verified'

if __name__ == '__main__':
    asyncio.run(main())
