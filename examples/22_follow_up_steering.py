"""22 · A second send joins the running job; duplicate message IDs are recalled."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService


class SteeringModel(LLMService):
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            await self.release.wait()
            return LLMResponse(raw_text='Considering the first input.')
        text = '\n'.join(str(m.get('content', '')) for m in request.messages)
        assert 'STEERED' in text
        return LLMResponse(raw_text='STEERED: patience helps.')

    async def close(self):
        pass

async def main():
    model = SteeringModel()
    async with demo_runtime(model) as (runtime, loop, target, actor):
        receipt = await runtime.client.send(target, 'Discuss patience', message_id='first', actor=actor)
        await asyncio.wait_for(model.entered.wait(), 10)
        added = await runtime.client.send(target, 'Include STEERED in your answer', message_id='guidance', actor=actor)
        duplicate = await runtime.client.send(target, 'Include STEERED in your answer', message_id='guidance', actor=actor)
        assert added.input_ref == duplicate.input_ref
        model.release.set()
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and 'STEERED' in result.final_text
        print(result.final_text)

if __name__ == '__main__':
    asyncio.run(main())
