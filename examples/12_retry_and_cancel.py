"""12 · Unknown calls are not replayed; explicit new input and cancellation."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService


class FailingOnce(LLMService):
    def __init__(self):
        self.calls = 0
        self.entered = asyncio.Event()

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError('original result unknown')
        if self.calls == 3:
            self.entered.set()
            await asyncio.Event().wait()
        return LLMResponse(raw_text='New input succeeded.')

    async def close(self):
        pass


async def main():
    model = FailingOnce()
    async with demo_runtime(model) as (runtime, loop, target, actor):
        failed = await loop.send(target, 'first', message_id='first', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        assert failed.status == 'failed' and model.calls == 1
        good = await loop.send(target, 'new input', message_id='second', actor=actor,
                               deadline_ms=int(time.time()*1000)+10000)
        assert good.status == 'completed' and model.calls == 2
        receipt = await runtime.client.send(target, 'long work', message_id='third', actor=actor)
        await asyncio.wait_for(model.entered.wait(), 5)
        status = await runtime.client.inspect(receipt.input_ref, actor=actor)
        await runtime.client.cancel(status.job_ref, actor=actor)
        cancelled = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert cancelled.status == 'cancelled'
        print(failed.status, good.status, cancelled.status)


if __name__ == '__main__':
    asyncio.run(main())
