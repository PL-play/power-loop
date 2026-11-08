"""00 · One Runtime input and its durable result; no credentials or Redis."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService


class HelloModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Hello from Power Loop 11.')

    async def close(self):
        pass


async def main():
    async with demo_runtime(HelloModel()) as (runtime, loop, target, actor):
        result = await loop.send(target, 'hello', message_id='hello-1', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
