"""01 · Reuse a TargetRef and unique message IDs; inspect persisted history."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService


class HistoryModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='teal' if 'teal' in str(request.messages) else 'unknown')

    async def close(self):
        pass


async def main():
    async with demo_runtime(HistoryModel()) as (runtime, loop, target, actor):
        for index, body in enumerate(['My favorite color is teal.', 'What is my favorite color?']):
            result = await loop.send(target, body, message_id=f'turn-{index}', actor=actor,
                                     deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == 'teal'
            print(result.final_text)
        rows = await runtime.store.load_all_messages(target.session_id)
        assert [row.role for row in rows] == ['user', 'assistant', 'user', 'assistant']
        print('history rows:', len(rows))
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
