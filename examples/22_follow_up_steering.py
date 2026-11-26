"""22 · Follow-up guidance queues for the next send turn; duplicate message IDs are deduplicated."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService


class SteeringModel(LLMService):
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.seen_requests = []

    async def complete(self, request, **kwargs):
        self.calls += 1
        messages = list(request.messages)
        self.seen_requests.append(messages)
        if self.calls == 1:
            self.entered.set()
            await self.release.wait()
            return LLMResponse(raw_text='Considering the first input.')
        text = '\n'.join(str(m.get('content', '')) for m in messages)
        assert 'STEERED' in text
        return LLMResponse(raw_text='STEERED: patience helps.')

    async def close(self):
        pass


async def main():
    model = SteeringModel()
    guidance_text = 'Include STEERED in your answer'
    async with demo_runtime(model) as (runtime, loop, target, actor):
        receipt = await runtime.client.send(target, 'Discuss patience', message_id='first', actor=actor)
        await asyncio.wait_for(model.entered.wait(), 10)
        added = await runtime.client.send(target, guidance_text, message_id='guidance', actor=actor)
        duplicate = await runtime.client.send(target, guidance_text, message_id='guidance', actor=actor)
        assert added.input_ref == duplicate.input_ref
        assert added.newly_accepted is True and duplicate.newly_accepted is False
        model.release.set()

        deadline = int(time.time() * 1000) + 10000
        first_result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline)
        guidance_result = await runtime.client.wait(added.input_ref, actor=actor, deadline_ms=deadline)

        assert first_result.status == 'completed'
        assert first_result.final_text == 'Considering the first input.'
        assert guidance_result.status == 'completed' and 'STEERED' in guidance_result.final_text
        assert first_result.send_index is not None and guidance_result.send_index is not None
        assert first_result.send_index < guidance_result.send_index

        assert model.calls == 2
        guidance_occurrences = sum(
            str(m.get('content', '')).count(guidance_text)
            for req in model.seen_requests
            for m in req
            if m.get('role') == 'user'
        )
        assert guidance_occurrences == 1, model.seen_requests
        assert model.seen_requests[1][-1] == {'role': 'user', 'content': guidance_text}
        print(guidance_result.final_text)

if __name__ == '__main__':
    asyncio.run(main())
