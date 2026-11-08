# Human input and approval

SDK 11.0.0 · schema 15 · protocol 42 · descriptor 1.

Use request_user_input for a persisted interaction. Answer its exact interaction_id and wait_round through RuntimeClient.answer. TOOL_BEFORE approval controls whether a tool runs; its timeout/cancellation is not successful approval.

The following source is executable from this checkout. It imports the shared example composition helper, [examples/_runtime.py](../../../examples/_runtime.py); keep that file alongside the example. The helper starts and closes a real Runtime. For independent application setup, follow [Quickstart](../user-guide/quickstart.md).

```bash
.venv/bin/python examples/21_request_user_input.py
```

```python
"""21 · A paused task is answered by JobRef, interaction ID and wait round.

The demo supplies an answer. A UI may use the same visible pending state and
submit a different choice; the original input remains the query identity.
"""
import asyncio
import json
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, create_default_tool_registry


class AskingModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m.get('role') == 'tool' for m in request.messages):
            return LLMResponse(raw_text=next(m['content'] for m in reversed(request.messages) if m.get('role') == 'tool'))
        return LLMResponse(raw_text='', tool_calls=[{'id':'question','type':'function','function':{
            'name':'request_user_input','arguments':json.dumps({'kind':'choice','prompt':'Choose tone',
                'options':[{'id':'gentle','label':'Gentle'}, {'id':'direct','label':'Direct'}]})}}])

    async def close(self):
        pass


async def main(answers=None):
    registry = create_default_tool_registry(include=['request_user_input'])
    async with demo_runtime(AskingModel(), registry) as (runtime, loop, target, actor):
        receipt = await runtime.client.send(target, 'Ask for a tone', message_id='question', actor=actor)
        async with asyncio.timeout(10):
            while True:
                status = await runtime.client.inspect(receipt.input_ref, actor=actor)
                if status.state == 'waiting_input':
                    break
                await asyncio.sleep(.02)
        pending = (await runtime.store.get_state(target.session_id)).pending
        interaction = pending['pending_interactions'][0]
        await runtime.client.answer(status.job_ref, interaction_id=interaction['interaction_id'],
            wait_round=pending['round_index'], value={'choice':(answers or ['gentle'])[0]}, message_id='answer', actor=actor)
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
```

[hooks](../user-guide/hooks.md) · [API](../api/index.md)
