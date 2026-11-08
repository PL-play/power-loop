# 用户输入与授权

SDK 11.0.0 · schema 15 · protocol 42 · descriptor 1.

request_user_input 创建持久交互。通过 RuntimeClient.answer 回答对应 interaction_id 和 wait_round。TOOL_BEFORE 授权决定工具是否执行，超时或取消不能视为授权成功。

以下源码可在本源码目录运行，依赖同目录的 [examples/_runtime.py](../../../examples/_runtime.py)。该辅助文件启动并关闭实际 Runtime；独立应用装配见[快速开始](../user-guide/quickstart.md)。

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
