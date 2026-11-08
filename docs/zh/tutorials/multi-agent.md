# 多 Agent 协作

SDK 11.0.0 · schema 15 · protocol 42 · descriptor 1.

在执行节点注册 spawn_agent。受理的子任务通过 Runtime 独立运行，按原 JobRef 查询控制。子任务继承允许的资源与有界配置，不继承父 Python 对象。工作流把任务组织到明确的归属和额度之下。

以下源码可在本源码目录运行，依赖同目录的 [examples/_runtime.py](../../../examples/_runtime.py)。该辅助文件启动并关闭实际 Runtime；独立应用装配见[快速开始](../user-guide/quickstart.md)。

```bash
.venv/bin/python examples/03_subagent_delegation.py
```

```python
"""03 · Capacity-one parent accepts a child, finishes, and receives its result."""
import asyncio
import json
import time

from _runtime import demo_runtime

from power_loop import JobRef, LLMResponse, LLMService, ToolRegistry, register_spawn_agent


class DelegatingModel(LLMService):
    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith('child'):
            return LLMResponse(raw_text='Tokyo')
        if not any(m.get('role') == 'tool' for m in request.messages):
            return LLMResponse(raw_text='', tool_calls=[{'id':'spawn','type':'function','function':{
                'name':'spawn_agent','arguments':json.dumps({'task':'Capital of Japan?',
                    'system_prompt':'child', 'tools':[], 'max_rounds':1})}}])
        return LLMResponse(raw_text='Parent accepted the task.')

    async def close(self):
        pass


async def main():
    registry = ToolRegistry()
    register_spawn_agent(registry)
    async with demo_runtime(DelegatingModel(), registry) as (runtime, loop, target, actor):
        parent = await loop.send(target, 'Delegate the question', message_id='delegate', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        assert parent.status == 'completed'
        rows = await runtime.store.load_all_messages(target.session_id)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id == 'spawn'))
        result = await runtime.client.wait(JobRef(**receipt['job_ref']), actor=actor,
                                            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and result.final_text == 'Tokyo'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
```

[subagents](../user-guide/subagents.md) · [API](../api/index.md)
