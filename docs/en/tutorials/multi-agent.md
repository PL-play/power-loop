# Multi-agent workflows

SDK 11.0.0 · schema 15 · protocol 42 · descriptor 1.

Register spawn_agent on the execution node. Accepted child jobs run independently through Runtime; query/control them by their original JobRef. The child inherits permitted resources and bounded options, not the parent Python object. Workflows compose these jobs under explicit owners and budgets.

The following source is executable from this checkout. It imports the shared example composition helper, [examples/_runtime.py](../../../examples/_runtime.py); keep that file alongside the example. The helper starts and closes a real Runtime. For independent application setup, follow [Quickstart](../user-guide/quickstart.md).

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
