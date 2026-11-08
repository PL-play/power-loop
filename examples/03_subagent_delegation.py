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
