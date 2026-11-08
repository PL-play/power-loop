"""41 · A host job schedules a read-only result check through the timer backend.

The host owns its job and task lifetime. The timer only asks for the saved result;
an unavailable external result is unknown and is never automatically re-executed.
Deployments replace this demo service with their own service and result lookup.
"""
import asyncio
import json
import time

from _runtime import demo_runtime

from power_loop import (
    InputRef,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
    timer_context,
)
from power_loop.runtime.pricing import Free


class JobService:
    def __init__(self):
        self.task = None
        self.result = 'unknown'
        self.timer = None
        self.checks = 0

    async def start(self, label):
        if self.task is not None:
            raise ValueError('This demo service accepts one job')
        async def work():
            await asyncio.sleep(.05)
            self.result = f'processed {label}'
        self.task = asyncio.create_task(work())
        context = get_tool_runtime_context(required=True)
        self.timer = await context.loop.timer_backend.create(await timer_context(), delay_s=.15,
            note='Read the original job result with check_job')
        return 'Accepted host-job-1'

    async def check(self):
        self.checks += 1
        return self.result

    async def close(self):
        if self.task:
            await self.task

class JobModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m.get('tool_call_id') == 'check' for m in request.messages):
            return LLMResponse(raw_text=next(m['content'] for m in request.messages if m.get('tool_call_id') == 'check'))
        if 'Read the original job result' in str(request.messages):
            name, args, call_id = 'check_job', {}, 'check'
        elif not any(m.get('role') == 'tool' for m in request.messages):
            name, args, call_id = 'start_job', {'label':'nightly-report'}, 'start'
        else:
            return LLMResponse(raw_text='Job accepted; awaiting the timer.')
        return LLMResponse(raw_text='', tool_calls=[{'id':call_id, 'type':'function',
            'function':{'name':name, 'arguments':json.dumps(args)}}])
    async def close(self):
        pass

async def main():
    service = JobService()
    registry = ToolRegistry()
    registry.register(ToolDefinition(name='start_job', description='Start one host job', pricing=Free('local demo'),
        input_schema={'type':'object','properties':{'label':{'type':'string'}},'required':['label']}), service.start)
    registry.register(ToolDefinition(name='check_job', description='Read the original result or unknown', pricing=Free('local demo')), service.check)
    try:
        async with demo_runtime(JobModel(), registry) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Start the report', message_id='start', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and service.timer
            timer = service.timer
            ref = InputRef(runtime.store.store_id, target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0')
            async with asyncio.timeout(10):
                while (await runtime.store.get_timer(target.session_id, timer.timer_id)).status != 'fired':
                    await asyncio.sleep(.05)
            result = await runtime.client.wait(ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == 'processed nightly-report'
            assert service.checks == 1
            print(result.final_text)
            return result.final_text
    finally:
        await service.close()

if __name__ == '__main__':
    asyncio.run(main())
