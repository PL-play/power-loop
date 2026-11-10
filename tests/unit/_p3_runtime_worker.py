"""Real worker used by the P3 multi-interpreter contract test."""
import asyncio
import json
import logging
import sys

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import TargetRef, TrustedActor
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
from power_loop.tools import create_default_tool_registry
from power_loop.tools.spawn_agent import register_spawn_agent


def report(**value):
    print(json.dumps(value), flush=True)


async def main():
    logging.basicConfig(level=logging.WARNING)
    dsn, prefix, sid, coordinator_path, role, workspace = sys.argv[1:]
    role, _, scenario = role.partition("-")
    store = await open_store(dsn, table_prefix=prefix, schema='tasks')
    actor = TrustedActor('tenant', 'principal', None)
    gate = asyncio.Event()
    class Coordinator(DatabaseSessionOwnershipCoordinator):
        async def acquire(self, key, **kwargs):
            if (key[1] == sid) != (role != 'B'):
                return None
            return await super().acquire(key, **kwargs)

    registry = create_default_tool_registry(workspace_dir=workspace)
    register_spawn_agent(registry)
    from power_loop.tools.jobs import register_jobs_tools
    register_jobs_tools(registry)
    async def hold():
        from power_loop.core.agent_context import get_session_id
        report(event='child_running', sid=get_session_id())
        await gate.wait()
        return 'original tool completed'
    registry.register(ToolDefinition(name='hold', description='wait for test gate', interrupt='abort',
        input_schema={'type': 'object'}, pricing=Free('local')), hold)

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith('CHILD'):
                if not any(m.get('role') == 'tool' for m in request.messages):
                    return LLMResponse(raw_text='', token_usage=LLMTokenUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2),
                        tool_calls=[{'id': 'hold', 'type': 'function', 'function': {'name': 'hold', 'arguments': '{}'}}])
                return LLMResponse(raw_text='child completed: ' + '|'.join(str(m.get('content','')) for m in request.messages if m.get('role') == 'user'),
                    token_usage=LLMTokenUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))
            messages = request.messages
            latest = next((m.get('content','') for m in reversed(messages) if m.get('role') == 'user'), '')
            replies = [m for m in messages if m.get('role') == 'tool']
            def call(name, args, ident):
                return {'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
            if scenario == 'background' and latest == 'create children' and not any(m.get('tool_call_id')=='bg-one' for m in replies):
                return LLMResponse(raw_text='',tool_calls=[call('background_run',{'action':'run',
                    'command':f'echo started >> {workspace}/marker-{name}; sleep 60; echo finished >> {workspace}/marker-{name}'},'bg-'+name)
                    for name in ('one','two')])
            if scenario != 'background' and latest == 'create children' and not any(m.get('tool_call_id') == 'spawn-one' for m in replies):
                return LLMResponse(raw_text='', tool_calls=[call('spawn_agent', {'task': task, 'system_prompt': 'CHILD', 'tools':['hold']}, 'spawn-'+task) for task in ('one','two')])
            if isinstance(latest,str) and latest.startswith('control:'):
                action = json.loads(latest[8:])
                ident = action['id']
                if not any(m.get('tool_call_id') == ident for m in replies):
                    return LLMResponse(raw_text='', tool_calls=[call(action['tool'], action['args'], ident)])
            return LLMResponse(raw_text='parent handled')
        async def close(self):
            pass

    model = Model()
    task_runtime = TaskRuntime(JobManager(store), tenant_id='tenant', principal_id='principal', executor_id=role, boot_id=role,
        boot_contract_hash='a'*64)
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(model, registry, AgentLoopConfig(compactor=None,max_tokens=10), task_runtime=task_runtime)
    factory.register('p3.agent@1',build)
    runtime = Runtime(store=store, factory=factory, node_id=role, boot_id=role, capacity=2,
        coordinator=Coordinator(coordinator_path), transport=DatabasePollingTransport(RuntimeInputStore(store),interval_s=.03,control_interval_s=.03),
        session_lease_ttl_s=2, recovery_interval_s=.2, shared_execution=True)
    try:
        await runtime.worker.start()
        target = TargetRef(store.store_id, sid)
        if role == 'A':
            receipt = await runtime.client.send(target,'create children',message_id='create',actor=actor)
            result = await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=now_ms()+10000)
            rows = await store.load_all_messages(sid)
            outputs = [m.content for m in rows if (m.tool_call_id or '').startswith(('spawn-','bg-'))]
            try:
                children = [json.loads(value) for value in outputs]
            except ValueError:
                raise AssertionError(outputs) from None
            report(event='parent_finished',job_id=result.job_ref.job_id,children=children)
            if scenario == 'parent_lost':
                await asyncio.to_thread(sys.stdin.readline)
        else:
            report(event='ready')
            while True:
                command = (await asyncio.to_thread(sys.stdin.readline)).strip()
                if command == 'quit' or not command:
                    break
                if command == 'release':
                    gate.set()
                    report(event='released')
                else:
                    data = json.loads(command)
                    receipt = await runtime.client.send(target,'control:'+json.dumps(data),message_id=data['id'],actor=actor)
                    result = await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=now_ms()+15000)
                    rows = await store.load_all_messages(sid)
                    outputs = [m.content for m in rows if m.tool_call_id == data['id']]
                    report(event='controlled',action=data['id'],status=result.status,output=outputs[-1])
    finally:
        await task_runtime.shutdown(timeout_s=2)
        await runtime.aclose(timeout_s=2,cancel_running=True)


if __name__ == '__main__':
    asyncio.run(main())
