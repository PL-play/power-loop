"""29 · Two current-runtime sessions use one database-atomic board (no credentials)."""
import asyncio
import json
import time

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    RuntimeEnv,
    SessionStore,
    StoreBlackboard,
    ToolRegistry,
    TrustedActor,
    register_blackboard_tools,
)


class Writer(LLMService):
    def __init__(self, actions):
        self.actions = iter(actions)
    async def complete(self, request, **kwargs):
        action = next(self.actions, None)
        if action:
            return LLMResponse(raw_text='', tool_calls=[{'id':f'board-{action["action"]}', 'type':'function',
                'function':{'name':'board','arguments':json.dumps(dict(action))}}])
        return LLMResponse(raw_text='posted')
    async def close(self):
        pass


async def main():
    store = await SessionStore.open(':memory:')
    board, factory = StoreBlackboard(store), RegistryRuntimeFactory()
    async def build(descriptor, context):
        registry = ToolRegistry()
        register_blackboard_tools(registry)
        return ExecutionEnvironment(Writer(descriptor.config['actions']),registry,
            AgentLoopConfig(max_rounds=5),runtime_env=RuntimeEnv(blackboard=board,blackboard_id='demo'))
    factory.register('board-writer@1',build)
    runtime = Runtime(store=store,factory=factory,node_id='example',timers_enabled=False)
    actor = TrustedActor('demo','host',None)
    try:
        await runtime.worker.start()
        async def write(name, actions):
            target = await runtime.open_session(RuntimeDescriptor(1,'agent','board-writer@1',None,
                'board@1','builtin.none@1','board@1',{'actions':actions},{}),actor=actor)
            await store.merge_session_metadata(target.session_id, {'name':name})
            receipt = await runtime.client.send(target,'post your note',message_id='post',actor=actor)
            result = await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=int(time.time()*1000)+10000)
            assert result.status=='completed', result
        await write('planner', [{'action':'post','text':'Plan the work','kind':'task','status':'open'}])
        entries = await board.read('demo')
        await write('worker', [{'action':'read'}, {'action':'update','entry_id':entries[0].id,'status':'done'},
                               {'action':'post','text':'Review completed','kind':'note'}])
        entries=await board.read('demo')
        assert len(entries)==2 and entries[0].status == 'done'
        assert {entry.author for entry in entries} == {'planner','worker'}
        assert {entry.kind for entry in entries} == {'task','note'}
        for entry in entries:
            print(entry.id,entry.text)
        await board.delete('demo')  # Host ends the shared board scope.
    finally:
        await runtime.aclose()


if __name__=='__main__':
    asyncio.run(main())
