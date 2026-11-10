"""Isolated test worker reconstructs model and runtime from explicit input/env."""
import asyncio
import json
import sys

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    ToolRegistry,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import plain
from tests.real._llm import make_llm


async def main():
    options=json.load(sys.stdin)
    store=await SessionStore.open(options['path'])
    factory=RegistryRuntimeFactory()
    async def build(descriptor,context):
        return ExecutionEnvironment(make_llm(max_tokens=2048,temperature=0),ToolRegistry(),
            AgentLoopConfig(max_rounds=3,max_tokens=2048,compactor=None))
    factory.register('real.agent@1',build)
    runtime=Runtime(store=store,factory=factory,node_id='isolated-real')
    actor=TrustedActor('isolated-test','host',None)
    try:
        await runtime.worker.start()
        target=await runtime.open_session(RuntimeDescriptor(1,'agent','real.agent@1',None,
            'builtin.empty@1','builtin.none@1','builtin.none@1',{},{}),actor=actor)
        if options['mode']=='workflow':
            receipt=await runtime.client.workflow(target,{'name':'real','input':'France',
                'root':{'type':'agent','id':'capital','spec':{'name':'capital','tools':[],
                    'system_prompt':'Answer in one short sentence.'},'input':'What is the capital of {{input}}?'}},
                message_id='real-workflow',actor=actor)
        else:
            receipt=await runtime.client.send(target,'Name the capital of France in one sentence.',message_id='real-send',actor=actor)
        result=await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=now_ms()+120000)
        while runtime.worker._tasks:
            await asyncio.wait(tuple(runtime.worker._tasks))
        print(json.dumps({'result':plain(result),'root':plain(target)}),flush=True)
    finally:
        await runtime.aclose(timeout_s=5,cancel_running=True)


if __name__=='__main__':
    asyncio.run(main())
