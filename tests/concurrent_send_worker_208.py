import asyncio
import json
import os

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.store.factory import open_store


class Model(LLMService):
    async def complete(self,request,**kwargs):
        print('model_entered',flush=True)
        await asyncio.to_thread(input)
        return LLMResponse(raw_text='first')
    async def close(self): pass


async def main():
    store=await open_store(os.environ['P7A_DB_DSN'],table_prefix=os.environ['P7A_TABLE_PREFIX'],schema='tasks')
    factory=RegistryRuntimeFactory()
    async def build(*args):
        return ExecutionEnvironment(Model(),None,AgentLoopConfig(compactor=None))
    factory.register('test.agent@1',build)
    runtime=Runtime(store=store,factory=factory,node_id='process',timers_enabled=False)
    try:
        try:
            await runtime.worker.start()
        except RuntimeRequestError as exc:
            print(exc.code,flush=True)
            return
        actor=TrustedActor(**json.loads(os.environ['P7A_ACTOR']))
        target=TargetRef(store.store_id,os.environ['P7A_SESSION'])
        receipt=await runtime.client.send(target,'first',message_id='first',actor=actor)
        result=await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=now_ms()+30000)
        print(result.status,flush=True)
    finally:
        await runtime.aclose(timeout_s=1,cancel_running=True)


if __name__ == '__main__':
    asyncio.run(main())
