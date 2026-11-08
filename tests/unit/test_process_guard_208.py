import asyncio
import os
import sys

import pytest

from power_loop import MemoryMessageTransport, RegistryRuntimeFactory, Runtime, RuntimeClient, RuntimeRequestError
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']


async def child(store):
    kind = store._db.dialect.name
    dsn = store._db.path if kind == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if kind == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.process_guard_worker_208',
        env={**os.environ, 'P7A_DB_DSN': dsn, 'P7A_TABLE_PREFIX': store.table_prefix},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    async with asyncio.timeout(20):
        line = (await process.stdout.readline()).decode().strip()
    return process, line


async def finish(process):
    if process.returncode is None:
        process.stdin.write(b'close\n')
        await process.stdin.drain()
    async with asyncio.timeout(15):
        _, error = await process.communicate()
    assert process.returncode == 0, error.decode()


async def test_two_default_processes_and_release(registry_store):
    store, _ = registry_store
    first, status = await child(store)
    try:
        assert status == 'ready'
        second, status = await child(store)
        assert status == 'invalid_options'
        await second.wait()
        assert second.returncode == 0, (await second.stderr.read()).decode()
        client = RuntimeClient(store=store, transport=MemoryMessageTransport())
        assert client._accepting
        await client.aclose()
    finally:
        await finish(first)
    third, status = await child(store)
    assert status == 'ready'
    await finish(third)


async def test_shared_memory_domain_refcount(registry_store):
    store, connect = registry_store
    other = await connect()
    coordinator, transport = MemorySessionOwnershipCoordinator(), MemoryMessageTransport()
    a = Runtime(store=other, factory=RegistryRuntimeFactory(), node_id='a', coordinator=coordinator, transport=transport, timers_enabled=False)
    b = Runtime(store=store, factory=RegistryRuntimeFactory(), node_id='b', coordinator=coordinator, transport=transport, timers_enabled=False)
    await a.worker.start()
    await b.worker.start()
    await a.worker.shutdown()
    # A guard shared by live workers must not pin the first worker's data pool.
    await asyncio.wait_for(other.close(), 2)
    c = Runtime(store=store, factory=RegistryRuntimeFactory(), node_id='c', timers_enabled=False)
    with pytest.raises(RuntimeRequestError, match='coordination domain'):
        await c.worker.start()
    await b.worker.shutdown()
    await c.worker.start()
    await c.worker.shutdown()
    for runtime in (a,b,c):
        await runtime.client.aclose()
    await transport.aclose()
    await c.transport.aclose()


@pytest.mark.parametrize('active',[False,True])
async def test_lock_connection_loss_stops_admission(registry_store,active):
    from power_loop import AgentLoopConfig, ExecutionEnvironment, LLMResponse, LLMService
    from power_loop.agent.runtime_store import now_ms

    from .test_runtime_messaging_203 import fixture
    store,_=registry_store
    _,target,actor=await fixture(store)
    entered=asyncio.Event()
    class Model(LLMService):
        async def complete(self,request,**kwargs):
            entered.set()
            await asyncio.Event().wait()
            return LLMResponse(raw_text='must not finish')
        async def close(self):
            pass
    factory=RegistryRuntimeFactory()
    async def build(descriptor,context):
        return ExecutionEnvironment(Model(),None,AgentLoopConfig(compactor=None))
    factory.register('test.agent@1',build)
    runtime=Runtime(store=store,factory=factory,node_id='lost',timers_enabled=False)
    await runtime.worker.start()
    try:
        if active:
            receipt=await runtime.client.send(target,'hold',message_id='hold',actor=actor)
            await asyncio.wait_for(entered.wait(),5)
        guard=runtime.process_guard.entry
        if guard.kind=='sqlite':
            guard.file.close()
        elif guard.kind=='postgres':
            await guard.connection.close()
        else:
            guard.connection.close()
        async with asyncio.timeout(10):
            while runtime.client._accepting:
                await asyncio.sleep(.05)
        assert runtime.worker.last_error.startswith('store_lock_lost')
        with pytest.raises(RuntimeRequestError):
            await runtime.client.send(target,'new',message_id='after-loss',actor=actor)
        if active:
            result=await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=now_ms()+5000)
            assert result.status=='interrupted'
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()
