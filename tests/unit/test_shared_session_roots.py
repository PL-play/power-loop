"""Separate workers serialize accepted inputs for one session on each database."""
import asyncio
import os
import sys
from pathlib import Path

import pytest

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeClient,
    RuntimeDescriptor,
    TargetRef,
    TrustedActor,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import plain
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import _SEND
from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def worker(dsn, prefix, directory, role):
    directory = Path(directory)
    store = await open_store(dsn, table_prefix=prefix, schema='tasks')
    class Model(LLMService):
        async def complete(self, request, **kwargs):
            body = next(row['content'] for row in reversed(request.messages) if row['role'] == 'user')
            # Exclusive creation is an assertion of actual non-overlap, not scheduling.
            fd = os.open(directory/'active', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            try:
                (directory/(body+'.entered')).write_text(_SEND.get().lease.job_id)
                async with asyncio.timeout(20):
                    while body == 'first' and not (directory/'release').exists():
                        await asyncio.sleep(.01)
                return LLMResponse(raw_text='answer-'+body)
            finally:
                (directory/'active').unlink()
        async def close(self):
            pass
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(Model(), None, AgentLoopConfig(compactor=None))
    factory.register('serialize@1', build)
    transport = DatabasePollingTransport(RuntimeInputStore(store), interval_s=.03)
    runtime = Runtime(store=store, factory=factory, node_id=role, capacity=1,
        transport=transport, coordinator=DatabaseSessionOwnershipCoordinator(directory/'ownership.db'),
        shared_execution=True, timers_enabled=False, recovery_interval_s=None)
    try:
        await runtime.worker.start()
        print('ready', flush=True)
        await asyncio.to_thread(input)
    finally:
        await runtime.aclose(timeout_s=2, cancel_running=True)


async def test_distinct_roots_queue_in_separate_processes_without_overlapping_models(registry_store, tmp_path):
    store, _ = registry_store
    backend = store._db.dialect.name
    dsn = store._db.path if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    descriptor = RuntimeDescriptor(1, 'agent', 'serialize@1', None,
        'builtin.empty@1', 'builtin.none@1', 'builtin.none@1', {}, {})
    actor = TrustedActor('shared-session', 'host', None)
    sid = await store.create_session(metadata={'tenant_id':actor.tenant_id, 'principal_id':actor.principal_id,
        'execution_role':'main', 'runtime_descriptor':plain(descriptor)})
    target = TargetRef(store.store_id, sid)
    transport = DatabasePollingTransport(RuntimeInputStore(store), interval_s=.03)
    client = RuntimeClient(store=store, transport=transport)
    children = []
    async def launch(role):
        process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit.test_shared_session_roots',
            dsn, store.table_prefix, str(tmp_path), role,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        children.append(process)
        line = await asyncio.wait_for(process.stdout.readline(), 15)
        assert line == b'ready\n', (await process.stderr.read()).decode()
    try:
        await launch('one')
        first = await client.send(target, 'first', actor=actor, message_id='first')
        async with asyncio.timeout(15):
            while not (tmp_path/'first.entered').exists():
                await asyncio.sleep(.01)
        await launch('two')
        second = await client.send(target, 'second', actor=actor, message_id='second')
        assert second.newly_accepted and second.input_ref != first.input_ref
        await asyncio.sleep(.15)
        assert not (tmp_path/'second.entered').exists()
        assert [row.content for row in await store.load_all_messages(sid)] == ['first']
        (tmp_path/'release').touch()
        results = await asyncio.gather(*(client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+15000)
                                        for receipt in (first, second)))
        assert [result.final_text for result in results] == ['answer-first', 'answer-second']
        assert all(result.status == 'completed' for result in results)
        assert results[0].job_ref.job_id != results[1].job_ref.job_id
        rows = await store.load_all_messages(sid)
        assert [(row.content, row.send_index) for row in rows if row.role == 'user'] == [('first', 1), ('second', 2)]
        for process in children:
            process.stdin.write(b'close\n')
            await process.stdin.drain()
            _, error = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == 0, error.decode()
    finally:
        (tmp_path/'release').touch()
        for process in children:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await client.aclose()
        await transport.aclose()


if __name__ == '__main__':
    asyncio.run(worker(*sys.argv[1:]))
