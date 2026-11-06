"""The existing session lease serializes sends from separate root task trees."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, SessionBusy, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def loop_for(store, model, executor, directory):
    from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
    runtime = TaskRuntime(JobManager(store), tenant_id='shared-session', principal_id='host',
                          executor_id=executor, boot_id=executor)
    return StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        session_coordinator=DatabaseSessionOwnershipCoordinator(str(Path(directory)/"ownership.db")),
        config=AgentLoopConfig(compactor=None, delegation_policy='single_level'))


class Model(LLMService):
    async def complete(self, request, **kwargs):
        self.root = _SEND.get().lease.job_id
        return LLMResponse(raw_text='done')

    def stream(self, request):
        raise AssertionError('nonstream fixture')

    async def close(self):
        pass


async def worker(dsn, prefix, sid, directory):
    path = Path(directory)

    class Paused(Model):
        async def complete(self, request, **kwargs):
            response = await super().complete(request, **kwargs)
            (path/'entered').write_text(self.root)
            while not (path/'release').exists():
                await asyncio.sleep(.01)
            return response

    store = await open_store(dsn, table_prefix=prefix, schema='tasks')
    loop = loop_for(store, Paused(), 'first', directory)
    try:
        assert (await loop.send('first input', sid)).status == 'completed'
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_distinct_roots_cannot_claim_one_sessions_input_concurrently(registry_store, tmp_path):
    store, _ = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path/'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    sid = await store.create_session()
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit.test_shared_session_roots', dsn, store.table_prefix,
        sid, str(tmp_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    model = Model()
    loop = loop_for(store, model, 'second', tmp_path)
    try:
        async with asyncio.timeout(15):
            while not (tmp_path/'entered').exists():
                if process.returncode is not None:
                    output, _ = await process.communicate()
                    pytest.fail(output.decode())
                await asyncio.sleep(.01)
        before = await store.get_state(sid)
        with pytest.raises(SessionBusy):
            await loop.send('second input', sid)
        assert await store.get_state(sid) == before
        assert [row.content for row in await store.load_all_messages(sid)] == ['first input']
        (tmp_path/'release').touch()
        output, _ = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, output.decode()
        assert (await loop.send('second input', sid)).status == 'completed'
        assert model.root != (tmp_path/'entered').read_text()
        rows = await store.load_all_messages(sid)
        assert [(row.content, row.send_index) for row in rows if row.role == 'user'] == [
            ('first input', 1), ('second input', 2)]
    finally:
        (tmp_path/'release').touch()
        if process.returncode is None:
            process.kill()
            await process.wait()
        await loop.aclose(drain_timeout_s=.1)


if __name__ == '__main__':
    asyncio.run(worker(*sys.argv[1:]))
