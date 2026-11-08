"""Two independent processes share only the database, on all three backends."""
import asyncio
import os
import sys

import pytest

from power_loop import StoreBlackboard, StoreTimerBackend, TimerContext

from .test_job_registry_192 import registry_store
from .test_runtime_messaging_203 import fixture

__all__ = ['registry_store']


@pytest.mark.parametrize('mode', ['timers', 'board'])
async def test_independent_writers(registry_store, mode):
    store, _ = registry_store
    _, target, actor = await fixture(store)
    if mode == 'timers':
        backend = StoreTimerBackend(store)
        for i in range(10):
            await backend.create(TimerContext(store.store_id, target.session_id, actor), due_at_ms=1, note=str(i))
    kind = store._db.dialect.name
    dsn = store._db.path if kind == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if kind == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    children = []
    try:
        for writer in ('a', 'b'):
            process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.timer_worker_208',
                env={**os.environ, 'P7A_DB_DSN': dsn, 'P7A_TABLE_PREFIX': store.table_prefix,
                     'P7A_MODE': mode, 'P7A_WRITER': writer},
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            children.append(process)
        async with asyncio.timeout(30):
            assert await asyncio.gather(*(p.stdout.readline() for p in children)) == [b'ready\n']*2
            for p in children:
                p.stdin.write(b'go\n')
                await p.stdin.drain()
            results = await asyncio.gather(*(p.communicate() for p in children))
        for p, (_, error) in zip(children, results):
            assert p.returncode == 0, error.decode()
        if mode == 'timers':
            assert sum(int(out) for out, _ in results) == 10
            rows = await store.list_timers(target.session_id, statuses=('fired',))
            assert len(rows) == 10 and all(r.fire_count == 1 for r in rows)
            queued = await store._db.fetchall(f'SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?', (target.session_id,))
            assert len(queued) == len({r['item_id'] for r in queued}) == 10
        else:
            entries = await StoreBlackboard(store).read('shared')
            assert len({e.id for e in entries}) == 20
            assert {e.text for e in entries} == {f'{w}:{i}' for w in ('a','b') for i in range(10)}
    finally:
        for p in children:
            if p.returncode is None:
                p.kill()
                await p.wait()
