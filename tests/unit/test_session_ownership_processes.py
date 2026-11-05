"""Two actual interpreters contend, then repair a paused owner's pending tool."""
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("managed", ["no", "yes"])
async def test_second_process_is_excluded_then_takes_over_with_clean_history(registry_store, tmp_path, managed):
    store, _ = registry_store
    sid = await store.create_session()
    backend = store._db.dialect.name
    dsn = str(tmp_path / 'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    processes = []

    async def start(role):
        process = await asyncio.create_subprocess_exec(sys.executable, '-m',
            'tests.unit._session_ownership_worker', dsn, store.table_prefix, sid,
            str(tmp_path / 'coordinator.json'), role, managed, cwd=Path(__file__).resolve().parents[2],
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        processes.append(process)
        return process

    async def event(process, expected):
        line = await asyncio.wait_for(process.stdout.readline(), 15)
        assert line, (await process.stderr.read()).decode()[-3000:]
        data = json.loads(line)
        assert data['event'] == expected, data
        return data

    async def command(process, value):
        process.stdin.write((value + '\n').encode())
        await process.stdin.drain()

    old = None
    try:
        new = await start('new')
        assert (await event(new, 'warmed'))['next_seq'] == 1
        old = await start('old')
        await event(old, 'running')
        await command(new, 'try')
        assert (await event(new, 'busy'))['calls'] == 0
        await command(old, 'pause')
        await event(old, 'pausing')
        await asyncio.sleep(1)
        await command(new, 'takeover')
        healed = await event(new, 'healed')
        assert healed == dict(event='healed', status='completed', pending=None, calls=1)
        old.send_signal(signal.SIGCONT)
        assert (await event(old, 'old_write_denied'))['denied'] == [True] * 4
        await event(old, 'stopped')
        await command(new, 'quit')
        for process in processes:
            await asyncio.wait_for(process.wait(), 10)
            assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
        if managed == 'yes':
            old_jobs = await store._db.fetchall(f"SELECT status,outcome FROM {store.table_prefix}jobs WHERE executor_id='old'")
            assert old_jobs and all(row['outcome'] == 'interrupted' for row in old_jobs)
        rows = await store.load_all_messages(sid)
        assert all(row.content != 'late-old-write' for row in rows)
        assert [row.tool_call_id for row in rows if row.role == 'tool'] == ['original-call']
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
