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


async def test_delayed_acquire_cannot_overwrite_an_epoch_installed_in_another_interpreter(registry_store, tmp_path):
    store, _ = registry_store
    sid = await store.create_session()
    backend = store._db.dialect.name
    dsn = str(tmp_path / 'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    processes = []
    async def launch(role):
        process = await asyncio.create_subprocess_exec(sys.executable, '-m',
            'tests.unit._session_ownership_worker', dsn, store.table_prefix, sid,
            str(tmp_path / 'coordinator.db'), role, cwd=Path(__file__).resolve().parents[2],
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        processes.append(process)
        return process
    async def event(process):
        line = await asyncio.wait_for(process.stdout.readline(), 15)
        assert line, (await process.stderr.read()).decode()[-3000:]
        return json.loads(line)
    try:
        old = await launch('acquire-old')
        assert await event(old) == {'event': 'acquire_pending'}
        new = await launch('acquire-new')
        assert await event(new) == dict(event='installed', epoch=1, grants=1, first_released=False)
        await asyncio.wait_for(new.wait(), 10)
        assert new.returncode == 0, (await new.stderr.read()).decode()[-3000:]
        old.stdin.write(b'resume\n')
        await old.stdin.drain()
        assert await event(old) == dict(event='installed', epoch=2, grants=2, first_released=True)
        await asyncio.wait_for(old.wait(), 10)
        assert old.returncode == 0, (await old.stderr.read()).decode()[-3000:]
        assert await store.session_owner_epoch(sid) == 2
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def test_second_process_is_excluded_then_takes_over_with_clean_history(registry_store, tmp_path):
    store, _ = registry_store
    from .test_runtime_messaging_203 import fixture
    _, target, _ = await fixture(store)
    sid = target.session_id
    backend = store._db.dialect.name
    dsn = str(tmp_path / 'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    processes = []

    async def start(role):
        process = await asyncio.create_subprocess_exec(sys.executable, '-m',
            'tests.unit._session_ownership_worker', dsn, store.table_prefix, sid,
            str(tmp_path / 'coordinator.db'), role, cwd=Path(__file__).resolve().parents[2],
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
        old = await start('old')
        await event(old, 'running')
        new = await start('new')
        await event(new, 'ready')
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
