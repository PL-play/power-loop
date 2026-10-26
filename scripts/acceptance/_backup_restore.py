"""Offline acceptance copies only; never called by the runtime or production CLI."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import unquote, urlparse


def database_name(dsn):
    parsed = urlparse(dsn)
    name = unquote(parsed.path.lstrip('/'))
    if (parsed.scheme not in ('postgres', 'postgresql') or not name.endswith('_test')
            or not re.fullmatch(r'[a-zA-Z0-9_]+', name)):
        raise ValueError('backup rehearsal requires an explicit disposable PostgreSQL test database')
    return name


async def command(args, *, input_data=None):
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, error = await asyncio.wait_for(process.communicate(input_data), 40)
        if process.returncode:
            # No DSN, credentials or provider payload in error messages.
            raise RuntimeError('backup utility failed: ' + error.decode()[-2000:])
        return output, process.pid
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()


async def restore_copy(source, destination, *, prefix, archive, backend):
    if not re.fullmatch(r'bridge_[0-9a-f]{8}_', prefix) or source == destination:
        raise ValueError('restore requires a distinct target and owned test prefix')
    archive = Path(archive)
    if archive.exists():
        raise ValueError('refuse to overwrite backup evidence')
    utility_pids = []
    if backend == 'sqlite':
        source_path, destination_path = Path(source), Path(destination)
        if source_path.resolve() == destination_path.resolve() or not source_path.is_file():
            raise ValueError('invalid SQLite backup source or target')
        # Reserve both output names. A collision can never overwrite another
        # fixture, an original store, or an earlier backup.
        with archive.open('xb'):
            pass
        with destination_path.open('xb'):
            pass
        def copy():
            with closing(sqlite3.connect(source_path)) as original, closing(sqlite3.connect(archive)) as backup:
                original.backup(backup)
            with closing(sqlite3.connect(archive)) as backup, closing(sqlite3.connect(destination_path)) as restored:
                backup.backup(restored)
                assert restored.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
        await asyncio.to_thread(copy)
    elif backend == 'postgres':
        origin, target = database_name(source), database_name(destination)
        if origin == target:
            raise ValueError('restore must not target the source database')
        parsed = urlparse(source)
        other = urlparse(destination)
        if (parsed.hostname, parsed.port, parsed.username) != (other.hostname, other.port, other.username):
            raise ValueError('test runner must own both databases on the same isolated server')
        container = os.environ['POWER_LOOP_BACKUP_PG_CONTAINER']
        user = unquote(parsed.username or '')
        if not user or not re.fullmatch(r'[a-zA-Z0-9_-]+', container):
            raise ValueError('invalid test container binding')
        # The runner creates both databases; the source writer has exited.
        # Docker's existing local DB authentication is used, not credentials in argv.
        dumped, dump_pid = await command(['docker', 'exec', container, 'pg_dump', '-U', user,
            '-d', origin, '--format=custom', '--no-owner', '--no-acl', '--table=' + prefix + '*'])
        with archive.open('xb') as output:
            output.write(dumped)
        # No --clean: a nonempty/conflicting target is an error, never an overwrite.
        _, restore_pid = await command(['docker', 'exec', '-i', container, 'pg_restore', '-U', user,
            '-d', target, '--exit-on-error', '--no-owner', '--no-acl'], input_data=dumped)
        utility_pids = [dump_pid, restore_pid]
    else:
        raise ValueError('unsupported or paused backup backend')
    return {'backend': backend, 'utility_pids': utility_pids, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_bytes': archive.stat().st_size, 'source_preserved': True}
