"""A separate database writer for current-read vs coherent-snapshot tests."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.store.factory import _make_database


async def main():
    dsn, table, ready, release, attempted, output = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    db = await _make_database(dsn)
    try:
        Path(ready).write_text(json.dumps({'pid': os.getpid()}))
        async with asyncio.timeout(30):
            while not Path(release).exists():
                await asyncio.sleep(.01)
        Path(attempted).write_text('writing')
        async with db.transaction() as tx:
            lock = '' if db.dialect.name == 'sqlite' else ' FOR UPDATE'
            root = await tx.fetchone(f'SELECT generation FROM {table} WHERE node_id=?'+lock, ('root',))
            assert root['generation'] == 1
            changed = await tx.execute(f'UPDATE {table} SET generation=generation+1 WHERE node_id=? AND generation=?', ('parent', 1))
        Path(output).write_text(json.dumps({'changed': changed}))
    finally:
        await db.close()


asyncio.run(main())
