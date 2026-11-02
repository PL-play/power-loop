"""Commit a gate version and its stop barrier while another writer waits."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.store.factory import _make_database


async def main():
    dsn, table, ready, release = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    db = await _make_database(dsn)
    try:
        async with db.transaction() as tx:
            assert await tx.execute(f"UPDATE {table} SET value=value+1 WHERE kind='gate'") == 1
            assert await tx.execute(f"UPDATE {table} SET value=value+1 WHERE kind='barrier'") == 1
            Path(ready).write_text(json.dumps({'pid': os.getpid()}))
            async with asyncio.timeout(20):
                while not Path(release).exists():
                    await asyncio.sleep(.01)
    finally:
        await db.close()


asyncio.run(main())
