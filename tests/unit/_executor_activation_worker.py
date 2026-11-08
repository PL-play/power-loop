"""Exit at actual executor-admission/publication commit boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, stage, marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    manager = JobManager(store, now_ms=lambda: config['now'])
    identity = registry.ExecutorIdentity(**config['identity'])
    begin, publish, save = registry._begin_activation, registry._publish_activation, registry._save

    def die():
        Path(marker).write_text(json.dumps({'stage': stage, 'pid': os.getpid()}))
        os._exit(93)

    async def begun(*args, **kwargs):
        result = await begin(*args, **kwargs)
        if stage == 'barrier':
            die()
        return result

    async def published(*args, **kwargs):
        if stage == 'prepared':
            die()
        if stage == 'hold-prepared':
            Path(marker).write_text(json.dumps({'stage': stage, 'pid': os.getpid()}))
            async with asyncio.timeout(30):
                while not Path(marker).with_suffix('.release').exists():
                    await asyncio.sleep(.01)
        result = await publish(*args, **kwargs)
        if stage == 'activated':
            die()
        return result

    async def saved(current, tx, actual, *args, **kwargs):
        result = await save(current, tx, actual, *args, **kwargs)
        if ((stage == 'old-closure-uncommitted' and actual != identity)
                or (stage == 'activation-uncommitted' and actual == identity)):
            die()
        return result

    registry._begin_activation = begun
    registry._publish_activation = published
    registry._save = saved
    result = await registry.activate(manager, identity, expected_version=1, system_key='startup')
    if stage == 'hold-prepared':
        Path(marker).with_suffix('.done').write_text(json.dumps(result))
        await store.close()
        return
    raise AssertionError('activation crash boundary not reached')


asyncio.run(main())
