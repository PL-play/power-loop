"""Hold actual claim/registration before commit without the old store mutex."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, mode, ready, release, output = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    manager = JobManager(store, now_ms=lambda: config['now'])
    actor = JobActor(**config['actor'])
    identity = registry.ExecutorIdentity(**config['identity'])

    @asynccontextmanager
    async def native():
        async with manager.db.transaction() as tx:
            yield tx

    manager._control_transaction = native

    async def hold():
        Path(ready).write_text(json.dumps({'mode': mode, 'pid': os.getpid()}))
        async with asyncio.timeout(20):
            while not Path(release).exists():
                await asyncio.sleep(.01)

    try:
        if mode.startswith('registration-'):
            save = registry._save

            def die():
                Path(ready).write_text(json.dumps({'mode': mode, 'pid': os.getpid()}))
                os._exit(96)

            async def recorded(*args, **kwargs):
                result = await save(*args, **kwargs)
                if mode == 'registration-uncommitted':
                    die()
                return result

            registry._save = recorded
            await registry.register(manager, identity, contract_hash='a'*64)
            die()
        elif mode == 'hold-claim':
            require = registry.require_accepting

            async def accepted(*args, **kwargs):
                result = await require(*args, **kwargs)
                await hold()
                return result

            registry.require_accepting = accepted
            lease = await manager.claim_job(actor, config['job_id'], executor_id=identity.executor_id,
                boot_id=identity.boot_id, expected_version=1)
            result = {'job_id': lease.job_id, 'generation': lease.generation}
        else:
            save = registry._save

            async def saved(*args, **kwargs):
                result = await save(*args, **kwargs)
                await hold()
                return result

            registry._save = saved
            result = await registry.register(manager, identity, contract_hash='a'*64)
        Path(output).write_text(json.dumps(result))
    finally:
        await store.close()


asyncio.run(main())
