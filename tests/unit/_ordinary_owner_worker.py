"""Retain a live owner's frozen permission across a real workflow handoff.

The parent copies its actual current execution before the response returns.
This process publishes with that old identity;it never claims new authority.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from power_loop import SessionLeaseLost
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import _SEND, Execution, TaskRuntime


async def wait(path):
    async with asyncio.timeout(30):
        while not path.exists():
            await asyncio.sleep(.01)


async def main(dsn, prefix, raw, folder):
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config, out = json.loads(raw), Path(folder)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    actor, lease = JobActor(**config['actor']), JobLease(**config['lease'])
    runtime = TaskRuntime(JobManager(store), tenant_id=actor.tenant_id, principal_id=actor.principal_id,
        executor_id=lease.executor_id, boot_id=lease.boot_id)
    binding = Execution(runtime, actor, lease)
    binding.resource_evidence['_message_write_proofs'] = tuple(tuple(p) for p in config['proofs'])
    token = _SEND.set(binding)
    try:
        async with store.send_read_scope(actor.session_id):
            await store.get_state(actor.session_id)
            (out/'owner.ready').write_text(json.dumps({'pid':os.getpid(),'lease':asdict(lease)}))
            await wait(out/'owner.probe')
            state = await store.get_state(actor.session_id)
            # Late factual observations were allowed after clean model closure.
            # Prove that closure alone does not reject this old permission.
            await store.set_round_index(actor.session_id, state.round_index)
            state = await store.get_state(actor.session_id)
            (out/'owner.probed').write_text(json.dumps({'pid':os.getpid(),'next_seq':state.next_seq}))
            await wait(out/'owner.append')
            (out/'owner.appending').touch()
            try:
                seq = await store.append_message(actor.session_id, role='assistant', content='stale-process',
                    send_index=config['send_index'], round_index=state.round_index, pending=None)
                result = {'committed':seq}
            except SessionLeaseLost:
                result = {'error':'SessionLeaseLost'}
            (out/'owner.done').write_text(json.dumps(result))
    finally:
        _SEND.reset(token)
        await store.close()


if __name__ == '__main__':
    asyncio.run(main(*sys.argv[1:]))
