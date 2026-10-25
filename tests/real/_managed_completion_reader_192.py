"""Fresh SQLite process repairs original results; no runtime/provider/network."""
import asyncio
import json
import socket
import sys

from power_loop import SessionStore
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.managed_completion import restore_result
from power_loop.runtime.managed_effects import read_raw_receipt


def forbidden_network(*args, **kwargs):
    raise AssertionError('Mechanical result repair must not contact any provider or network')


async def main():
    socket.create_connection = forbidden_network
    socket.socket.connect = forbidden_network
    store = await SessionStore.open(sys.argv[1], schema='compatible')
    try:
        manager = JobManager(store)
        actor, job = JobActor(**json.loads(sys.argv[2])), sys.argv[3]
        before = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        first = await restore_result(manager, actor, job)
        assert await restore_result(manager, actor, job) == first
        assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq') == before
        try:
            raw = await read_raw_receipt(manager, actor, job)
        except JobError as error:
            if len(sys.argv) != 5 or sys.argv[4] != 'allow_missing_receipt' or error.code != 'managed_effect_evidence_missing':
                raise
            raw = None
        print(json.dumps({'repaired': first, 'raw': raw, 'network_disabled': True}))
    finally:
        await store.close()


asyncio.run(main())
