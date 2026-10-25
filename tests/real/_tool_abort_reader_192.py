"""A fresh network-disabled process reads the original positive checkpoint."""
import asyncio
import json
import socket
import sys

from power_loop import SessionStore
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.tool_call_abort import read_abort


def forbidden(*args, **kwargs):
    raise AssertionError('A checkpoint reader must not contact a network or provider')


async def main():
    socket.create_connection = forbidden
    socket.socket.connect = forbidden
    store = await SessionStore.open(sys.argv[1], schema='compatible')
    try:
        manager = JobManager(store)
        actor = JobActor(**json.loads(sys.argv[2]))
        before = await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id')
        value = await read_abort(manager, actor, sys.argv[3], assistant_seq=int(sys.argv[4]))
        assert await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id') == before
        print(json.dumps({'network_disabled': True, 'evidence_unchanged': True, 'checkpoint': value}))
    finally:
        await store.close()


asyncio.run(main())
