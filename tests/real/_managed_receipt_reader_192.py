"""Fresh mechanical process: recover original coordinates; models/network disabled."""

import asyncio
import json
import socket
import sys

from power_loop import SessionStore
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.managed_effects import reconcile_raw_receipt, terminate_unstarted


def forbidden_network(*args, **kwargs):
    raise AssertionError("A mechanical SQLite receipt reader must not contact any provider or network")


async def main():
    socket.create_connection = forbidden_network
    socket.socket.connect = forbidden_network
    store = await SessionStore.open(sys.argv[1], schema="compatible")
    try:
        manager = JobManager(store)
        actor, job = JobActor(**json.loads(sys.argv[2])), sys.argv[3]
        observed = await reconcile_raw_receipt(manager, actor, job)
        negative = await terminate_unstarted(manager, actor, job)
        print(json.dumps({"observed": observed, "negative": negative, "network_disabled": True}))
    finally:
        await store.close()


asyncio.run(main())
