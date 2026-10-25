"""Read one original host step and completed result without constructing a model."""
import asyncio
import json
import sys

from power_loop import SessionStore
from power_loop.runtime.host_send_steps import lookup
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.send_receipt import read


async def main():
    path, actor_json, step_json, source_json = sys.argv[1:]
    actor, step, source = JobActor(**json.loads(actor_json)), json.loads(step_json), json.loads(source_json)
    store = await SessionStore.open(path, schema="compatible")
    try:
        manager = JobManager(store)
        binding = await lookup(manager, actor, **step, expected_host_context=source)
        receipt = await read(manager, actor, binding["job_id"], expected_host_context=source)
        print(json.dumps({"binding": binding, "receipt": receipt, "real_provider_requests": 0}))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
