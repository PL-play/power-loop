"""Mechanical fresh-process read only: no runtime, provider or model factory."""
import asyncio
import json
import sys

from power_loop import SessionStore
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.send_receipt import read


async def main():
    path, prefix, actor_json, job_id, context_json = sys.argv[1:]
    store = await SessionStore.open(path, table_prefix=prefix, schema="compatible")
    try:
        receipt = await read(JobManager(store), JobActor(**json.loads(actor_json)), job_id,
            expected_host_context=json.loads(context_json))
        print(json.dumps({"receipt": receipt, "real_provider_requests": 0}))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
