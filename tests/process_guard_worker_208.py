"""Independent interpreter used by the default-runtime startup contract tests."""
import asyncio
import os

from power_loop import RegistryRuntimeFactory, Runtime, RuntimeRequestError
from power_loop.runtime.store.factory import open_store


async def main():
    store = await open_store(os.environ['P7A_DB_DSN'], table_prefix=os.environ['P7A_TABLE_PREFIX'], schema='tasks')
    runtime = Runtime(store=store, factory=RegistryRuntimeFactory(), node_id='independent', timers_enabled=False)
    try:
        try:
            await runtime.worker.start()
        except RuntimeRequestError as exc:
            print(exc.code, flush=True)
            return
        print('ready', flush=True)
        await asyncio.to_thread(input)
    finally:
        await runtime.aclose(timeout_s=1)


if __name__ == '__main__':
    asyncio.run(main())
