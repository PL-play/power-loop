"""An independent driver/board writer: no worker and no model execution."""
import asyncio
import os

from power_loop import MemoryMessageTransport, RuntimeClient, StoreBlackboard, TimerRunner
from power_loop.runtime.store.factory import open_store


async def main():
    store = await open_store(os.environ['P7A_DB_DSN'], table_prefix=os.environ['P7A_TABLE_PREFIX'], schema='tasks')
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    try:
        print('ready', flush=True)
        await asyncio.to_thread(input)
        if os.environ['P7A_MODE'] == 'timers':
            print(await TimerRunner(store=store, client=client).scan_once(), flush=True)
        else:
            board = StoreBlackboard(store)
            for i in range(10):
                await board.post('shared', text=f"{os.environ['P7A_WRITER']}:{i}")
            print('10', flush=True)
    finally:
        await client.aclose()
        await transport.aclose()
        await store.close()


if __name__ == '__main__':
    asyncio.run(main())
