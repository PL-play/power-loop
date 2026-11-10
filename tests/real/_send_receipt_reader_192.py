"""Mechanical fresh-process read only: no runtime, provider or model factory."""
import asyncio
import json
import sys

from power_loop import SessionStore


async def main():
    path, prefix, session_id = sys.argv[1:]
    store = await SessionStore.open(path, table_prefix=prefix, schema="verify")
    try:
        rows = await store.load_all_messages(session_id)
        print(json.dumps({
            "assistant": [row.content for row in rows if row.role == "assistant"],
            "real_provider_requests": 0,
        }))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
