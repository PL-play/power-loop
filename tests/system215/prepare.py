"""Create a private env and fresh shared test prefix; never print credentials."""
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit

from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.session_native_ddl import CONTRACT_TABLES


async def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN"]
    if not urlsplit(dsn).path.endswith("_test"):
        raise ValueError("P8 refuses non-test databases")
    if len(sys.argv) > 2 and sys.argv[2] == "--clean":
        manifest = json.loads((output / "cluster.json").read_text())
        prefix = manifest["prefix"]
        if not prefix.startswith("p8_215_") or not prefix.replace("_", "").isalnum():
            raise ValueError("invalid P8 prefix")
        store = await open_store(dsn, table_prefix=prefix, schema="tasks")
        for name in reversed((*CONTRACT_TABLES, "schema_migrations")):
            await store._db.execute(f"DROP TABLE IF EXISTS {prefix}{name} CASCADE")
        await store.close()
        print("Removed isolated P8 prefix", prefix)
        return
    prefix = "p8_215_" + secrets.token_hex(4) + "_"
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    manifest = {"store_id": store.store_id, "prefix": prefix,
                "namespace": "p8-215-" + secrets.token_hex(4), "database": urlsplit(dsn).path[1:]}
    (output / "cluster.json").write_text(json.dumps(manifest, indent=2))
    env_path = output / "cluster.env"
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("P8_PG_DSN=" + dsn.replace("127.0.0.1", "host.docker.internal") + "\n")
        stream.write("P8_PREFIX=" + prefix + "\nP8_NAMESPACE=" + manifest["namespace"] + "\n")
    await store.close()
    print("Prepared isolated test prefix", prefix, "in", manifest["database"])


if __name__ == "__main__":
    asyncio.run(main())
