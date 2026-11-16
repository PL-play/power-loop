"""P8 runner: private test DSNs, current source only, and unconditional container cleanup.

Use the SDK venv from its one development checkout. P8_TEST_CONFIG points to the
existing private P0-R1 .env (contains P6_DATABASE_PASSWORD); it is never copied.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent


def environment():
    config = Path(os.environ["P8_TEST_CONFIG"])
    values = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
    from urllib.parse import quote
    password = quote(values["P6_DATABASE_PASSWORD"], safe="")
    return {**os.environ,
        "POWER_LOOP_TASK_PG_DSN": f"postgresql://p6:{password}@127.0.0.1:35432/design203_p6_test",
        "POWER_LOOP_TASK_MYSQL_DSN": f"mysql://root:{password}@127.0.0.1:33306/design203_p6_test",
        "P5A_TEST_REDIS_URL": "redis://127.0.0.1:36379/0",
        "P5A_TEST_KAFKA_SERVERS": "127.0.0.1:39092",
        "PYTHONPATH": str(ROOT)}


async def auxiliary_database(dsn, database, *, drop=False):
    import asyncpg

    connection = await asyncpg.connect(dsn)
    try:
        if drop:
            await connection.execute(f'DROP DATABASE "{database}" WITH (FORCE)')
        else:
            await connection.execute(f'CREATE DATABASE "{database}"')
    finally:
        await connection.close()


class BackendSelection:
    def __init__(self, backend):
        self.backend = backend

    def pytest_collection_modifyitems(self, config, items):
        selected, excluded = [], []
        for item in items:
            params = getattr(getattr(item, "callspec", None), "params", {})
            # Other unparameterized and dialect-arithmetic tests still run in each pass.
            backend = params.get("registry_store")
            (excluded if backend in {"sqlite", "postgres", "mysql"}
             and backend != self.backend else selected).append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=excluded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["suite", "containers", "pytest"])
    parser.add_argument("--db", choices=["sqlite", "postgres", "mysql"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args, extra = parser.parse_known_args()
    output = args.output.resolve()
    (output / "raw").mkdir(parents=True, exist_ok=True)
    env = environment()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if args.require_clean and dirty:
        raise RuntimeError("final P8 verification requires a committed SDK tree")
    import power_loop
    source = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip(),
        "dirty": bool(dirty), "import_path": power_loop.__file__,
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
        "python": sys.version, "mode": args.mode, "database": args.db, "pytest_arguments": extra,
    }
    (output / f"source-{args.mode}-{args.db or 'all'}.json").write_text(json.dumps(source, indent=2))
    os.environ.update(env)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import pytest
    if args.mode != "containers":
        plugins = [BackendSelection(args.db)] if args.db else []
        if args.mode == "pytest":
            return pytest.main(["-q", "--no-real", "-rs", "-p", "no:cacheprovider", *extra], plugins=plugins)
        database = "p8_suite_" + uuid.uuid4().hex[:12] + "_test"
        dsn = env["POWER_LOOP_TASK_PG_DSN"]
        asyncio.run(auxiliary_database(dsn, database))
        os.environ["POWER_LOOP_TEST_PG_DSN"] = dsn.rsplit("/", 1)[0] + "/" + database
        from urllib.parse import unquote, urlsplit
        os.environ["POSTGRES_PASSWORD"] = unquote(urlsplit(dsn).password)
        try:
            return pytest.main(["-q", "--no-real", "-rs", "-p", "no:cacheprovider", *extra], plugins=plugins)
        finally:
            asyncio.run(auxiliary_database(dsn, database, drop=True))
    env["P8_ENV_FILE"] = str(output / "cluster.env")
    env["P8_OUTPUT"] = str(output)
    prepare = [sys.executable, "tests/system215/prepare.py", str(output)]
    compose = ["sudo", "-n", "--preserve-env=P8_ENV_FILE", "docker", "compose", "-f",
               str(ROOT / "tests/system215/docker-compose.p8.yml"), "-p", "p8-215"]
    subprocess.run(prepare, env=env, check=True)
    try:
        subprocess.run([*compose, "build", "a"], env=env, check=True)
        subprocess.run([*compose, "up", "-d"], env=env, check=True)
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                                "tests/system215/test_three_node.py", *extra], env=env)
    finally:
        subprocess.run([*compose, "down", "-v"], env=env, check=True)
        subprocess.run([*prepare, "--clean"], env=env, check=True)
        (output / "cluster.env").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
