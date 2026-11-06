"""Driver replacement preserves independent leaves and their original bills."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import DatabasePollingTransport, RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import JobRef, RuntimeDescriptor, TargetRef, TrustedActor

from ._workflow_fixture_203 import leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("scenario", ["live", "dispatch_gap", "parallel", "unknown_leaf"])
async def test_new_driver_reuses_live_leaf_and_completed_dependency(registry_store, tmp_path, scenario):
    store, _ = registry_store
    descriptor = RuntimeDescriptor(1, "agent", "p4.agent@1", None, "p4.tools@1", "p4.hooks@1", "p4.resources@1", {}, {})
    actor = TrustedActor("tenant", "principal", None)
    transport = DatabasePollingTransport(RuntimeInputStore(store), interval_s=0.03)
    client = RuntimeClient(store=store, transport=transport)
    from power_loop.agent.runtime_types import plain

    sid = await store.create_session(
        metadata=dict(
            tenant_id="tenant", principal_id="principal", execution_role="main", runtime_descriptor=plain(descriptor)
        )
    )
    target = TargetRef(store.store_id, sid)
    backend = store._db.dialect.name
    dsn = (
        str(tmp_path / "registry.db")
        if backend == "sqlite"
        else os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    )
    processes = []

    async def launch(role):
        path = tmp_path / (role + ".log")
        with path.open("wb") as err:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.unit._p4_runtime_worker",
                dsn,
                store.table_prefix,
                str(tmp_path / "coordinator.db"),
                role,
                str(tmp_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=err,
                cwd=Path(__file__).resolve().parents[2],
            )
        process.log = path
        processes.append(process)
        await event(process, "ready")
        return process

    async def event(process, kind):
        try:
            line = await asyncio.wait_for(process.stdout.readline(), 20)
        except TimeoutError:
            raise AssertionError("\n".join(p.log.read_text()[-10000:] for p in processes)) from None
        assert line, process.log.read_text()
        value = json.loads(line)
        assert value["event"] == kind, value
        return value

    async def command(process, text):
        process.stdin.write((text + "\n").encode())
        await process.stdin.drain()

    try:
        a, b = leaf("A"), leaf("B")
        for node in (a, b):
            node["spec"]["tools"] = ["hold"]
            node["spec"]["max_rounds"] = 3
        if scenario == "parallel":
            root = {"type": "parallel", "branches": [a, b]}
        else:
            b["inputs_from"] = ["A"]
            root = {"type": "sequence", "steps": [a, b]}
        if scenario == "unknown_leaf":
            a["retry"] = {"max_attempts": 2, "on": ["failed"]}
        handle = await client.workflow(target, {"name": "rebuild", "root": root}, message_id="run", actor=actor)
        c = await launch("C")
        if scenario == "dispatch_gap":
            for _ in range(250):
                node = await store._db.fetchone(
                    f"SELECT generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
                )
                if node:
                    break
                await asyncio.sleep(0.02)
            assert node["generation"] == 0
            a = None
        else:
            a = await launch("A")
            first = await event(a, "running")
            assert first["calls"] == first["used"] == 1
        if scenario == "unknown_leaf":
            a.kill()
            await a.wait()
            await launch("A2")
            result = await client.wait(
                JobRef(store.store_id, handle.job_id, 1), actor=actor, deadline_ms=now_ms() + 15000
            )
            assert result.status == "failed", result
            nodes = await store._db.fetchall(f"SELECT * FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
            assert len(nodes) == 1 and nodes[0]["outcome"] == "interrupted"
            assert (tmp_path / "A-actions").read_text().splitlines() == ["external action"]
            return
        if scenario == "parallel":
            b = await launch("B")
            await event(b, "running")
        c.kill()
        await c.wait()
        c2 = await launch("C2")
        for _ in range(250):
            driver = await store._db.fetchone(
                f"SELECT * FROM {store.table_prefix}jobs WHERE job_id=?", (handle.job_id,)
            )
            if driver["generation"] == 2:
                break
            await asyncio.sleep(0.02)
        assert driver["generation"] == 2, c2.log.read_text()
        old = await client.wait(JobRef(store.store_id, handle.job_id, 1), actor=actor, deadline_ms=now_ms() + 3000)
        assert old.status == "interrupted"
        if a is None:
            a = await launch("A")
            await event(a, "running")
        await command(a, "release")
        await event(a, "completed")
        if scenario != "parallel":
            b = await launch("B")
            await event(b, "running")
        await command(b, "release")
        await event(b, "completed")
        result = await client.wait(JobRef(store.store_id, handle.job_id, 2), actor=actor, deadline_ms=now_ms() + 20000)
        assert result.status == "completed", result
        nodes = await store._db.fetchall(
            f"SELECT * FROM {store.table_prefix}jobs WHERE kind='workflow_node' ORDER BY title"
        )
        assert len(nodes) == 2
        assert all(n["generation"] == 1 and n["outcome"] == "succeeded" for n in nodes)
        for process in (a, b):
            await command(process, "budget")
            budget = await event(process, "budget")
            assert budget["reserved"] == budget["settled"] == 2
        assert result.usage["calls"] == 4
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await client.aclose()
        await transport.aclose()
