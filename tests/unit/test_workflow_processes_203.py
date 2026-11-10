"""Driver replacement preserves independent leaves and their original bills."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import DatabasePollingTransport, RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import JobRef, RuntimeDescriptor, RuntimeRequestError, TargetRef, TrustedActor
from power_loop.workflow import journal

from ._workflow_fixture_203 import leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("scenario", ["live", "dispatch_gap", "parallel", "unknown_leaf", "auto_completed", "auto_pending"])
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
    automatic = scenario.startswith("auto_")

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
                scenario,
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
            line = await asyncio.wait_for(process.stdout.readline(), 30)
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
        spec = {"name": "rebuild", "root": root}
        if automatic:
            a["spec"].update(max_rounds=1, tools=["progress"])
            a["continuation"] = {"max_continuations": 2, "extra_rounds": 3, "gate": "todo_remaining"}
            spec["budget"] = {"max_tokens": 12, "stop_at_remaining_pct": 0}
        handle = await client.workflow(target, spec, message_id="run", actor=actor)
        c = await launch("C")
        if scenario == "dispatch_gap":
            for _ in range(1500):
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
            if automatic:
                await event(a, "completed" if scenario == "auto_completed" else "continuing")
            else:
                await event(a, "running")
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
            next_leaf = leaf("B")
            next_leaf["spec"].update(tools=["hold"], max_rounds=3)
            following = await client.workflow(target, {"name": "after unknown", "root": next_leaf},
                                              message_id="after-unknown", actor=actor)
            b = await launch("B")
            await event(b, "running")
            await command(b, "release")
            await event(b, "completed")
            continued = await client.wait(JobRef(store.store_id, following.job_id, 0),
                                          actor=actor, deadline_ms=now_ms() + 20000)
            assert continued.status == "completed", continued
            assert continued.usage["calls"] == 2
            assert (tmp_path / "A-actions").read_text().splitlines() == ["external action"]
            return
        if scenario in {"parallel", "auto_completed"}:
            b = await launch("B")
            await event(b, "running")
        if automatic:
            original = await store._db.fetchall(
                f"SELECT title,job_id,session_id,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
            )
            a_row = next(n for n in original if n["title"] == "A")
            assert a_row["generation"] == 2
            assert all(item["status"] == "completed" for item in (
                await store.get_runtime_state(a_row["session_id"], "todo")
            )["items"])
        c.kill()
        await c.wait()
        c2 = await launch("C2")
        for _ in range(1500):
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
        if scenario != "auto_completed":
            await command(a, "release")
            await event(a, "completed")
        if scenario not in {"parallel", "auto_completed"}:
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
        assert all(n["generation"] == (2 if automatic and n["title"] == "A" else 1)
                   and n["outcome"] == "succeeded" for n in nodes)
        assert result.usage["calls"] == (6 if automatic else 4)
        if automatic:
            assert result.usage["total_tokens"] == 12
            assert result.structured_output["results"]["A"]["text"] == "A done"
            assert result.structured_output["results"]["A"]["usage"]["calls"] == 4
            for before in original:
                after = next(n for n in nodes if n["title"] == before["title"])
                assert all(after[k] == before[k] for k in ("job_id", "session_id", "generation"))
            for generation in (1, 2):
                old_a = await client.wait(JobRef(store.store_id, a_row["job_id"], generation),
                                          actor=actor, deadline_ms=now_ms() + 3000)
                assert old_a.usage["calls"] == 2 and old_a.usage["total_tokens"] == 4
            saved = await journal.read(store, target.session_id, handle.run_id)
            assert next(n for n in saved["nodes"] if n["path"] == "A")["result_ref"]["generation"] == 2
            # The round-limit response is a model call but executes no tool.
            assert (tmp_path / "A-progress").read_text().splitlines() == ["progress"] * 2
            assert (tmp_path / "B-actions").read_text().splitlines() == ["external action"]
            with pytest.raises(RuntimeRequestError, match="budget_exceeded"):
                await handle.resume(generation=2, node_id="A", message="over budget", message_id="over")
        # A fresh invocation after replacement uses new work, without changing
        # any of the original identities, result or billing assertions above.
        next_leaf = leaf("B")
        next_leaf["spec"].update(tools=["hold"], max_rounds=3)
        following = await client.workflow(target, {"name": "next", "root": next_leaf},
                                          message_id="next", actor=actor)
        next_result = await client.wait(JobRef(store.store_id, following.job_id, 0),
                                       actor=actor, deadline_ms=now_ms() + 20000)
        assert next_result.status == "completed", next_result
        assert next_result.usage["calls"] == 2
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await client.aclose()
        await transport.aclose()
