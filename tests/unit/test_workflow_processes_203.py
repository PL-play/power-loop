"""Real two-process driver replacement preserves a healthy independent leaf."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import DatabasePollingTransport, RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import RuntimeDescriptor, TargetRef, TrustedActor

from ._workflow_fixture_203 import leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _event(process, expected):
    async with asyncio.timeout(30):
        while True:
            line = await process.stdout.readline()
            assert line, (await process.stderr.read()).decode()[-4000:]
            value = json.loads(line)
            if value["event"] == expected:
                return value


async def _launch(store, coordinator, role):
    backend = store._db.dialect.name
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.unit._p4_runtime_worker",
        dsn, store.table_prefix, str(coordinator), role,
        cwd=Path(__file__).resolve().parents[2],
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await _event(process, "ready")
    return process


async def _command(process, command):
    process.stdin.write((command + "\n").encode())
    await process.stdin.drain()


async def _quit(process):
    if process.returncode is None:
        await _command(process, "quit")
        await asyncio.wait_for(process.wait(), 20)
        assert process.returncode == 0, (await process.stderr.read()).decode()[-4000:]


async def test_real_driver_process_crash_recovers_with_healthy_node_process(registry_store, tmp_path):
    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("real driver/node processes use a shared server DB")
    descriptor = RuntimeDescriptor(
        2, "agent", "p4.agent@215", None, "p4.tools@215", "p4.hooks@215", "p4.resources@215", {}, {},
    )
    actor = TrustedActor("tenant", "principal", None)
    port = RuntimeInputStore(store)
    root = await port.create_session(descriptor, actor=actor, options={"session_id": "workflow-root"})
    transport = DatabasePollingTransport(port, interval_s=0.03)
    client = RuntimeClient(store=store, transport=transport)
    processes = []
    try:
        handle = await client.workflow(
            root, {"name": "two processes", "root": leaf("A")},
            message_id="two-process-run", actor=actor,
        )
        driver = await _launch(store, tmp_path / "coordinator.db", "C")
        processes.append(driver)
        assert (await _event(driver, "driver_claimed"))["send_index"] == 1
        node = await _launch(store, tmp_path / "coordinator.db", "N")
        processes.append(node)
        await _event(node, "node_started")
        children = await client.list_children(handle.target, actor=actor)
        assert len(children.items) == 1
        node_status = await client.inspect(children.items[0], actor=actor)
        assert node_status.execution_state == "running" and not node_status.cancel_requested

        driver.kill()
        await driver.wait()
        assert node.returncode is None
        replacement = await _launch(store, tmp_path / "coordinator.db", "C2")
        processes.append(replacement)
        assert (await _event(replacement, "driver_claimed"))["send_index"] == 2
        assert node.returncode is None
        await _command(node, "release")
        result = await client.wait(handle.input_ref, actor=actor, deadline_ms=now_ms() + 20000)
        assert result.status == "completed" and result.final_text == "two process done"
        assert result.send_index == 2
        continuation = await store._db.fetchone(
            f"SELECT continuation_of,continuation_index,terminal_status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue'",
            (handle.target.session_id,),
        )
        assert continuation == {
            "continuation_of": handle.input_ref.message_id,
            "continuation_index": 1,
            "terminal_status": "completed",
        }
        sessions = await store._db.fetchall(
            f"SELECT session_id,session_kind FROM {store.t.sessions} ORDER BY created_at",
        )
        assert [row["session_kind"] for row in sessions] == ["agent", "workflow_driver", "agent"]
        assert isinstance(handle.target, TargetRef)
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await client.aclose()
        await transport.aclose()
