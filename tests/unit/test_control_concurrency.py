"""Prove existing global-lock behavior before considering a lock replacement."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import LLMResponse, LLMService
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store, runtime
from .test_round_change_signals import fixture_loop

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def stop_process(process):
    if process.returncode is None:
        process.kill()
    await process.communicate()


async def test_separate_processes_never_dispatch_two_models_for_one_session(registry_store, tmp_path):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    sid = await store.create_session()
    release = tmp_path / "release"
    first_result, second_result = tmp_path / "first.json", tmp_path / "second.json"

    async def spawn(path, role):
        return await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._concurrent_send_worker",
            dsn, store.table_prefix, sid, str(path), str(release), role,
            cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    first = await spawn(first_result, "first")
    second = None
    try:
        async with asyncio.timeout(15):
            while not first_result.exists():
                if first.returncode is not None:
                    output, _ = await first.communicate()
                    raise AssertionError(output.decode())
                await asyncio.sleep(.01)
        assert json.loads(first_result.read_text()) == {"status": "model_entered", "calls": 1}
        second = await spawn(second_result, "second")
        output, _ = await asyncio.wait_for(second.communicate(), 15)
        assert second.returncode == 0, output.decode()
        assert json.loads(second_result.read_text()) == {"status": "busy", "calls": 0}
        release.touch()
        output, _ = await asyncio.wait_for(first.communicate(), 15)
        assert first.returncode == 0, output.decode()
        assert json.loads(first_result.read_text()) == {"status": "completed", "calls": 1}
        cold = await connect(schema="compatible")
        messages = await cold.load_all_messages(sid)
        assert [row.role for row in messages] == ["user", "assistant"]
        assert [row.content for row in messages] == ["first", "first"]
        jobs = await cold._db.fetchall(f"SELECT status,supervision_status FROM {cold.table_prefix}jobs WHERE session_id=?", (sid,))
        assert [(row["status"], row["supervision_status"]) for row in jobs] == [("model_closed", "closed")]
    finally:
        release.touch()
        await stop_process(first)
        if second is not None:
            await stop_process(second)


async def test_different_sessions_enter_models_concurrently_and_close_without_deadlock(registry_store):
    store, connect = registry_store
    stores = [store, await connect(schema="compatible")]
    entered, release = asyncio.Event(), asyncio.Event()
    arrivals = []

    class Model(LLMService):
        def __init__(self, label):
            self.label = label
        async def complete(self, request, **kwargs):
            arrivals.append(self.label)
            if len(arrivals) == 2:
                entered.set()
            await release.wait()
            return LLMResponse(raw_text=self.label)
        def stream(self, request):
            raise AssertionError("nonstream fixture")
        async def close(self):
            pass

    pairs = [fixture_loop(s, Model(str(i))) for i, s in enumerate(stores)]
    sessions = [await loop.new_session() for _, loop in pairs]
    tasks = [asyncio.create_task(loop.send(str(i), sid)) for i, ((_, loop), sid) in enumerate(zip(pairs, sessions, strict=True))]
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert sorted(arrivals) == ["0", "1"]
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 15)
        assert [result.final_text for result in results] == ["0", "1"]
        for i, (s, sid) in enumerate(zip(stores, sessions, strict=True)):
            assert [row.content for row in await s.load_all_messages(sid)] == [str(i), str(i)]
        assert all(not runtime.manager._execution_controls for runtime, _ in pairs)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        for runtime, loop in pairs:
            await runtime.shutdown(timeout_s=1)
            await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("order", ["stop_first", "results_first", "interleaved"])
async def test_same_tree_stop_new_send_and_duplicate_child_completions_do_not_deadlock(runtime, order):
    _store, manager, actor, root, parent, connect, _now = runtime
    others = [JobManager(await connect(schema="compatible"), now_ms=manager.now) for _ in range(3)]
    children = []
    for i in range(2):
        registered = await manager.register_job(actor, kind="subagent", parent=parent, system_key=f"child-{i}")
        lease = await manager.claim_job(actor, registered.job.job_id, executor_id=f"worker-{i}",
                                        boot_id="boot", expected_version=1)
        children.append(lease)
    barrier = asyncio.Event()

    async def finish(i, m):
        await barrier.wait()
        return await m.record_execution_result(actor, children[i], outcome="succeeded",
            result={"value": i}, resource_state="stopped")

    async def stop(m):
        await barrier.wait()
        return await m.request_cancel(actor, root.job.job_id, system_key="stop-tree", reason="stop original tree")

    async def new_send():
        await barrier.wait()
        try:
            return await others[2].register_job(actor, kind="send", session_id=actor.session_id,
                parent=parent, system_key="concurrent-send", request={"model_attempt_protocol": 1})
        except JobError as exc:
            assert exc.code == "cancel_requested"
            return None

    completions = [finish(i, m) for i, m in ((0, manager), (1, others[0]), (0, others[1]), (1, others[2]))]
    stops = [stop(manager), stop(others[1])]
    actions = stops + completions + [new_send()] if order == "stop_first" else (
        completions + [new_send()] + stops if order == "results_first" else [completions[0], stops[0], *completions[1:], new_send(), stops[1]])
    tasks = [asyncio.create_task(action) for action in actions]
    barrier.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), 15)
    assert (await manager.get_job(actor, root.job.job_id)).cancel_requested_at is not None
    for i, lease in enumerate(children):
        row = await manager.get_job(actor, lease.job_id)
        assert row.result == {"value": i} and row.resource_state == "stopped"
        assert row.outcome in ("succeeded", "cancelled")
        events = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='result'",
                                         (lease.job_id,))
        assert len(events) == 1
        assert await others[2].record_execution_result(actor, lease, outcome="succeeded",
            result={"value": i}, resource_state="stopped") == events[0]["event_id"]
    stops = [value for value in results if getattr(value, "action", None) == "stop"]
    assert len(stops) == 2 and stops[0] == stops[1]
    for registered in (value for value in results if getattr(value, "job", None) is not None):
        with pytest.raises(JobError, match="cancel_requested"):
            await manager.claim_job(actor, registered.job.job_id, executor_id="too-late", boot_id="boot", expected_version=1)
