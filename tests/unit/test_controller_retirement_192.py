"""A finished original controller must not be reopened with its still-valid lease."""

import pytest

from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import Execution

from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("entry", ["fresh_manager", "copied_binding"])
async def test_exited_controller_lease_cannot_authorize_replacement_work(registry_store, entry):
    c = await original_owner_case(registry_store, "subagent", allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send("Launch the original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        binding = c.state.owner_binding
        assert binding.stop_renewal.is_set()
        assert all(local["binding"] is not binding for local in c.runtime._local_executions.values())
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), binding.lease.job_id)
        assert owner.result_status == "none" and owner.supervision_status == "open"
        assert c.state.counts["BUILD"] == 1 and not c.state.effects
        if entry == "fresh_manager":
            manager = JobManager(await registry_store[1]())
            with pytest.raises(
                JobError, match="job_intake_closed|unresolved_prior_attempt|controller_execution_exited"
            ):
                await manager.check_execution(binding.actor, binding.lease)
        else:
            replacement = Execution(c.runtime, binding.actor, binding.lease)
            with pytest.raises(
                JobError, match="job_intake_closed|unresolved_prior_attempt|controller_execution_exited"
            ):
                async with c.runtime.execution(replacement, owner=True):
                    pass
        # Retiring control permission is distinct from handling the original
        # result or stopping arbitrary descendant resources.
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), owner.job_id)
        assert owner.outcome is None and owner.result_status == "none"
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def read_in_process(c, binding, record_property):
    import asyncio
    import json
    import os
    import sys
    from dataclasses import asdict

    if c.manager.db.dialect.name == "sqlite":
        rows = await c.manager.db.fetchall("PRAGMA database_list")
        dsn = next(row["file"] for row in rows if row["name"] == "main")
    else:
        dsn = os.environ["POWER_LOOP_TASK_PG_DSN"]
        from urllib.parse import urlparse

        assert urlparse(dsn).path.endswith("_test")
    code = """import asyncio,json,sys
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.jobs import JobManager,JobActor,JobLease,JobError
async def main():
 data=json.load(sys.stdin)
 store=await open_store(data['dsn'],table_prefix=data['prefix'],schema='tasks')
 try:
  manager=JobManager(store)
  try:
   await manager.check_execution(JobActor(**data['actor']),JobLease(**data['lease']))
  except JobError as error:
   print(json.dumps({'error':error.code}))
  else:
   print(json.dumps({'authorized':True}))
 finally:
  await store.close()
asyncio.run(main())"""
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    record_property("controller_reader_pid", child.pid)
    try:
        output, error = await asyncio.wait_for(
            child.communicate(
                json.dumps(
                    {
                        "dsn": dsn,
                        "prefix": c.manager.prefix,
                        "actor": asdict(binding.actor),
                        "lease": asdict(binding.lease),
                    }
                ).encode()
            ),
            20,
        )
        assert child.returncode == 0, error[-300:]
        return json.loads(output)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


async def test_another_process_observes_closed_controller_admission_without_releasing_results(
    registry_store, record_property
):
    c = await original_owner_case(registry_store, "subagent", allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send("Launch the original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        before = await c.manager.execution_usage(
            c.runtime.actor(c.state.root), c.state.owner_binding.lease.job_id, generation=1
        )
        observed = await read_in_process(c, c.state.owner_binding, record_property)
        assert observed.get("error") in ("job_intake_closed", "unresolved_prior_attempt")
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_binding.lease.job_id)
        assert (
            owner.result_status == "none" and owner.supervision_status == "open" and owner.resource_state == "unknown"
        )
        assert await c.manager.execution_usage(c.runtime.actor(c.state.root), owner.job_id, generation=1) == before
        assert not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def test_foreign_runtime_cannot_borrow_a_live_controller_lease(registry_store):
    from power_loop.runtime.task_runtime import TaskRuntime

    c = await original_owner_case(registry_store, "subagent")
    try:
        await c.loop.send("Launch original task once.", c.state.root)
        import asyncio

        await asyncio.wait_for(c.state.entered.wait(), 5)
        original = c.state.owner_binding
        foreign = TaskRuntime(
            c.manager,
            tenant_id=c.runtime.tenant_id,
            principal_id=c.runtime.principal_id,
            executor_id="foreign-executor",
            boot_id="foreign-boot",
        )
        copied = Execution(foreign, original.actor, original.lease)
        with pytest.raises(JobError, match="execution_binding_mismatch"):
            async with foreign.execution(copied, owner=True):
                pass
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def test_retirement_cancellation_still_restores_local_context_and_heartbeat(registry_store, monkeypatch):
    import asyncio

    from power_loop.runtime import controller_retirement
    from power_loop.runtime.task_runtime import _OWNER

    c = await original_owner_case(registry_store, "subagent", allowed_calls=1)
    initial = set(delegation._TASKS)

    async def cancelled(*args):
        raise asyncio.CancelledError("cancel retirement write")

    monkeypatch.setattr(controller_retirement, "retire", cancelled)
    try:
        await c.loop.send("Launch original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        assert not c.runtime._local_executions and _OWNER.get() is None
        binding = c.state.owner_binding
        with pytest.raises(JobError, match="controller_execution_exited"):
            async with c.runtime.execution(binding, owner=True):
                pass
        assert all(
            task.done() or task.get_name() != "job-lease-" + binding.lease.job_id for task in asyncio.all_tasks()
        )
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


@pytest.mark.parametrize("fault", ["before_commit", "after_commit", "cancelled"])
async def test_failed_retirement_is_retained_and_retry_does_not_replay_work(
    registry_store, monkeypatch, record_property, fault
):
    import asyncio

    from power_loop.runtime import controller_retirement

    c = await original_owner_case(registry_store, "subagent", allowed_calls=1)
    initial = set(delegation._TASKS)
    original = controller_retirement.retire
    failed = False

    async def fail_once(runtime, binding, local):
        nonlocal failed
        if not failed and binding.lease.job_id == c.state.owner_binding.lease.job_id:
            failed = True
            if fault == "after_commit":
                await original(runtime, binding, local)
            if fault == "cancelled":
                raise asyncio.CancelledError("interrupted controller receipt")
            raise OSError("controller receipt unavailable")
        return await original(runtime, binding, local)

    monkeypatch.setattr(controller_retirement, "retire", fail_once)
    try:
        await c.loop.send("Launch original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        binding = c.state.owner_binding
        usage = await c.manager.execution_usage(binding.actor, binding.lease.job_id, generation=1)
        assert failed and not c.runtime._local_executions
        # The synchronous local witness survives even cancellation before SQL.
        copied = Execution(c.runtime, binding.actor, binding.lease)
        with pytest.raises(JobError, match="controller_execution_exited"):
            async with c.runtime.execution(copied, owner=True):
                pytest.fail("exited controller reopened")
        status = await c.runtime.executor_status()
        assert status["controller_retirement_pending"] == 1
        assert status["store_safe_to_close"] is False
        # Concurrent new callers share one retry; lost ACK creates no duplicate.
        await asyncio.gather(*(c.runtime.ensure_started() for _ in range(8)))
        await c.runtime.ensure_started()
        status = await c.runtime.executor_status()
        assert status["controller_retirement_pending"] == 0
        assert (await read_in_process(c, binding, record_property))["error"] in (
            "job_intake_closed", "unresolved_prior_attempt"
        )
        events = await c.manager.db.fetchall(
            f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='controller_retired'",
            (binding.lease.job_id,),
        )
        assert len(events) == 1
        owner = await c.manager.get_job(binding.actor, binding.lease.job_id)
        assert owner.result_status == "none" and owner.supervision_status == "open"
        assert c.state.counts["BUILD"] == 1 and not c.state.effects
        assert await c.manager.execution_usage(binding.actor, binding.lease.job_id, generation=1) == usage
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


@pytest.mark.parametrize("shutdown_timeout", [0, 3])
async def test_repeated_write_failure_blocks_new_work_but_preserves_query_and_stop(
    registry_store, monkeypatch, shutdown_timeout
):
    from power_loop.runtime import controller_retirement

    c = await original_owner_case(registry_store, "subagent", allowed_calls=1)
    original = controller_retirement._persist
    initial = set(delegation._TASKS)

    writes = []

    async def unavailable(*args):
        writes.append(True)
        raise OSError("receipt database unavailable")

    monkeypatch.setattr(controller_retirement, "_persist", unavailable)
    try:
        await c.loop.send("Launch original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        with pytest.raises(JobError, match="controller_retirement_requires_reconciliation"):
            c.runtime.check_local_admission()
        with pytest.raises(OSError):
            await c.runtime.ensure_started()
        binding = c.state.owner_binding
        # JobManager remains available for mechanical reads and durable stop.
        owner = await c.manager.get_job(binding.actor, binding.lease.job_id)
        assert owner.result_status == "none" and owner.supervision_status == "open"
        before_shutdown = len(writes)
        status = await c.runtime.shutdown(timeout_s=shutdown_timeout)
        if shutdown_timeout == 0:
            assert len(writes) == before_shutdown
        assert status["controller_retirement_pending"] == 1 and not status["store_safe_to_close"]
        assert c.state.counts["BUILD"] == 1 and not c.state.effects
        monkeypatch.setattr(controller_retirement, "_persist", original)
        await c.runtime._controller_retirements.flush()
        assert not c.runtime._controller_retirements.pending
    finally:
        monkeypatch.setattr(controller_retirement, "_persist", original)
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)
