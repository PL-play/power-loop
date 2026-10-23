"""Do not permit different controlled boots to share one cancellable host task."""

import asyncio

import pytest

from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_no_foreign_boot_body_is_interrupted_by_local_shutdown(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    runtimes = [
        TaskRuntime(
            manager,
            tenant_id="cross-runtime",
            principal_id="host",
            executor_id=e,
            boot_id="boot",
            boot_contract_hash="a" * 64,
        )
        for e in ("a", "b")
    ]
    bindings = []
    for i, runtime in enumerate(runtimes):
        sid = await store.create_session()
        actor = runtime.actor(sid)
        reg = await manager.register_job(actor, kind="send", session_id=sid, system_key=f"send-{i}")
        bindings.append(await runtime.claim(actor, reg))
    ready, release = asyncio.Event(), asyncio.Event()
    cancelled = []

    async def sharing_task():
        async with runtimes[0].execution(bindings[0]):
            try:
                async with runtimes[1].execution(bindings[1]):
                    ready.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.append("foreign-boot-b")
                        raise
            except JobError as exc:
                assert exc.code == "executor_task_owner_conflict"
                ready.set()

    task = asyncio.create_task(sharing_task())
    try:
        await asyncio.wait_for(ready.wait(), 2)
        await runtimes[0].shutdown(timeout_s=0.2, grace_s=0.01)
        assert not cancelled, "boot A shutdown cancelled an active independent boot B body"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtimes[1].shutdown(timeout_s=0.3)


@pytest.mark.parametrize("operation", ["claim", "model_send"])
async def test_foreign_task_ownership_rejected_before_admission(registry_store, operation):
    store, _ = registry_store
    manager = JobManager(store)
    a, b = [
        TaskRuntime(
            manager, tenant_id="owners", principal_id="host", executor_id=e, boot_id="boot", boot_contract_hash="a" * 64
        )
        for e in ("a", "b")
    ]
    sid = await store.create_session()
    actor = a.actor(sid)
    first = await manager.register_job(actor, kind="send", session_id=sid, system_key="a")
    binding = await a.claim(actor, first)
    second = await manager.register_job(actor, kind="send", session_id=sid, system_key="b")
    try:
        async with a.execution(binding):
            with pytest.raises(JobError, match="executor_task_owner_conflict"):
                if operation == "claim":
                    await b.claim(actor, second)
                else:
                    async with b.model_send(None, sid, 1):
                        pytest.fail("foreign model body entered")
        assert b._lifecycle._start_task is None
        assert (await manager.get_job(actor, second.job.job_id)).generation == 0
    finally:
        await a.shutdown(timeout_s=0.5)


async def test_independent_task_with_inherited_context_survives_other_boot_shutdown(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    a, b = [
        TaskRuntime(
            manager, tenant_id="owners", principal_id="host", executor_id=e, boot_id="boot", boot_contract_hash="a" * 64
        )
        for e in ("a", "b")
    ]
    bindings = []
    for runtime in (a, b):
        sid = await store.create_session()
        actor = runtime.actor(sid)
        reg = await manager.register_job(actor, kind="send", session_id=sid, system_key=runtime.executor_id)
        bindings.append(await runtime.claim(actor, reg))
    release, ready = asyncio.Event(), asyncio.Event()
    foreign = []

    async def separate():
        async with b.execution(bindings[1]):
            ready.set()
            await release.wait()

    async def creator():
        async with a.execution(bindings[0]):
            foreign.append(asyncio.create_task(separate()))
            await asyncio.Event().wait()

    parent = asyncio.create_task(creator())
    try:
        await asyncio.wait_for(ready.wait(), 3)
        result = await a.shutdown(timeout_s=1)
        assert result["durable_drain_confirmed"] and parent.done()
        assert len(foreign) == 1 and not foreign[0].done() and not foreign[0].cancelling()
        assert not b._lifecycle.closed
    finally:
        release.set()
        await asyncio.gather(parent, *foreign, return_exceptions=True)
        await b.shutdown(timeout_s=1)


async def test_legacy_uncontrolled_runtimes_preserve_nested_context_behavior(registry_store):
    store, _ = registry_store
    manager = JobManager(store)
    bindings = []
    for executor in ("a", "b"):
        runtime = TaskRuntime(manager, tenant_id="legacy", principal_id="host", executor_id=executor, boot_id="boot")
        sid = await store.create_session()
        actor = runtime.actor(sid)
        reg = await manager.register_job(actor, kind="send", session_id=sid, system_key=executor)
        bindings.append((runtime, await runtime.claim(actor, reg)))
    async with bindings[0][0].execution(bindings[0][1]), bindings[1][0].execution(bindings[1][1]):
        assert all(runtime._local_executions for runtime, _ in bindings)
    assert all(not runtime._local_executions for runtime, _ in bindings)
