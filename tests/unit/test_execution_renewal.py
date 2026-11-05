"""Conditional heartbeat fencing and the unavoidable cost of finite leases."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop
from power_loop.runtime.control_runtime import current
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import _SEND, Execution, TaskRuntime

from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import issue_nudge
from .test_send_sql_profile import Profiled, _counts
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import finish, managed

__all__ = ["registry_store", "runtime", "ready", "managed"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def owner(case):
    _store, manager, actor, _root, lease, _connect, _now = case
    task_runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
                              executor_id=lease.executor_id, boot_id=lease.boot_id)
    return task_runtime, Execution(task_runtime, actor, lease)


async def test_owned_renewal_has_one_update_and_one_root_lock_even_at_the_same_time(runtime):
    task_runtime, binding = owner(runtime)
    manager = task_runtime.manager
    async with task_runtime.execution(binding, initialize_model=False):
        original = manager.db
        manager.db = measured = Profiled(original)
        try:
            for _ in range(3):
                measured.records.clear()
                await manager.renew_execution(binding.actor, binding.lease)
                assert len(measured.records) == 2
                assert measured.records[0]["sql"].startswith("SELECT root.*")
                assert measured.records[1]["sql"].startswith("WITH RECURSIVE lineage")
        finally:
            manager.db = original


@pytest.mark.parametrize("fault", ["expired", "cancelled", "actor", "tenant"])
async def test_owned_renewal_cannot_erase_a_durable_barrier(runtime, fault):
    task_runtime, binding = owner(runtime)
    store, manager, actor, root, lease, connect, clock = runtime
    async with task_runtime.execution(binding, initialize_model=False):
        if fault == "expired":
            clock[0] += 30001
        elif fault == "cancelled":
            await JobManager(await connect(), now_ms=manager.now).request_cancel(actor, root.job.job_id, system_key="stop", reason="stop")
        other = replace(actor, session_id=await store.create_session()) if fault == "actor" else (
            replace(actor, tenant_id="foreign") if fault == "tenant" else actor)
        before = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,))
        with pytest.raises(JobError, match={"expired": "expired", "cancelled": "cancel_requested", "actor": "access_denied", "tenant": "access_denied"}[fault]):
            await manager.renew_execution(other, lease)
        assert await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,)) == before


async def test_ancestor_stop_is_returned_by_the_same_child_update(runtime):
    task_runtime, _binding = owner(runtime)
    _store, manager, actor, root, parent, connect, _clock = runtime
    registered = await manager.register_job(actor, kind="subagent", parent=parent, system_key="owned-child")
    lease = await manager.claim_job(actor, registered.job.job_id, executor_id=parent.executor_id, boot_id=parent.boot_id, expected_version=1)
    binding = Execution(task_runtime, actor, lease)
    async with task_runtime.execution(binding, initialize_model=False):
        await JobManager(await connect(), now_ms=manager.now).request_cancel(actor, root.job.job_id, system_key="stop-tree", reason="stop")
        original = manager.db
        manager.db = measured = Profiled(original)
        try:
            with pytest.raises(JobError, match="cancel_requested"):
                await manager.renew_execution(actor, lease)
            assert len(measured.records) == 2
        finally:
            manager.db = original


async def test_owned_renewal_does_not_extend_the_original_completion_window(runtime, monkeypatch):
    from power_loop.runtime import completion_windows
    monkeypatch.setattr(completion_windows, "DEFAULT_TIMEOUT_MS", 1000)
    task_runtime, _binding = owner(runtime)
    _store, manager, actor, _root, parent, _connect, clock = runtime
    registered = await manager.register_job(actor, kind="subagent", parent=parent, independent=True,
        system_key="finite-owner", request={completion_windows.PROTOCOL: 1})
    lease = await manager.claim_job(actor, registered.job.job_id, executor_id=parent.executor_id,
                                   boot_id=parent.boot_id, expected_version=1, lease_ms=300000)
    binding = Execution(task_runtime, actor, lease)
    async with task_runtime.execution(binding):
        clock[0] += 1001
        before = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,))
        with pytest.raises(JobError, match="owner_completion_deadline_exceeded"):
            await manager.renew_execution(actor, lease)
        assert await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,)) == before


async def test_owned_job_renewal_requires_its_still_live_original_executor(registry_store):
    store, _connect = registry_store
    clock = [1_000_000]
    manager = JobManager(store, now_ms=lambda: clock[0])
    task_runtime = TaskRuntime(manager, tenant_id="executor-renewal", principal_id="host",
        executor_id="original", boot_id="boot", lease_ms=300000, boot_contract_hash="a"*64)
    await task_runtime.start()
    actor = task_runtime.actor(await store.create_session())
    registered = await manager.register_job(actor, kind="send", session_id=actor.session_id, system_key="original")
    binding = await task_runtime.claim(actor, registered)
    try:
        async with task_runtime.execution(binding):
            clock[0] += 299000
            await manager.renew_execution(actor, binding.lease, lease_ms=300000)
            clock[0] += 1001  # Job lease is live; the original executor lease has expired.
            before = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (binding.lease.job_id,))
            with pytest.raises(JobError, match="executor_lease_expired"):
                await manager.renew_execution(actor, binding.lease)
            assert await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (binding.lease.job_id,)) == before
    finally:
        await manager.close_model_execution(actor, binding.lease, evidence={"exit": "completed"})
        await task_runtime.shutdown(timeout_s=1)


@pytest.mark.parametrize("committed", [False, True])
async def test_renewal_commit_failure_and_unknown_ack_retry_the_same_lease(runtime, monkeypatch, committed):
    task_runtime, binding = owner(runtime)
    _store, manager, actor, _root, lease, _connect, clock = runtime
    async with task_runtime.execution(binding, initialize_model=False):
        original = manager.db.transaction

        @asynccontextmanager
        async def failed():
            async with original() as tx:
                yield tx
                if not committed:
                    raise OSError("renewal commit failed")
            raise OSError("renewal acknowledgement lost")

        before = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,))
        with monkeypatch.context() as patch:
            patch.setattr(manager.db, "transaction", failed)
            with pytest.raises(OSError, match="renewal"):
                await manager.renew_execution(actor, lease, lease_ms=60000)
        after = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,))
        assert after["lease_until_ms"] == (clock[0]+60000 if committed else before["lease_until_ms"])
        await manager.renew_execution(actor, lease, lease_ms=60000)
        final = await manager.db.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (lease.job_id,))
        assert final == {**before, "lease_until_ms": clock[0]+60000, "updated_at": clock[0]}


async def test_old_executor_cannot_renew_after_actual_generation_takeover(ready, monkeypatch):
    c = ready
    await issue_nudge(c.fixture, monkeypatch)
    try:
        assert (await c.manager.get_job(c.runtime.actor(c.parent), c.old.job_id)).generation == 2
        before = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.old.job_id,))
        stale = Execution(c.runtime, c.runtime.actor(c.sid), c.old_lease)
        token = _SEND.set(stale)
        try:
            with pytest.raises(JobError, match="stale_execution"):
                await c.manager.renew_execution(stale.actor, stale.lease)
        finally:
            _SEND.reset(token)
        assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.old.job_id,)) == before
    finally:
        c.model.release.set()
        await finish(c.fixture)


async def test_finite_durable_leases_require_writes_as_send_time_advances(registry_store, record_property):
    async def measure(ticks):
        store = await registry_store[1](suffix=f"e{ticks}_")
        original = store._db
        store._db = measured = Profiled(original)
        clock = [1_000_000]
        manager = JobManager(store, now_ms=lambda: clock[0])
        task_runtime = TaskRuntime(manager, tenant_id="elapsed", principal_id="host", executor_id="executor", boot_id="boot",
                                   lease_ms=300000, boot_contract_hash="a"*64)

        class Model(LLMService):
            ticks = 0
            calls = 0

            async def complete(self, request, **kwargs):
                self.calls += 1
                binding = _SEND.get()
                for _ in range(self.ticks):
                    from power_loop.runtime import executor_registry
                    clock[0] += 100000  # The configured five-minute leases renew at TTL/3.
                    lifecycle = task_runtime._lifecycle
                    receipt = await executor_registry.heartbeat(manager, lifecycle.identity,
                        expected_version=lifecycle._version, lease_ms=300000)
                    lifecycle._version = receipt["executor"]["version"]
                    start = len(measured.records)
                    await manager.renew_execution(binding.actor, binding.lease, lease_ms=300000)
                    assert len(measured.records)-start == 2
                    current(binding).renew(clock[0]+300000)
                return LLMResponse(raw_text="original response")

            def stream(self, request):
                raise AssertionError("complete only")

            async def close(self):
                pass

        model = Model()
        loop = StatefulAgentLoop(llm=model, store=store, task_runtime=task_runtime,
            config=AgentLoopConfig(compactor=None, max_rounds=2, delegation_policy="single_level"))
        sid = await loop.new_session()
        try:
            assert (await loop.send("warm", sid)).status == "completed"
            measured.records.clear()
            measured.tx_sites.clear()
            model.ticks = ticks
            assert (await loop.send("original long send", sid)).status == "completed"
            assert model.calls == 2
            renewals = [record for record in measured.records if record["sql"].startswith("WITH RECURSIVE lineage")]
            assert len(renewals) == ticks
            return _counts(measured.records, measured.tx_sites)
        finally:
            await task_runtime.shutdown(timeout_s=1)
            await loop.aclose(drain_timeout_s=.1)
            store._db = original

    short, long = await measure(0), await measure(30)
    assert long["control"] >= short["control"]+60
    assert long["control_tx"] >= short["control_tx"]+30
    record_property("finite_lease_elapsed_sql", json.dumps({"0s": short, "3000s": long}))
    target = os.environ.get("POWER_LOOP_SQL_PROFILE")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / f"elapsed-{registry_store[0]._db.dialect.name}.json").write_text(
            json.dumps({"0s": short, "3000s": long, "job_renewals": 30, "executor_renewals": 30,
                        "ttl_ms": 300000, "cadence_ms": 100000, "sql_per_job_renewal": 2}, indent=2)+"\n")


async def test_checkpoint_interval_does_not_drive_execution_lease_writes(registry_store, monkeypatch):
    import asyncio

    from power_loop.runtime.execution_state import CheckpointPolicy

    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='renewal-frequency', principal_id='host',
        executor_id='executor', boot_id='boot', lease_ms=3000,
        checkpoint_policy=CheckpointPolicy(interval_s=.05))
    renewals = []
    original = manager.renew_execution

    async def counted(*args, **kwargs):
        renewals.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(manager, 'renew_execution', counted)

    class SlowModel(LLMService):
        async def complete(self, request, **kwargs):
            await asyncio.sleep(.15)
            return LLMResponse(raw_text='done')

        def stream(self, request):
            raise AssertionError('nonstream fixture')

        async def close(self):
            return None

    loop = StatefulAgentLoop(llm=SlowModel(), store=store, task_runtime=runtime,
        config=AgentLoopConfig(compactor=None, delegation_policy='single_level'))
    try:
        sid = await loop.new_session()
        assert (await loop.send('slow call', sid)).status == 'completed'
        assert not renewals  # Frequent checkpoints must not turn into heartbeat writes.
    finally:
        await loop.aclose(drain_timeout_s=.1)
