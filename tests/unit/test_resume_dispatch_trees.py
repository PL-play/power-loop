"""Dispatch permission survives crashes without reconstructing an executor."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop.runtime.session_controls import ControlTransaction

from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import nudge_receipt, ready
from .test_task_runtime_192 import finish, managed, receipt

__all__ = ["registry_store", "managed", "ready"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def exercise_writer(c, monkeypatch, method):
    original, write = getattr(c.manager, method), c.manager._write
    one, many = ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault("_resume_read_jobs", set()).update(
            row["job_id"] for row in rows if row is not None and "job_id" in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    async def observed(*args, **kwargs):
        roots = {row["job_id"]: row["root_job_id"] for row in await c.manager.db.fetchall(
            f"SELECT job_id,root_job_id FROM {c.manager.jobs}")}

        @asynccontextmanager
        async def scoped():
            async with write() as tx:
                yield tx
                jobs = tx.touched_jobs | tx.__dict__.get("_resume_read_jobs", set())
                footprints.append({roots[job] for job in jobs})

        with monkeypatch.context() as patch:
            patch.setattr(ControlTransaction, "fetchone", fetchone)
            patch.setattr(ControlTransaction, "fetchall", fetchall)
            patch.setattr(c.manager, "_write", scoped)
            return await original(*args, **kwargs)

    monkeypatch.setattr(c.manager, method, observed)
    await issue_nudge(c.fixture, monkeypatch)
    receipt = await nudge_receipt(c)
    await finish(c.fixture)
    assert receipt["status"] == "accepted" and receipt["generation"] == 2
    assert c.model.child_calls == 2
    assert footprints and all(len(roots) <= 1 for roots in footprints), footprints
    return footprints


async def test_dispatch_permission_target_and_acknowledgement_use_separate_tree_writers(ready, monkeypatch):
    footprints = await exercise_writer(ready, monkeypatch, "claim_resume_dispatch")
    assert len(footprints) == 3 and footprints[0] == footprints[2] and footprints[0] != footprints[1]




@asynccontextmanager
async def held_dispatch(c, monkeypatch):
    entered = asyncio.Event()
    args = {}

    async def withhold(actor, operation_id, **kwargs):
        args.update(actor=actor, operation_id=operation_id, **kwargs)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(c.manager, "claim_resume_dispatch", withhold)
    task = asyncio.create_task(issue_nudge(c.fixture, monkeypatch))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=task)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await finish(c.fixture)


@pytest.fixture
async def pending(ready, monkeypatch):
    async with held_dispatch(ready, monkeypatch) as value:
        yield value


@pytest.fixture
async def registered_pending(managed, monkeypatch):
    from power_loop.runtime import executor_registry as registry

    store, manager, runtime, loop, model, parent, connect = managed
    actor = runtime.actor(parent)
    identity = registry.ExecutorIdentity(actor.tenant_id, actor.principal_id, runtime.executor_id, runtime.boot_id)
    await registry.register(manager, identity, contract_hash='a'*64, startup_lease_ms=300000)
    await registry.activate(manager, identity, expected_version=1, system_key='start', lease_ms=300000)
    await loop.send('delegate', parent)
    created = await receipt(managed)
    await finish(managed)
    await loop.flush_follow_ups(parent)
    old = await manager.get_job(actor, created['job_id'])
    assert old.generation == 1 and old.result_status == 'handled'
    model.release.clear()
    model.entered.clear()
    c = SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model,
        parent=parent, connect=connect, fixture=managed, old=old)
    async with held_dispatch(c, monkeypatch) as value:
        yield value
