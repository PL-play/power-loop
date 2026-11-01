"""Each continuation admission and dispatch writer belongs to just one tree."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop.runtime.session_controls import ControlTransaction

from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import nudge_receipt, ready
from .test_task_runtime_192 import finish, managed

__all__ = ["registry_store", "managed", "ready"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("method", ["resume_subagent", "claim_resume_dispatch"])
async def test_each_continuation_writer_reads_and_writes_only_its_own_root(ready, monkeypatch, method):
    c = ready
    original = getattr(c.manager, method)
    write = c.manager._write
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
