"""An original continuation grant can finish after exit without new dispatch."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.session_controls import ControlTransaction

from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import finish, managed

__all__ = ["registry_store", "managed", "ready"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def pending(ready, monkeypatch):
    c, args = ready, {}
    entered = asyncio.Event()
    admit = c.manager.resume_subagent

    async def hold(actor, job_id, **kwargs):
        args.update(actor=actor, job_id=job_id, **kwargs)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(c.manager, 'resume_subagent', hold)
    source = asyncio.create_task(issue_nudge(c.fixture, monkeypatch))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=source, admit=admit)
    finally:
        source.cancel()
        await asyncio.gather(source, return_exceptions=True)
        await finish(c.fixture)














async def test_a_committed_inbox_input_cannot_hide_behind_an_earlier_admission_snapshot(pending, monkeypatch):
    p, c = pending, pending.c
    peer = await c.connect()
    entered, release, put_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lookup = ControlTransaction.fetchall
    paused = []

    async def after_lookup(tx, sql, params=()):
        rows = await lookup(tx, sql, params)
        if not paused and sql.startswith('SELECT * FROM '+c.manager.events+' WHERE operation_key=? OR event_id=?'):
            paused.append(True)
            entered.set()
            await release.wait()
        return rows

    async def put_input():
        put_started.set()
        return await peer.inbox_put(c.sid, [{'item_id':'concurrent-input', 'content':'must not be skipped'}])

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, 'fetchall', after_lookup)
        action = asyncio.create_task(p.admit(**p.args))
        put = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            put = asyncio.create_task(put_input())
            await asyncio.wait_for(put_started.wait(), 3)
            try:
                await asyncio.wait_for(asyncio.shield(put), .2)
                before_admission = True
            except TimeoutError:
                before_admission = False
            release.set()
            if before_admission:
                with pytest.raises(JobError, match='resume_input_pending'):
                    await asyncio.wait_for(action, 5)
                assert (await c.manager.get_job(p.args['actor'], c.old.job_id)).generation==1
            else:
                operation, _ = await asyncio.wait_for(action, 5)
                assert operation.result['generation']==2
            await asyncio.wait_for(put, 5)
            items = await c.store.inbox_items(c.sid)
            assert any(item['item_id']=='concurrent-input' and item['status']=='pending' for item in items)
            assert len(items)==(1 if before_admission else 2)
            assert c.model.child_calls==1
        finally:
            release.set()
            await asyncio.gather(action, return_exceptions=True)
            if put is not None:
                await asyncio.wait_for(put, 5)
