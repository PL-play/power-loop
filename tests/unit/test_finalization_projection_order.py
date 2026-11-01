"""Retained closure must survive a later derived processing projection."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.runtime import session_obligations
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.result_revisions import result_id

from .test_job_registry_192 import registry_store, runtime
from .test_processing_decisions_192 import processing_case

__all__ = ["registry_store", "runtime", "processing_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_original_handler_exit_allows_closure_before_its_later_target_projection(processing_case, monkeypatch):
    c = processing_case
    target, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    decisions = await c.commit(snapshot, response)
    entered, release = asyncio.Event(), asyncio.Event()
    project = session_obligations.refresh_closed_decisions
    calls = []

    async def delay(owner, selected):
        assert owner._control_intents.get() is None
        if owner is c.manager and not calls:
            calls.append(selected)
            entered.set()
            await release.wait()
        return await project(owner, selected)

    with monkeypatch.context() as patch:
        patch.setattr(session_obligations, 'refresh_closed_decisions', delay)
        exiting = asyncio.create_task(c.manager.close_model_execution(c.actor, c.caller,
            evidence={'exit':'completed', 'last_assistant_ref':response.to_dict(), 'processing_decisions':decisions}))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            cold = JobManager(await c.connect(), now_ms=c.manager.now)
            source = await cold.db.fetchone(f"SELECT event_id FROM {cold.events} WHERE event_id=? AND event_type='model_closed'",
                ('model_closed_'+c.caller.job_id,))
            assert source is not None
            before = await cold.db.fetchone(f"SELECT delivery_status FROM {cold.events} WHERE event_id=?", (result_id(target, 1),))
            assert before['delivery_status']=='pending_delivery'
            receipt = await cold.finalize_supervision(c.actor, target, generation=1)
            release.set()
            await asyncio.wait_for(exiting, 5)
            after = await cold.db.fetchone(f"SELECT delivery_status FROM {cold.events} WHERE event_id=?", (result_id(target, 1),))
            assert after['delivery_status']=='handled'
            assert await cold.finalize_supervision(c.actor, target, generation=1)==receipt
            await session_obligations.verify(cold, c.actor)
            saved = await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?", (receipt['event_id'],))
            refs = json.loads(saved['payload_json'])['proof']['references']
            assert all('delivery_status' not in ref for ref in refs)
            assert any(ref['event_id']==result_id(target, 1) for ref in refs)
            original = await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?", (result_id(target, 1),))
            changed = json.loads(original['payload_json'])
            changed['tampered'] = True
            damaged = json.dumps(changed)
            await cold.db.execute(f"UPDATE {cold.events} SET payload_json=? WHERE event_id=?", (damaged, result_id(target, 1)))
            try:
                with pytest.raises(JobError, match='finalization_evidence_changed'):
                    await cold.finalize_supervision(c.actor, target, generation=1)
                assert (await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?",
                    (result_id(target, 1),)))['payload_json']==damaged
            finally:
                await cold.db.execute(f"UPDATE {cold.events} SET payload_json=? WHERE event_id=?",
                    (original['payload_json'], result_id(target, 1)))
        finally:
            release.set()
            await asyncio.wait_for(exiting, 5)
