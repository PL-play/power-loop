"""Neither a projected retirement nor an original raw-row ACK is invented exit."""
import asyncio

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.pending_calls import recover_runtime, retire_pending

from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_actual_live_owner_cannot_be_retired_from_a_closed_intake_projection(registry_store, monkeypatch):
    from power_loop.runtime import controller_retirement
    entered, release = asyncio.Event(), asyncio.Event()
    original = controller_retirement.retire
    async def hold(runtime, binding, local):
        row = await runtime.manager.get_job(binding.actor, binding.lease.job_id)
        if row.kind == 'subagent':
            entered.set()
            await release.wait()
        return await original(runtime, binding, local)
    monkeypatch.setattr(controller_retirement, 'retire', hold)
    c = await owned_pending_case(registry_store, 'subagent', owner_exit_gate=entered)
    loop = runtime = None
    try:
        owner = c.state.owner_binding
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET intake='closed',resource_state='unknown',lease_until_ms=NULL WHERE job_id=?", (owner.lease.job_id,))
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        await enqueue(loop, actor, c.effect['job_id'])
        before = (await c.store.get_state(c.state.source_binding.actor.session_id)).pending
        with pytest.raises(JobError):
            await retire_pending(loop, actor, c.state.source)
        assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending == before
        assert any(slot.lease == owner.lease and not slot.task.done() and slot.exit is None
                   for slot in c.runtime._controller_task_exits.slots.values())
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='pending_owner_transfer'")
    finally:
        release.set()
        await cleanup(c, loop, runtime)


async def test_returned_subset_is_preserved_while_only_unreturned_calls_are_reviewed(registry_store):
    c = await owned_pending_case(registry_store, 'workflow_node', raw_fail_at=2, effect_calls=2)
    loop = runtime = None
    try:
        sid = c.state.source_binding.actor.session_id
        raw = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (sid,))
        assert len(raw) == 1
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review' and not result['acknowledgement_created']
        assert len(result['notices']) == 1
        await loop.send('Handle the original known outcome; do not repeat its action.', c.state.root)
        assert (await runtime.manager.get_job(actor, c.effect['job_id'])).supervision_status == 'closed'
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (sid,)) == raw
        assert c.state.executed == len(c.state.effects) == c.state.prepared == 2
        assert c.state.lookups == c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c, loop, runtime)
