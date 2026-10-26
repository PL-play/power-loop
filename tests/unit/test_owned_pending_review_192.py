"""Original v2 child pending is explicitly reviewed by its actual main."""
import json

import pytest

from power_loop.runtime.controller_recovery import reconcile as recover_owner
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.pending_calls import SNAPSHOT, retire_pending

from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def cleanup(c, loop=None, runtime=None):
    c.state.release.set()
    if runtime is not None:
        await runtime.shutdown(timeout_s=3)
        await runtime.flush_control_checkpoints()
        await loop.aclose(drain_timeout_s=.1)
    await c.runtime.shutdown(timeout_s=3)
    await c.runtime.flush_control_checkpoints()
    await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_original_main_handles_v2_child_pending_without_replaying_effect_or_promoting_child(registry_store, kind):
    c = await owned_pending_case(registry_store, kind)
    loop = runtime = None
    try:
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        closed = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', ('model_closed_'+source.job_id,))
        assert json.loads(closed['payload_json'])['evidence'][SNAPSHOT]['version'] == 2
        original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY session_id,seq')
        usage = await c.manager.execution_usage(c.state.owner_binding.actor, c.state.owner_binding.lease.job_id, generation=1)
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        await enqueue(loop, actor, c.effect['job_id'])
        disposed = await retire_pending(loop, actor, source.job_id)
        assert disposed == await retire_pending(loop, actor, source.job_id)
        assert not disposed['acknowledgement_created'] and not disposed['execution_authority']
        assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending is None
        assert (await c.store.get_state(c.state.root)).pending is None
        await loop.send('Read and handle the original known outcome without repeating work.', c.state.root, tools=[])
        effect = await runtime.manager.get_job(actor, c.effect['job_id'])
        assert effect.result_status == 'handled' and effect.supervision_status == 'closed'
        assert (await runtime.manager.get_job(actor, source.job_id)).supervision_status == 'closed'
        await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
        await loop.send('Handle the original interrupted controller result without repeating work.', c.state.root, tools=[])
        owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
        assert owner.supervision_status == 'closed' and owner.outcome == 'interrupted' and owner.execution_role == kind
        assert await runtime.manager.execution_usage(actor, owner.job_id, generation=1) == usage
        for row in original:
            assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
        assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (c.state.source_binding.actor.session_id,))
        assert c.state.prepared == c.state.executed == len(c.state.effects) == c.state.counts['BUILD'] == 1
        assert c.state.lookups == c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c, loop, runtime)
