"""Current controller keys fit every store and preserve immutable originals."""
import pytest

from power_loop.runtime import controller_handoff, delegation
from power_loop.runtime.abort_outcomes import handled_references

from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_current_controller_events_and_source_messages_are_immutable_after_review(registry_store):
    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        await c.runtime.flush_control_checkpoints()
        actor = c.runtime.actor(c.state.root)
        await controller_handoff.transfer_review(c.loop, actor, c.state.source)
        saved = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('controller_retired','controller_review_handoff') ORDER BY event_id")
        original_messages = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} WHERE session_id=? ORDER BY seq', (c.state.source_binding.actor.session_id,))
        assert len(saved) == 2 and all(len(row['event_id']) <= 64 for row in saved)
        await controller_handoff.transfer_review(c.loop, actor, c.state.source)
        await c.loop.send('Read and handle the original result without repeating work.', c.state.root, tools=[])
        row = await c.manager.get_job(actor, c.state.source)
        assert row.supervision_status == 'closed' and row.result_status == 'handled'
        async with c.manager._read() as tx:
            source = await c.manager._authorize(tx, actor, c.state.source)
            assert await handled_references(c.manager, tx, source)
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('controller_retired','controller_review_handoff') ORDER BY event_id") == saved
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} WHERE session_id=? ORDER BY seq', (c.state.source_binding.actor.session_id,)) == original_messages
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)
