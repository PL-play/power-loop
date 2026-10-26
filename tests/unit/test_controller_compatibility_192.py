"""Original long event keys remain readable; new keys never exceed SQL limits."""
import pytest

from power_loop.runtime import controller_handoff, controller_retirement, delegation
from power_loop.runtime.abort_outcomes import handled_references, result_event
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_original_legacy_ids_are_consumed_without_rewriting_their_events_or_raw_messages(registry_store, monkeypatch):
    retirement_id, handoff_id = controller_retirement.identity, controller_handoff.identity
    with monkeypatch.context() as patch:
        patch.setattr(controller_retirement, 'identity', lambda manager, lease, **kwargs: retirement_id(manager, lease, legacy=True))
        patch.setattr(controller_handoff, 'identity', lambda manager, source, **kwargs: handoff_id(manager, source, legacy=True))
        c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
        initial = set(delegation._TASKS)
        try:
            await c.loop.send('Launch original work once.', c.state.root)
            await wait_owned(c, 'subagent', initial)
            await c.runtime.flush_control_checkpoints()
            actor = c.runtime.actor(c.state.root)
            await controller_handoff.transfer_review(c.loop, actor, c.state.source)
        except BaseException:
            c.state.release.set()
            await c.runtime.shutdown(timeout_s=3)
            await c.loop.aclose(drain_timeout_s=.1)
            raise
    try:
        saved = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type IN ('controller_retired','controller_review_handoff') ORDER BY event_id")
        original_messages = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} WHERE session_id=? ORDER BY seq', (c.state.source_binding.actor.session_id,))
        assert saved and all(len(row['event_id']) > 64 for row in saved)
        await controller_handoff.transfer_review(c.loop, actor, c.state.source)
        await c.loop.send('Read and handle the legacy original result without repeating work.', c.state.root, tools=[])
        row = await c.manager.get_job(actor, c.state.source)
        assert row.supervision_status == 'closed' and row.result_status == 'handled'
        async with c.manager._write() as tx:
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


async def test_ambiguous_handoff_aliases_cannot_select_an_arbitrary_original_review(registry_store):
    import json

    from power_loop.runtime.tool_call_plans import _event
    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        actor = c.runtime.actor(c.state.root)
        await controller_handoff.transfer_review(c.loop, actor, c.state.source)
        async with c.manager._write() as tx:
            source = await c.manager._authorize(tx, actor, c.state.source)
            original = await tx.fetchone(f'SELECT * FROM {c.manager.events} WHERE event_id=?', (controller_handoff.identity(c.manager, source),))
            await _event(c.manager, tx, controller_handoff.identity(c.manager, source, legacy=True), source,
                         'controller_review_handoff', json.loads(original['payload_json']))
        async with c.manager._write() as tx:
            source = await c.manager._authorize(tx, actor, c.state.source)
            result = await result_event(c.manager, tx, source)
            with pytest.raises(JobError, match='controller_handoff_ambiguous'):
                await controller_handoff.read(c.manager, tx, source, result)
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)
