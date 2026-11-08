"""Original native callback results do not acknowledge asynchronous work."""
import asyncio

import pytest

from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import finish, jobs, managed

__all__ = ['registry_store', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]














async def test_actual_spawn_acceptance_does_not_close_running_child_or_parent_supervision(managed, monkeypatch):
    store, manager, runtime, loop, model, sid, _ = managed
    append = store.append_message
    failed = False
    async def fail_receipt(session_id, **kwargs):
        nonlocal failed
        if not failed and session_id == sid and kwargs.get('role') == 'tool':
            failed = True
            raise OSError('original spawn receipt write failed')
        return await append(session_id, **kwargs)
    monkeypatch.setattr(store, 'append_message', fail_receipt)
    try:
        result = await loop.send('delegate')
        assert result.status == 'failed' and result.error == 'OSError'
        await asyncio.wait_for(model.entered.wait(), 5)
        records = await jobs(managed)
        root = next(j for j in records if j.kind == 'send' and j.session_id == sid)
        child = next(j for j in records if j.kind == 'subagent')
        actor = runtime.actor(sid)
        source = await manager.get_job(actor, root.job_id)
        assert (await manager.get_job(actor, child.job_id)).resource_state == 'running'
        assert source.supervision_status == 'open'
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, root.job_id, generation=1)
        assert model.child_calls == 1 and (await manager.get_job(actor, child.job_id)).result_status == 'none'
        assert not [r for r in await store.load_all_messages(sid) if r.role == 'tool']
    finally:
        monkeypatch.setattr(store, 'append_message', append)
        await finish(managed)
