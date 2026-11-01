"""The original leaf adapter cannot silently adopt another workflow execution."""
from __future__ import annotations

import asyncio

import pytest

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ['registry_store', 'managed_workflow']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_another_workflow_execution_cannot_use_the_original_leaf_resume_adapter(managed_workflow, monkeypatch):
    from power_loop.workflow import runner
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.resume import resume_detached

    from .test_node_nudge_runtime_192 import client
    from .test_workflow_restore_plan_192 import finished
    from .test_workflow_task_runtime_192 import leaf

    c = await client(managed_workflow, monkeypatch)
    c.state['spec']['root'] = {'type':'sequence', 'steps':[
        {'type':'parallel', 'branches':[leaf('BUILD'), leaf('AUDIT')]}, leaf('REVIEW', inputs_from=['BUILD'])]}
    first = await finished(managed_workflow)
    actor = c.runtime.actor(c.parent)
    before = await c.manager.get_job(actor, first.job_id)
    sid = before.result['results']['BUILD']['session_id']
    entered, release = asyncio.Event(), asyncio.Event()
    execute = WorkflowEngine._exec

    async def hold_sibling(engine, node, *args, **kwargs):
        if (getattr(node, 'id', None)=='AUDIT' and engine._task_execution is not None
                and engine._task_execution.lease.generation==2):
            entered.set()
            await release.wait()
        return await execute(engine, node, *args, **kwargs)

    async def restore():
        handle = await resume_detached(c.loop, c.parent, first.run_id, continuation=('BUILD', 'recompute BUILD', 3))
        c.state['handles'].append(handle)
        return {'operation_id':handle.operation_id}

    monkeypatch.setattr(WorkflowEngine, '_exec', hold_sibling)
    try:
        await c.issue(handler=restore)
        assert len(c.state['handles'])==2
        handle = c.state['handles'][-1]
        await asyncio.wait_for(entered.wait(), 4)
        engine = runner._ENGINES[first.run_id]
        for _ in range(300):
            prior = engine._results.get('BUILD')
            frames = [frame for frame in engine._frames.values() if frame['result'] is not None and frame['result'].node_id=='BUILD']
            if prior is not None and prior.status=='completed' and len(frames)==1 and not frames[0]['active']:
                break
            await asyncio.sleep(.01)
        else:
            pytest.fail('restored BUILD did not settle while its real parallel sibling was paused')
        from .test_node_nudge_runtime_192 import target
        [(binding, _)] = [(row, data) for row, data in await target(c) if data['session_id']==sid and data.get('generation')==2]
        node = await c.manager._row(c.manager.db, binding['job_id'])
        assert node['generation']==2 and (await c.manager.get_job(actor, first.job_id)).generation==2
        previous = await c.manager.get_job(actor, node['job_id'])
        originals = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND action='nudge' ORDER BY event_id", (node['job_id'],))
        admit, errors = c.manager.resume_workflow_node, []
        async def observed(*args, **kwargs):
            try:
                return await admit(*args, **kwargs)
            except Exception as exc:
                errors.append(str(exc))
                raise
        monkeypatch.setattr(c.manager, 'resume_workflow_node', observed)
        c.run_id, c.sid = first.run_id, sid
        async def steer():
            return await runner.steer_node(first.run_id, 'BUILD', sid, 'NUDGED-43')
        await c.issue(handler=steer)
        assert errors==['resume_target_mismatch']
        assert not [value for value in c.replies if value.get('mode')=='continue']
        assert await c.manager.get_job(actor, node['job_id'])==previous
        assert c.model.counts['BUILD']==2
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND action='nudge' ORDER BY event_id", (node['job_id'],))==originals
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('execution','dispatch') AND job_id=?", (node['job_id'],))
        from .test_node_admission_trees import records
        assert not (await records(c))[0]
        release.set()
        await asyncio.wait_for(handle.task, 10)
        result = await c.manager.get_job(actor, node['job_id'])
        assert result.generation==2 and result.outcome=='succeeded' and result.result==previous.result
        assert (await c.manager.get_job(actor, first.job_id)).outcome=='succeeded'
        assert c.model.counts['BUILD']==2 and c.model.counts['AUDIT']==1
        assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node'"))==3
    finally:
        release.set()
