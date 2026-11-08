"""Resume keeps one logical child and immutable per-generation results."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import InputRef, JobRef, RuntimeRequestError
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobLease

from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_task_runtime_192 import finish, managed, receipt

__all__ = ['registry_store', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def ready(managed):
    store,manager,runtime,agent,model,parent,connect=managed
    await agent.send('delegate')
    created=await receipt(managed)
    await finish(managed)
    old=await manager.get_job(runtime.actor(parent),created['job_id'])
    assert old.generation==1 and old.result_status=='handled'
    first=await agent.runtime.client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+1000)
    model.release.clear()
    model.entered.clear()
    return SimpleNamespace(store=store,manager=manager,runtime=runtime,loop=agent,model=model,parent=parent,
        connect=connect,created=created,old=old,old_lease=JobLease(old.job_id,'fixture','boot',1),
        sid=created['session_id'],fixture=managed,first=first)


async def latest(c):
    return await c.manager.get_job(c.runtime.actor(c.parent),c.old.job_id)


async def nudge_receipt(c):
    rows=await c.store.load_all_messages(c.parent)
    return json.loads(next(r.content for r in reversed(rows) if r.role=='tool'))


async def test_original_job_and_session_resume_once_with_distinct_results_and_usage(ready,monkeypatch):
    c=ready
    _,sent,_=await issue_nudge(c.fixture,monkeypatch,repeat=True)
    assert sent.status=='completed'
    accepted=await nudge_receipt(c)
    assert accepted['target']['job_id']==c.old.job_id and accepted['input_ref']['session_id']==c.sid
    await asyncio.wait_for(c.model.entered.wait(),5)
    await finish(c.fixture)
    result=await c.loop.runtime.client.wait(InputRef(**accepted['input_ref']),actor=c.loop.actor,deadline_ms=now_ms()+1000)
    assert result.status=='completed' and result.final_text=='NUDGED-42'
    assert result.job_ref==replace(c.first.job_ref,generation=2)
    assert result.usage['total_tokens']==10 and c.model.child_calls==2
    assert await c.loop.runtime.client.wait(c.first.job_ref,actor=c.loop.actor,deadline_ms=now_ms()+1000)==c.first
    assert len(await c.store.list_children(c.parent))==1
    current=await latest(c)
    assert current.result_status=='handled' and current.supervision_status=='closed'


@pytest.mark.parametrize('block',['foreign','stale','tools','rounds','model'])
async def test_resume_invalid_authority_or_widened_contract_never_starts_model(ready,block):
    c=ready
    client,actor,ref=c.loop.runtime.client,c.loop.actor,c.first.job_ref
    options=None
    if block=='foreign':
        actor=replace(actor,principal_id='foreign')
    elif block=='stale':
        ref=replace(ref,generation=7)
    else:
        options={'tools':['spawn_agent']} if block=='tools' else {'max_rounds':9} if block=='rounds' else {'model':'unbound-model'}
    with pytest.raises(RuntimeRequestError):
        await client.resume(ref,body='forbidden',message_id='blocked',actor=actor,options=options)
    assert c.model.child_calls==1 and (await latest(c)).generation==1
    assert await client.wait(c.first.job_ref,actor=c.loop.actor,deadline_ms=now_ms()+1000)==c.first


async def test_three_generations_preserve_prior_results_and_charge_each_actual_call(ready):
    c=ready
    client,actor=c.loop.runtime.client,c.loop.actor
    c.model.release.set()
    previous=c.first
    for generation in (2,3):
        accepted=await client.resume(previous.job_ref,body='next answer',message_id='resume-'+str(generation),actor=actor)
        result=await client.wait(accepted.input_ref,actor=actor,deadline_ms=now_ms()+10000)
        await c.loop.drain()
        assert result.status=='completed' and result.job_ref.generation==generation
        assert result.usage['total_tokens']==10
        assert await client.wait(previous.job_ref,actor=actor,deadline_ms=now_ms()+1000)==replace(previous,input_ref=None)
        previous=result
    assert c.model.child_calls==3 and len(await c.store.list_children(c.parent))==1


async def test_duplicate_resume_input_is_same_request_and_changed_body_conflicts(ready):
    c=ready
    client,actor=c.loop.runtime.client,c.loop.actor
    first=await client.resume(c.first.job_ref,body='next',message_id='same',actor=actor)
    again=await client.resume(c.first.job_ref,body='next',message_id='same',actor=actor)
    assert first.input_ref==again.input_ref and not again.newly_accepted
    with pytest.raises(RuntimeRequestError,match='duplicate_input_conflict'):
        await client.resume(c.first.job_ref,body='changed',message_id='same',actor=actor)
    c.model.release.set()
    result=await client.wait(first.input_ref,actor=actor,deadline_ms=now_ms()+10000)
    await c.loop.drain()
    assert result.job_ref.generation==2 and c.model.child_calls==2


@pytest.mark.parametrize('queued',[False,True])
async def test_cancel_new_resume_does_not_rewrite_original_result(ready,monkeypatch,queued):
    c=ready
    client,actor=c.loop.runtime.client,c.loop.actor
    acquire=c.loop.runtime.coordinator.acquire
    async def hold(key,**kwargs):
        return None if key[1]==c.sid else await acquire(key,**kwargs)
    if queued:
        monkeypatch.setattr(c.loop.runtime.coordinator,'acquire',hold)
    accepted=await client.resume(c.first.job_ref,body='cancel this continuation',message_id='cancel-resume',actor=actor)
    if not queued:
        await asyncio.wait_for(c.model.entered.wait(),5)
    await client.cancel(accepted.input_ref,actor=actor)
    result=await client.wait(accepted.input_ref,actor=actor,deadline_ms=now_ms()+10000)
    assert result.status=='cancelled'
    assert await client.wait(c.first.job_ref,actor=actor,deadline_ms=now_ms()+1000)==c.first
    assert c.model.child_calls==(1 if queued else 2)
    if not queued:
        with pytest.raises(RuntimeRequestError,match='target_cancelled'):
            await client.resume(result.job_ref,body='must stay cancelled',message_id='no-restart',actor=actor)
