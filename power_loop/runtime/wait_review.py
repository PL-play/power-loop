"""Finite child waits reuse persisted interactions, timers and original job controls."""
from __future__ import annotations

import asyncio
import contextvars
import json
from dataclasses import dataclass

from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import (
    InputRef,
    JobRef,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    canonical,
    plain,
)
from power_loop.core.execution_context import get_execution_context

from .human_input import HumanInputRequired
from .jobs import JobManager
from .timer_backend import TimerContext


@dataclass(frozen=True)
class WaitPolicy:
    """Trusted factory policy, never taken from model tool arguments."""
    deadline_at_ms: int
    max_wait_ms: int = 1800000

    def __post_init__(self):
        if type(self.deadline_at_ms) is not int or self.deadline_at_ms <= 0:
            raise ValueError('deadline_at_ms must be positive')
        if type(self.max_wait_ms) is not int or not 0 < self.max_wait_ms <= 30*86400000:
            raise ValueError('max_wait_ms must be within 30 days')


async def request(loop, *, due_at_ms, note, continuation, policy):
    if policy is None:
        raise RuntimeRequestError('capability_unavailable', 'trusted wait policy is required')
    now = now_ms()
    if type(due_at_ms) is not int or not now <= due_at_ms < policy.deadline_at_ms:
        raise RuntimeRequestError('invalid_options', 'review must precede the original deadline')
    if due_at_ms-now > policy.max_wait_ms:
        raise RuntimeRequestError('invalid_options', 'review exceeds wait policy')
    if not isinstance(note, str) or not note.strip() or len(note)>4000:
        raise RuntimeRequestError('invalid_options', 'note must contain 1..4000 characters')
    if not isinstance(continuation, str) or not continuation.strip() or len(continuation)>64000:
        raise RuntimeRequestError('invalid_options', 'continuation must contain 1..64000 characters')
    ctx = get_execution_context()
    worker = loop._runtime_worker
    original_deadline = loop._runtime_worker_context.expires_at_ms
    if original_deadline is not None and policy.deadline_at_ms > original_deadline:
        raise RuntimeRequestError("invalid_options", "wait cannot extend the original task deadline")
    if worker.runtime.timer_backend is None:
        raise RuntimeRequestError('capability_unavailable', 'timer backend is required for review')
    if len(worker._wait_deadlines) >= 1024:
        raise RuntimeRequestError('capacity_unavailable', 'too many waiting reviews')
    manager = JobManager(loop.store)
    owner = await manager._row(manager.db, ctx.job_ref.job_id)
    if owner['kind'] not in {'subagent','workflow_node'}:
        raise RuntimeRequestError('forbidden', 'only finite child tasks request review')
    root = await manager._row(manager.db, owner['root_job_id'])
    review = dict(main_session_id=root['created_by_session_id'], due_at_ms=due_at_ms,
        note=note, continuation=continuation, deadline_at_ms=policy.deadline_at_ms,
        max_wait_ms=policy.max_wait_ms, revision=0, decision=None, timer_id=None)
    raise HumanInputRequired(kind='task_review', prompt=note,
        metadata={'expires_at_ms':policy.deadline_at_ms, 'wait_review':review})


def _interaction(pending):
    return next((item for item in pending.get('pending_interactions', [])
                 if 'wait_review' in item.get('metadata', {})), None)


def event_id(interaction):
    return interaction['interaction_id']+':'+str(interaction['metadata']['wait_review']['revision'])


def public(ref, pending, interaction):
    review=interaction['metadata']['wait_review']
    return dict(job_id=ref.job_id,generation=ref.generation,event_id=event_id(interaction),
        main_session_id=review['main_session_id'],due_at_ms=review['due_at_ms'],
        deadline_at_ms=review['deadline_at_ms'],now_ms=now_ms(),note=review['note'],
        continuation=review['continuation'],decision=review['decision'],
        schedule_error=review.get('schedule_error'),wait_round=pending['round_index'])


async def arm(worker, sid, *, recover=False):
    """After the waiting state commits, schedule only its original main review."""
    port=worker.inputs
    state=await port.db.fetchone(f'SELECT pending_json FROM {port.state} WHERE session_id=?',(sid,))
    pending=json.loads(state['pending_json'] or '{}')
    interaction=_interaction(pending)
    if interaction is None or not pending.get('runtime_wait'):
        return
    ref=JobRef(**pending['runtime_wait']['job_ref'])
    review=interaction['metadata']['wait_review']
    original=await port.input(port.db, InputRef(**pending['runtime_wait']['input_ref']))
    host_context=original.meta['descriptor']['host_context']
    metadata=await port.db.fetchone(f'SELECT metadata_json FROM {worker.runtime.store.t.sessions} WHERE session_id=?',(sid,))
    identity=json.loads(metadata['metadata_json'])
    actor=TrustedActor(identity['tenant_id'],identity['principal_id'],None)
    watch_deadline(worker,ref,actor,review['deadline_at_ms'])
    # A failed/unknown external create is never automatically replayed on reopen.
    if recover or review['timer_id'] is not None or review.get('schedule_error'):
        return
    source=TimerContext(ref.store_id,review['main_session_id'],actor,ref,
        interaction['tool_call_id'],'wait-review-'+event_id(interaction),host_context)
    payload=public(ref,pending,interaction)
    payload["continuation"]=review["continuation"][:1000]
    payload["continuation_length"]=len(review["continuation"])
    note=('Review the original waiting child with wait_handoff inspect/decide. '
          'Choose continue, defer or cancel; never start replacement work. Task data:\n'+canonical(payload))
    try:
        timer=await worker.runtime.timer_backend.create(source,delay_s=None,due_at_ms=review['due_at_ms'],note=note,interval_s=None)
        updates={'timer_id':timer.timer_id}
    except Exception as exc:
        updates={'schedule_error':getattr(exc,'code',type(exc).__name__)}
        await _update_schedule(worker,sid,event_id(interaction),updates)
        raise
    await _update_schedule(worker,sid,event_id(interaction),updates)


async def _update_schedule(worker,sid,expected,updates):
    port=worker.inputs
    async with port.db.transaction() as tx:
        state=await port.db.dialect.lock_state(tx,port.state,sid)
        pending=json.loads(state['pending_json'] or '{}')
        interaction=_interaction(pending)
        if interaction is None or event_id(interaction)!=expected:
            return  # A normal answer/cancel already completed this wait.
        interaction['metadata']['wait_review'].update(updates)
        await tx.execute(f'UPDATE {port.state} SET pending_json=? WHERE session_id=?',(canonical(pending),sid))


def watch_deadline(worker,ref,actor,deadline):
    if ref in worker._wait_deadlines:
        return
    if len(worker._wait_deadlines)>=1024:
        raise RuntimeRequestError('capacity_unavailable','too many waiting reviews')
    async def expire():
        try:
            await asyncio.sleep(max(0,(deadline-now_ms())/1000))
            worker._wait_deadlines.pop(ref,None)
            await worker.runtime.client.cancel(ref,actor=actor,reason='wait review deadline')
        except RuntimeRequestError as exc:
            if exc.code not in {'stale_generation','not_found'}:
                worker.last_error='wait deadline: '+str(exc)[:500]
        except Exception as exc:
            worker.last_error='wait deadline: '+str(exc)[:500]
        finally:
            worker._wait_deadlines.pop(ref,None)
    worker._wait_deadlines[ref]=asyncio.create_task(expire(),context=contextvars.Context())


async def inspect(client,ref,actor,expected,*,field='summary',char_offset=0,max_chars=8000):
    job=await client.inputs.resolve_job(client.inputs.db,ref,actor)
    row=await client.inputs.db.fetchone(f'SELECT pending_json FROM {client.inputs.state} WHERE session_id=?',(job['session_id'],))
    pending=json.loads(row['pending_json'] or '{}')
    item=_interaction(pending)
    require_review(pending,item,ref,actor,expected)
    value=public(ref,pending,item)
    if field!='summary':
        if field not in {'note','continuation'} or type(char_offset) is not int or char_offset<0 or type(max_chars) is not int or not 256<=max_chars<=16000:
            raise RuntimeRequestError('invalid_options')
        text=value[field]
        value=dict(field=field,text=text[char_offset:char_offset+max_chars],
            next_offset=char_offset+max_chars if char_offset+max_chars<len(text) else None)
    else:
        value["continuation"]=value["continuation"][:1000]
        value["continuation_length"]=len(item["metadata"]["wait_review"]["continuation"])
    return value


def require_review(pending,item,ref,actor,expected):
    if item is None or pending.get('runtime_wait',{}).get('job_ref')!=plain(ref):
        raise RuntimeRequestError('interaction_expired')
    review=item['metadata']['wait_review']
    if actor.session_id is not None and actor.session_id!=review['main_session_id']:
        raise RuntimeRequestError('forbidden','only the original main may decide')
    if event_id(item)!=expected or review['decision'] is not None:
        raise RuntimeRequestError('stale_wait_review')
    if now_ms()>=review['deadline_at_ms']:
        raise RuntimeRequestError('interaction_expired')


async def decide(worker,ref,actor,expected,*,decision,next_review_at_ms=None):
    client,port=worker.runtime.client,worker.inputs
    if decision not in {'continue','defer','cancel'}:
        raise RuntimeRequestError('invalid_options')
    job=await port.resolve_job(port.db,ref,actor)
    sid=job['session_id']
    # Freeze the explicit business decision under the same state lock used by answer/cancel.
    async with port.db.transaction() as tx:
        await port.authorize(tx,TargetRef(ref.store_id,sid,ref.job_id),actor)
        state=await port.db.dialect.lock_state(tx,port.state,sid)
        pending=json.loads(state['pending_json'] or '{}')
        item=_interaction(pending)
        require_review(pending,item,ref,actor,expected)
        review=item['metadata']['wait_review']
        previous=dict(review)
        if decision=='defer':
            if (type(next_review_at_ms) is not int or not now_ms()<next_review_at_ms<review['deadline_at_ms']
                    or next_review_at_ms-now_ms()>review['max_wait_ms']):
                raise RuntimeRequestError('invalid_options','defer must precede the original deadline')
            review.update(revision=review['revision']+1,due_at_ms=next_review_at_ms,timer_id=None)
            review.pop('schedule_error',None)
        else:
            review['decision']=decision
        await tx.execute(f'UPDATE {port.state} SET pending_json=? WHERE session_id=?',(canonical(pending),sid))
    if decision=='continue':
        from .interactions import answer
        receipt=await answer(client,ref,interaction_id=item['interaction_id'],wait_round=pending['round_index'],
            value=review['continuation'],message_id='wait-continue-'+expected,actor=actor,review_id=expected)
        response={'decision':decision,'input_ref':plain(receipt.input_ref),'status':'accepted'}
    elif decision=='cancel':
        receipt=await client.cancel(ref,actor=actor,reason='main cancelled wait review')
        response={'decision':decision,'status':receipt.state}
    else:
        response={'decision':decision,'event_id':event_id(item),'status':'accepted'}
    if previous['timer_id'] is not None and previous['due_at_ms']>now_ms():
        original=await port.input(port.db,InputRef(**pending['runtime_wait']['input_ref']))
        source=TimerContext(ref.store_id,previous['main_session_id'],actor,ref,item['tool_call_id'],
            'wait-cancel-'+expected,original.meta['descriptor']['host_context'])
        try:
            await worker.runtime.timer_backend.cancel(source,previous['timer_id'])
        except Exception as exc:
            # The decision already happened; an unknown cleanup is never replayed.
            response['timer_cleanup_error']=getattr(exc,'code',type(exc).__name__)
    if decision=='defer':
        await arm(worker,sid)
    return response
