"""Publish source model permission, target wait decision and source receipt separately."""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import control_requests as requests
from . import timer_waits as waits
from . import wait_notifications as notices
from .finalization_plan import PlanChanged
from .jobs import JobError, JobLease, _hash, _json, _operation, verify_stored

ACTION = 'decide'
TARGET_BARRIERS = frozenset({'wait_decision_already_recorded', 'wait_due_state_changed',
    'wait_state_changed', 'wait_deadline_exceeded', 'wait_review_out_of_bounds',
    'wait_review_limit_exceeded', 'cancel_requested', 'unresolved_prior_attempt',
    'stale_execution', 'stale_parent_execution', 'job_intake_closed',
    'owner_completion_deadline_exceeded', 'owner_wait_deadline_exceeded'})
CLIENT_FIELDS = ('event_id', 'caller', 'call_ref', 'tool_call_id', 'decision', 'reason', 'next_review_at_ms')


@dataclass(frozen=True)
class PriorTree:
    root_id: str
    generation: int
    count: int
    version_sum: int


def client(request):
    return {key:request[key] for key in CLIENT_FIELDS}


def _proof(due, saved, seen):
    return {'due_hash':due['request_hash'], 'observation':seen, 'original_owner':saved['binding']['owner'],
        'continuation_hash':_hash(saved['request']['request']['continuation']),
        'deadline_at_ms':saved['request']['policy']['deadline_at_ms'],
        'review_cycle':waits._object(due['payload_json']).get('review', {}).get('cycle', 0)}


def _response(manager, request):
    return {'status':'accepted', 'event_id':request['event_id'],
        'decision_event_id':notices.decision_id(manager, request['event_id']), 'decision':request['decision'],
        'effect_status':'pending', 'execution_authority':False}


async def _mutable(manager, reader, actor, request):
    job = await manager._authorize(reader, actor, request['job_id'])
    proof = request['proof']
    now = manager.now()
    if now >= proof['deadline_at_ms']:
        raise JobError('wait_deadline_exceeded')
    if (job['generation']!=0 or job['status']!='queued' or job['resource_state']!='not_started'
            or job['intake']!='open' or job['supervision_status']!='open' or job['outcome'] is not None):
        raise JobError('wait_state_changed')
    await manager._supervision_barriers(reader, job)
    await manager._control_owners(reader, job)
    owner = await manager._row(reader, proof['original_owner']['job_id'])
    manager._check_proof(owner, JobLease(**proof['original_owner']))
    if request['decision']=='defer':
        from .wait_reviews import MAX_REVIEWS
        if proof['review_cycle']>=MAX_REVIEWS:
            raise JobError('wait_review_limit_exceeded')
        if not now < request['next_review_at_ms'] < proof['deadline_at_ms']:
            raise JobError('wait_review_out_of_bounds')
    return job


async def _prepare(manager, reader, actor, request, *, notification=None):
    job, due, saved, receipt = notification or await notices.notification(manager, reader, actor, request['event_id'])
    if request['job_id']!=job['job_id'] or request['proof']!=_proof(due, saved, request['proof']['observation']):
        raise JobError('wait_decision_evidence_corrupt')
    await _mutable(manager, reader, actor, request)
    # The due and route readers validate all retained review/model originals in
    # this readonly snapshot. Publication only needs its tree's mutable guards.
    current, _, _ = await waits._inspect_due(manager, reader, actor, job['job_id'])
    if current['status']!='decision_pending' or current['event_id']!=request['event_id']:
        raise JobError('wait_due_state_changed')
    root = job['root_job_id']
    summary = await reader.fetchone(f"SELECT COUNT(*) AS job_count,COALESCE(SUM(version),0) AS version_sum,"
        f"MAX(CASE WHEN job_id=? THEN generation END) AS root_generation FROM {manager.jobs} WHERE root_job_id=?", (root, root))
    return job, PriorTree(root, summary['root_generation'], summary['job_count'], summary['version_sum'])


async def _retained_notice(manager, reader, actor, request):
    """Verify this frozen due without recursively walking its prior reviews."""
    due = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (request['event_id'],))
    if due is None or due['event_type']!='wait_due' or due['job_id']!=request['job_id']:
        raise JobError('decide_request_corrupt')
    payload = waits._object(due['payload_json'])
    job, _, saved = await waits._read(manager, reader, actor, request['job_id'])
    if due['request_hash']!=_hash(payload) or payload['wait']!=saved:
        raise JobError('decide_request_corrupt')
    return await notices.notification(manager, reader, actor, request['event_id'], _due=(job,due,saved))


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None, *, _notification=None):
    from .wait_decisions import _intent, _observation
    expected = _intent(request['event_id'], JobLease(**request['caller']), waits._ref(request['call_ref']),
        request['tool_call_id'], request['decision'], request['reason'], request['next_review_at_ms'])
    if set(request)!=set(CLIENT_FIELDS)|{'job_id', 'proof'} or client(request)!=expected:
        raise JobError('decide_request_corrupt')
    accepted = await requests.stored(manager, reader, actor, event, request, action=ACTION, completion=completion)
    job, due, saved, receipt = _notification or await _retained_notice(manager, reader, actor, request)
    observed = await _observation(manager, reader, actor, JobLease(**request['caller']),
        waits._ref(request['call_ref']), due, receipt)
    if (request['job_id']!=job['job_id'] or request['proof']!=_proof(due, saved, observed)
            or accepted.data['source']['execution_role']!='main' or accepted.source['parent_job_id'] is not None
            or accepted.data['target']!={'job_id':job['job_id'], 'root_job_id':job['root_job_id'], 'generation':0}):
        raise JobError('decide_request_corrupt')
    return accepted


def _receipt(manager, event, request, request_ref):
    expected = {'version':1, 'request':client(request), 'response':_response(manager, request), 'request_ref':request_ref}
    if (event['event_type']!='operation' or event['action']!='wait_decision' or event['job_id']!=request['job_id']
            or event['caller_session_id']!=request['call_ref']['session_id']
            or event['tool_call_id']!=request['tool_call_id'] or json.loads(event['call_ref_json'])!=request['call_ref']
            or event['request_hash']!=_hash(client(request)) or json.loads(event['payload_json'])!=expected):
        raise JobError('wait_decision_evidence_corrupt')
    return _operation(event)


async def _publish(manager, actor, accepted, prepared):
    request = accepted.data['request']
    ref = waits._ref(request['call_ref'])
    key = manager._origin_key(actor, 'wait_decision', ref, request['tool_call_id'], None)
    canonical_id = notices.decision_id(manager, request['event_id'])
    async with manager.tree_write(accepted.data['target']['job_id']) as tx:
        old = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?",
            (key, requests.rejection_id(accepted.event['event_id'])))
        if old:
            return old[0]
        row = await manager._authorize(tx, actor, request['job_id'])
        try:
            if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (canonical_id,)):
                raise JobError('wait_decision_already_recorded')
            job = await _mutable(manager, tx, actor, request)
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return await requests.reject(manager, tx, actor, accepted.event, accepted.data, row, exc.code)
        if prepared is None:
            raise PlanChanged()
        version, now = job['version']+1, manager.now()
        request_ref = requests.reference(accepted.event)
        operation = await manager._insert_operation(tx, job_id=job['job_id'], version=version,
            actor=actor, action='wait_decision', key=key, fingerprint=_hash(client(request)), call_ref=ref,
            tool_call_id=request['tool_call_id'], payload={'version':1, 'request':client(request),
                'response':_response(manager, request), 'request_ref':request_ref}, tree=prepared[1])
        if operation is None:
            raise PlanChanged()
        value = {'version':1, 'event_id':request['event_id'], 'due_hash':request['proof']['due_hash'],
            'operation_id':operation.operation_id, 'request':client(request),
            'observation':request['proof']['observation'], 'original_owner':request['proof']['original_owner'],
            'continuation_hash':request['proof']['continuation_hash'], 'request_ref':request_ref}
        await tx.append_event(('event_id','job_id','event_type','job_version','caller_session_id','request_hash','payload_json','delivery_status','created_at'),
            (canonical_id,job['job_id'],'wait_decision',version,actor.session_id,_hash(value),_json(value),'pending_delivery',now))
        if await tx.write_job(job['job_id'], f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                'WHERE job_id=? AND root_job_id=? AND generation=? AND version=?',
                (now,job['job_id'],job['root_job_id'],job['generation'],job['version']))!=1:
            raise PlanChanged()
        await tx.update_event(job['job_id'], f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id IN (?,?)",
            (request['event_id'], notices.notification_id(manager, request['event_id'])))
        return await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))


async def _finish(manager, actor, accepted, outcome, prepared=None):
    requests.remember_event(manager, accepted.event)
    while outcome is None:
        if prepared is None:
            async with manager._read() as reader:
                try:
                    prepared = await _prepare(manager, reader, actor, accepted.data['request'])
                except JobError as exc:
                    if exc.code not in TARGET_BARRIERS:
                        raise
        try:
            outcome = await _publish(manager, actor, accepted, prepared)
        except PlanChanged:
            prepared = None
    operation = requests.read_outcome(manager, accepted, outcome)
    await requests.complete(manager, actor, accepted, outcome)
    if operation is None:
        raise JobError(json.loads(outcome['payload_json'])['error'])
    return operation


async def record(manager, actor, original):
    from .wait_decisions import _observation
    caller, ref = JobLease(**original['caller']), waits._ref(original['call_ref'])
    key = manager._origin_key(actor, 'wait_decision', ref, original['tool_call_id'], None)
    prepared = None
    async with manager._read() as reader:
        old = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (requests.identity(ACTION,key),))
        if old is not None:
            request = json.loads(old['payload_json'])['request']
            if client(request)!=original:
                raise JobError('operation_conflict')
        else:
            notification = await notices.notification(manager, reader, actor, original['event_id'])
            job, due, saved, receipt = notification
            if await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?",
                    (notices.decision_id(manager, original['event_id']),)):
                raise JobError('wait_decision_already_recorded')
            seen = await _observation(manager, reader, actor, caller, ref, due, receipt)
            request = {**original, 'job_id':job['job_id'], 'proof':_proof(due,saved,seen)}
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
        if accepted is None:
            prepared = await _prepare(manager, reader, actor, request, notification=notification)
    repeated = accepted is not None
    if accepted is None:
        async def authorize(tx):
            from .task_control import _source
            source = await _source(manager, tx, actor, caller, ref, original['tool_call_id'])
            if source['execution_role']!='main' or source['parent_job_id'] is not None:
                raise JobError('role_forbids_wait_decision')
            await manager._fence(tx, actor, caller)
            return source
        job = prepared[0]
        accepted, repeated = await requests.accept(manager, actor, request,
            {'job_id':job['job_id'],'root_job_id':job['root_job_id'],'generation':0}, action=ACTION, authorize=authorize)
    return await _finish(manager, actor, accepted, outcome, prepared), repeated


@verify_stored
async def reconcile_request(manager, actor, request_id):
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        original = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_id,))
        if original is None:
            raise JobError('decide_request_original_missing')
        request = json.loads(original['payload_json'])['request']
        accepted, outcome = await requests.discover(manager, reader, actor, request, action=ACTION)
    return await _finish(manager, actor, accepted, outcome)
