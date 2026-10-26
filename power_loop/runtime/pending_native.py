"""Freeze original dispatch evidence at actual model exit, before later review.

An absent start row alone is not an outcome. The sealed original plan, captured
exit snapshot, actual stopped model and closed dispatch gate must all agree.
"""
from __future__ import annotations

from .jobs import JobActor, JobError, _hash
from .model_exit import content_reference

FIELD = 'pending_native_dispatch_snapshot'


def identity(manager, source):
    return 'pc_native_snap_' + _hash([manager.store.store_id, source['job_id'], source['generation']])[:48]


async def _expected(manager, tx, source, snapshot):
    from .managed_effects import _ref
    from .tool_call_plans import _read
    from .tool_call_plans import identity as plan_id
    from .tool_call_plans import read as read_plan
    from .workflow_controller import proof

    if snapshot.get('version') != 2:
        return None
    ref = _ref(snapshot['assistant_ref'])
    plan = await read_plan(manager, tx, source, ref)
    pending = set(snapshot['pending']['tool_call_ids'])
    if not any(entry['call_id'] in pending and entry['recovery'] is None for entry in plan['entries']):
        return None
    dispatches = []
    for entry in plan['entries']:
        key = plan_id(manager, source, ref) + '_start_' + _hash(entry['call_id'])[:24]
        started = await _read(manager, tx, key, source, 'tool_call_dispatch_started', optional=True)
        expected = {'version': 1, 'scope': plan['scope'], 'plan_id': plan_id(manager, source, ref),
                    'plan_hash': _hash(plan), 'entry_hash': _hash(entry)}
        if started is not None and _hash(started) != _hash(expected):
            raise JobError('pending_native_dispatch_changed')
        event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,)) if started is not None else None
        dispatches.append({'call_id': entry['call_id'], 'started': content_reference(event) if event else None})
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (plan_id(manager, source, ref),))
    return {'version': 1, 'scope': proof(source), 'assistant_ref': ref.to_dict(), 'pending_hash': _hash(snapshot),
            'plan': content_reference(event), 'dispatches': dispatches, 'execution_authority': False}


async def capture(manager, tx, source, snapshot):
    """Called only inside actual model closure, in the same control transaction."""
    from .tool_call_plans import _event, _read
    data = await _expected(manager, tx, source, snapshot)
    if data is None:
        return None
    old = await _read(manager, tx, identity(manager, source), source, 'pending_native_snapshot', optional=True)
    if old is None:
        await _event(manager, tx, identity(manager, source), source, 'pending_native_snapshot', data)
    elif _hash(old) != _hash(data):
        raise JobError('pending_native_snapshot_changed')
    return content_reference(await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (identity(manager, source),)))


async def validate(manager, tx, source, closed, *, required=False):
    from .pending_calls import SNAPSHOT
    from .tool_call_plans import _read
    anchor = closed.get('evidence', {}).get(FIELD)
    if anchor is None:
        if required:
            raise JobError('pending_native_snapshot_required')
        return None
    snapshot = closed['evidence'].get(SNAPSHOT)
    if not isinstance(snapshot, dict):
        raise JobError('pending_native_snapshot_changed')
    data = await _read(manager, tx, identity(manager, source), source, 'pending_native_snapshot')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (identity(manager, source),))
    if _hash(anchor) != _hash(content_reference(event)) or _hash(data) != _hash(await _expected(manager, tx, source, snapshot)):
        raise JobError('pending_native_snapshot_changed')
    return event, data


async def undispatched(manager, tx, source):
    """Only the native subset; other original calls may already have executed."""
    from .managed_effects import _ref
    from .pending_calls import SNAPSHOT, model
    from .tool_call_plans import read as read_plan
    closed = await model(manager, tx, source)
    saved = await validate(manager, tx, source, closed, required=True)
    snapshot = closed['evidence'][SNAPSHOT]
    ref = _ref(snapshot['assistant_ref'])
    plan = await read_plan(manager, tx, source, ref)
    pending = set(snapshot['pending']['tool_call_ids'])
    entries = [entry for entry in plan['entries'] if entry['call_id'] in pending and entry['recovery'] is None]
    dispatches = {entry['call_id']: entry['started'] for entry in saved[1]['dispatches']}
    actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
    for entry in entries:
        if dispatches[entry['call_id']] is not None:
            raise JobError('pending_native_already_dispatched')
        import json
        operations = await tx.fetchall(f"SELECT call_ref_json FROM {manager.events} WHERE event_type='operation' "
            'AND caller_session_id=? AND tool_call_id=? LIMIT 1001', (actor.session_id, entry['call_id']))
        if len(operations) > 1000 or any(_hash(json.loads(event['call_ref_json'] or 'null')) == _hash(ref.to_dict()) for event in operations):
            raise JobError('pending_native_operation_conflict')
    if not entries:
        raise JobError('pending_native_calls_required')
    return closed, snapshot, ref, entries, saved


async def requested(manager, tx, source):
    """A negative source result describes only the proven native subset."""
    from .abort_outcomes import FIELD as RESULT_FIELD
    from .original_owner_review import read as original_owner
    from .pending_owner import read as transferred_owner
    from .workflow_controller import proof
    closed, snapshot, ref, entries, saved = await undispatched(manager, tx, source)
    binding = await original_owner(manager, tx, source)
    if binding is not None:
        await transferred_owner(manager, tx, source)
    outcome = closed['evidence'].get('exit')
    if outcome not in ('cancelled', 'interrupted', 'timed_out', 'budget_exceeded'):
        outcome = 'failed'
    exited = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', ('model_closed_'+source['job_id'],))
    return outcome, {'status': 'not_dispatched', 'final_text': 'Only the listed original calls were not dispatched.',
        RESULT_FIELD: {'version': 2, 'outcome_scope': 'original_undispatched_calls', 'scope': proof(source),
        'original_call': ref.to_dict(), 'calls': [{'tool_call_id': entry['call_id'], 'tool_name': entry['tool_name'],
            'arguments_hash': entry['arguments_hash'], 'resolution': 'not_dispatched'} for entry in entries],
        'checkpoint': content_reference(saved[0]), 'model_exit': content_reference(exited),
        'new_effect_requests': 0, 'source_model_usage': 'retained_in_original_model_attempt_ledger',
        'execution_authority': False, 'acknowledgement_created': False, 'billing_settled': False}}


async def prepare(manager, tx, actor, source):
    """Inside pending retirement: atomic native result/notice, never a tool row."""
    from .abort_outcomes import _notice_value, _Publication, notice, notice_id, render, result_event
    from .jobs import JobLease
    from .managed_delivery import _main
    from .tool_call_plans import _event
    from .workflow_controller import proof
    main = await _main(manager, tx, source)
    if main['session_id'] != actor.session_id:
        raise JobError('pending_call_requires_original_main')
    outcome, value = await requested(manager, tx, source)
    old = await result_event(manager, tx, source, required=False)
    if old is None:
        await manager._record_result_tx(tx, actor, JobLease(**proof(source)), outcome=outcome, result=value,
            resource_state='stopped', evidence_only=True, publication_proof=_Publication())
        source = await manager._row(tx, source['job_id'])
        old = await result_event(manager, tx, source)
    delivery = await notice(manager, tx, source, old, required=False, original=True)
    if delivery is None:
        data, key = await _notice_value(manager, tx, source, old), notice_id(manager, source, native=True)
        await _event(manager, tx, key, source, 'source_abort_delivery', data)
        await tx.execute(f'UPDATE {manager.events} SET caller_session_id=? WHERE event_id=?', (actor.session_id, key))
        columns = ('session_id', 'item_id', 'kind', 'mode', 'status', 'content', 'created_at')
        await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, columns),
            (actor.session_id, key, 'original_call_outcome', 'queue', 'pending', render(data['public']), manager.now()))
    return await manager._row(tx, source['job_id'])
