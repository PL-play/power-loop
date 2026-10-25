"""Retain original pending calls when an actual model executor exits.

A known result is not physical exit, queue acceptance or processing ACK. The
host records actual model exit through close_model_execution first. This module
then retires only the captured, resolved managed calls; no tool reply is made.
"""
from __future__ import annotations

import json

from .jobs import JobActor, JobError, _hash, _json

SNAPSHOT = 'pending_call_snapshot'
ERROR = 'pending_call_snapshot_error'


def identity(manager, source):
    return 'pc_retire_' + _hash([manager.store.store_id, source['job_id'], source['generation']])[:48]


async def _object(manager, tx, table, column, where, parameters, limit):
    length = f'length(CAST({column} AS BLOB))' if manager.db.dialect.name == 'sqlite' else f'octet_length({column})'
    size = await tx.fetchone(f'SELECT {length} AS n FROM {table} WHERE {where}', parameters)
    if size is None or size['n'] is None:
        return None
    if size['n'] > limit:
        raise JobError('pending_call_evidence_too_large')
    row = await tx.fetchone(f'SELECT {column} AS value FROM {table} WHERE {where}', parameters)
    try:
        value = json.loads(row['value'])
        if type(value) is not dict:
            raise ValueError()
        _json(value)
        return value
    except (ValueError, TypeError, RecursionError):
        raise JobError('pending_call_evidence_corrupt') from None


async def current(manager, tx, sid):
    return await _object(manager, tx, manager.store.t.session_state, 'pending_json', 'session_id=?', (sid,), 32768)


async def _verify(manager, tx, source, snapshot):
    from .managed_completion import _scope
    from .managed_effects import _ref, _verify_call
    from .workflow_controller import proof

    await _scope(manager, tx, source)
    try:
        pending = snapshot['pending']
        calls = pending['tool_calls']
        ref = _ref(snapshot['assistant_ref'])
        ids = [call['id'] for call in calls]
        if (set(snapshot) != {'version', 'scope', 'assistant_ref', 'pending'}
                or type(snapshot['version']) is not int or snapshot['version'] != 1
                or snapshot['scope'] != proof(source) or source['kind'] != 'send'
                or type(calls) is not list or not 0 < len(calls) <= 64
                or any(type(i) is not str or not i for i in ids) or len(set(ids)) != len(ids)
                or type(pending.get('tool_call_ids')) is not list or sorted(pending['tool_call_ids']) != sorted(ids)
                or type(pending.get('assistant_seq')) is not int or pending['assistant_seq'] != ref.seq
                or type(pending.get('round_index')) is not type(ref.round_index)
                or pending.get('round_index') != ref.round_index or ref.session_id != source['session_id']):
            raise ValueError()
        actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
        for call in calls:
            await _verify_call(manager, tx, actor, source, ref, call['id'], call['function']['name'],
                               _hash(json.loads(call['function']['arguments'])))
        # A rewritten pending subset must not discard another unreturned call
        # in the same original assistant message.
        raw = await tx.fetchone(f'SELECT tool_calls_json FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (ref.session_id, ref.seq))
        original = json.loads(raw['tool_calls_json'])
        original_ids = [call['id'] for call in original]
        if len(original_ids) > 64 or len(set(original_ids)) != len(original_ids):
            raise ValueError()
        replies = await tx.fetchall(f"SELECT tool_call_id FROM {manager.store.t.messages} WHERE session_id=? AND send_index=? AND role='tool' AND seq>? ORDER BY seq LIMIT 1001", (ref.session_id, ref.send_index, ref.seq))
        if len(replies) > 1000 or not set(original_ids) - {r['tool_call_id'] for r in replies} <= set(ids):
            raise ValueError()
    except (ValueError, TypeError, KeyError, RecursionError):
        raise JobError('pending_call_snapshot_mismatch') from None
    return pending, calls, ref


async def capture(manager, tx, source):
    pending = await current(manager, tx, source['session_id'])
    if pending is None:
        return None
    from .message_ref import MessageRef
    from .workflow_controller import proof

    seq = pending.get('assistant_seq')
    if type(seq) is not int or seq <= 0:
        raise JobError('pending_call_snapshot_mismatch')
    row = await tx.fetchone(f'SELECT id,session_id,seq,role,tool_call_id,send_index,round_index FROM {manager.store.t.messages} WHERE session_id=? AND seq=?',
                            (source['session_id'], seq))
    if row is None:
        raise JobError('pending_call_snapshot_mismatch')
    ref = MessageRef(manager.store.store_id, row['session_id'], row['id'], row['seq'], row['role'],
                     row['tool_call_id'], row['send_index'], row['round_index'])
    snapshot = {'version': 1, 'scope': proof(source), 'assistant_ref': ref.to_dict(), 'pending': pending}
    await _verify(manager, tx, source, snapshot)
    return snapshot


async def _closed(manager, tx, source):
    if source is None:
        raise JobError('pending_call_model_exit_required')
    value = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?',
                          ('model_closed_' + source['job_id'],), 4 * 1024 * 1024)
    event = await tx.fetchone(f'SELECT job_id,event_type,delivery_status FROM {manager.events} WHERE event_id=?', ('model_closed_' + source['job_id'],))
    if (value is None or event['job_id'] != source['job_id'] or event['event_type'] != 'model_closed'
            or event['delivery_status'] != 'not_applicable'
            or set(value) != {'generation', 'resource_state', 'evidence'}
            or type(value['generation']) is not int or type(value['evidence']) is not dict):
        raise JobError('pending_call_model_exit_required')
    return value


async def model(manager, tx, source):
    value = await _closed(manager, tx, source)
    if (ERROR in value.get('evidence', {}) or source['status'] != 'model_closed'
            or source['resource_state'] != 'stopped' or source['intake'] != 'closed'):
        raise JobError('pending_call_model_exit_required')
    from .model_exit import require_stopped
    await require_stopped(manager, tx, source, value)
    return value


async def _expected(manager, tx, source, closed):
    from .finalization import _reference
    from .managed_delivery import _inbox, read, result_event
    from .managed_effects import descriptor, job_id, resolved_execution

    snapshot = closed.get('evidence', {}).get(SNAPSHOT)
    if snapshot is None:
        raise JobError('pending_call_snapshot_required')
    pending, calls, ref = await _verify(manager, tx, source, snapshot)
    if pending.get('pending_interactions'):
        raise JobError('pending_call_interaction_unresolved')
    actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
    refs = []
    for call in calls:
        row = await manager._authorize(tx, actor, job_id(manager, actor, ref, call['id']))
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError('pending_call_adapter_required')
        creation, _binding = found
        if (row['parent_job_id'] != source['job_id'] or json.loads(creation['call_ref_json']) != ref.to_dict()
                or creation['tool_call_id'] != call['id']):
            raise JobError('pending_call_source_mismatch')
        result = await result_event(manager, tx, row)
        original, _known, _payload, _found = await resolved_execution(manager, tx, row, result)
        notice = await read(manager, tx, row, result)
        await _inbox(manager, tx, row, notice)
        original = [{k: v for k, v in item.items() if k != 'delivery_status'} for item in original]
        refs.append({'job_id': row['job_id'], 'generation': row['generation'], 'tool_call_id': call['id'],
            'evidence_hash': _hash(original), 'notice_id': notice[0]['event_id'], 'notice_hash': _hash(notice[1])})
    return {'version': 1, 'snapshot': snapshot,
        'model_exit': await _reference(manager, tx, 'model_closed_' + source['job_id']),
        'calls': refs, 'disposition': 'retired_for_result_review', 'execution_authority': False,
        'acknowledgement_created': False, 'billing_settled': False}


async def require_disposition(manager, tx, source, closed=None):
    if closed is None:
        closed = await _closed(manager, tx, source)
    if ERROR in closed.get('evidence', {}):
        raise JobError('pending_call_model_exit_required')
    if SNAPSHOT not in closed.get('evidence', {}):
        return None
    if _hash(closed) != _hash(await model(manager, tx, source)):
        raise JobError('pending_call_model_exit_required')
    key = identity(manager, source)
    value = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (key,), 262144)
    event = await tx.fetchone(f'SELECT job_id,event_type,delivery_status,request_hash FROM {manager.events} WHERE event_id=?', (key,))
    if (value is None or event['job_id'] != source['job_id'] or event['event_type'] != 'pending_call_disposition'
            or event['delivery_status'] != 'not_applicable' or event['request_hash'] != _hash(value)
            or _hash(value) != _hash(await _expected(manager, tx, source, closed))
            or await current(manager, tx, source['session_id']) == value['snapshot']['pending']):
        raise JobError('pending_call_disposition_required')
    from .finalization import _reference
    return await _reference(manager, tx, key)


async def retire_pending(loop, actor, source_job_id):
    """Host-only, atomic retirement after actual exit; never executes/replies/ACKs."""
    from .managed_completion import _host_only
    _host_only()
    runtime = getattr(loop, 'task_runtime', None)
    if runtime is None or runtime.actor(actor.session_id) != actor or await loop.ensure_store() is not runtime.manager.store:
        raise JobError('task_runtime_required')
    manager = runtime.manager
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, source_job_id)
        closed = await model(manager, tx, source)
        expected = await _expected(manager, tx, source, closed)
        if source['root_job_id'] != source_job_id or source['session_id'] != actor.session_id:
            # Child pending recovery must go through its explicit owner adapter;
            # this original-main port cannot silently continue a workflow node.
            raise JobError('pending_call_requires_original_main')
        key = identity(manager, source)
        old = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (key,), 262144)
        if old is None:
            if await current(manager, tx, source['session_id']) != expected['snapshot']['pending']:
                raise JobError('pending_call_state_changed')
            encoded = _json(expected)
            if len(encoded.encode('utf-8')) > 262144:
                raise JobError('pending_call_evidence_too_large')
            await tx.execute(f'INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)',
                (key, source['job_id'], 'pending_call_disposition', source['version'], _hash(expected), encoded, 'not_applicable', manager.now()))
            await tx.execute(f'UPDATE {manager.store.t.session_state} SET pending_json=NULL WHERE session_id=?', (source['session_id'],))
        reference = await require_disposition(manager, tx, source, closed)
        return {'source_job_id': source_job_id, 'disposition': reference, 'execution_authority': False,
            'acknowledgement_created': False, 'billing_settled': False}
