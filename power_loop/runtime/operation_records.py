"""Record acceptance metadata through source/target/source tree transactions.

This port never enqueues input,dispatches a model or executes an external action.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from . import control_requests as requests
from .jobs import KINDS, JobError, JobLease, _hash, _json, _operation, verify_stored
from .message_ref import MessageRef

TARGET_BARRIERS = frozenset({
    'stale_execution', 'cancel_requested', 'job_intake_closed', 'execution_lease_expired',
    'execution_not_running', 'unresolved_prior_attempt', 'stale_parent_execution',
    'owner_completion_deadline_exceeded', 'owner_wait_deadline_exceeded',
    'executor_not_accepting', 'executor_lease_expired', 'executor_registration_required',
    'executor_binding_missing', 'executor_binding_mismatch',
})


class _TargetChanged(Exception):
    pass


def _call_ref(value):
    """Current raw metadata calls may have no containing send index."""
    try:
        ref = MessageRef(value['store_id'], value['session_id'], int(value['id']), int(value['seq']),
            value['role'], value['tool_call_id'],
            int(value['send_index']) if value['send_index'] is not None else None, value['round_index'])
        if ref.to_dict() != value:
            raise ValueError()
        return ref
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError('record_call_ref_corrupt') from exc


def _intent(target, caller, action, ref, call_id, request, result):
    if action not in ('nudge', 'steer'):
        raise JobError('unsupported_operation')
    return json.loads(_json({'job_id': target.job_id, 'target': asdict(target), 'caller': asdict(caller),
        'operation_action': action, 'call_ref': ref.to_dict(), 'tool_call_id': call_id,
        'request': dict(request), 'result': dict(result)}))


def _fingerprint(request):
    return _hash([request['job_id'], request['request'], request['call_ref']])


def _outcome_hash(request, source_ref):
    return _hash([_fingerprint(request), source_ref])


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    saved = _saved_request(event, request)
    accepted = await requests.stored(manager, reader, actor, event, saved, action='record',
        completion=completion, source_kinds=KINDS, source_created_by=True)
    if accepted.data['target']['generation'] != saved['target']['generation']:
        raise JobError('record_request_corrupt')
    target = await manager._authorize(reader, actor, saved['job_id'])
    _role(accepted.data['source']['execution_role'], target['kind'])
    return accepted


def _saved_request(event, requested):
    """The first accepted result survives replay,as with the original API."""
    try:
        saved = json.loads(event['payload_json'])['request']
        if (saved['operation_action'] != requested['operation_action']
                or _fingerprint(saved) != _fingerprint(requested)):
            raise JobError('operation_conflict')
        expected = _intent(JobLease(**saved['target']), JobLease(**saved['caller']),
            saved['operation_action'], _call_ref(saved['call_ref']), saved['tool_call_id'],
            saved['request'], saved['result'])
        if expected != saved:
            raise JobError('record_request_corrupt')
        return saved
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError('record_request_corrupt') from exc


def _receipt(manager, event, request, source_ref):
    try:
        if (event['event_type'] != 'operation' or event['action'] != request['operation_action']
                or event['job_id'] != request['job_id'] or event['caller_session_id'] != request['call_ref']['session_id']
                or event['tool_call_id'] != request['tool_call_id']
                or json.loads(event['call_ref_json']) != request['call_ref']
                or event['request_hash'] != _outcome_hash(request, source_ref)
                or json.loads(event['payload_json']) != request['result']):
            raise JobError('record_receipt_mismatch')
        return _operation(event)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError('record_receipt_mismatch') from exc


def _role(role, target_kind):
    if (target_kind == 'workflow' and role != 'main' or
            target_kind == 'subagent' and role not in ('main', 'workflow_node')):
        raise JobError('role_forbids_delegation')


async def _source(manager, tx, actor, request, target_kind):
    source = await manager._fence(tx, actor, JobLease(**request['caller']))
    if (source['session_id'] or source['created_by_session_id']) != actor.session_id:
        raise JobError('caller_is_not_executing_parent')
    _role(source['execution_role'], target_kind)
    await manager._validate_call(tx, actor, _call_ref(request['call_ref']), request['tool_call_id'])
    return source


async def _apply(manager, actor, accepted):
    request = accepted.data['request']
    ref, target = _call_ref(request['call_ref']), JobLease(**request['target'])
    key = requests._origin(manager, actor, request, 'record')
    async with manager.tree_write(target.job_id) as tx:
        old = await tx.fetchall(f'SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?',
            (key, requests.rejection_id(accepted.event['event_id'])))
        if old:
            await manager._authorize(tx, actor, request['job_id'])
            return requests.read_outcome(manager, accepted, old[0]), old[0]
        current = await manager._authorize(tx, actor, target.job_id)
        try:
            row = await manager._fence(tx, actor, target)
        except JobError as exc:
            if exc.code not in TARGET_BARRIERS:
                raise
            return None, await requests.reject(manager, tx, actor, accepted.event, accepted.data, current, exc.code)
        if row['root_job_id'] != accepted.data['target']['root_job_id']:
            raise JobError('record_request_target_changed')
        now, version = manager.now(), row['version']+1
        changed = await tx.write_job(target.job_id, f'UPDATE {manager.jobs} SET version=version+1,updated_at=? '
            'WHERE job_id=? AND root_job_id=? AND generation=? AND executor_id=? AND boot_id=? AND version=? '
            "AND intake='open' AND cancel_requested_at IS NULL AND lease_until_ms>?",
            (now, target.job_id, row['root_job_id'], target.generation, target.executor_id,
             target.boot_id, row['version'], now))
        if changed != 1:
            raise _TargetChanged()
        fingerprint = _outcome_hash(request, requests.reference(accepted.event))
        operation = await manager._insert_operation(tx, job_id=target.job_id, version=version,
            actor=actor, action=request['operation_action'], key=key, fingerprint=fingerprint,
            call_ref=ref, tool_call_id=request['tool_call_id'], payload=request['result'])
        outcome = {'event_id': operation.operation_id, 'event_type': 'operation', 'job_id': target.job_id,
            'job_version': version, 'operation_key': key, 'request_hash': fingerprint,
            'action': request['operation_action'], 'caller_session_id': actor.session_id,
            'tool_call_id': request['tool_call_id'], 'call_ref_json': _json(request['call_ref']),
            'receipt_ref_json': None, 'payload_json': _json(request['result']),
            'delivery_status': 'not_applicable', 'created_at': now}
    return operation, outcome


async def _finish(manager, actor, accepted, outcome):
    requests.remember_event(manager, accepted.event)
    if outcome is None:
        while True:
            try:
                operation, outcome = await _apply(manager, actor, accepted)
                break
            except _TargetChanged:
                continue
    else:
        operation = requests.read_outcome(manager, accepted, outcome)
    await requests.complete(manager, actor, accepted, outcome)
    if operation is None:
        raise JobError(json.loads(outcome['payload_json'])['error'])
    return operation


async def record(manager, actor, target, *, caller, action, call_ref, tool_call_id, request, result):
    requested = _intent(target, caller, action, call_ref, tool_call_id, request, result)
    key = requests._origin(manager, actor, requested, 'record')
    event_id = requests.identity('record', key)
    async with manager._read() as reader:
        original = await reader.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (event_id,))
        if original is not None:
            saved = _saved_request(original, requested)
            accepted, outcome = await requests.discover(manager, reader, actor, saved, action='record')
        else:
            if await reader.fetchone(f'SELECT event_id FROM {manager.events} WHERE operation_key=?', (key,)):
                # Another current control domain already owns this exact call.
                raise JobError('operation_conflict')
            row = await manager._authorize(reader, actor, target.job_id)
            metadata = {'job_id': row['job_id'], 'root_job_id': row['root_job_id'], 'generation': target.generation}
            target_kind = row['kind']
            accepted, outcome = None, None
    if accepted is None:
        async def authorize(tx):
            return await _source(manager, tx, actor, requested, target_kind)
        accepted, _ = await requests.accept(manager, actor, requested, metadata, action='record', authorize=authorize)
    return await _finish(manager, actor, accepted, outcome)


@verify_stored
async def reconcile_request(manager, actor, request_id):
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        original = await reader.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=? AND event_type=?',
            (request_id, 'task_control_request'))
        if original is None:
            raise JobError('record_request_original_missing')
        request = json.loads(original['payload_json'])['request']
        accepted, outcome = await requests.discover(manager, reader, actor, request, action='record')
    return await _finish(manager, actor, accepted, outcome)
