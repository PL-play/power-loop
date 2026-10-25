"""Positive evidence from the actual SDK stack before batch admission.

A missing plan is not a proof. Only the original preparation failure path can
write this checkpoint; readers never recreate grants or declare model exit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .jobs import JobActor, JobError, _hash, _json


@dataclass(frozen=True)
class _Attempt:
    pipeline: object
    execution: object
    scope: str
    ref: object
    entries: str


def _freeze(pipeline, execution, source, ref, entries):
    from .workflow_controller import proof
    return _Attempt(pipeline, execution, _json(proof(source)), ref, _json(normalize(entries)))


def normalize(entries):
    return [{**entry, 'job_id': None, 'operation_id': None} for entry in entries]


def identity(manager, source, ref):
    from .tool_call_plans import identity as plan_identity
    return plan_identity(manager, source, ref) + '_abort'


async def _facts(manager, tx, source, ref, entries):
    from .managed_completion import _scope
    from .managed_effects import _never_claimed, _verify_call, descriptor
    from .tool_call_plans import _read
    from .tool_call_plans import identity as plan_identity
    from .tool_call_plans import read as read_plan
    from .tool_call_plans import required as plan_required

    if not await plan_required(manager, tx, source):
        raise JobError('tool_call_abort_source_protocol_required')
    await _scope(manager, tx, source)
    actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
    length = 'length(CAST(tool_calls_json AS BLOB))' if manager.db.dialect.name == 'sqlite' else 'octet_length(tool_calls_json)'
    size = await tx.fetchone(f'SELECT {length} AS n FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (ref.session_id, ref.seq))
    if size is None or size['n'] is None or size['n'] > 131072:
        raise JobError('tool_call_abort_raw_unavailable')
    raw = await tx.fetchone(f'SELECT tool_calls_json FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (ref.session_id, ref.seq))
    calls = json.loads(raw['tool_calls_json'])
    fields = {'call_id', 'tool_name', 'arguments_hash', 'registration', 'recovery', 'job_id', 'operation_id'}
    if (type(calls) is not list or not 0 < len(calls) <= 64 or type(entries) is not list
            or len(entries) != len(calls) or any(type(e) is not dict or set(e) != fields or e['job_id'] is not None or e['operation_id'] is not None for e in entries)
            or [e['call_id'] for e in entries] != [c.get('id') if type(c) is dict else None for c in calls]):
        raise JobError('tool_call_abort_call_set_changed')
    key = plan_identity(manager, source, ref)
    plan = await read_plan(manager, tx, source, ref, optional=True)
    if plan is None:
        if await _read(manager, tx, key + '_seal', source, 'tool_call_plan_seal', optional=True) is not None:
            raise JobError('tool_call_abort_plan_changed')
    elif _hash(normalize(plan['entries'])) != _hash(entries):
        raise JobError('tool_call_abort_plan_changed')
    for entry in entries:
        await _verify_call(manager, tx, actor, source, ref, entry['call_id'], entry['tool_name'], entry['arguments_hash'])
        started = key + '_start_' + _hash(entry['call_id'])[:24]
        if await _read(manager, tx, started, source, 'tool_call_dispatch_started', optional=True) is not None:
            raise JobError('tool_call_abort_already_admitted')
        operation_key = manager._origin_key(actor, 'create', ref, entry['call_id'], None)
        operation = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE operation_key=?', (operation_key,))
        if operation is not None:
            row = await manager._authorize(tx, actor, operation['job_id'])
            found = await descriptor(manager, tx, row)
            if (entry['recovery'] is None or found is None
                    or entry['recovery'] != {'kind': 'managed_effect', 'request': json.loads(found[0]['payload_json'])['request']}):
                raise JobError('tool_call_abort_registration_changed')
            await _never_claimed(manager, tx, row)
    return None if plan is None else {'event_id': key, 'payload_hash': _hash(plan)}


async def read(manager, tx, source, ref, *, required=False):
    from .managed_effects import _ref
    from .tool_call_plans import _read
    from .workflow_controller import proof

    key = identity(manager, source, ref)
    value = await _read(manager, tx, key, source, 'tool_call_abort', optional=not required)
    if value is None:
        # A lost checkpoint cannot leave an orphan anchor looking like no abort.
        if await _read(manager, tx, key + '_anchor', source, 'tool_call_abort_anchor', optional=True) is not None:
            raise JobError('tool_call_abort_checkpoint_missing')
        return None
    anchor = await _read(manager, tx, key + '_anchor', source, 'tool_call_abort_anchor')
    fields = {'version', 'scope', 'assistant_ref', 'entries', 'plan', 'decision', 'execution_authority', 'acknowledgement_created', 'billing_settled'}
    if (type(value) is not dict or set(value) != fields or type(value['version']) is not int or value['version'] != 1
            or _hash(value['scope']) != _hash(proof(source)) or _hash(value['assistant_ref']) != _hash(ref.to_dict())
            or value['decision'] != 'admission_not_granted'
            or value['execution_authority'] is not False or value['acknowledgement_created'] is not False or value['billing_settled'] is not False):
        raise JobError('tool_call_abort_checkpoint_corrupt')
    _ref(value['assistant_ref'])
    expected = {'version': 1, 'scope': proof(source), 'abort_id': key, 'abort_hash': _hash(value), 'assistant_ref': ref.to_dict()}
    if _hash(anchor) != _hash(expected) or _hash(value['plan']) != _hash(await _facts(manager, tx, source, ref, value['entries'])):
        raise JobError('tool_call_abort_anchor_mismatch')
    return value


async def _record(attempt):
    """Only called while unwinding the actual pre-admission SDK stack."""
    from .task_runtime import _SEND
    from .tool_call_plans import _event
    from .tool_call_plans import identity as plan_identity
    from .workflow_controller import proof

    execution = _SEND.get()
    if (type(attempt) is not _Attempt or execution is not attempt.execution
            or getattr(attempt.pipeline, '_native_abort_attempt', None) is not attempt):
        raise JobError('tool_call_abort_original_stack_required')
    pipeline, runtime, actor = attempt.pipeline, execution.runtime, execution.actor
    manager = runtime.manager
    retained = getattr(pipeline, '_native_tool_batch_bindings', None)
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, execution.lease.job_id)
        manager._check_proof(source, execution.lease)
        if _json(proof(source)) != attempt.scope or pipeline.session_id != actor.session_id:
            raise JobError('tool_call_abort_scope_changed')
        if retained is not None and retained.plan_id == plan_identity(manager, source, attempt.ref):
            raise JobError('tool_call_abort_already_admitted')
        entries = json.loads(attempt.entries)
        plan = await _facts(manager, tx, source, attempt.ref, entries)
        key = identity(manager, source, attempt.ref)
        value = {'version': 1, 'scope': proof(source), 'assistant_ref': attempt.ref.to_dict(), 'entries': entries,
                 'plan': plan, 'decision': 'admission_not_granted', 'execution_authority': False,
                 'acknowledgement_created': False, 'billing_settled': False}
        old = await read(manager, tx, source, attempt.ref)
        if old is not None:
            if _hash(old) != _hash(value):
                raise JobError('tool_call_abort_checkpoint_conflict')
            return old
        await _event(manager, tx, key, source, 'tool_call_abort', value)
        await _event(manager, tx, key + '_anchor', source, 'tool_call_abort_anchor',
                     {'version': 1, 'scope': proof(source), 'abort_id': key, 'abort_hash': _hash(value), 'assistant_ref': attempt.ref.to_dict()})
    return value


async def read_abort(manager, actor, source_job_id, *, assistant_seq):
    """Host-only original evidence reader; never stops/dispatches/ACKs anything."""
    from .managed_completion import _host_only
    from .message_ref import MessageRef
    _host_only()
    if type(assistant_seq) is not int or assistant_seq <= 0:
        raise JobError('persisted_tool_call_required')
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, source_job_id)
        row = await tx.fetchone(f'SELECT * FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (source['session_id'], assistant_seq))
        if row is None:
            raise JobError('tool_call_abort_raw_unavailable')
        ref = MessageRef(manager.store.store_id, source['session_id'], row['id'], row['seq'], row['role'], row['tool_call_id'], row['send_index'], row['round_index'])
        return await read(manager, tx, source, ref, required=True)


async def require_claim(manager, tx, row):
    if row['kind'] != 'background_tool' or row['generation'] != 0:
        return
    from .managed_effects import _ref, descriptor
    found = await descriptor(manager, tx, row)
    if found is None:
        return
    source = await manager._row(tx, row['parent_job_id'])
    ref = _ref(json.loads(found[0]['call_ref_json']))
    if await read(manager, tx, source, ref) is not None:
        raise JobError('tool_call_batch_aborted')
