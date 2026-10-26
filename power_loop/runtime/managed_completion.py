"""Actual finite callback exit checkpoints and query-only result repair.

Only the SDK callback return path creates a local completion. A remote business
lookup, missing task handle, lease expiry or a new runtime cannot invent one.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from .jobs import JobError, JobLease, _hash, _json
from .result_proof import ResultPublicationProof


@dataclass(eq=False)
class _Slot:
    key: tuple
    actor: object
    lease: object = None
    document: str | None = None
    claim: object = None


@dataclass
class CompletionCache:
    runtime: object = field(repr=False)

    def __post_init__(self):
        from .root_context import _authority

        self._authority = _authority(self.runtime)
        self._slots = {}
        self._claim_lock = asyncio.Lock()

    def check(self):
        from .root_context import _authority

        actual = _authority(self.runtime)
        if (any(a is not b for a, b in zip(actual[:3], self._authority[:3], strict=True))
                or actual[3:] != self._authority[3:]):
            raise JobError('managed_completion_runtime_changed')

    def reserve(self, actor, seq, call_id):
        self.check()
        if type(seq) is not int or seq <= 0 or type(call_id) is not str or not call_id:
            raise JobError('persisted_tool_call_required')
        key = (actor.session_id, seq, call_id)
        if key in self._slots:
            raise JobError('managed_effect_original_requires_reconciliation')
        # Bounds active callbacks plus completed-but-unsaved checkpoints. Fail
        # before claiming/prepare/dispatch; never drop an unresolved entry.
        if len(self._slots) >= 1000:
            raise JobError('managed_completion_capacity_reached')
        slot = _Slot(key, actor)
        self._slots[key] = slot
        return slot

    def require(self, slot):
        self.check()
        if type(slot) is not _Slot or self._slots.get(slot.key) is not slot:
            raise JobError('managed_completion_handle_missing')

    def complete(self, slot, execution, document):
        from .managed_effects import _document

        self.require(slot)
        if slot.actor != execution.actor or execution.runtime is not self.runtime:
            raise JobError('managed_completion_runtime_changed')
        saved = _json(_document(document, 65536))
        if slot.document is not None and slot.document != saved:
            raise JobError('managed_completion_conflict')
        slot.lease, slot.document = execution.lease, saved

    def release_uncompleted(self, slot):
        from .managed_claim import exited
        if self._slots.get(slot.key) is slot and slot.document is None and not exited(slot):
            del self._slots[slot.key]

    def discard_saved(self, slot):
        self.require(slot)
        del self._slots[slot.key]

    def pending_count(self):
        from .managed_claim import exited
        return sum(s.document is not None or exited(s) for s in self._slots.values())

    def claim_exit_count(self):
        from .managed_claim import exited
        return sum(exited(s) for s in self._slots.values())

    def find(self, job):
        self.check()
        return next((s for s in self._slots.values() if (s.lease is not None and s.lease.job_id == job)
                     or (s.claim is not None and s.claim.registration.job.job_id == job)), None)


async def _scope(manager, tx, row):
    from .execution_membership import _snapshot, scope

    actual = await scope(manager, tx, row['job_id'], row['generation'])
    expected = _snapshot(row)
    normalized = {k: v for k, v in actual.items() if k != 'executor_binding'}
    normalized['version'] = 1
    if _hash(normalized) != _hash(expected):
        raise JobError('managed_completion_scope_changed')
    return actual


def _payload(row, binding, value):
    from .managed_effects import ManagedToolResult, _document, _proof

    value = _document(value, 65536)
    if (set(value) != {'version', 'lease', 'exit', 'dispatch_started', 'completion'}
            or type(value['version']) is not int or value['version'] != 2
            or value['lease'] != _proof(row) or value['exit'] not in ('returned', 'raised', 'cancelled')
            or type(value['dispatch_started']) is not bool):
        raise JobError('managed_completion_invalid')
    completion = value['completion']
    if (type(completion) is not dict or set(completion) != {'outcome', 'result', 'resource_state'}
            or completion['resource_state'] != 'stopped'
            or completion['outcome'] != ('succeeded' if value['exit'] == 'returned' else 'interrupted')):
        raise JobError('managed_completion_invalid')
    result = completion['result']
    if (type(result) is not dict
            or set(result) != {'status', 'managed_effect_pending', 'managed_effect_result', 'display', 'usage'}
            or result['status'] != ('completed' if value['exit'] == 'returned' else 'interrupted')
            or type(result['managed_effect_pending']) is not bool
            or result['managed_effect_pending'] != (value['dispatch_started'] and value['exit'] != 'returned')
            or type(result['display']) is not str or not result['display']
            or len(result['display']) > binding['contract']['max_result_chars']
            or result['usage'] != {}):
        raise JobError('managed_completion_invalid')
    if result['managed_effect_result'] is not None:
        known = ManagedToolResult.from_dict(result['managed_effect_result'])
        if known.display != result['display']:
            raise JobError('managed_completion_invalid')
        if not value['dispatch_started'] and value['exit'] != 'returned' and known.resolution != 'not_dispatched':
            raise JobError('managed_completion_invalid')
    elif not result['managed_effect_pending']:
        raise JobError('managed_completion_invalid')
    return value


async def _original(manager, tx, row, found, value):
    from .managed_effects import _evidence

    scope = await _scope(manager, tx, row)
    creation, binding = found
    _payload(row, binding, value)
    prepared = await _evidence(manager, tx, row, 'prepared', required=False)
    dispatched = await _evidence(manager, tx, row, 'dispatch', required=False)
    if prepared is not None and prepared[1]['adapter'] != binding:
        raise JobError('managed_completion_binding_changed')
    if value['dispatch_started']:
        if (prepared is None or dispatched is None
                or prepared[1]['preparation']['local_result'] is not None
                or dispatched[1]['preparation_hash'] != _hash(prepared[1]['preparation'])):
            raise JobError('managed_completion_dispatch_mismatch')
    elif dispatched is not None:
        raise JobError('managed_completion_dispatch_mismatch')
    elif value['exit'] == 'returned':
        if (prepared is None or prepared[1]['preparation']['local_result'] is None
                or prepared[1]['preparation']['local_result'] != value['completion']['result']['managed_effect_result']):
            raise JobError('managed_completion_preparation_mismatch')
    return scope


def _anchor(row, creation, exit_id, value, scope):
    from .managed_effects import _proof

    return {'version': 1, 'lease': _proof(row), 'creation_id': creation['event_id'],
        'creation_hash': creation['request_hash'], 'exit_id': exit_id, 'exit_hash': _hash(value),
        'scope_hash': _hash(scope)}


async def validate_exit(manager, tx, row, found):
    from .managed_effects import _evidence, _id, _read_event

    if row['generation'] != 1:
        raise JobError('managed_completion_generation_unsupported')
    exited = await _evidence(manager, tx, row, 'resource_exit')
    if exited[1]['version'] != 2:
        raise JobError('managed_completion_checkpoint_unavailable')
    scope = await _original(manager, tx, row, found, exited[1])
    anchor = await _read_event(manager, tx, _id(manager, 'completion', row['job_id'], row['generation']))
    if (anchor[0]['job_id'] != row['job_id'] or anchor[0]['event_type'] != 'managed_effect_completion'
            or anchor[0]['delivery_status'] != 'not_applicable'
            or _hash(anchor[1]) != _hash(_anchor(row, found[0], exited[0]['event_id'], exited[1], scope))):
        raise JobError('managed_completion_anchor_mismatch')
    return exited[1]


@dataclass(frozen=True)
class _CheckpointProof(ResultPublicationProof):
    original_hash: str

    async def validate(self, manager, transaction, execution_row, requested_result):
        from .managed_effects import descriptor

        found = await descriptor(manager, transaction, execution_row)
        if found is None:
            raise JobError('managed_effect_evidence_missing')
        value = await validate_exit(manager, transaction, execution_row, found)
        if _hash(value) != self.original_hash or _hash(value['completion']) != _hash(requested_result):
            raise JobError('managed_completion_result_mismatch')
        return True


async def _record_tx(manager, tx, actor, row, found):
    from .managed_effects import _event, _id

    if found is None:
        raise JobError('managed_effect_evidence_missing')
    value = await validate_exit(manager, tx, row, found)
    completion = value['completion']
    event_id = await manager._record_result_tx(tx, actor, JobLease(**value['lease']),
        outcome=completion['outcome'], result=completion['result'], resource_state='stopped', evidence_only=True,
        publication_proof=_CheckpointProof(_hash(value)))
    saved = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (event_id,))
    await _event(manager, tx, await manager._row(tx, row['job_id']), 'result_recovery', {
        'version': 1, 'lease': value['lease'], 'creation_id': found[0]['event_id'],
        'exit_id': _id(manager, 'resource_exit', row['job_id'], row['generation']),
        'exit_hash': _hash(value), 'result_id': event_id, 'result_hash': _hash(json.loads(saved['payload_json']))})
    return {'job_id': row['job_id'], 'generation': row['generation'], 'result_id': event_id,
        'owned_callback_exited': True, 'execution_authority': False, 'billing_settled': False,
        'acknowledgement_created': False, 'tool_receipt_written': False}


async def _persist_local(runtime, actor, slot):
    from .managed_claim import exited, resolve_local

    cache = runtime._managed_completions
    if not exited(slot):
        return await _persist_checkpoint(runtime, actor, slot)
    # Original exception handling and host/shutdown retries share one writer.
    async with cache._claim_lock:
        cache.check()
        attempt = slot.claim
        if attempt.saved is not None:
            await runtime.manager.get_job(actor, attempt.registration.job.job_id, include_result=False)
            return attempt.saved
        if not await resolve_local(runtime, actor, slot):
            attempt.saved = {'job_id': attempt.registration.job.job_id, 'claim_committed': False,
                             'execution_authority': False, 'acknowledgement_created': False}
        else:
            attempt.saved = await _persist_checkpoint(runtime, actor, slot)
        return attempt.saved


async def _persist_checkpoint(runtime, actor, slot):
    from .managed_effects import _event, _id, descriptor

    cache, manager = runtime._managed_completions, runtime.manager
    cache.require(slot)
    if slot.document is None:
        raise JobError('managed_completion_exit_unconfirmed')
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, slot.lease.job_id)
        cache.require(slot)
        manager._check_proof(row, slot.lease)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError('managed_effect_evidence_missing')
        value = json.loads(slot.document)
        scope = await _original(manager, tx, row, found, value)
        cache.require(slot)
        await _event(manager, tx, row, 'resource_exit', value)
        await _event(manager, tx, row, 'completion', _anchor(row, found[0],
            _id(manager, 'resource_exit', row['job_id'], row['generation']), value, scope))
    # A separate checkpoint intentionally survives a following result write
    # failure. That exact original result can be restored in a fresh process.
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, slot.lease.job_id)
        cache.require(slot)
        manager._check_proof(row, slot.lease)
        found = await descriptor(manager, tx, row)
        result = await _record_tx(manager, tx, actor, row, found)
    cache.discard_saved(slot)
    return result


def _host_only():
    from .task_runtime import _OWNER, _SEND

    if _OWNER.get() is not None or _SEND.get() is not None:
        raise JobError('managed_completion_requires_host')


async def flush_local(runtime, actor, job):
    """Retry only an SDK-owned, actually exited local callback's exact checkpoint."""
    _host_only()
    await runtime.manager.get_job(actor, job, include_result=False)
    slot = runtime._managed_completions.find(job)
    if slot is None:
        raise JobError('managed_completion_handle_missing')
    return await _persist_local(runtime, actor, slot)


async def restore_result(manager, actor, job):
    """No runtime/provider/callback: restore one original result from durable exit."""
    _host_only()
    from .managed_effects import descriptor

    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError('managed_effect_evidence_missing')
        return await _record_tx(manager, tx, actor, row, found)
