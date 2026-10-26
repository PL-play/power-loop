"""Original managed invocation exits before prepare, including claim lost ACK.

The local witness belongs to the actual invocation Task. A candidate lease is
captured inside the original claim transaction, before its commit can lose an
ACK. Neither the candidate nor a missing prepared row alone proves an exit.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from .jobs import JobError

_CURRENT = ContextVar('managed_original_claim', default=None)


@dataclass
class _Attempt:
    cache: object
    slot: object
    registration: object
    task: object
    lease: object = None
    exit: str | None = None
    saved: object = None


@contextmanager
def claiming(cache, slot, registration):
    cache.require(slot)
    attempt = _Attempt(cache, slot, registration, asyncio.current_task())
    slot.claim = attempt
    token = _CURRENT.set(attempt)
    try:
        yield
    except BaseException as exc:
        # Synchronous: cancellation cannot erase this original frame exit.
        attempt.exit = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'raised'
        raise
    finally:
        _CURRENT.reset(token)


def capture(manager, actor, lease, expected_version):
    """Called by the actual claim transaction, before leaving its context."""
    attempt = _CURRENT.get()
    if attempt is None or attempt.registration.job.job_id != lease.job_id:
        return
    cache, slot = attempt.cache, attempt.slot
    cache.require(slot)
    if (attempt.task is not asyncio.current_task() or cache.runtime.manager is not manager
            or slot.actor != actor or attempt.registration.job.version != expected_version
            or attempt.exit is not None or attempt.lease is not None
            or (lease.executor_id, lease.boot_id) != (cache.runtime.executor_id, cache.runtime.boot_id)):
        raise JobError('managed_claim_original_mismatch')
    attempt.lease = lease


def exited(slot):
    return slot.claim is not None and slot.claim.exit is not None


async def resolve_local(runtime, actor, slot):
    """Validate a retained original witness; never claim or enter execution."""
    from .jobs import _json
    from .managed_completion import _original
    from .managed_effects import ManagedToolResult, _evidence, _proof, descriptor

    cache, manager = runtime._managed_completions, runtime.manager
    cache.require(slot)
    attempt = slot.claim
    if not exited(slot) or attempt.cache is not cache or attempt.slot is not slot:
        raise JobError('managed_claim_exit_unconfirmed')
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, attempt.registration.job.job_id)
        found = await descriptor(manager, tx, row)
        cache.require(slot)
        if found is None or found[0]['event_id'] != attempt.registration.operation.operation_id:
            raise JobError('managed_claim_registration_changed')
        if (slot.document is None and row['generation'] == 0 and row['status'] == 'queued'
                and row['version'] == attempt.registration.job.version):
            # A rolled-back/unentered claim has no resource. The existing
            # sealed-plan unstarted path still owns its eventual disposition.
            cache.discard_saved(slot)
            return False
        if attempt.lease is None or row['generation'] != 1 or _proof(row) != asdict(attempt.lease):
            raise JobError('managed_claim_execution_unconfirmed')
        for kind in ('prepared', 'dispatch'):
            if await _evidence(manager, tx, row, kind, required=False) is not None:
                raise JobError('managed_claim_preparation_conflict')
        result = ManagedToolResult(
            'The managed effect was not dispatched because its claim did not return.'[:found[1]['contract']['max_result_chars']],
            {'decision': 'not_dispatched', 'job_id': row['job_id']}, 'not_dispatched')
        value = {'version': 2, 'lease': asdict(attempt.lease), 'exit': attempt.exit,
                 'dispatch_started': False, 'completion': {'outcome': 'interrupted',
                 'resource_state': 'stopped', 'result': {'status': 'interrupted',
                 'managed_effect_pending': False, 'managed_effect_result': result.to_dict(),
                 'display': result.display, 'usage': {}}}}
        await _original(manager, tx, row, found, value)
        cache.require(slot)
        if slot.document is not None and slot.document != _json(value):
            raise JobError('managed_completion_conflict')
        slot.lease, slot.document = attempt.lease, _json(value)
    return True


async def flush_local(runtime):
    from .managed_completion import _persist_local

    cache = runtime._managed_completions
    cache.check()
    for slot in list(cache._slots.values()):
        if exited(slot):
            await _persist_local(runtime, slot.actor, slot)
