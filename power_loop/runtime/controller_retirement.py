"""Retire an actual SDK controller's admission; never claim resource completion."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from .jobs import JobError, _hash


def identity(manager, lease, *, legacy=False):
    value = asdict(lease) if not isinstance(lease, dict) else lease
    return ('controller_retired_' if legacy else 'ctl_retire_') + _hash([manager.store.store_id, value])[:48]


async def read_retirement(manager, tx, owner, lease, *, optional=False):
    from .tool_call_plans import _read
    keys = [identity(manager, lease), identity(manager, lease, legacy=True)]
    rows = await tx.fetchall(f'SELECT * FROM {manager.events} WHERE event_id IN (?,?)', tuple(keys))
    if not rows:
        if optional:
            return None
        raise JobError('controller_retirement_required')
    if len(rows) != 1:
        raise JobError('controller_retirement_ambiguous')
    value = await _read(manager, tx, rows[0]['event_id'], owner, 'controller_retired')
    return rows[0], value


@dataclass(eq=False)
class _Exit:
    actor: object
    lease: object
    failed: bool = False


class RetirementCache:
    """Only actual local controller exit creates a retryable metadata write."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.pending = {}
        self._flush_lock = asyncio.Lock()

    def check(self):
        self.runtime._managed_completions.check()

    def capture(self, binding, local):
        self.check()
        if (local.get('binding') is not binding or local.get('task') is not asyncio.current_task()
                or self.runtime._local_executions.get(id(local)) is not local
                or not binding.stop_renewal.is_set()):
            raise JobError('controller_exit_context_required')
        key = _hash(asdict(binding.lease))
        slot = self.pending.setdefault(key, _Exit(binding.actor, binding.lease))
        local['retirement'] = slot
        self.runtime._controller_task_exits.capture(binding, local)
        return slot

    def contains(self, lease):
        self.check()
        return _hash(asdict(lease)) in self.pending

    def discard(self, slot):
        key = _hash(asdict(slot.lease))
        if self.pending.get(key) is slot:
            del self.pending[key]

    def failed_count(self):
        return sum(slot.failed for slot in self.pending.values())

    async def flush(self):
        self.check()
        if not self.failed_count():
            return
        async with self._flush_lock:
            self.check()
            # Healthy in-flight closure never holds up an unrelated controller.
            for slot in list(self.pending.values()):
                if slot.failed:
                    await _persist(self.runtime, slot)
                    self.discard(slot)


async def retire(runtime, binding, local):
    slot = local.get('retirement')
    if slot is None or slot.actor != binding.actor or slot.lease != binding.lease:
        raise JobError('controller_exit_context_required')
    await _persist(runtime, slot)


async def _persist(runtime, binding):
    cache = runtime._controller_retirements
    cache.check()
    if cache.pending.get(_hash(asdict(binding.lease))) is not binding:
        raise JobError('controller_exit_context_required')
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._authorize(tx, binding.actor, binding.lease.job_id)
        if row['kind'] not in ('workflow', 'workflow_node', 'subagent'):
            runtime._controller_task_exits.exclude(binding.lease)
            return
        try:
            manager._check_proof(row, binding.lease)
        except JobError as exc:
            if exc.code == 'stale_execution':
                return  # A newer admitted generation already fences this controller.
            raise
        if row['intake'] == 'closed' and row['resource_state'] != 'unknown':
            return
        from .tool_call_plans import _event
        data = {'version': 1, 'lease': asdict(binding.lease), 'kind': row['kind'],
                'execution_role': row['execution_role'], 'session_id': row['session_id'],
                'reason': 'sdk_controller_context_exited', 'execution_authority': False,
                'resource_stop_confirmed': False, 'result_handled': False}
        saved = await read_retirement(manager, tx, row, binding.lease, optional=True)
        if saved is not None:
            if _hash(saved[1]) != _hash(data):
                raise JobError('controller_retirement_conflict')
            return
        key = identity(manager, binding.lease)
        await _event(manager, tx, key, row, 'controller_retired', data)
        # Close admission atomically with the receipt. Existing SDK readers
        # already fence closed intake and unknown resources; no new permission
        # rule or inferred physical-exit proof is needed.
        await tx.execute(f"UPDATE {manager.jobs} SET intake='closed',status='settling',"
                         "resource_state='unknown',lease_until_ms=NULL,version=version+1,updated_at=? WHERE job_id=?",
                         (manager.now(), row['job_id']))
