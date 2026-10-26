"""Actual retained SDK Task completion, independent of result and fee handling."""
from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import asdict, dataclass

from .jobs import JobError, _hash


def identity(manager, lease):
    return 'ctl_exit_' + _hash([manager.store.store_id, asdict(lease)])[:48]


@dataclass(eq=False)
class _Slot:
    actor: object
    lease: object
    task: object
    exit: dict | None = None
    failed: bool = False


class TaskExitCache:
    def __init__(self, runtime):
        self.runtime = runtime
        self.slots = {}
        self._lock = asyncio.Lock()
        self._writer = None

    def check(self):
        self.runtime._managed_completions.check()

    def capture(self, binding, local):
        self.check()
        task = asyncio.current_task()
        if task not in self.runtime._adapter_tasks:
            return  # An inline host scope is not an independently retained SDK Task.
        key = _hash(asdict(binding.lease))
        if (binding.runtime is not self.runtime or local.get('task') is not task
                or local.get('binding') is not binding or self.runtime._local_executions.get(id(local)) is not local
                or self.runtime._controller_retirements.pending.get(key) is not local.get('retirement')):
            raise JobError('controller_exit_context_required')
        if key in self.slots:
            if self.slots[key].task is not task:
                raise JobError('controller_exit_task_conflict')
            return
        slot = _Slot(binding.actor, binding.lease, task)
        self.slots[key] = slot
        task.add_done_callback(lambda done: self._finished(key, slot, done))

    def exclude(self, lease):
        self.slots.pop(_hash(asdict(lease)), None)

    def failed_count(self):
        return sum(slot.failed for slot in self.slots.values())

    def _finished(self, key, slot, done):
        if self.slots.get(key) is not slot:
            return
        if done is not slot.task or not done.done():
            slot.failed = True
            return
        error = None if done.cancelled() else done.exception()
        if slot.exit is None:
            slot.exit = {'state': 'cancelled' if done.cancelled() else 'raised' if error is not None else 'returned',
                         'error_type': type(error).__name__ if error is not None else None,
                         'observed_at_ms': self.runtime.manager.now()}
        if self._writer is not None and not self._writer.done():
            return
        coroutine = self._write_ready()
        writer = None
        try:
            writer = asyncio.create_task(coroutine, context=contextvars.Context(), name='controller-exit-checkpoints')
            self._writer = writer
            writer.add_done_callback(self._writer_finished)
            self.runtime.track_task(writer)
        except BaseException:
            if writer is None:
                coroutine.close()
            else:
                writer.cancel()
            slot.failed = True
            logging.getLogger(__name__).warning('Controller Task exit checkpoint retained for reconciliation')

    def _writer_finished(self, done):
        if done.cancelled() or done.exception() is not None:
            for slot in self.slots.values():
                if slot.exit is not None:
                    slot.failed = True
            return
        # A second controller may finish after the active writer took its
        # snapshot. Drain that new witness too, without retrying failed writes.
        if not self.failed_count():
            for key, slot in self.slots.items():
                if slot.exit is not None:
                    self._finished(key, slot, slot.task)
                    break

    async def _write_ready(self):
        try:
            async with asyncio.timeout(2):
                await self.runtime._controller_retirements.flush()
                await self.flush()
        except Exception as error:
            for slot in self.slots.values():
                if slot.exit is not None:
                    slot.failed = True
            logging.getLogger(__name__).warning('Controller Task exit checkpoint pending: %s', type(error).__name__)

    async def flush(self):
        self.check()
        async with self._lock:
            self.check()
            for key, slot in list(self.slots.items()):
                if slot.exit is None:
                    continue
                try:
                    await persist(self.runtime, slot)
                except BaseException:
                    slot.failed = True
                    raise
                if self.slots.get(key) is slot:
                    del self.slots[key]


async def persist(runtime, slot):
    from .execution_membership import scope, scope_id
    from .model_exit import content_reference
    from .tool_call_plans import _event, _read

    cache = runtime._controller_task_exits
    cache.check()
    if (type(slot) is not _Slot or cache.slots.get(_hash(asdict(slot.lease))) is not slot
            or slot.exit is None or not slot.task.done()):
        raise JobError('actual_controller_task_exit_required')
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._authorize(tx, slot.actor, slot.lease.job_id)
        original = await scope(manager, tx, row['job_id'], slot.lease.generation)
        if _hash(original['lease']) != _hash(asdict(slot.lease)):
            raise JobError('controller_exit_scope_changed')
        if original['kind'] not in ('subagent', 'workflow_node', 'workflow'):
            return
        frozen = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',
                                   (scope_id(manager, row['job_id'], slot.lease.generation),))
        value = {'version': 1, 'lease': asdict(slot.lease), 'scope': content_reference(frozen),
            'task_exit': dict(slot.exit), 'execution_authority': False,
            'descendant_stop_confirmed': False, 'billing_settled': False}
        key = identity(manager, slot.lease)
        old = await _read(manager, tx, key, row, 'controller_task_exit', optional=True)
        if old is None:
            await _event(manager, tx, key, row, 'controller_task_exit', value)
        elif old != value:
            raise JobError('controller_task_exit_conflict')


async def read(manager, tx, row, lease):
    from .execution_membership import scope, scope_id
    from .model_exit import content_reference
    from .tool_call_plans import _read

    frozen = await scope(manager, tx, row['job_id'], lease.generation)
    if _hash(frozen['lease']) != _hash(asdict(lease)) or frozen['kind'] not in ('subagent', 'workflow_node', 'workflow'):
        raise JobError('controller_exit_scope_changed')
    data = await _read(manager, tx, identity(manager, lease), row, 'controller_task_exit')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',
                              (scope_id(manager, row['job_id'], lease.generation),))
    keys = {'version', 'lease', 'scope', 'task_exit', 'execution_authority', 'descendant_stop_confirmed', 'billing_settled'}
    if (set(data) != keys or type(data['version']) is not int or data['version'] != 1
            or _hash(data['lease']) != _hash(asdict(lease)) or _hash(data['scope']) != _hash(content_reference(event))
            or any(data[k] is not False for k in ('execution_authority', 'descendant_stop_confirmed', 'billing_settled'))
            or type(data['task_exit']) is not dict or set(data['task_exit']) != {'state', 'error_type', 'observed_at_ms'}
            or data['task_exit']['state'] not in ('returned', 'raised', 'cancelled')
            or type(data['task_exit']['observed_at_ms']) is not int or not 0 <= data['task_exit']['observed_at_ms'] < 2**63
            or (data['task_exit']['state'] == 'raised' and (type(data['task_exit']['error_type']) is not str
                or not 1 <= len(data['task_exit']['error_type']) <= 128))
            or (data['task_exit']['state'] != 'raised' and data['task_exit']['error_type'] is not None)):
        raise JobError('controller_task_exit_corrupt')
    return data
