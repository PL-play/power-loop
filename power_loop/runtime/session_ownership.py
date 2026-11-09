"""Application-provided session ownership, with a single-process default.

Shared implementations live in the application. Keys must identify the store
and the individual session, never an entire task tree. Implementations
notify an involuntary loss with ``lease.lose()``; a failed renewal must do so
before returning False or raising. Releasing an older lease cannot release a
new owner's lease.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, field

from power_loop.contracts.errors import SessionBusy, SessionLeaseLost


@dataclass(eq=False)
class SessionOwnership:
    """A local handle, valid only until its monotonic deadline or notification.

An implementation must set the deadline from the time *before* its acquire or
renew request. This prevents a delayed reply extending local permission past
the shared lease's expiry. Loss callbacks are synchronous and must not raise.
"""

    key: tuple[str, str]
    owner_id: str
    expires_at: float
    on_lost: Callable[[], None] = field(repr=False)
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    lost: bool = False
    released: bool = False
    epoch: int = 0
    node_id: str = ""
    boot_id: str = ""
    installs_fence: bool = False

    def lose(self) -> None:
        if self.lost or self.released:
            return
        self.lost = True
        self.on_lost()

    def require_owned(self) -> None:
        if time.monotonic() >= self.expires_at:
            self.lose()
        if self.lost or self.released:
            raise SessionLeaseLost(self.key)


class SessionOwnershipCoordinator(ABC):
    """Exclusive acquisition, renewal, release, and local loss notification.

Every acquisition is exclusive, including repeated calls with the same owner
ID. The caller retains the returned handle for its whole ownership period.
An application's implementation must pass SessionOwnershipContractTests.
"""

    scope = "process"

    @abstractmethod
    async def acquire(self, key: tuple[str, str], *, owner_id: str, ttl_s: float,
                      on_lost: Callable[[], None]) -> SessionOwnership | None:
        """Return a new handle, or None while another handle owns the key."""

    @abstractmethod
    async def renew(self, lease: SessionOwnership, *, ttl_s: float) -> bool:
        """Extend this handle, or notify loss and return False."""

    @abstractmethod
    async def release(self, lease: SessionOwnership) -> None:
        """Release this handle only; repeated release has no effect."""


class MemorySessionOwnershipCoordinator(SessionOwnershipCoordinator):
    """One process, no database or Redis. Share an instance between loops."""

    def __init__(self):
        self._leases: dict[tuple[str, str], SessionOwnership] = {}

    async def acquire(self, key, *, owner_id, ttl_s, on_lost):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        now = time.monotonic()
        previous = self._leases.get(key)
        if previous is not None:
            if not previous.lost and not previous.released and previous.expires_at > now:
                return None
            previous.lose()
        lease = SessionOwnership(key, owner_id, now + ttl_s, on_lost)
        self._leases[key] = lease
        return lease

    async def renew(self, lease, *, ttl_s):
        if ttl_s <= 0:
            raise ValueError("session ownership ttl_s must be positive")
        now = time.monotonic()
        if (self._leases.get(lease.key) is not lease or lease.lost
                or lease.released or lease.expires_at <= now):
            if self._leases.get(lease.key) is lease:
                del self._leases[lease.key]
            lease.lose()
            return False
        lease.expires_at = now + ttl_s
        return True

    async def release(self, lease):
        if self._leases.get(lease.key) is lease:
            del self._leases[lease.key]
        lease.released = True


DEFAULT_SESSION_COORDINATOR = MemorySessionOwnershipCoordinator()

# Only live session periods are retained. Independent children acquire their own.
_CURRENT: ContextVar[_OwnershipPeriod | None] = ContextVar("session_ownership", default=None)
_PERIODS: dict[tuple, _OwnershipPeriod] = {}


class _OwnershipPeriod:
    def __init__(self, loop, coordinator, store, session_id, ttl_s):
        self.loop = loop
        self.coordinator, self.store, self.session_id = coordinator, store, session_id
        self.key = (store.store_id, session_id)
        self.registry_key = id(coordinator), self.key
        self.ttl_s = ttl_s
        self.lease = None
        self.tasks = {}
        self.idle = asyncio.Event()
        self.writer = None
        self.checked = False

    def lost(self):
        self.loop._cache_invalidate(self.session_id)
        if _PERIODS.get(self.registry_key) is self:
            del _PERIODS[self.registry_key]
        for task in tuple(self.tasks):
            task.cancel("session ownership lost")

    def enter(self, task):
        self.lease.require_owned()
        self.idle.clear()
        self.tasks[task] = self.tasks.get(task, 0) + 1

    def leave(self, task):
        count = self.tasks[task] - 1
        if count:
            self.tasks[task] = count
        else:
            del self.tasks[task]
        if not self.tasks:
            self.idle.set()

    async def maintain(self):
        try:
            while self.tasks:
                try:
                    await asyncio.wait_for(self.idle.wait(), timeout=self.ttl_s / 3)
                except TimeoutError:
                    try:
                        async with asyncio.timeout(max(0, self.lease.expires_at - time.monotonic())):
                            renewed = await self.coordinator.renew(self.lease, ttl_s=self.ttl_s)
                    except Exception:
                        self.lease.lose()
                        logging.getLogger(__name__).exception("session renewal failed: %s", self.key)
                        break
                    if not renewed:
                        self.lease.lose()
                        break
                    self.lease.require_owned()
        finally:
            if _PERIODS.get(self.registry_key) is self:
                del _PERIODS[self.registry_key]
            try:
                if not self.lease.installs_fence:
                    await self.store.release_session_owner(self.session_id, self.lease)
            finally:
                await self.coordinator.release(self.lease)


def current_ownership(store=None, session_id=None):
    period = _CURRENT.get()
    if period is None or (store is not None and period.store.store_id != store.store_id):
        return None
    if session_id is not None and period.session_id != session_id:
        return None
    return period.lease


def detached_context():
    """Retain ownership for metadata writes without inheriting transactions."""
    context = Context()
    context.run(_CURRENT.set, _CURRENT.get())
    return context


def ownership_lost(lease):
    return lease is not None and (lease.lost or lease.released or time.monotonic() >= lease.expires_at)


def require_current_ownership(store=None):
    period = _CURRENT.get()
    if period is not None and (store is None or period.store is store or
                              store.store_id is not None and period.store.store_id == store.store_id):
        period.lease.require_owned()


def track_owned_task(task):
    """Retain this session only for an explicitly attached metadata writer."""
    period = _CURRENT.get()
    if period is not None:
        period.enter(task)
        task.add_done_callback(period.leave)


async def recover_acquired_session(loop, session_id, *, cache_valid):
    """Interrupt only this acquired session; preserve children and billing."""
    period = _CURRENT.get()
    if period is None or period.session_id != session_id or period.checked:
        return
    period.checked = True
    runtime = loop.task_runtime
    if runtime is None:
        return
    manager = runtime.manager
    rows = await manager.db.fetchall(
        f"SELECT job_id,generation,kind,parent_job_id FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? "
        "AND session_id=? AND outcome IS NULL AND status IN ('running','settling')",
        (runtime.tenant_id, runtime.principal_id, session_id))
    for row in rows:
        from types import SimpleNamespace

        from power_loop.agent.runtime_store import RuntimeInputStore
        from power_loop.agent.runtime_types import JobRef, Result

        port = RuntimeInputStore(loop.store)
        ref = JobRef(loop.store.store_id, row['job_id'], row['generation'])
        public_job = row['kind'] != 'send' or row['parent_job_id'] is None
        async with manager._write() as tx:
            if public_job:
                route = await port._lock_result_sessions(tx, ref)
            changed = await tx.write_job(row['job_id'],
                f"UPDATE {manager.jobs} SET status='interrupted',outcome='interrupted',"
                "intake='closed',resource_state='unknown',version=version+1,updated_at=? "
                "WHERE job_id=? AND generation=? AND outcome IS NULL AND status IN ('running','settling')",
                (manager.now(), row['job_id'], row['generation']))
            if changed and public_job:
                await port._commit_result(tx, ref, Result(ref, None, 'interrupted', error='execution owner lost'),
                    SimpleNamespace(epoch=period.lease.epoch, lease=period.lease), route)


@asynccontextmanager
async def _hold_session_ownership(loop, session_id):
    store = loop.store
    period = _CURRENT.get()
    if (period is not None and period.store.store_id == store.store_id
            and period.session_id == session_id and period.loop._owner_id == loop._owner_id):
        period.lease.require_owned()
        acquired = False
    else:
        acquired = True
        period = _OwnershipPeriod(loop, loop.session_coordinator, store, session_id,
                                  loop.config.session_lease_ttl_s)
        # The expected epoch belongs to this acquire request, never its reply.
        while True:
            runtime = loop.task_runtime
            node_id = runtime.executor_id if runtime else loop._owner_id
            boot_id = runtime.boot_id if runtime else loop._owner_id
            coordinator = loop.session_coordinator
            if getattr(coordinator, "installs_session_fence", False):
                lease = await coordinator.acquire(
                    period.key, owner_id=loop._owner_id, ttl_s=period.ttl_s, on_lost=period.lost,
                    node_id=node_id, boot_id=boot_id)
                if lease is None:
                    raise SessionBusy(f"session {session_id} is held by another owner")
                period.lease = lease
                break
            expected = await store.session_owner_epoch(session_id)
            lease = await coordinator.acquire(
                period.key, owner_id=loop._owner_id, ttl_s=period.ttl_s, on_lost=period.lost)
            if lease is None:
                raise SessionBusy(f"session {session_id} is held by another owner")
            period.lease = lease
            lease.node_id, lease.boot_id = node_id, boot_id
            try:
                installed = await store.install_session_owner(session_id, lease, expected)
            except BaseException:
                await coordinator.release(lease)
                raise
            if installed:
                break
            await coordinator.release(lease)
        _PERIODS[period.registry_key] = period
    task = asyncio.current_task()
    period.enter(task)
    token = _CURRENT.set(period)
    if acquired:
        period.writer = asyncio.create_task(period.maintain(), name="session-lease-" + session_id)
        # Mutable store hints are discarded; the window and control caches
        # use their existing next_seq/compaction and control-version checks.
        store._forget_appended(session_id)
    try:
        async with store.send_read_scope(session_id, ownership=period.lease):
            yield period.lease
    except asyncio.CancelledError:
        if period.lease.lost:
            raise SessionLeaseLost(session_id) from None
        raise
    finally:
        _CURRENT.reset(token)
        period.leave(task)
        if not period.tasks:
            await period.writer


@asynccontextmanager
async def hold_session_ownership(loop, session_id):
    from contextlib import ExitStack

    from power_loop.core.execution_context import (
        ExecutionContext,
        _bind_execution_context,
        get_execution_context,
        independent_execution_scope,
    )
    previous = _CURRENT.get()
    with ExitStack() as scopes:
        if previous is not None and previous.key != (loop.store.store_id, session_id):
            scopes.enter_context(independent_execution_scope())
        async with _hold_session_ownership(loop, session_id) as lease:
            if get_execution_context(required=False) is None:
                def snapshot():
                    from power_loop.agent.runtime_types import JobRef

                    from .task_runtime import _SEND
                    execution = _SEND.get()
                    return ExecutionContext(loop.store.store_id, session_id, lease.node_id, lease.boot_id,
                        job_ref=(JobRef(loop.store.store_id, execution.lease.job_id, execution.lease.generation)
                                 if execution else None), owner_epoch=lease.epoch, owner_token=lease.token,
                        actor=execution.actor if execution else None)
                scopes.enter_context(_bind_execution_context(snapshot))
            yield lease
