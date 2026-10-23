"""Opt-in leaf admission. Branch scopes retain their historical semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import monotonic

GATES: ContextVar[tuple[asyncio.Semaphore, ...]] = ContextVar("workflow_leaf_gates", default=())
REQUEST_GATES: ContextVar[tuple[asyncio.Semaphore, ...]] = ContextVar("workflow_request_gates", default=())
CURRENT: ContextVar[Lease | None] = ContextVar("workflow_leaf_lease", default=None)


class AdmissionCancelled(Exception):
    pass


class Lease:
    def __init__(self, gates, stop=None):
        self.gates = gates
        self.stop = stop
        self.held = []
        self.queue_s = 0.0
        self.idle_s = 0.0
        self.metrics: dict[str, float] = {}
        self.owner_task = None

    async def acquire(self):
        task = asyncio.current_task()
        if self.owner_task is None:
            self.owner_task = task
        elif self.owner_task is not task:
            raise RuntimeError("admission_lease_task_changed")
        started = monotonic()
        try:
            for gate in self.gates:
                if self.stop is None:
                    await gate.acquire()
                    self.held.append(gate)
                    continue
                waiter = asyncio.create_task(gate.acquire())
                stopper = asyncio.create_task(self.stop.wait())
                try:
                    await asyncio.wait((waiter, stopper), return_when=asyncio.FIRST_COMPLETED)
                finally:
                    stopper.cancel()
                    if not waiter.done():
                        waiter.cancel()
                    await asyncio.gather(waiter, stopper, return_exceptions=True)
                    if not waiter.cancelled() and waiter.exception() is None:
                        self.held.append(gate)
                if self.stop.is_cancelled():
                    raise AdmissionCancelled()

        except BaseException:
            self.release()
            raise
        finally:
            self.queue_s += monotonic() - started

    def release(self):
        for gate in reversed(self.held):
            gate.release()
        self.held.clear()


@asynccontextmanager
async def admit(stop=None):
    lease = Lease(GATES.get(), stop)
    await lease.acquire()
    token = CURRENT.set(lease)
    try:
        yield lease
    finally:
        CURRENT.reset(token)
        lease.release()


@asynccontextmanager
async def idle():
    """Host-only: release slots while the entire leaf is idle after its send ends.

    Do not use inside individual parallel tools: sibling tools may still run.
    """
    lease = CURRENT.get()
    if lease is None or lease.owner_task is not asyncio.current_task() or not lease.held:
        yield
        return
    lease.release()
    started = monotonic()
    completed = False
    try:
        yield
        completed = True
    finally:
        lease.idle_s += monotonic() - started
        if completed:
            await lease.acquire()


def count_metric(name: str, value: float = 1):
    lease = CURRENT.get()
    if lease is not None:
        lease.metrics[name] = lease.metrics.get(name, 0) + value


@asynccontextmanager
async def request_slot(stop=None):
    """Limit model requests including nested helpers; never hold across tool waits."""
    gates = REQUEST_GATES.get()
    if not gates:
        yield
        return
    lease = Lease(gates, stop)
    try:
        await lease.acquire()
    except AdmissionCancelled as exc:
        raise asyncio.CancelledError() from exc
    count_metric("model_queue_s", lease.queue_s)
    started = monotonic()
    try:
        yield
    finally:
        count_metric("model_request_s", monotonic() - started)
        count_metric("model_requests")
        lease.release()
