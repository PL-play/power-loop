"""Opt-in leaf admission. Branch scopes retain their historical semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import monotonic

GATES: ContextVar[tuple[asyncio.Semaphore, ...]] = ContextVar("workflow_leaf_gates", default=())
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

    async def acquire(self):
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
    if lease is None or not lease.held:
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
