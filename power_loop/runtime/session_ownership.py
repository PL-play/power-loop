"""Application-provided session ownership, with a single-process default.

Shared implementations live in the application. Keys must identify the store
and the individual session, never an entire task tree. Implementations
notify an involuntary loss with ``lease.lose()``; a failed renewal must do so
before returning False or raising. Releasing an older lease cannot release a
new owner's lease.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

from power_loop.contracts.errors import SessionLeaseLost


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


# A worker binds the actual handle obtained from its Runtime claim.
_CURRENT: ContextVar[SessionOwnership | None] = ContextVar("session_ownership", default=None)


def current_ownership(store=None, session_id=None):
    lease = _CURRENT.get()
    if lease is None or (store is not None and lease.key[0] != store.store_id):
        return None
    if session_id is not None and lease.key[1] != session_id:
        return None
    # A matching retired handle must reach require_owned(), never become None.
    return lease


def require_current_ownership(store=None):
    lease = current_ownership(store)
    if lease is not None:
        lease.require_owned()
