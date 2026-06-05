"""Unified cancellation primitive for power-loop.

Why
---
Pipelines and tool handlers come from many places — sync threads, asyncio
tasks, hook callbacks. Different callers naturally hold different "cancel
signals":

* a long-running CLI may use ``threading.Event``;
* an asyncio server may use ``asyncio.Event``;
* a UI framework may expose only ``is_cancelled()`` as a callable.

``CancellationToken`` is the **one shape** the pipeline checks. Callers pass
whatever they have; ``from_any`` lifts it into a token. There is also a
plain "owned" token (``CancellationToken()``) that callers can ``cancel()``
themselves — used by hook ``HookDirective.CANCEL`` (M1.5) and by
``StatefulAgentLoop.cancel(sid)``-style helpers.

The token is **read-only from the pipeline's side**: it never *creates*
cancellation, only observes it. The only mutating method, ``cancel()``,
is for callers that explicitly created an owned token.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Union

from power_loop.contracts.errors import CancellationRequested

# What callers may hand us. ``None`` means "no cancellation".
CancellationLike = Union["CancellationToken", asyncio.Event, threading.Event, Callable[[], bool], None]


class CancellationToken:
    """Observable cancellation flag.

    Construct it three ways:

    * ``CancellationToken()`` — owned; flip with ``cancel()``.
    * ``CancellationToken.from_any(obj)`` — wrap an existing event / callable.
    * ``CancellationToken.never()`` — sentinel that is never cancelled
      (sugar for "the caller passed None").
    """

    __slots__ = ("_check", "_reason", "_owned_event")

    def __init__(self, *, _check: Callable[[], bool] | None = None) -> None:
        # Owned mode: a private threading.Event so ``cancel()`` works and
        # waiters relying on Event semantics keep working.
        self._owned_event = threading.Event()
        if _check is None:
            self._check = self._owned_event.is_set
        else:
            self._check = _check
        self._reason: str = "cancelled"

    # ── Factories ───────────────────────────────────────────────────────

    @classmethod
    def from_any(cls, source: CancellationLike) -> CancellationToken:
        """Lift any cancel-like object into a token. ``None`` → ``never()``."""
        if source is None:
            return cls.never()
        if isinstance(source, CancellationToken):
            return source
        if isinstance(source, threading.Event):
            tok = cls.__new__(cls)
            tok._owned_event = source
            tok._check = source.is_set
            tok._reason = "cancelled"
            return tok
        if isinstance(source, asyncio.Event):
            tok = cls.__new__(cls)
            tok._owned_event = threading.Event()  # unused, kept for shape
            tok._check = source.is_set
            tok._reason = "cancelled"
            return tok
        if callable(source):
            tok = cls.__new__(cls)
            tok._owned_event = threading.Event()
            tok._check = source
            tok._reason = "cancelled"
            return tok
        raise TypeError(
            f"CancellationToken.from_any: unsupported source type {type(source).__name__}"
        )

    @classmethod
    def never(cls) -> CancellationToken:
        """Sentinel token that is never cancelled."""
        tok = cls.__new__(cls)
        tok._owned_event = threading.Event()
        tok._check = lambda: False
        tok._reason = "cancelled"
        return tok

    # ── Observation ─────────────────────────────────────────────────────

    def is_cancelled(self) -> bool:
        try:
            return bool(self._check())
        except Exception:
            # A user-supplied callable that raises is treated as "not cancelled"
            # rather than corrupting loop control flow. Errors in cancellation
            # checks should never abort the loop.
            return False

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancellationRequested(self._reason)

    # ── Mutation (owned tokens only) ────────────────────────────────────

    def cancel(self, reason: str = "cancelled") -> None:
        """Flip an owned token. No-op (and safe) on wrapped tokens whose
        underlying source isn't an Event we own — the underlying source is
        the canonical signal there."""
        self._reason = reason
        # We only know how to flip the owned threading.Event. Wrapped
        # asyncio.Event / callable sources must be flipped by their owner.
        self._owned_event.set()


__all__ = ["CancellationToken", "CancellationLike"]
