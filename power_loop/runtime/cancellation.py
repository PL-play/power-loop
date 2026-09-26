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

    __slots__ = ("_check", "_reason", "_owned_event", "_parent", "_never", "_any")

    def __init__(self, *, _check: Callable[[], bool] | None = None) -> None:
        # Owned mode: a private threading.Event so ``cancel()`` works and
        # waiters relying on Event semantics keep working.
        self._owned_event = threading.Event()
        if _check is None:
            self._check = self._owned_event.is_set
        else:
            self._check = _check
        self._reason: str = "cancelled"
        self._parent: CancellationToken | None = None
        self._never = False
        self._any: list[CancellationToken] | None = None

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
            tok._parent, tok._never, tok._any = None, False, None
            return tok
        if isinstance(source, asyncio.Event):
            tok = cls.__new__(cls)
            tok._owned_event = threading.Event()  # unused, kept for shape
            tok._check = source.is_set
            tok._reason = "cancelled"
            tok._parent, tok._never, tok._any = None, False, None
            return tok
        if callable(source):
            tok = cls.__new__(cls)
            tok._owned_event = threading.Event()
            tok._check = source
            tok._reason = "cancelled"
            tok._parent, tok._never, tok._any = None, False, None
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
        tok._parent, tok._never, tok._any = None, True, None
        return tok

    @classmethod
    def any_of(cls, *tokens: CancellationToken | None) -> CancellationToken:
        """Cancelled as soon as ANY of ``tokens`` is (``None`` entries ignored). For work with
        more than one owner — a workflow run awaited by a tool call: its own cancel() stops just
        it, the calling run's stop stops it too. Its reason is the first cancelled one's."""
        toks = [t for t in tokens if t is not None]
        if len(toks) == 1:
            return toks[0]
        tok = cls()
        own = tok._owned_event

        def _check() -> bool:
            return own.is_set() or any(t.is_cancelled() for t in toks)

        tok._check = _check
        tok._any = toks
        return tok

    def child(self) -> CancellationToken:
        """A token cancelled when EITHER it is cancelled itself OR this one is (design/124 §8.2:
        the stop tree). Stopping a node stops its whole subtree; stopping a child leaves the
        parent and its other children running. Cheap: no registration, the check walks up."""
        tok = CancellationToken()
        own = tok._owned_event
        parent = self
        tok._parent = parent
        tok._check = lambda: own.is_set() or parent.is_cancelled()
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

    @property
    def reason(self) -> str:
        """Why it was cancelled — its own reason, or the ancestor's that cancelled it."""
        if self._owned_event.is_set():
            return self._reason
        if self._any:
            return next((t.reason for t in self._any if t.is_cancelled()), self._reason)
        if self._parent is not None and self._parent.is_cancelled():
            return self._parent.reason
        return self._reason

    @property
    def is_never(self) -> bool:
        """The ``never()`` sentinel — nothing can cancel it, so nobody should wait on it."""
        return self._never

    async def wait(self, *, poll_s: float = 0.1) -> None:
        """Return once cancelled (polled: sources are threading/asyncio events or callables, and a
        child checks up its ancestors — there's no single thing to await). ``never()`` tokens
        block forever; callers racing a token should skip ``is_never`` ones."""
        while not self.is_cancelled():
            await asyncio.sleep(poll_s)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise CancellationRequested(self.reason)

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
