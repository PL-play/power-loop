"""Single-owner control state; reducers never perform database I/O.

The journal owns recoverable facts. This object owns only the current execution
and a bounded, coalesced set of changes awaiting a checkpoint. A recovered open
execution is *not* permission to repeat its uncertain external actions.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .jobs import JobError, _hash, _json


@dataclass(frozen=True)
class CheckpointPolicy:
    """When buffered control facts are saved; never per model/tool action by default.

    Raw messages and host fee captures are durable independently, so the
    checkpoint cost of a send depends on its duration and size, not its rounds.
    """
    actions: int | None = None
    bytes: int = 262144
    interval_s: float = 5.0

    def __post_init__(self):
        if self.actions is not None and (type(self.actions) is not int or not 1 <= self.actions <= 1024):
            raise ValueError("checkpoint actions must be between 1 and 1024")
        if type(self.bytes) is not int or not 1024 <= self.bytes <= 16 * 1024 * 1024:
            raise ValueError("checkpoint bytes must be between 1 KiB and 16 MiB")
        if type(self.interval_s) not in (int, float) or not 0.05 <= self.interval_s <= 300:
            raise ValueError("checkpoint interval must be between 0.05 and 300 seconds")


@dataclass(frozen=True)
class ExecutionIdentity:
    store_id: str
    job_id: str
    generation: int
    executor_id: str
    boot_id: str
    tenant_id: str
    principal_id: str
    session_id: str

    def __post_init__(self):
        if type(self.generation) is not int or self.generation < 1:
            raise JobError("invalid_execution_identity")
        if any(type(value) is not str or not value or len(value) > 255
               for key, value in asdict(self).items() if key != "generation"):
            raise JobError("invalid_execution_identity")

    @property
    def digest(self):
        return _hash(asdict(self))


@dataclass
class LocalAllocation:
    """One authority-owned, bounded allocation, never a new wallet balance."""
    allocation_id: str
    calls: int
    tokens: int
    max_inflight: int
    used_calls: int = 0
    used_tokens: int = 0
    held_tokens: int = 0
    inflight: int = 0
    holds: dict[str, int] = field(default_factory=dict)
    finished: dict[str, int] = field(default_factory=dict)
    observed: dict[str, tuple[int | None, bool]] = field(default_factory=dict)
    report_version: int = 0

    def __post_init__(self):
        if not self.allocation_id or any(type(v) is not int or v < 0
                                        for v in (self.calls, self.tokens, self.max_inflight)):
            raise JobError("invalid_execution_allocation")

    def reserve(self, attempt_id: str, token_bound: int):
        if type(token_bound) is not int or token_bound < 0:
            raise JobError("invalid_execution_token_bound")
        if attempt_id in self.holds or attempt_id in self.finished:
            raise JobError("execution_attempt_already_admitted")
        if self.used_calls >= self.calls:
            raise JobError("budget_max_calls_exhausted")
        if self.inflight >= self.max_inflight:
            raise JobError("budget_max_inflight_exhausted")
        if self.used_tokens + self.held_tokens + token_bound > self.tokens:
            raise JobError("budget_max_tokens_exhausted")
        self.used_calls += 1
        self.inflight += 1
        self.held_tokens += token_bound
        self.holds[attempt_id] = token_bound

    def observe(self, attempt_id: str, actual_tokens: int | None, *, returned=True):
        if actual_tokens is not None and (type(actual_tokens) is not int or actual_tokens < 0):
            raise JobError("invalid_execution_token_usage")
        if attempt_id in self.observed:
            if self.observed[attempt_id] != (actual_tokens, returned):
                raise JobError("execution_usage_conflict")
            return
        if attempt_id not in self.holds:
            raise JobError("execution_allocation_hold_missing")
        bound = self.holds[attempt_id]
        # Real usage must be recorded even when a provider exceeds its bound.
        if actual_tokens is not None:
            self.used_tokens += actual_tokens
            self.held_tokens -= bound
        if returned:
            self.inflight -= 1
        if actual_tokens is not None and returned:
            self.holds.pop(attempt_id)
            self.finished[attempt_id] = actual_tokens
        self.observed[attempt_id] = (actual_tokens, returned)

    def release_unstarted(self, attempt_id):
        if attempt_id in self.observed or attempt_id not in self.holds:
            raise JobError("budget_release_not_permitted")
        self.used_calls -= 1
        self.held_tokens -= self.holds.pop(attempt_id)
        self.inflight -= 1
        # A released identity is never a new admission key.
        self.finished[attempt_id] = 0

    def summary(self):
        return {key: getattr(self, key) for key in (
            "allocation_id", "calls", "tokens", "max_inflight", "used_calls",
            "used_tokens", "held_tokens", "inflight",
        )}


class ExecutionState:
    """Memory reducer and checkpoint coordinator for one admitted owner.

    Mutations have no await and therefore remain atomic within an asyncio loop,
    including concurrent tool callbacks. Only checkpoint I/O uses a lock. A
    failed checkpoint retains its exact immutable batch for an idempotent retry.
    """

    def __init__(self, identity: ExecutionIdentity, journal, *, lease_until_ms: int,
                 policy: CheckpointPolicy | None = None, now_ms=None):
        self.identity = identity
        self.journal = journal
        self.policy = policy or CheckpointPolicy()
        self.now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self.lease_until_ms = lease_until_ms
        self.deadline = time.monotonic() + max(0, (lease_until_ms - self.now_ms()) / 1000)
        self.cancel_requested = False
        self.closed = False
        self.needs_reconciliation = False
        self.checkpoint_failed = False
        # Durable managed-effect children are verified before the next ordinary
        # admission; memory never infers that an external effect resolved.
        self.managed_check = False
        self.pending_obligations: set[str] = set()
        self.max_pending_obligations = 1
        self.cursor = 0
        self.chain_hash = identity.digest
        self.models = 0
        self.tools = 0
        self.completed = 0
        self.unknown = 0
        self.budget_attempts = 0
        self.budget_unsettled = 0
        self.active: dict[str, dict[str, Any]] = {}
        self.allocations: dict[str, LocalAllocation] = {}
        self._records: dict[tuple[str, str], dict] = {}
        self._dirty: dict[tuple[str, str], dict] = {}
        self._dirty_sizes: dict[tuple[str, str], int] = {}
        self._dirty_bytes = 0
        self._completed_since_flush = 0
        self._pending = None
        self._pending_batches = []
        self._staged_digests = []
        self._staged_transaction = None
        self._flush_lock = asyncio.Lock()
        self._last_flush = time.monotonic()
        self.before_checkpoint = None
        self._staged = False

    def require_admission(self):
        if self.closed:
            raise JobError("execution_control_closed")
        if self.cancel_requested:
            raise JobError("cancel_requested")
        if self.now_ms() >= self.lease_until_ms or time.monotonic() >= self.deadline:
            raise JobError("execution_lease_expired")
        if self.needs_reconciliation or self.unknown:
            raise JobError("unresolved_prior_attempt")
        if len(self.pending_obligations) >= self.max_pending_obligations:
            raise JobError("budget_reconciliation_required")
        if self.checkpoint_failed:
            raise JobError("execution_checkpoint_pending")

    def renew(self, lease_until_ms: int):
        if type(lease_until_ms) is not int or lease_until_ms <= self.now_ms():
            raise JobError("execution_lease_expired")
        self.lease_until_ms = lease_until_ms
        self.deadline = time.monotonic() + (lease_until_ms - self.now_ms()) / 1000

    def record(self, kind: str, key: str, value: dict):
        if type(kind) is not str or not kind or type(key) is not str or not key or type(value) is not dict:
            raise JobError("invalid_execution_record")
        # Canonicalize before retaining caller input: subsequent mutation cannot
        # change a sealed request, pending checkpoint, or accepted observation.
        encoded = _json(value)
        value = json.loads(encoded)
        identity = (kind, key)
        old = self._records.get(identity)
        if old == value:
            return False
        if old is not None and kind.endswith("_immutable"):
            raise JobError("execution_record_conflict")
        if kind == "budget":
            def unsettled(item):
                return item is not None and (item["pending"] or item["phase"] not in ("released", "settling") or item["ack"] is None)
            self.budget_attempts += old is None
            self.budget_unsettled += int(unsettled(value)) - int(unsettled(old))
        self._records[identity] = value
        self._dirty[identity] = {"kind": kind, "key": key, "value": value}
        self._dirty_bytes += len(encoded.encode()) - self._dirty_sizes.get(identity, 0)
        self._dirty_sizes[identity] = len(encoded.encode())
        return True

    def get(self, kind: str, key: str):
        value = self._records.get((kind, key))
        return None if value is None else copy.deepcopy(value)

    def returned_models(self, round_index):
        return sorted(identity for (kind, identity), value in self._records.items()
                      if kind == "attempt" and value["kind"] == "model"
                      and value["phase"] == "finished" and value["observation"]["status"] == "returned"
                      and value["prepared"]["round_index"] == round_index)

    def begin_attempt(self, attempt_id: str, kind: str, prepared: dict, *, control_only=False):
        if type(control_only) is not bool or control_only and kind != "tool":
            raise JobError("invalid_execution_attempt_kind")
        if control_only:
            if self.closed:
                raise JobError("execution_control_closed")
        else:
            self.require_admission()
        if kind not in ("model", "tool"):
            raise JobError("invalid_execution_attempt_kind")
        previous = self.get("attempt", attempt_id)
        value = {"kind": kind, "prepared": prepared, "phase": "prepared"}
        if control_only:
            value["control_only"] = True
        if previous is not None:
            if (previous.get("kind") != kind or previous.get("prepared") != prepared
                    or bool(previous.get("control_only")) != control_only):
                raise JobError("execution_attempt_conflict")
            return previous
        if kind == "model" and any(a["kind"] == "model" for a in self.active.values()):
            raise JobError("unresolved_prior_attempt")
        if self.models + self.tools >= 1000000:
            raise JobError("execution_attempt_limit")
        self.record("attempt", attempt_id, value)
        self.active[attempt_id] = self.get("attempt", attempt_id)
        if kind == "model":
            self.models += 1
        else:
            self.tools += 1
        return self.get("attempt", attempt_id)

    def dispatch(self, attempt_id: str):
        current = self.get("attempt", attempt_id)
        if current is None:
            raise JobError("execution_attempt_missing")
        if current.get("control_only"):
            if self.closed:
                raise JobError("execution_control_closed")
        else:
            self.require_admission()
        if current["phase"] != "prepared":
            raise JobError("execution_attempt_already_dispatched")
        current["phase"] = "dispatched"
        self.record("attempt", attempt_id, current)
        self.active[attempt_id] = current

    def abort_prepared(self, attempt_id):
        current = self.get("attempt", attempt_id)
        if current is None or current["phase"] not in ("prepared", "aborted"):
            raise JobError("budget_release_not_permitted")
        current["phase"] = "aborted"
        self.record("attempt", attempt_id, current)
        self.active.pop(attempt_id, None)

    def observe(self, attempt_id: str, observation: dict):
        """Accept the original result after stop; never grant a new execution."""
        current = self.get("attempt", attempt_id)
        if current is None or current["phase"] == "prepared":
            raise JobError("execution_attempt_not_dispatched")
        if current["phase"] == "finished":
            if current["observation"] != observation:
                raise JobError("execution_observation_conflict")
            return False
        if observation.get("status") not in ("returned", "unknown"):
            raise JobError("invalid_execution_observation")
        current.update(phase="finished", observation=observation)
        self.record("attempt", attempt_id, current)
        self.active.pop(attempt_id, None)
        self.completed += 1
        self.unknown += observation["status"] == "unknown"
        self._completed_since_flush += 1
        return True

    def summary(self):
        return {"version": 1, "models": self.models, "tools": self.tools,
                "completed": self.completed, "unknown": self.unknown,
                "inflight": len(self.active), "closed": self.closed,
                "cancel_requested": self.cancel_requested,
                "needs_reconciliation": self.needs_reconciliation,
                "pending_obligations": len(self.pending_obligations),
                "budget_attempts": self.budget_attempts, "budget_unsettled": self.budget_unsettled,
                "allocations": [a.summary() for a in self.allocations.values()]}

    def due(self):
        return bool(self._dirty) and (
            self.policy.actions is not None and self._completed_since_flush >= self.policy.actions
            or self._dirty_bytes >= self.policy.bytes
            or time.monotonic() - self._last_flush >= self.policy.interval_s
        )

    async def checkpoint(self, *, force=False, transaction=None, defer=None):
        """With ``defer``, the batch joins ``transaction`` and the head moves on commit."""
        if transaction is None:
            if self._pending is None and (not self._dirty or not force and not self.due()):
                return False
            # Closing a send already holds a database transaction before this
            # lock. Independent flushes use the same order, and retain the lock
            # through commit so another flush cannot reuse an uncommitted head.
            acquired = False
            hooks, undo = [], []
            try:
                async with self.journal.db.transaction() as tx:
                    await self._flush_lock.acquire()
                    acquired = True
                    result = await self._checkpoint_locked(force=force, transaction=tx,
                        defer=lambda commit, rollback: (hooks.append(commit), undo.append(rollback)))
                for hook in hooks:
                    hook()
                return result
            except BaseException:
                for hook in reversed(undo):
                    hook()
                self.checkpoint_failed = True
                raise
            finally:
                if acquired:
                    self._staged_transaction = None
                    self._flush_lock.release()
        if self._staged_transaction is transaction:
            return await self._checkpoint_locked(force=force, transaction=transaction, defer=defer)
        await self._flush_lock.acquire()
        if defer is None:
            try:
                return await self._checkpoint_locked(force=force, transaction=transaction, defer=defer)
            finally:
                self._flush_lock.release()
        self._staged_transaction = transaction
        def release():
            self._staged_transaction = None
            self._flush_lock.release()
        # Keep the memory lock through the enclosing transaction's outcome.
        # This transaction may reenter; another transaction must wait.
        defer(release, release)
        return await self._checkpoint_locked(force=force, transaction=transaction, defer=defer)

    async def _checkpoint_locked(self, *, force, transaction, defer):
        index = len(self._staged_digests)
        if index == len(self._pending_batches):
            if not self._dirty or not force and not self.due():
                return False
            if self.before_checkpoint is not None:
                await self.before_checkpoint(transaction)
            records = list(self._dirty.values())
            summary = self.summary()
            number = (self._pending_batches[-1][0] if self._pending_batches else self.cursor) + 1
            previous = self._staged_digests[-1][1] if self._staged_digests else self.chain_hash
            self._pending_batches.append((number, previous, records, summary))
            self._pending = self._pending_batches[0]
            self._dirty = {}
            self._dirty_sizes = {}
            self._dirty_bytes = 0
            self._completed_since_flush = 0
        number, previous, records, summary = self._pending_batches[index]
        try:
            digest = await self.journal.append(self.identity, number, previous, records, summary, transaction=transaction)
        except BaseException:
            self.checkpoint_failed = True
            raise
        self._staged_digests.append((number, digest))
        self._staged = True
        if defer is None:
            self._applied(number, digest)
        else:
            defer(lambda: self._applied(number, digest), lambda: self._unstage(number))
        return True

    def _applied(self, number, digest):
        self.cursor, self.chain_hash = number, digest
        self._pending_batches.pop(0)
        self._staged_digests.pop(0)
        self._pending = self._pending_batches[0] if self._pending_batches else None
        self._staged = bool(self._staged_digests)
        self.checkpoint_failed = bool(self._pending_batches)
        self._last_flush = time.monotonic()

    def _unstage(self, number):
        # Rollback callbacks run in reverse, including a savepoint rollback.
        # Retain every exact batch so a lost ACK can replay the same chain.
        self._staged_digests.pop()
        self._staged = bool(self._staged_digests)
        self.checkpoint_failed = True

    async def close(self, *, transaction=None, defer=None):
        self.closed = True
        for key, value in list(self.active.items()):
            if value["phase"] == "prepared":
                self.abort_prepared(key)
        self.needs_reconciliation |= bool(self.active or self.unknown or any(a.holds for a in self.allocations.values()))
        self.record("lifecycle", "close", {"closed": True,
                    "needs_reconciliation": self.needs_reconciliation})
        # A previous failed flush may be followed by late observations. Commit
        # both batches in order, rather than dropping the late result on close.
        while self._pending_batches or self._dirty:
            if transaction is not None and len(self._staged_digests) == len(self._pending_batches) and not self._dirty:
                break
            await self.checkpoint(force=True, transaction=transaction, defer=defer)
