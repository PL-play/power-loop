"""Prepare shared immutable policy originals before a single-tree writer."""
from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .jobs import JobError

_SOURCES = ContextVar('budget_policy_sources', default=None)


@dataclass(frozen=True)
class PolicySource:
    marker: dict | None


class PolicySourceChanged(JobError):
    def __init__(self, request_hash):
        self.request_hash = request_hash
        self.source = None
        super().__init__('budget_policy_source_changed')


def current(runtime):
    return (_SOURCES.get() or {}).get(id(runtime))


async def _read(runtime):
    from .model_budgets import policy_key
    marker_id = 'mb_policy_'+policy_key(runtime)[3:]
    return await runtime.manager.db.fetchone(
        f'SELECT payload_json,request_hash FROM {runtime.manager.events} WHERE event_id=?', (marker_id,))


@asynccontextmanager
async def prepared(runtime, *, source=None):
    existing = _SOURCES.get() or {}
    identity = id(runtime)
    if identity in existing:
        yield
        return
    manager = runtime.manager
    if manager._control_intents.get() is not None:
        raise JobError('budget_policy_source_required_before_writer')
    if source is None:
        from .execution_membership import _trusted
        from .model_budgets import policy_key
        known = _trusted(manager, '_budget_markers', policy_key(runtime))
        marker = {'payload_json': None, 'request_hash': known} if known else await _read(runtime)
        source = PolicySource(marker)
    token = _SOURCES.set({**existing, identity: source})
    try:
        yield
    except PolicySourceChanged as changed:
        # The writer has rolled back before this outer scope exits. Resolve
        # a concurrent first policy here;missing originals remain an error.
        marker = await _read(runtime)
        if marker is None or marker['request_hash'] != changed.request_hash:
            raise JobError('budget_policy_evidence_corrupt') from changed
        changed.source = PolicySource(marker)
        raise
    finally:
        _SOURCES.reset(token)
