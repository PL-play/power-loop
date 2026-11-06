"""Read-only SDK execution identity and explicitly allowed host extensions."""
from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class ExecutionContext:
    store_id: str
    session_id: str
    node_id: str
    boot_id: str
    input_ref: object = None
    job_ref: object = None
    parent_job_id: str | None = None
    root_job_id: str | None = None
    owner_epoch: int = 0
    owner_token: str | None = None
    actor: object = None
    options: object = None
    resources: object = None
    cancel_token: object = None
    host_context: Mapping = field(default_factory=lambda: MappingProxyType({}))


_CURRENT_EXECUTION = ContextVar("power_loop_execution_context", default=None)
_HOST_CONTEXT = ContextVar("power_loop_host_context", default=None)
_HOST_FIELDS = ContextVar("power_loop_host_fields", default=None)


def get_execution_context(required=True):
    provider = _CURRENT_EXECUTION.get()
    if provider is None:
        if required:
            raise RuntimeError("execution context is not bound")
        return None
    return replace(provider(), host_context=_HOST_CONTEXT.get())


@contextmanager
def _bind_execution_context(provider, host_fields=None):
    tokens = [(var, var.set(value)) for var, value in (
        (_CURRENT_EXECUTION, provider), (_HOST_CONTEXT, MappingProxyType({})),
        (_HOST_FIELDS, {k: frozenset(v) for k, v in (host_fields or {}).items()}))]
    try:
        yield get_execution_context()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@contextmanager
def bind_host_context(namespace, values):
    """Bind JSON host fields for this scope; SDK identity is never writable."""
    get_execution_context()
    allowed = (_HOST_FIELDS.get() or {}).get(namespace)
    if allowed is None or not set(values) <= allowed:
        raise ValueError("host context namespace or field is not allowed")
    frozen = _freeze(json.loads(json.dumps(dict(values), allow_nan=False)))
    token = _HOST_CONTEXT.set(MappingProxyType({**_HOST_CONTEXT.get(), namespace: frozen}))
    try:
        yield get_execution_context().host_context[namespace]
    finally:
        _HOST_CONTEXT.reset(token)


@contextmanager
def independent_execution_scope():
    """Remove inherited call authority before entering another session."""
    from power_loop.core import agent_context as ctx
    from power_loop.runtime.invocation import _INVOCATION
    from power_loop.runtime.session_ownership import _CURRENT
    from power_loop.runtime.store.send_reads import ACTIVE
    from power_loop.runtime.task_runtime import _OWNER, _SEND, _TASK_OWNERS

    values = ((_CURRENT, None), (_SEND, None), (_OWNER, None), (_TASK_OWNERS, ()),
              (_INVOCATION, None), (ACTIVE, None), (_CURRENT_EXECUTION, None),
              (_HOST_CONTEXT, None), (_HOST_FIELDS, None), (ctx._current_cancel_token, None),
              (ctx._current_loop, None), (ctx._current_ctx, None), (ctx._current_session_id, None),
              (ctx._effective_tools, None))
    tokens = [(var, var.set(value)) for var, value in values]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
