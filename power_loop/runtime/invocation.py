"""Identity of the actual persisted assistant call currently invoking a tool."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

_INVOCATION: ContextVar[tuple[str | None, int | None, str | None] | None] = ContextVar(
    "power_loop_tool_invocation", default=None
)


_SCOPE_FACTORIES = {}


def register_invocation_scope(name, factory, *, exporter=None):
    """Install a trusted host context manager around each actual invocation."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("invalid invocation scope provider")
    _SCOPE_FACTORIES[name] = (factory, exporter)


@contextmanager
def invocation_context(session_id, assistant_seq, tool_call_id, tool_name=None):
    token = _INVOCATION.set((session_id, assistant_seq, tool_call_id))
    try:
        with ExitStack() as scopes:
            for factory, _exporter in tuple(_SCOPE_FACTORIES.values()):
                scopes.enter_context(
                    factory(
                        session_id=session_id,
                        assistant_seq=assistant_seq,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
                )
            yield
    finally:
        _INVOCATION.reset(token)


def current_invocation(session_id):
    current = _INVOCATION.get()
    # A child's model run inherits ContextVars but must not inherit its parent's
    # tool identity. A real child tool invocation will install its own context.
    return current[1:] if current and current[0] == session_id else (None, None)


def capture_host_contexts():
    """Audit-only host snapshots; never part of task authority or request input."""
    import json

    values = {}
    for name, (_factory, exporter) in tuple(_SCOPE_FACTORIES.items()):
        value = exporter() if exporter is not None else None
        if value is not None:
            values[name] = json.loads(json.dumps(value, allow_nan=False))
    return values
