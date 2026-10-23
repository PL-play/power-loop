"""Identity of the actual persisted assistant call currently invoking a tool."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_INVOCATION: ContextVar[tuple[str | None, int | None, str | None] | None] = ContextVar(
    "power_loop_tool_invocation", default=None)


@contextmanager
def invocation_context(session_id, assistant_seq, tool_call_id):
    token = _INVOCATION.set((session_id, assistant_seq, tool_call_id))
    try:
        yield
    finally:
        _INVOCATION.reset(token)


def current_invocation(session_id):
    current = _INVOCATION.get()
    # A child's model run inherits ContextVars but must not inherit its parent's
    # tool identity. A real child tool invocation will install its own context.
    return current[1:] if current and current[0] == session_id else (None, None)
