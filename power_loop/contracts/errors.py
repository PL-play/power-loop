"""Public error hierarchy for power-loop.

All errors raised by the library inherit from :class:`PowerLoopError` so
callers can ``except PowerLoopError`` as a single catch-all.
"""

from __future__ import annotations

from typing import Any


class PowerLoopError(Exception):
    """Base for all power-loop raised exceptions."""


class SessionNotFoundError(PowerLoopError):
    """Raised when a ``session_id`` does not exist in the store."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session not found: {session_id}")
        self.session_id = session_id


class SessionPendingError(PowerLoopError):
    """Raised when a session has unresolved tool_calls from a previous run.

    The previous loop crashed (or was killed) after the assistant emitted
    ``tool_calls`` but before all matching ``tool`` messages were appended.
    The OpenAI/Anthropic message protocol forbids us from sending the next
    LLM request in this state. The caller must explicitly choose:

      * ``StatefulAgentLoop.resume(sid)`` — finish executing the pending
        tool_calls and continue the loop, or
      * ``StatefulAgentLoop.abort_pending(sid)`` — append synthetic
        ``<aborted>`` tool messages, restoring protocol validity, then
        proceed with the new user input.
    """

    def __init__(
        self,
        session_id: str,
        *,
        assistant_seq: int,
        pending_tool_calls: list[dict[str, Any]],
    ) -> None:
        self.session_id = session_id
        self.assistant_seq = assistant_seq
        self.pending_tool_calls = pending_tool_calls
        names = ",".join(
            str((tc.get("function") or {}).get("name") or tc.get("name") or "?")
            for tc in pending_tool_calls
        )
        super().__init__(
            f"session {session_id} has {len(pending_tool_calls)} unresolved tool_calls"
            f" from round (assistant_seq={assistant_seq}): {names}."
            " Recover with resume(sid) to finish them, abort_pending(sid) to"
            " discard them, or call send(..., heal_pending=True) to auto-abort"
            " and proceed"
        )


class LLMTimeout(PowerLoopError):
    """Raised when an LLM call (or a sequence of retries) exceeds ``total_timeout``.

    ``elapsed`` is the wall-clock seconds spent since the policy started
    counting (before the timeout fired). ``attempts`` is how many physical
    LLM calls were made before giving up.
    """

    def __init__(self, *, elapsed: float, attempts: int, total_timeout: float) -> None:
        self.elapsed = elapsed
        self.attempts = attempts
        self.total_timeout = total_timeout
        super().__init__(
            f"LLM call timed out after {elapsed:.2f}s "
            f"(total_timeout={total_timeout:.2f}s, attempts={attempts})"
        )


class LLMRetryExhausted(PowerLoopError):
    """Raised when ``LLMRetryPolicy.max_attempts`` is reached without success.

    ``last_error`` is the exception raised by the most recent attempt — the
    direct cause of the give-up. Always set as ``__cause__`` via ``raise from``.
    """

    def __init__(self, *, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM retry exhausted after {attempts} attempt(s); "
            f"last error: {type(last_error).__name__}: {last_error}"
        )


class CancellationRequested(PowerLoopError):
    """Raised when a ``CancellationToken`` fires while the loop is awaiting work.

    This is a controlled, expected exit path — callers cancelling via
    ``stop_event.set()`` / ``token.cancel()`` / ``HookDirective.CANCEL`` should
    catch it (or let ``StatefulAgentLoop`` translate it to a ``cancelled``
    result for them).
    """

    def __init__(self, reason: str = "cancelled") -> None:
        self.reason = reason
        super().__init__(reason)


class CompactionFailed(PowerLoopError):
    """Reserved for the compactor's hard-fail path (M2.5). Soft fail today
    still emits ``compact.failed`` event and continues — this exception is
    only raised when the loop must be aborted because the un-compacted
    history blew the context window."""


class ToolNotFound(PowerLoopError):
    """Raised when a tool name is not registered in the :class:`ToolRegistry`."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"tool not found: {tool_name}")


class ToolValidationError(PowerLoopError):
    """Raised when tool arguments fail schema / required-param validation."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"tool {tool_name!r}: {message}")


class SpecValidationError(PowerLoopError):
    """Raised when an :class:`~power_loop.runtime.spec.AgentSpec` fails validation.

    Replaces the previous ``AgentSpecError(ValueError)``. ``field`` is the
    spec key that caused the failure (or ``None`` for a general error).
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)
