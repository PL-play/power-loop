"""Public error hierarchy for power-loop.

All errors raised by the library inherit from :class:`PowerLoopError` so
callers can ``except PowerLoopError`` as a single catch-all.
"""

from __future__ import annotations


class PowerLoopError(Exception):
    """Base for all power-loop raised exceptions.

    Each subclass carries a stable, machine-readable ``code`` (a dotted string) so
    callers can branch on ``exc.code`` instead of class identity — robust across
    refactors and easy to log/translate (H4.4). ``code`` is a class attribute.
    """

    code: str = "power_loop.error"


class SessionNotFoundError(PowerLoopError):
    """Raised when a ``session_id`` does not exist in the store."""

    code = "session.not_found"

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session not found: {session_id}")
        self.session_id = session_id


class SessionBusy(PowerLoopError):
    """Another owner holds this session; retry on the holder or after release.

    Foreign input is not accepted or parked by the SDK. The application keeps
    the wake-up pending and routes or retries it.
    """

    code = "session.busy"


class InboxClaimChanged(PowerLoopError):
    """The entire claimed batch must be reclaimed before it can enter history."""

    code = "inbox.claim_changed"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"inbox claim changed: {session_id}")


class SessionLeaseLost(PowerLoopError):
    """A previous turn owner cannot write after expiry or takeover."""

    code = "session.lease_lost"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"session lease lost: {session_id}")


class LLMTimeout(PowerLoopError):
    """Raised when an LLM call (or a sequence of retries) exceeds ``total_timeout``.

    ``elapsed`` is the wall-clock seconds spent since the policy started
    counting (before the timeout fired). ``attempts`` is how many physical
    LLM calls were made before giving up.
    """

    code = "llm.timeout"

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

    code = "llm.retry_exhausted"

    def __init__(self, *, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM retry exhausted after {attempts} attempt(s); "
            f"last error: {type(last_error).__name__}: {last_error}"
        )


class LLMNonRetryable(LLMRetryExhausted):
    """The LLM call failed with an error retrying cannot fix (the retry policy's ``give_up_on``
    said so — e.g. HTTP 402 insufficient balance, 401 bad key, 404 unknown model), so it was
    given up after ``attempts`` call(s) instead of burning the whole retry budget.

    A subclass of :class:`LLMRetryExhausted` so every existing ``except LLMRetryExhausted``
    degrades it the same way; the pipeline reports ``reason="non_retryable"``."""

    code = "llm.non_retryable"


class CancellationRequested(PowerLoopError):
    """Raised when a ``CancellationToken`` fires while the loop is awaiting work.

    This is a controlled, expected exit path — callers cancelling via
    ``stop_event.set()`` / ``token.cancel()`` / ``HookDirective.CANCEL`` should
    catch it (or let ``StatefulAgentLoop`` translate it to a ``cancelled``
    result for them).
    """

    code = "cancelled"

    def __init__(self, reason: str = "cancelled") -> None:
        self.reason = reason
        super().__init__(reason)


class CompactionFailed(PowerLoopError):
    """Reserved for the compactor's hard-fail path (M2.5). Soft fail today
    still emits ``compact.failed`` event and continues — this exception is
    only raised when the loop must be aborted because the un-compacted
    history blew the context window."""

    code = "compaction.failed"


class ToolNotFound(PowerLoopError):
    """Raised when a tool name is not registered in the :class:`ToolRegistry`."""

    code = "tool.not_found"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"tool not found: {tool_name}")


class ToolValidationError(PowerLoopError):
    """Raised when tool arguments fail schema / required-param validation."""

    code = "tool.invalid_args"

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"tool {tool_name!r}: {message}")


class SpecValidationError(PowerLoopError):
    """Raised when an :class:`~power_loop.runtime.spec.AgentSpec` fails validation.

    Replaces the previous ``AgentSpecError(ValueError)``. ``field`` is the
    spec key that caused the failure (or ``None`` for a general error).
    """

    code = "spec.invalid"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)
