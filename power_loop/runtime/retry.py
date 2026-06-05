"""LLM retry / total-timeout policy.

The policy is intentionally **library-internal** in scope: it only wraps
``LLMService.complete`` (see :meth:`AgentPipeline.call_llm`). Tools, hooks,
and message I/O do not retry — they fail loudly because they are usually
domain logic where silent retry hides bugs.

Behaviour
---------
``with_retry(call, *, policy, emit, token)``:

1. Computes a wall-clock deadline = ``now + policy.total_timeout`` (if set).
2. Loops up to ``policy.max_attempts``::

     try:
         return await call()
     except policy.retry_on as exc:
         emit(LLM_RETRY_ATTEMPTED, …)
         if last_attempt or deadline_exceeded or cancelled:
             stop
         await sleep(backoff)
   3. On *retryable* exhaustion → ``raise LLMRetryExhausted(...)``.
3. On deadline crossed → ``raise LLMTimeout(...)``.
4. On token cancel between attempts → ``raise CancellationRequested(...)``.

The caller is expected to catch ``LLMRetryExhausted`` / ``LLMTimeout`` and
**degrade** the loop (status=``"degraded"``, append a synthetic assistant
message explaining the failure). ``CancellationRequested`` is the same
exit path as a user-requested stop and is translated by the pipeline to
status=``"cancelled"``.

The retry sleep itself is async-cancellable: if ``token`` flips during the
backoff sleep, we raise ``CancellationRequested`` immediately rather than
waiting the full backoff.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from power_loop.contracts.errors import CancellationRequested, LLMRetryExhausted, LLMTimeout
from power_loop.runtime.cancellation import CancellationToken

T = TypeVar("T")


def _default_retryable() -> tuple[type[BaseException], ...]:
    """Default retryable error set.

    We catch ``Exception`` minus ``CancellationRequested`` minus
    ``asyncio.CancelledError`` minus ``KeyboardInterrupt`` (the last two
    are already excluded because they're ``BaseException``, not
    ``Exception``). This is intentionally broad: LLM SDKs differ wildly
    in their exception taxonomies (httpx errors, openai.RateLimitError,
    anthropic.APITimeoutError, plain ``TimeoutError``, json decode
    errors from malformed stream chunks…). Callers wanting tighter
    control pass an explicit tuple.
    """
    return (Exception,)


@dataclass
class LLMRetryPolicy:
    """Retry policy for ``LLMService.complete`` calls.

    Attributes
    ----------
    max_attempts:
        Total physical LLM calls including the first. ``1`` disables retry.
    backoff_initial:
        Sleep before the second attempt, in seconds. Each subsequent attempt
        doubles, capped at ``backoff_max``.
    backoff_max:
        Cap for the exponential backoff sleep.
    total_timeout:
        Wall-clock seconds across all attempts (including backoff sleeps).
        ``None`` means no overall deadline; ``max_attempts`` is the only
        stop condition.
    retry_on:
        Exception types that trigger a retry. Anything outside this tuple
        bubbles up immediately (and never counts as a retry).
    """

    max_attempts: int = 3
    backoff_initial: float = 0.5
    backoff_max: float = 8.0
    total_timeout: float | None = 60.0
    retry_on: tuple[type[BaseException], ...] = field(default_factory=_default_retryable)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("LLMRetryPolicy.max_attempts must be >= 1")
        if self.backoff_initial < 0 or self.backoff_max < 0:
            raise ValueError("LLMRetryPolicy backoff fields must be >= 0")
        if self.total_timeout is not None and self.total_timeout <= 0:
            raise ValueError("LLMRetryPolicy.total_timeout must be > 0 or None")

    def backoff_for_attempt(self, attempt: int) -> float:
        """Seconds to sleep before attempt index ``attempt`` (0-based).

        ``attempt=0`` is the first call: no sleep. ``attempt=1`` sleeps
        ``backoff_initial``; ``attempt=2`` doubles, etc., capped at
        ``backoff_max``.
        """
        if attempt <= 0:
            return 0.0
        backoff = self.backoff_initial * (2 ** (attempt - 1))
        return min(backoff, self.backoff_max)


RetryAttemptCallback = Callable[[int, BaseException, float], None]
"""(attempt_index_just_failed, exception, next_sleep_seconds) -> None"""


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: LLMRetryPolicy,
    token: CancellationToken,
    on_retry: RetryAttemptCallback | None = None,
) -> T:
    """Drive ``call()`` under ``policy``; honour ``token``.

    See module docstring for full semantics.
    """
    start = time.monotonic()
    deadline = start + policy.total_timeout if policy.total_timeout is not None else None
    last_error: BaseException | None = None

    for attempt in range(policy.max_attempts):
        token.raise_if_cancelled()

        if deadline is not None and time.monotonic() >= deadline:
            raise LLMTimeout(
                elapsed=time.monotonic() - start,
                attempts=attempt,
                total_timeout=policy.total_timeout or 0.0,
            )

        try:
            return await call()
        except CancellationRequested:
            raise
        except asyncio.CancelledError:
            raise
        except policy.retry_on as exc:
            last_error = exc
            is_last = attempt == policy.max_attempts - 1
            if is_last:
                break

            sleep_for = policy.backoff_for_attempt(attempt + 1)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMTimeout(
                        elapsed=time.monotonic() - start,
                        attempts=attempt + 1,
                        total_timeout=policy.total_timeout or 0.0,
                    ) from exc
                sleep_for = min(sleep_for, remaining)

            if on_retry is not None:
                try:
                    on_retry(attempt, exc, sleep_for)
                except Exception:
                    # Hook errors must not corrupt the retry loop.
                    pass

            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            token.raise_if_cancelled()

    assert last_error is not None  # loop only exits via return / raise / last_error set
    raise LLMRetryExhausted(attempts=policy.max_attempts, last_error=last_error) from last_error


__all__ = ["LLMRetryPolicy", "RetryAttemptCallback", "with_retry"]
