"""Direct model calls made inside a task-managed send.

Such a call goes through the send's own admission, budget, response capture and
observation, booked through the original task. Without the matching send binding it is
refused, never sent unmanaged. Fold usage is not added to the send's result usage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any

from power_loop._vendor.llm_client.interface import LLMTokenUsage
from power_loop.contracts.errors import CancellationRequested
from power_loop.contracts.event_payloads import LlmCallCompletedPayload, LlmCallStartedPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.jobs import JobError
from power_loop.runtime.usage_estimate import estimate_completion_tokens, estimate_prompt_tokens

log = logging.getLogger(__name__)


class ModelCalls:
    """The model calls of one fold. The first failure is sticky: a strategy's own
    fallback must not buy another call after a failed or unconfirmed one."""

    def __init__(self, *, session_id: str, bus: Any = None, cancel_token: Any = None, purpose: str | None = None) -> None:
        if purpose not in (None, 'fold'):
            raise JobError('model_call_purpose_invalid')
        self.purpose = purpose
        self.session_id = session_id
        self.bus = bus
        self.cancel_token = None if cancel_token is None else CancellationToken.from_any(cancel_token)
        self.failure: BaseException | None = None
        self.calls = 0

    def wrap(self, llm: Any) -> Any:
        return None if llm is None else ManagedModelLLM(llm, self)

    def raise_if_cancelled(self) -> None:
        """Strategies swallow model errors; a cancellation must still reach the send."""
        if isinstance(self.failure, (CancellationRequested, asyncio.CancelledError)):
            raise self.failure

    def emit(self, event_type: AgentEventType, data: Any) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish(AgentEvent(type=event_type, data=data, session_id=self.session_id,
                                        round_index=0, stream_id=self.purpose or 'llm'))
        except Exception as exc:
            log.warning("fold call observer failed: %s", type(exc).__name__)


class ManagedModelLLM:
    def __init__(self, inner: Any, calls: ModelCalls) -> None:
        self._inner = inner
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def complete(self, request: Any, **kwargs: Any) -> Any:
        if self._calls.failure is not None:
            raise self._calls.failure
        try:
            return await self._complete(request, **kwargs)
        except BaseException as exc:
            self._calls.failure = exc
            raise

    async def _complete(self, request: Any, **kwargs: Any) -> Any:
        from power_loop._vendor.llm_client.attempt_scope import single_attempt
        from power_loop.core.pipeline import _failure_outcome, _llm_endpoint
        from power_loop.runtime.model_attempts import (
            event_identity,
            observe,
            provider_deadline,
            response_observation,
            start_call,
        )
        from power_loop.runtime.response_capture import record
        from power_loop.runtime.task_runtime import _SEND, check_dispatch
        from power_loop.workflow.admission import request_slot

        calls, sid = self._calls, self._calls.session_id
        binding = _SEND.get()
        if binding is None or binding.actor.session_id != sid:
            raise JobError(f"{calls.purpose or 'llm'}_call_unmanaged")
        if calls.cancel_token is not None:
            calls.cancel_token.raise_if_cancelled()
        model = (getattr(request, "model", None)
                 or getattr(getattr(self._inner, "capabilities", None), "model", None) or "")
        async with request_slot(calls.cancel_token):
            await check_dispatch(sid)
            attempt = await start_call(SimpleNamespace(session_id=sid, llm=self._inner), request,
                                       round_index=0, retry_index=1, model=model, purpose=calls.purpose)
            if attempt is None:
                raise JobError(f"{calls.purpose or 'llm'}_call_unmanaged")
            identity = event_identity(attempt)
            calls.calls += 1
            call_id = f"{calls.purpose or 'llm'}.{calls.calls}"
            base = dict(call_id=call_id, round_index=0, attempt=1, model=model, purpose=calls.purpose or 'chat', **identity)
            calls.emit(AgentEventType.LLM_CALL_STARTED, LlmCallStartedPayload(**base))
            t0 = time.perf_counter()
            policy_clock = None
            provider_started = False
            try:
                token = single_attempt.set(True)
                try:
                    deadline = await provider_deadline(attempt)
                    if deadline is not None and deadline <= asyncio.get_running_loop().time():
                        raise JobError("owner_wait_deadline_exceeded")
                    policy_clock = asyncio.timeout_at(deadline)
                    async with policy_clock:
                        provider_started = True
                        response = await self._inner.complete(request, **kwargs)
                finally:
                    single_attempt.reset(token)
            except BaseException as exc:
                error = exc
                if isinstance(exc, TimeoutError) and policy_clock is not None and policy_clock.expired():
                    error = JobError("owner_wait_deadline_exceeded")
                prompt = estimate_prompt_tokens(messages=request.messages, system_prompt=request.system_prompt,
                                                tools=request.tools) if provider_started else 0
                await observe(attempt, usage={"prompt_tokens": prompt, "completion_tokens": 0,
                                              "total_tokens": prompt}, estimated=True, error=error)
                calls.emit(AgentEventType.LLM_CALL_COMPLETED, LlmCallCompletedPayload(
                    **base, duration_ms=(time.perf_counter() - t0) * 1000.0, success=False,
                    error_type=type(error).__name__, prompt_tokens=prompt, completion_tokens=0,
                    total_tokens=prompt, estimated=True, outcome=_failure_outcome(error),
                    endpoint=_llm_endpoint(self._inner)))
                if error is not exc:
                    raise error from exc
                raise
            usage = getattr(response, "token_usage", None)
            reported = response_observation(usage, response)
            estimated = usage is None or getattr(usage, "prompt_tokens", None) is None
            if estimated:
                prompt = estimate_prompt_tokens(messages=request.messages, system_prompt=request.system_prompt,
                                                tools=request.tools)
                completion = estimate_completion_tokens(
                    text=getattr(response, "raw_text", "") or "", think=getattr(response, "think", "") or "",
                    tool_calls=getattr(response, "tool_calls", None) or ())
                usage = LLMTokenUsage(prompt_tokens=prompt, completion_tokens=completion,
                                      total_tokens=prompt + completion)
                response.token_usage = usage
            await record(attempt, reported=reported, observation=response_observation(usage, response),
                         estimated=estimated)
            calls.emit(AgentEventType.LLM_CALL_COMPLETED, LlmCallCompletedPayload(
                **base, duration_ms=(time.perf_counter() - t0) * 1000.0, success=True,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                prompt_cached_tokens=getattr(usage, "prompt_cached_tokens", None),
                prompt_cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None),
                estimated=estimated, endpoint=_llm_endpoint(self._inner),
                provider_cost=getattr(usage, "cost", None),
                prompt_image_tokens=getattr(usage, "prompt_image_tokens", None)))
            if policy_clock is not None and policy_clock.expired():
                raise JobError("owner_wait_deadline_exceeded")
            return response
