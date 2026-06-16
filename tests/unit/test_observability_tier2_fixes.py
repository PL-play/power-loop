"""Regression tests for Tier-2 observability fixes:

- C11: in thread dispatch mode, a raising sync subscriber must NOT kill the worker
  (which would silently drop every subsequent event under the default suppress=False).
- C15: the metrics sink must count a failed tool call ONCE (success=false), not also
  as success=true — the pipeline emits TOOL_CALL_FAILED then an unconditional
  TOOL_CALL_COMPLETED (now carrying failed=True).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    ToolDefinition,
    ToolRegistry,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contrib.metrics_sink import attach_metrics_sink
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager

pytestmark = pytest.mark.unit


def _evt(i: int = 0) -> AgentEvent:
    return AgentEvent(type=AgentEventType.SYSTEM_LOG, payload={"i": i})


# ── C11: the thread-mode worker survives a raising subscriber ────────────────


def test_thread_mode_worker_survives_raising_subscriber() -> None:
    # default suppress_subscriber_errors=False → _invoke_handler re-raises
    bus = AgentEventBus(sync_dispatch="thread")
    seen: list[int] = []

    def boom(_e: AgentEvent) -> None:
        raise RuntimeError("subscriber boom")

    bus.subscribe(AgentEventType.SYSTEM_LOG, boom)
    bus.subscribe(AgentEventType.SYSTEM_LOG, lambda e: seen.append(e.payload["i"]))
    try:
        for i in range(5):
            bus.publish(_evt(i))
    finally:
        bus.shutdown()  # flushes the queue before returning

    # Pre-fix the worker died on the first boom and dropped the rest; the second
    # subscriber must still have received EVERY event.
    assert seen == [0, 1, 2, 3, 4]


# ── C15: failed tool call counted once (event-level) ─────────────────────────


@dataclass
class _FakeBackend:
    incrs: list[tuple[str, int, dict]] = field(default_factory=list)

    def incr(self, name, value=1, labels=None):
        self.incrs.append((name, value, labels or {}))

    def observe(self, name, value, labels=None):
        pass

    def gauge(self, name, value, labels=None):
        pass


def _tool_call_success_labels(be: _FakeBackend) -> list[str]:
    return [lab.get("success") for n, _v, lab in be.incrs if n.endswith("_tool_calls")]


def test_metrics_sink_counts_failed_tool_call_once() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    be = _FakeBackend()
    attach_metrics_sink(bus, be)

    # the real pipeline order for a failed call: FAILED then COMPLETED(failed=True)
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_FAILED, payload={"name": "search"}))
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_COMPLETED,
                           payload={"name": "search", "failed": True}))
    assert _tool_call_success_labels(be) == ["false"]  # was ["false", "true"]

    # a successful call still counts exactly one success=true
    be.incrs.clear()
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_COMPLETED,
                           payload={"name": "ok", "failed": False}))
    assert _tool_call_success_labels(be) == ["true"]


# ── C15: end-to-end — a real failing tool through the pipeline ───────────────


@dataclass
class _CallThenStop(LLMService):
    """First response calls the tool; subsequent responses end the run."""

    _i: int = 0

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        self._i += 1
        if self._i == 1:
            return LLMResponse(raw_text="", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "boom", "arguments": "{}"}}])
        return LLMResponse(raw_text="done", content_text="done")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def test_pipeline_failed_tool_metrics_counted_once() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    be = _FakeBackend()
    attach_metrics_sink(bus, be)

    reg = ToolRegistry()

    def _boom(**kwargs):
        raise RuntimeError("tool exploded")

    reg.register(ToolDefinition(name="boom", description="d", input_schema={"type": "object"}), _boom)

    p = AgentPipeline(
        llm=_CallThenStop(), config=AgentLoopConfig(max_rounds=3),
        tool_registry=reg, hooks=AgentHooks(), bus=bus, ctx=ContextManager(),
    )
    await p.run([{"role": "user", "content": "go"}])

    # the failing tool produced exactly ONE tool_calls increment, labeled failure
    assert _tool_call_success_labels(be) == ["false"]
