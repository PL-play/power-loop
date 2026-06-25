"""OBS-5: the OpenTelemetry span bridge builds session→round→llm/tool span trees.

Uses OTel's InMemorySpanExporter (skipped if opentelemetry isn't installed). Pins the
span hierarchy + parenting + error status from the paired lifecycle events.
"""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from power_loop import AgentEvent, AgentEventBus, AgentEventType  # noqa: E402
from power_loop.contrib.otel_sink import attach_otel_sink  # noqa: E402

pytestmark = pytest.mark.unit


def _bus_with_exporter():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    bus = AgentEventBus(suppress_subscriber_errors=True)
    attach_otel_sink(bus, tracer=tracer)
    return bus, exporter


def _ev(t, **payload):
    sid = payload.pop("session_id", "s")
    return AgentEvent(type=t, session_id=sid, payload=payload)


def test_tracer_failure_never_aborts_the_loop() -> None:
    """M-contrib-observability-2: a tracer that raises in span CREATION or set_attribute must NOT
    propagate out of publish() on a non-suppressing bus (the default) and abort the agent run."""

    class _BadSpan:
        def set_attribute(self, *a, **k):
            raise RuntimeError("attr backend down")

        def set_status(self, *a, **k):
            pass

        def end(self, *a, **k):
            pass

    class _BoomTracer:
        def __init__(self):
            self.calls = 0

        def start_span(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("tracer down")  # span CREATION failure
            return _BadSpan()                       # later: set_attribute failure

    bus = AgentEventBus()  # default suppress_subscriber_errors=False
    attach_otel_sink(bus, tracer=_BoomTracer())
    # Pre-fix these re-raised out of publish() (start_span / set_attribute were unguarded).
    bus.publish(_ev(AgentEventType.SESSION_STARTED, scope="main"))
    bus.publish(_ev(AgentEventType.LLM_CALL_STARTED, round_index=0, call_id="c1", model="m"))
    bus.publish(_ev(AgentEventType.LLM_CALL_COMPLETED, call_id="c1", success=True, duration_ms=1.0))


def test_session_round_llm_tool_span_tree() -> None:
    bus, exporter = _bus_with_exporter()
    bus.publish(_ev(AgentEventType.SESSION_STARTED, scope="main"))
    bus.publish(_ev(AgentEventType.ROUND_STARTED, round_index=0))
    bus.publish(_ev(AgentEventType.LLM_CALL_STARTED, round_index=0, call_id="r0.a1", model="m"))
    bus.publish(_ev(AgentEventType.LLM_CALL_COMPLETED, call_id="r0.a1", model="m",
                    success=True, duration_ms=12.0))
    bus.publish(_ev(AgentEventType.TOOL_CALL_STARTED, round_index=0, tool_call_id="tc1", name="search"))
    bus.publish(_ev(AgentEventType.TOOL_CALL_FAILED, tool_call_id="tc1", name="search"))
    bus.publish(_ev(AgentEventType.ROUND_COMPLETED, round_index=0))
    bus.publish(_ev(AgentEventType.SESSION_ENDED, reason="done"))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"power_loop.session", "power_loop.round",
            "power_loop.llm_call", "power_loop.tool_call"} <= set(spans)

    session, rnd = spans["power_loop.session"], spans["power_loop.round"]
    llm, tool = spans["power_loop.llm_call"], spans["power_loop.tool_call"]
    # parenting: round under session; llm + tool under round
    assert rnd.parent.span_id == session.context.span_id
    assert llm.parent.span_id == rnd.context.span_id
    assert tool.parent.span_id == rnd.context.span_id
    # attributes + error status
    assert llm.attributes["model"] == "m" and llm.attributes["duration_ms"] == 12.0
    assert tool.status.status_code == trace.StatusCode.ERROR  # TOOL_CALL_FAILED


def test_close_flushes_open_spans() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    bus = AgentEventBus(suppress_subscriber_errors=True)
    bridge = attach_otel_sink(bus, tracer=provider.get_tracer("t"))
    bus.publish(_ev(AgentEventType.SESSION_STARTED))
    bus.publish(_ev(AgentEventType.ROUND_STARTED, round_index=0))
    assert exporter.get_finished_spans() == ()  # still open
    bridge.close()
    names = {s.name for s in exporter.get_finished_spans()}
    assert {"power_loop.session", "power_loop.round"} <= names  # close ended them
