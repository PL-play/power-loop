"""OBS-4: the metrics sink maps the typed event stream to counter/observe/gauge calls.

Tested against a fake in-memory MetricsBackend — no Prometheus/StatsD dependency — so it
pins the *mapping*, which is the part that can regress. The shipped backends just forward
to their client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType
from power_loop.contrib.metrics_sink import MetricsBackend, attach_metrics_sink

pytestmark = pytest.mark.unit


@dataclass
class _FakeBackend:
    incrs: list[tuple[str, int, dict]] = field(default_factory=list)
    observes: list[tuple[str, float, dict]] = field(default_factory=list)
    gauges: list[tuple[str, float, dict]] = field(default_factory=list)

    def incr(self, name, value=1, labels=None):
        self.incrs.append((name, value, labels or {}))

    def observe(self, name, value, labels=None):
        self.observes.append((name, value, labels or {}))

    def gauge(self, name, value, labels=None):
        self.gauges.append((name, value, labels or {}))


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(_FakeBackend(), MetricsBackend)


def test_backend_failure_never_aborts_the_loop() -> None:
    """H7 (BUG_REVIEW_3.4): a raising backend on a non-suppressing bus (the default, incl.
    DEFAULT_EVENT_BUS) must NOT propagate out of publish() and abort the agent run."""

    class _BoomBackend:
        def incr(self, *a, **k):
            raise RuntimeError("statsd socket down")

        def observe(self, *a, **k):
            raise RuntimeError("statsd socket down")

        def gauge(self, *a, **k):
            raise RuntimeError("statsd socket down")

    bus = AgentEventBus()  # default suppress_subscriber_errors=False
    attach_metrics_sink(bus, _BoomBackend())
    # Pre-fix this re-raised RuntimeError out of publish(); now it's logged and swallowed.
    bus.publish(AgentEvent(type=AgentEventType.ROUND_COMPLETED, payload={}))
    bus.publish(AgentEvent(type=AgentEventType.LLM_CALL_COMPLETED,
                           payload={"model": "m", "success": True, "duration_ms": 1.0}))


def test_event_to_metric_mapping() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    be = _FakeBackend()
    attach_metrics_sink(bus, be)

    bus.publish(AgentEvent(type=AgentEventType.LLM_CALL_COMPLETED,
                           payload={"model": "m1", "success": True, "duration_ms": 42.0}))
    bus.publish(AgentEvent(type=AgentEventType.LLM_RETRY_ATTEMPTED, payload={}))
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_COMPLETED, payload={"name": "search"}))
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_FAILED, payload={"name": "search"}))
    bus.publish(AgentEvent(type=AgentEventType.ROUND_COMPLETED, payload={}))
    bus.publish(AgentEvent(type=AgentEventType.AGENT_ERROR, payload={"error_type": "ValueError"}))
    bus.publish(AgentEvent(type=AgentEventType.USAGE_UPDATED, payload={"usage": {"total_tokens": 123}}))

    incr_names = {(n, tuple(sorted(lab.items()))) for n, _v, lab in be.incrs}
    assert ("power_loop_llm_calls", (("model", "m1"), ("success", "true"))) in incr_names
    assert ("power_loop_llm_retries", ()) in incr_names
    assert ("power_loop_tool_calls", (("success", "true"), ("tool", "search"))) in incr_names
    assert ("power_loop_tool_calls", (("success", "false"), ("tool", "search"))) in incr_names
    assert ("power_loop_rounds", ()) in incr_names
    assert ("power_loop_errors", (("error_type", "ValueError"),)) in incr_names

    obs_names = {n for n, _v, _l in be.observes}
    assert "power_loop_llm_call_duration_ms" in obs_names
    assert "power_loop_usage_total_tokens" in obs_names
    dur = next(v for n, v, _l in be.observes if n == "power_loop_llm_call_duration_ms")
    assert dur == 42.0


def test_events_filter_restricts_mapping() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    be = _FakeBackend()
    attach_metrics_sink(bus, be, events={AgentEventType.ROUND_COMPLETED})
    bus.publish(AgentEvent(type=AgentEventType.ROUND_COMPLETED, payload={}))
    bus.publish(AgentEvent(type=AgentEventType.AGENT_ERROR, payload={"error_type": "X"}))
    names = {n for n, _v, _l in be.incrs}
    assert names == {"power_loop_rounds"}  # AGENT_ERROR filtered out


def test_custom_prefix() -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    be = _FakeBackend()
    attach_metrics_sink(bus, be, prefix="myapp")
    bus.publish(AgentEvent(type=AgentEventType.ROUND_COMPLETED, payload={}))
    assert be.incrs[0][0] == "myapp_rounds"


# ── shipped backends ─────────────────────────────────────────────────────────


def test_prometheus_backend_records() -> None:
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry

    from power_loop.contrib.metrics_sink import PrometheusBackend

    reg = CollectorRegistry()
    be = PrometheusBackend(registry=reg)
    be.incr("pl_calls", labels={"model": "m", "success": "true"})
    be.incr("pl_calls", labels={"model": "m", "success": "true"})
    be.observe("pl_dur_ms", 5.0, labels={"model": "m"})
    be.gauge("pl_active", 3.0)
    # counter value read back through the registry
    val = reg.get_sample_value("pl_calls_total", {"model": "m", "success": "true"})
    assert val == 2.0
    assert reg.get_sample_value("pl_active") == 3.0


@dataclass
class _FakeStatsd:
    calls: list = field(default_factory=list)

    def incr(self, name, value=1):
        self.calls.append(("incr", name, value))

    def timing(self, name, value):
        self.calls.append(("timing", name, value))

    def gauge(self, name, value):
        self.calls.append(("gauge", name, value))


def test_statsd_backend_folds_labels_into_name() -> None:
    from power_loop.contrib.metrics_sink import StatsDBackend

    fake = _FakeStatsd()
    be = StatsDBackend(client=fake)
    be.incr("pl_calls", labels={"model": "m", "success": "true"})
    be.observe("pl_dur", 7.0)
    assert fake.calls[0] == ("incr", "pl_calls.model_m.success_true", 1)
    assert fake.calls[1] == ("timing", "pl_dur", 7.0)
