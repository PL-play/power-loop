"""Metrics event sink (OBS-4): map the typed event stream to counters/histograms/gauges.

The mapping itself is dependency-free — it speaks to a tiny :class:`MetricsBackend`
Protocol, so you can plug in any metrics system (or a fake, in tests). Two backends ship
behind optional extras: :class:`PrometheusBackend` (``power-loop[prometheus]``) and
:class:`StatsDBackend` (``power-loop[statsd]``); both lazy-import their client so the core
stays SDK-free.

Usage::

    from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

    attach_metrics_sink(bus, PrometheusBackend())   # exposes power_loop_* metrics
    # or bring your own MetricsBackend (incr / observe / gauge).

Metrics emitted (prefix ``power_loop`` by default):
- ``…_llm_calls`` (counter, labels: model, success) + ``…_llm_call_duration_ms`` (observe)
- ``…_llm_retries`` (counter)
- ``…_tool_calls`` (counter, labels: tool, success)
- ``…_rounds`` (counter)
- ``…_errors`` (counter, labels: error_type)
- ``…_usage_total_tokens`` (observe, cumulative per round/run)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.core.events import AgentEventBus

__all__ = ["MetricsBackend", "attach_metrics_sink", "PrometheusBackend", "StatsDBackend"]


@runtime_checkable
class MetricsBackend(Protocol):
    """Minimal metrics seam. Implement these three and any backend works."""

    def incr(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None: ...
    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None: ...
    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None: ...


def attach_metrics_sink(
    bus: AgentEventBus,
    backend: MetricsBackend,
    *,
    prefix: str = "power_loop",
    events: Iterable[AgentEventType] | None = None,
) -> None:
    """Subscribe ``backend`` to the metric-bearing events on ``bus``. ``events`` (if
    given) restricts which of the mapped types are wired."""
    wanted = set(events) if events is not None else None

    def _on(t: AgentEventType) -> bool:
        return wanted is None or t in wanted

    def _handler(event: AgentEvent) -> None:
        p = event.payload or {}
        t = event.type
        if not _on(t):
            return
        if t is AgentEventType.LLM_CALL_COMPLETED:
            labels = {"model": str(p.get("model") or ""), "success": _b(p.get("success", True))}
            backend.incr(f"{prefix}_llm_calls", labels=labels)
            backend.observe(f"{prefix}_llm_call_duration_ms", float(p.get("duration_ms") or 0.0), labels)
        elif t is AgentEventType.LLM_RETRY_ATTEMPTED:
            backend.incr(f"{prefix}_llm_retries")
        elif t is AgentEventType.TOOL_CALL_COMPLETED:
            # The pipeline emits TOOL_CALL_FAILED *then* an unconditional
            # TOOL_CALL_COMPLETED for a failed call. Counting COMPLETED here would
            # double-count the failure AND label it success=true; skip it when the
            # payload marks the call failed (the FAILED branch already counted it).
            if not p.get("failed"):
                backend.incr(f"{prefix}_tool_calls", labels={"tool": str(p.get("name") or ""), "success": "true"})
        elif t is AgentEventType.TOOL_CALL_FAILED:
            backend.incr(f"{prefix}_tool_calls", labels={"tool": str(p.get("name") or ""), "success": "false"})
        elif t is AgentEventType.ROUND_COMPLETED:
            backend.incr(f"{prefix}_rounds")
        elif t is AgentEventType.AGENT_ERROR:
            backend.incr(f"{prefix}_errors", labels={"error_type": str(p.get("error_type") or "")})
        elif t is AgentEventType.USAGE_UPDATED:
            total = (p.get("usage") or {}).get("total_tokens")
            if total is not None:
                backend.observe(f"{prefix}_usage_total_tokens", float(total))

    bus.subscribe(None, _handler)


def _b(v: Any) -> str:
    return "true" if v else "false"


# ── shipped backends (behind extras) ────────────────────────────────────────


class PrometheusBackend:
    """`prometheus_client` backend. Lazily creates Counter/Histogram/Gauge per metric
    name (with the label keys seen first). Install via ``power-loop[prometheus]``."""

    def __init__(self, *, registry: Any | None = None) -> None:
        try:
            import prometheus_client  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "PrometheusBackend needs prometheus_client: pip install 'power-loop[prometheus]'"
            ) from exc
        self._prom = prometheus_client
        self._registry = registry
        self._metrics: dict[str, Any] = {}

    def _labels(self, labels: dict[str, str] | None) -> tuple[list[str], dict[str, str]]:
        labels = labels or {}
        return sorted(labels.keys()), labels

    def incr(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        keys, lab = self._labels(labels)
        m = self._metrics.get(name)
        if m is None:
            kw = {"registry": self._registry} if self._registry is not None else {}
            m = self._prom.Counter(name, name, keys, **kw)
            self._metrics[name] = m
        (m.labels(**lab) if keys else m).inc(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        keys, lab = self._labels(labels)
        m = self._metrics.get(name)
        if m is None:
            kw = {"registry": self._registry} if self._registry is not None else {}
            m = self._prom.Histogram(name, name, keys, **kw)
            self._metrics[name] = m
        (m.labels(**lab) if keys else m).observe(value)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        keys, lab = self._labels(labels)
        m = self._metrics.get(name)
        if m is None:
            kw = {"registry": self._registry} if self._registry is not None else {}
            m = self._prom.Gauge(name, name, keys, **kw)
            self._metrics[name] = m
        (m.labels(**lab) if keys else m).set(value)


class StatsDBackend:
    """`statsd` backend. Labels are appended to the metric name (StatsD has no native
    labels). Install via ``power-loop[statsd]``."""

    def __init__(self, *, client: Any | None = None, host: str = "127.0.0.1", port: int = 8125) -> None:
        if client is not None:
            self._client = client
            return
        try:
            import statsd
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "StatsDBackend needs statsd: pip install 'power-loop[statsd]'"
            ) from exc
        self._client = statsd.StatsClient(host, port)

    def _name(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        return name + "." + ".".join(f"{k}_{v}" for k, v in sorted(labels.items()))

    def incr(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        self._client.incr(self._name(name, labels), value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self._client.timing(self._name(name, labels), value)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self._client.gauge(self._name(name, labels), value)
