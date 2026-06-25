"""OpenTelemetry span bridge (OBS-5): turn the paired lifecycle events into spans.

Maps the already-emitted ``*_STARTED`` / ``*_COMPLETED`` event pairs into a span tree —
``session`` → ``round`` → (``llm_call`` | ``tool_call``) — so a run shows up as a trace in
any OTel backend (Jaeger, Tempo, an OTLP collector, …). Span durations use the event
envelope's monotonic clock semantics (OBS-6).

Behind the ``power-loop[otel]`` extra; ``opentelemetry`` is imported lazily so this module
is importable without it.

Usage::

    from power_loop.contrib.otel_sink import attach_otel_sink
    bridge = attach_otel_sink(bus)          # uses the global tracer provider
    ...                                      # run the loop → spans are exported
    bridge.close()                           # end any spans still open
"""

from __future__ import annotations

import logging
from typing import Any

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.core.events import AgentEventBus

__all__ = ["attach_otel_sink", "OtelSpanBridge"]

logger = logging.getLogger(__name__)


class OtelSpanBridge:
    """Holds the open spans and the event→span mapping. One per bus."""

    def __init__(self, tracer: Any) -> None:
        from opentelemetry import trace  # lazy

        self._trace = trace
        self._tracer = tracer
        self._session: dict[str | None, Any] = {}
        self._round: dict[tuple[str | None, int], Any] = {}
        self._call: dict[tuple[str | None, str], Any] = {}
        self._tool: dict[tuple[str | None, str], Any] = {}

    def _start(self, name: str, parent: Any, attrs: dict[str, Any]) -> Any:
        ctx = self._trace.set_span_in_context(parent) if parent is not None else None
        return self._tracer.start_span(name, context=ctx, attributes=attrs)

    def handle(self, event: AgentEvent) -> None:
        # Observability must NEVER break the loop (M-contrib-observability-2): span CREATION
        # (start_span) and set_attribute were unguarded, unlike _end/_error — a tracer that raises
        # there would unwind through publish() on a non-suppressing bus and abort the agent run.
        # Wrap the whole dispatch in a log-and-swallow guard.
        try:
            self._handle(event)
        except Exception:  # noqa: BLE001 — a tracing hiccup is a dropped span, not a failed run
            logger.exception("otel sink failed; dropping this span event and continuing")

    def _handle(self, event: AgentEvent) -> None:
        t = event.type
        sid = event.session_id
        p = event.payload or {}
        if t is AgentEventType.SESSION_STARTED:
            self._session[sid] = self._start(
                "power_loop.session", None, {"session_id": sid or "", "scope": str(p.get("scope") or "")}
            )
        elif t is AgentEventType.SESSION_ENDED:
            self._end(self._session.pop(sid, None))
        elif t is AgentEventType.ROUND_STARTED:
            ri = int(p.get("round_index") or 0)
            self._round[(sid, ri)] = self._start(
                "power_loop.round", self._session.get(sid), {"round_index": ri}
            )
        elif t is AgentEventType.ROUND_COMPLETED:
            ri = int(p.get("round_index") or 0)
            self._end(self._round.pop((sid, ri), None))
        elif t is AgentEventType.LLM_CALL_STARTED:
            ri = int(p.get("round_index") or 0)
            parent = self._round.get((sid, ri)) or self._session.get(sid)
            self._call[(sid, str(p.get("call_id") or ""))] = self._start(
                "power_loop.llm_call", parent,
                {"model": str(p.get("model") or ""), "attempt": int(p.get("attempt") or 1)},
            )
        elif t is AgentEventType.LLM_CALL_COMPLETED:
            span = self._call.pop((sid, str(p.get("call_id") or "")), None)
            if span is not None:
                span.set_attribute("duration_ms", float(p.get("duration_ms") or 0.0))
                span.set_attribute("success", bool(p.get("success", True)))
                if not p.get("success", True):
                    self._error(span, str(p.get("error_type") or "llm_error"))
                self._end(span)
        elif t is AgentEventType.TOOL_CALL_STARTED:
            ri = int(p.get("round_index") or event.round_index or 0)
            parent = self._round.get((sid, ri)) or self._session.get(sid)
            self._tool[(sid, str(p.get("tool_call_id") or ""))] = self._start(
                "power_loop.tool_call", parent, {"tool": str(p.get("name") or "")}
            )
        elif t in (AgentEventType.TOOL_CALL_COMPLETED, AgentEventType.TOOL_CALL_FAILED):
            span = self._tool.pop((sid, str(p.get("tool_call_id") or "")), None)
            if span is not None:
                if t is AgentEventType.TOOL_CALL_FAILED:
                    self._error(span, "tool_error")
                self._end(span)

    def _error(self, span: Any, desc: str) -> None:
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, desc))
        except Exception:  # pragma: no cover - never let observability break the loop
            pass

    def _end(self, span: Any) -> None:
        if span is not None:
            try:
                span.end()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        """End any spans still open (e.g. a session that never emitted SESSION_ENDED)."""
        for store in (self._call, self._tool, self._round, self._session):
            for span in list(store.values()):
                self._end(span)
            store.clear()


def attach_otel_sink(bus: AgentEventBus, *, tracer: Any | None = None) -> OtelSpanBridge:
    """Bridge ``bus`` events to OpenTelemetry spans. ``tracer`` defaults to the global
    provider's tracer. Returns the :class:`OtelSpanBridge`; call ``.close()`` to end any
    spans left open. Requires ``power-loop[otel]``."""
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "attach_otel_sink needs opentelemetry: pip install 'power-loop[otel]'"
        ) from exc
    bridge = OtelSpanBridge(tracer or trace.get_tracer("power_loop"))
    bus.subscribe(None, bridge.handle)
    return bridge
