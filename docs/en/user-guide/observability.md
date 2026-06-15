# Observability

[中文](../../zh/user-guide/observability.md) | [User Guide](../index.md)

Every lifecycle moment is a typed [event](events.md). This page is about getting those
events **out** — durably, with metrics, and as traces — and about not letting a sink
stall the agent loop.

## The event envelope

Each `AgentEvent` carries an envelope alongside its typed payload:

| Field | Meaning |
|---|---|
| `seq` | process-wide monotonic sequence — a total order across every session/sub-agent |
| `ts` | wall-clock creation time (epoch seconds) — readable/exportable, but non-monotonic |
| `mono` | `perf_counter` seconds — use this for durations/latency (survives NTP/clock steps) |

`event.to_dict()` / `AgentEvent.from_dict(d)` round-trip the whole envelope. `from_dict`
preserves a serialized `seq`/`ts`/`mono` exactly and does **not** advance the process
counter — so replayed events keep their original order.

## Durable events + replay

The in-process bus is ephemeral. Persist events to a rotating JSONL file and read them
back later:

```python
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay

sink = attach_jsonl_sink(bus, "events.jsonl")   # all events; or events={...}
...                                              # run the loop
sink.close()

for event in replay("events.jsonl"):             # oldest→newest, across rotations
    print(event.seq, event.type, event.payload)
```

Payloads are truncated + secret-redacted (shared policy with the logging sink; pass
`redact_keys=()` to disable). Rotation is size-based (`max_bytes`, `backup_count`).

The stdlib **logging sink** (`attach_logging_sink`) emits one JSON line per event
(now including `seq`/`ts`) to a `logging.Logger` — pipe it into jq/Loki/CloudWatch.

## Metrics

`attach_metrics_sink` maps the typed events to counters/observations through a tiny
`MetricsBackend` Protocol (`incr` / `observe` / `gauge`) — bring your own, or use a
shipped backend:

```python
from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

attach_metrics_sink(bus, PrometheusBackend())    # power-loop[prometheus]
# or StatsDBackend()                              # power-loop[statsd]
```

Emitted (prefix `power_loop`): `…_llm_calls` (+`…_llm_call_duration_ms`), `…_llm_retries`,
`…_tool_calls` (labels: tool, success), `…_rounds`, `…_errors`, `…_usage_total_tokens`.
The mapping is dependency-free; only the shipped backends import their client (lazily).

## Tracing (OpenTelemetry)

`attach_otel_sink` turns the paired `*_STARTED`/`*_COMPLETED` events into a span tree —
`session` → `round` → `llm_call` / `tool_call` — exportable to any OTel backend:

```python
from power_loop.contrib.otel_sink import attach_otel_sink   # power-loop[otel]

bridge = attach_otel_sink(bus)     # uses the global tracer provider
...                                 # run the loop → spans exported
bridge.close()                      # end any spans still open
```

Spans carry `model` / `duration_ms` / `success`; a failed tool/LLM call sets an error
status. `opentelemetry` is imported lazily, so this module is importable without the extra.

## Backpressure — don't block the loop

A **synchronous** subscriber runs inline on the publishing thread (usually the agent's
event loop), so it **must be fast**. For slow work, prefer an `async def` handler (it is
scheduled as a task and never blocks the loop). If you must run slow *sync* sinks, opt
into a background dispatcher:

```python
bus = AgentEventBus(sync_dispatch="thread", queue_maxsize=1000, on_overflow="drop_newest")
...
bus.shutdown()   # flushes the queue, then stops the worker
```

In thread mode, sync subscribers drain on a background thread via a bounded queue, so
`publish()` returns immediately; on overflow events are dropped per `on_overflow`
(`drop_newest` / `drop_oldest` / `block`) and counted in `bus.dropped`. Async subscribers
still schedule on the loop. Default is `inline` (no behavior change).

See [`examples/36_observability.py`](../../../examples/36_observability.py) for a JSONL
sink + metrics + replay end to end.
