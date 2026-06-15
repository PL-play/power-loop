"""Structured-logging event sink: one JSON line per agent event.

Every power-loop adopter ends up writing the same glue — subscribe to the
event bus, serialize events, ship them to logs/metrics. This sink is that
glue, stdlib-only. Pipe the lines into anything that speaks JSON (jq, Loki,
CloudWatch, an OTel collector's filelog receiver, ...).

Usage::

    import logging
    from power_loop import AgentEventBus, StatefulAgentLoop
    from power_loop.contrib.logging_sink import attach_logging_sink

    bus = AgentEventBus(suppress_subscriber_errors=True)
    attach_logging_sink(bus)                       # all events, INFO
    attach_logging_sink(bus, events={AgentEventType.USAGE_UPDATED})  # subset
    loop = StatefulAgentLoop(llm=llm, event_bus=bus, ...)

Each line looks like::

    {"event": "usage_updated", "session_id": "sess_…", "round_index": 1,
     "payload": {"usage": {"prompt_tokens": 123, ...}}}

Large payload fields (message bodies, prompts) are truncated to
``max_field_len`` so a chatty agent cannot blow up your log volume.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contrib._redact import (
    DEFAULT_REDACT_KEYS,
    REDACTED,
    resolve_redact,
    sanitize,
)
from power_loop.core.events import AgentEventBus

DEFAULT_LOGGER_NAME = "power_loop.events"

# Back-compat aliases — the redaction policy now lives in contrib/_redact.py (shared with
# the JSONL sink). Kept so existing imports of these names keep working.
_DEFAULT_REDACT_KEYS = DEFAULT_REDACT_KEYS
_REDACTED = REDACTED
_sanitize = sanitize


def attach_logging_sink(
    bus: AgentEventBus,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    events: Iterable[AgentEventType] | None = None,
    max_field_len: int = 500,
    redact_keys: Iterable[str] | None = None,
) -> None:
    """Subscribe a JSON-lines logger to ``bus``.

    :param events: only log these event types; ``None`` logs everything.
    :param max_field_len: truncate string payload values beyond this length.
    :param redact_keys: key-name substrings whose values are replaced with ``***``.
        Defaults to a common secret denylist (api_key/token/password/…); pass
        ``()`` to disable redaction, or your own iterable to override.
    """
    log = logger if logger is not None else logging.getLogger(DEFAULT_LOGGER_NAME)
    wanted = set(events) if events is not None else None
    redact_lower = resolve_redact(redact_keys)

    def _handler(event: AgentEvent) -> None:
        if wanted is not None and event.type not in wanted:
            return
        if not log.isEnabledFor(level):
            return
        # Include the envelope ordering keys (ts/seq) so log lines are orderable and
        # correlatable with a durable event stream — not just the event name + payload.
        record: dict[str, Any] = {
            "event": event.type.value,
            "seq": event.seq,
            "ts": event.ts,
            "session_id": event.session_id,
        }
        if event.round_index is not None:
            record["round_index"] = event.round_index
        if event.source:
            record["source"] = event.source
        payload = event.payload or {}
        record["payload"] = sanitize(payload, max_field_len, redact_lower)
        log.log(level, json.dumps(record, ensure_ascii=False, default=str))

    if wanted is None:
        bus.subscribe(None, _handler)  # global: every event type
    else:
        for event_type in wanted:
            bus.subscribe(event_type, _handler)
