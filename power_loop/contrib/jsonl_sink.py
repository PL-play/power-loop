"""Durable JSONL event sink + replay (OBS-2).

The in-process :class:`AgentEventBus` is ephemeral — events are gone when the process
exits. This sink persists the full typed envelope (``ts``/``seq``/``mono`` + payload, via
:meth:`AgentEvent.to_dict`) as one JSON object per line in a size-rotated file, and
:func:`replay` reads them back as :class:`AgentEvent` objects — for audit, debugging, or
feeding an offline pipeline.

Usage::

    from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay

    sink = attach_jsonl_sink(bus, "events.jsonl")   # all events
    ...                                              # run the loop
    sink.close()
    for event in replay("events.jsonl"):             # oldest → newest, across rotations
        ...

Payloads are redacted + truncated with the same policy as ``logging_sink`` (shared
``contrib/_redact``); pass ``redact_keys=()`` to disable.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contrib._redact import resolve_redact, sanitize
from power_loop.core.events import AgentEventBus

__all__ = ["attach_jsonl_sink", "replay", "JsonlSink"]


class JsonlSink:
    """A size-rotated JSON-lines writer. One ``AgentEvent.to_dict()`` per line; rotates
    to ``path.1``, ``path.2``, … (oldest dropped past ``backup_count``). Thread-safe."""

    def __init__(self, path: str | Path, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def write_line(self, line: str) -> None:
        with self._lock:
            if self._fh.closed:
                return
            if self.max_bytes and self._fh.tell() > 0 and self._fh.tell() + len(line) + 1 > self.max_bytes:
                self._rotate()
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()

    def _rotate(self) -> None:
        self._fh.close()
        if self.backup_count > 0:
            for i in range(self.backup_count - 1, 0, -1):
                src = self._backup(i)
                if src.exists():
                    src.replace(self._backup(i + 1))
            self.path.replace(self._backup(1))
        self._fh = self.path.open("w", encoding="utf-8")

    def _backup(self, i: int) -> Path:
        return self.path.with_name(self.path.name + f".{i}")

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def attach_jsonl_sink(
    bus: AgentEventBus,
    path: str | Path,
    *,
    events: Iterable[AgentEventType] | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    max_field_len: int = 2000,
    redact_keys: Iterable[str] | None = None,
) -> JsonlSink:
    """Persist events from ``bus`` to a rotating JSONL file at ``path``.

    :param events: only persist these types; ``None`` persists everything.
    :param max_bytes/backup_count: rotation (``0`` disables size rotation).
    :param max_field_len: truncate long string payload values.
    :param redact_keys: secret-key denylist (``None`` = default; ``()`` = no redaction).
    Returns the :class:`JsonlSink`; call ``.close()`` to flush + release the file.
    """
    sink = JsonlSink(path, max_bytes=max_bytes, backup_count=backup_count)
    wanted = set(events) if events is not None else None
    redact_lower = resolve_redact(redact_keys)

    def _handler(event: AgentEvent) -> None:
        if wanted is not None and event.type not in wanted:
            return
        d = event.to_dict()
        d["payload"] = sanitize(d.get("payload") or {}, max_field_len, redact_lower)
        sink.write_line(json.dumps(d, ensure_ascii=False, default=str))

    if wanted is None:
        bus.subscribe(None, _handler)
    else:
        for event_type in wanted:
            bus.subscribe(event_type, _handler)
    return sink


def replay(path: str | Path) -> Iterator[AgentEvent]:
    """Yield every persisted event as an :class:`AgentEvent`, oldest → newest, reading
    the rotated backups (``path.N`` … ``path.1``) before the live ``path``. Blank lines
    and unparseable lines are skipped. Original ``ts``/``seq``/``mono`` are preserved."""
    p = Path(path)
    files: list[Path] = []
    i = 1
    while True:
        backup = p.with_name(p.name + f".{i}")
        if not backup.exists():
            break
        files.append(backup)
        i += 1
    files.reverse()  # highest index = oldest
    if p.exists():
        files.append(p)
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield AgentEvent.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
