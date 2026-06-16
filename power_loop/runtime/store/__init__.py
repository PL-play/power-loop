"""Pluggable storage backends for power-loop.

The store is being refactored from a single local-SQLite class into a backend-neutral
async ``SessionStore`` facade written once against small ``Database`` + ``Dialect`` ports,
with SQLite (default, zero-dep), PostgreSQL, and MySQL backends. See
``docs/design/storage-backends.md``.

This package currently holds the backend-neutral row/enum **types** (shared by every
backend); the facade, ports, dialects, schema, and backends land in subsequent phases.
"""

from power_loop.runtime.store.types import (
    BackgroundTaskRow,
    CompactionRow,
    MessageRow,
    MessageState,
    NoteRow,
    SessionKind,
    SessionRow,
    SessionStateRow,
    SessionStatsRow,
    SessionStatus,
    SubagentLifecycle,
    TimerRow,
)

__all__ = [
    "BackgroundTaskRow",
    "CompactionRow",
    "MessageRow",
    "MessageState",
    "NoteRow",
    "SessionKind",
    "SessionRow",
    "SessionStateRow",
    "SessionStatsRow",
    "SessionStatus",
    "SubagentLifecycle",
    "TimerRow",
]
