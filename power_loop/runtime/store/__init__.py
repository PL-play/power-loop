"""Pluggable storage backends for power-loop.

A backend-neutral async ``SessionStore`` facade written once against small ``Database`` +
``Dialect`` ports, with SQLite (default, zero-dep), PostgreSQL (``[postgres]``), and MySQL
(``[mysql]``) backends. Open one with :func:`open_store` (DSN scheme → backend) or
``SessionStore.open`` (SQLite). See ``docs/design/storage-backends.md``.
"""

from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaPolicy,
    StoreSchemaError,
    provisioning_ddl,
)
from power_loop.runtime.store.store import (
    DEFAULT_MAX_SPAWN_DEPTH,
    DEFAULT_TABLE_PREFIX,
    SessionStore,
)
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
    # facade + provisioning
    "SessionStore",
    "open_store",
    "SchemaPolicy",
    "StoreSchemaError",
    "provisioning_ddl",
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_TABLE_PREFIX",
    "DEFAULT_MAX_SPAWN_DEPTH",
]
