# Storage backends (SQLite · PostgreSQL · MySQL)

power-loop's entire store — sessions, messages, timers, compaction journals, usage stats,
sub-agent trees, the shared blackboard — is written **once** against a tiny async
`Database` + `Dialect` port. You pick the backend with a **DSN**; nothing above the store
changes. SQLite is the zero-infrastructure default; PostgreSQL and MySQL are real
multi-writer servers behind optional driver extras.

```python
from power_loop import StatefulAgentLoop, open_store, SchemaPolicy

# Via the loop (it lazy-opens an owned store on first use):
StatefulAgentLoop(llm=llm, dsn="app.db")                          # SQLite (default)
StatefulAgentLoop(llm=llm, dsn="postgresql://u:p@host:5432/app")  # PostgreSQL
StatefulAgentLoop(llm=llm, dsn="mysql://u:p@host:3306/app")       # MySQL

# Or open a store directly (e.g. to share across loops):
store = await open_store("postgresql://u:p@host/app", table_prefix="pl_")
loop = StatefulAgentLoop(llm=llm, store=store)
```

| Param | Default | Meaning |
|---|---|---|
| `dsn` (alias `db_path`) | `./power_loop_sessions.db` | DSN or a bare/`sqlite://` path. Scheme picks the backend. |
| `table_prefix` | `pl_` | Prefix on every table/index — isolates power-loop on a shared database. |
| `schema` | `SchemaPolicy.AUTO_CREATE` | How tables are provisioned at open (below). |

Install the driver for the backend you use:

```bash
pip install 'power-loop[postgres]'   # asyncpg
pip install 'power-loop[mysql]'      # aiomysql (pure-Python)
```

> Pass **either** a pre-opened `store=` **or** a store config (`dsn`/`table_prefix`/`schema`) — not both (it raises). SQLite is the only backend the zero-dependency core can open without an extra.

---

## Choosing a backend

| | **SQLite** (default) | **PostgreSQL** | **MySQL** |
|---|---|---|---|
| Infra | none — one file | a server | a server |
| Driver extra | none (stdlib) | `[postgres]` (asyncpg) | `[mysql]` (aiomysql) |
| Writers | **one process per file** (shard across files) | multi-writer¹ | multi-writer¹ |
| Best for | local apps, demos, embedded, per-tenant files | shared server, many app instances | shared server, MySQL shops |
| Maintenance ops | `vacuum()` / `checkpoint()` (WAL) | no-ops² | no-ops² |

¹ Per-session sequence allocation is multi-writer-safe via a `SELECT … FOR UPDATE` row lock,
so two processes can append to *different* sessions concurrently and never collide. The
**pending-state machine still assumes one writer drives a given session at a time** — see
[Preconditions](#preconditions).
² `vacuum`/`checkpoint`/`backup` are a SQLite `Maintenance` capability; on PG/MySQL they are
no-ops (use your DB's native tooling).

The same backend-agnostic **conformance suite** (`tests/unit/test_store_parity*.py`) runs
every store behavior — gap-free sequence allocation, compaction ordering, cascade delete,
upsert accumulate, timer CAS, atomic rollback — against SQLite, PostgreSQL, and MySQL, with
the legacy local store as the oracle.

---

## Schema provisioning (`SchemaPolicy`)

Provisioning is a policy chosen at open time:

| Policy | Behavior |
|---|---|
| `AUTO_CREATE` (default) | Probe the version table; if absent, create every table + index and stamp the version. If the DDL fails (e.g. the role lacks `CREATE` rights), raise `StoreSchemaError` carrying the **complete DDL script** to run by hand. |
| `VERIFY` | Probe only. If the schema is missing or its version differs, raise `StoreSchemaError` (with the DDL). For DB roles with **no DDL rights** — provision out-of-band, then open with `VERIFY`. |

```python
from power_loop import SchemaPolicy, StoreSchemaError, open_store

# Zero-infra: tables appear on first use.
store = await open_store("postgresql://app@host/app")  # AUTO_CREATE (default)

# Locked-down role: verify only, and print the exact DDL to hand a DBA if it's missing.
try:
    store = await open_store("postgresql://readonly@host/app", schema=SchemaPolicy.VERIFY)
except StoreSchemaError as e:
    print(e)            # message + the full runnable provisioning script
    print(e.ddl)        # list[str] of CREATE/INSERT statements
```

`create_schema: bool` is a deprecated alias kept for the 1.x line (`True → AUTO_CREATE`,
`False → VERIFY`).

To get the DDL programmatically without opening (e.g. to generate a migration):

```python
from power_loop.runtime.store.schema import provisioning_ddl
from power_loop.runtime.store.backends.postgres import PostgresDatabase  # or sqlite/mysql

# (or just catch StoreSchemaError.ddl from a VERIFY open against a fresh DB)
```

---

## The DDL, per backend

This is exactly what `AUTO_CREATE` runs (and what `StoreSchemaError.ddl` prints) for the
default `pl_` prefix. It is generated from the dialect at runtime, so it never drifts from
the code. 12 tables + a version table; SQLite/PostgreSQL declare indexes separately, MySQL
inline (it has no `CREATE INDEX IF NOT EXISTS`).

### SQLite

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, system_prompt TEXT, model TEXT, config_json TEXT, status TEXT NOT NULL DEFAULT 'active', kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT, spawn_tool_call_id TEXT, spawn_depth INTEGER NOT NULL DEFAULT 0, lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT);
CREATE INDEX IF NOT EXISTS pl_idx_sessions_parent ON pl_sessions(parent_session_id);
CREATE TABLE IF NOT EXISTS pl_messages (session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT, round_index INTEGER, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, seq));
CREATE INDEX IF NOT EXISTS pl_idx_messages_session_state ON pl_messages(session_id, state, seq);
CREATE TABLE IF NOT EXISTS pl_compactions (session_id TEXT NOT NULL, compact_seq INTEGER NOT NULL, note_seq INTEGER NOT NULL, from_seq INTEGER NOT NULL, to_seq INTEGER NOT NULL, before_tokens INTEGER, after_tokens INTEGER, round_index INTEGER, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, compact_seq));
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id TEXT NOT NULL, round_index INTEGER NOT NULL, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, model TEXT, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, round_index));
CREATE TABLE IF NOT EXISTS pl_session_state (session_id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL DEFAULT 1, round_index INTEGER NOT NULL DEFAULT 0, last_compact_seq INTEGER NOT NULL DEFAULT 0, pending_json TEXT);
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, state_key));
CREATE TABLE IF NOT EXISTS pl_shared_state (owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at INTEGER NOT NULL, PRIMARY KEY (owner, state_key));
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL, return_code INTEGER, output_tail TEXT, output_path TEXT, last_seen_at INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, task_id));
CREATE INDEX IF NOT EXISTS pl_idx_background_tasks_session_status ON pl_background_tasks(session_id, status, updated_at);
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id TEXT PRIMARY KEY, sends INTEGER NOT NULL DEFAULT 0, rounds INTEGER NOT NULL DEFAULT 0, llm_calls INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0, first_send_at INTEGER, last_send_at INTEGER, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_timers (session_id TEXT NOT NULL, timer_id INTEGER NOT NULL, due_at INTEGER NOT NULL, note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s INTEGER, fire_count INTEGER NOT NULL DEFAULT 0, last_fired_at INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, timer_id));
CREATE INDEX IF NOT EXISTS pl_idx_timers_due ON pl_timers(status, due_at);
CREATE TABLE IF NOT EXISTS pl_notes (session_id TEXT NOT NULL, note_id INTEGER NOT NULL, content TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, note_id));
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

### PostgreSQL

Same shape; `INTEGER → BIGINT` for epoch-ms timestamps and counters, JSON kept as `TEXT`
(the store (de)serializes), `pinned SMALLINT`.

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id TEXT PRIMARY KEY, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, system_prompt TEXT, model TEXT, config_json TEXT, status TEXT NOT NULL DEFAULT 'active', kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT, spawn_tool_call_id TEXT, spawn_depth BIGINT NOT NULL DEFAULT 0, lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT);
CREATE INDEX IF NOT EXISTS pl_idx_sessions_parent ON pl_sessions(parent_session_id);
CREATE TABLE IF NOT EXISTS pl_messages (session_id TEXT NOT NULL, seq BIGINT NOT NULL, role TEXT NOT NULL, name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT, round_index BIGINT, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq));
CREATE INDEX IF NOT EXISTS pl_idx_messages_session_state ON pl_messages(session_id, state, seq);
CREATE TABLE IF NOT EXISTS pl_compactions (session_id TEXT NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL, from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT, after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, compact_seq));
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id TEXT NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT, completion_tokens BIGINT, total_tokens BIGINT, model TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index));
CREATE TABLE IF NOT EXISTS pl_session_state (session_id TEXT PRIMARY KEY, next_seq BIGINT NOT NULL DEFAULT 1, round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0, pending_json TEXT);
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key));
CREATE TABLE IF NOT EXISTS pl_shared_state (owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key));
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT, last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id));
CREATE INDEX IF NOT EXISTS pl_idx_background_tasks_session_status ON pl_background_tasks(session_id, status, updated_at);
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id TEXT PRIMARY KEY, sends BIGINT NOT NULL DEFAULT 0, rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0, tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0, completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0, first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL);
CREATE TABLE IF NOT EXISTS pl_timers (session_id TEXT NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL, note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s BIGINT, fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, timer_id));
CREATE INDEX IF NOT EXISTS pl_idx_timers_due ON pl_timers(status, due_at);
CREATE TABLE IF NOT EXISTS pl_notes (session_id TEXT NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL, pinned SMALLINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id));
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

### MySQL

String PK/index columns become `VARCHAR(255)` (MySQL can't index/PK a `TEXT` without a
prefix length), small enum-like columns `VARCHAR(32)`, `utf8mb4`, and indexes are declared
**inline** (MySQL has no `CREATE INDEX IF NOT EXISTS`). The `key` column is named `state_key`
(reserved word). The version-row stamp uses `INSERT IGNORE`; upserts use
`INSERT … AS new_row ON DUPLICATE KEY UPDATE` (MySQL 8.0.19+).

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id VARCHAR(255) NOT NULL, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, system_prompt TEXT, model VARCHAR(255), config_json TEXT, status VARCHAR(32) NOT NULL DEFAULT 'active', kind VARCHAR(32) NOT NULL DEFAULT 'root', parent_session_id VARCHAR(255), spawn_tool_call_id VARCHAR(255), spawn_depth BIGINT NOT NULL DEFAULT 0, lifecycle VARCHAR(32) NOT NULL DEFAULT 'ephemeral', metadata_json TEXT, PRIMARY KEY (session_id), KEY pl_idx_sessions_parent (parent_session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_messages (session_id VARCHAR(255) NOT NULL, seq BIGINT NOT NULL, role VARCHAR(32) NOT NULL, name VARCHAR(255), content TEXT, tool_calls_json TEXT, tool_call_id VARCHAR(255), round_index BIGINT, state VARCHAR(32) NOT NULL DEFAULT 'active', meta_json TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq), KEY pl_idx_messages_session_state (session_id, state, seq)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_compactions (session_id VARCHAR(255) NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL, from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT, after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, compact_seq)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id VARCHAR(255) NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT, completion_tokens BIGINT, total_tokens BIGINT, model VARCHAR(255), created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_state (session_id VARCHAR(255) NOT NULL, next_seq BIGINT NOT NULL DEFAULT 1, round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0, pending_json TEXT, PRIMARY KEY (session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_shared_state (owner VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id VARCHAR(255) NOT NULL, task_id VARCHAR(255) NOT NULL, command TEXT NOT NULL, status VARCHAR(32) NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT, last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id), KEY pl_idx_bgtasks_session_status (session_id, status, updated_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id VARCHAR(255) NOT NULL, sends BIGINT NOT NULL DEFAULT 0, rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0, tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0, completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0, first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_timers (session_id VARCHAR(255) NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL, note TEXT NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'armed', interval_s BIGINT, fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, timer_id), KEY pl_idx_timers_due (status, due_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_notes (session_id VARCHAR(255) NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL, pinned TINYINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

> Change `pl_` to your `table_prefix` throughout. `pl_schema_migrations` is the version
> table that lets `VERIFY` work and refuses a newer-than-code database.

---

## Resumable loops & the active-window cache

A `StatefulAgentLoop` holds **no authoritative session state** — all of it lives in the
store. So a loop is cheap to create and you resume any session by id from a cold process:

```python
loop = StatefulAgentLoop(llm=create_llm_service_from_env(), dsn=DSN)   # cheap; opens lazily
await loop.prewarm(session_id)                # optional: pre-load the active window
result = await loop.send(user_text, session_id=session_id)
```

To skip re-reading the whole active history on every send, the loop keeps a per-session
**active-window cache** — but it caches only the *durable* projection (the rows
`load_active_messages` returns) keyed by a monotonic `next_seq` token, and rebuilds the
working copy (recall, microcompaction) fresh each send. So it is a pure accelerator: a cold
loop with an empty cache feeds the model byte-for-byte the same prompts (verified by a
warm-vs-cold conformance test). It's LRU-bounded (`session_cache_size`, default 256, `0`
disables) and exposed via `loop.cache_stats`.

---

## Preconditions

- **Single-writer-per-session.** The per-session lock is in-process; it gives no
  cross-process mutual exclusion. With SQLite, run one writer process per file. With
  PostgreSQL/MySQL, sequence allocation is multi-writer-safe, but a given session's
  pending-state machine still assumes one writer at a time — serialize a session's sends in
  your dispatcher/queue layer. (The window cache is data-safe regardless: a stale token forces
  a reload, never serves wrong data.)
- **Concurrent first-boot.** `AUTO_CREATE` is idempotent and self-heals on retry, but it is
  not atomic on MySQL (DDL auto-commits) and takes no cross-process lock. If many instances
  may boot against a *fresh* server schema at once, provision out-of-band (run the DDL above)
  and open with `SchemaPolicy.VERIFY`.

See also: [Sessions](sessions.md) · [Scaling](scaling.md) · the design note in
`docs/design/storage-backends.md`.
