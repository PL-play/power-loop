# Design: Pluggable storage backends (SQLite · PostgreSQL · MySQL)

Status: **in progress** (Phase 1). Decisions locked: **async-first**, **hand-rolled thin dialect**, **PostgreSQL first**.

## Goal

Replace the single local-SQLite `SessionStore` with a **pluggable backend** abstraction that also
supports PostgreSQL and MySQL, with a clean, extensible seam and no backward-compat burden (the
library has a single user; breaking changes are acceptable, the old surface can be rebuilt).

## What the current store is

- One `SessionStore` class, ~50 **synchronous** methods, the **sole DB writer**; returns 8 frozen
  dataclasses (`SessionRow`, `MessageRow`, `SessionStateRow`, `CompactionRow`, `BackgroundTaskRow`,
  `NoteRow`, `TimerRow`, `SessionStatsRow`) + 4 enums.
- `messages` is a **plain table**, PK `(session_id, seq)`; `seq` is a **per-session, gap-free,
  monotonic counter** (the row identity) allocated from `session_state.next_seq`. No snowflake, no
  partitioning, no change_seq (those live in the separate DeepTalk `api` repo, not here).
- Callers depend only on the **method signatures + dataclasses + per-method atomicity** — nothing
  reaches into `_conn`/`_lock`/`_read_pool`/raw SQL (grep-verified). Clean seam to abstract behind.

## The five things welded to SQLite (and the fix)

1. **Seq allocation** — `SELECT next_seq → INSERT → UPDATE` is atomic only because one process holds
   an in-process `RLock`. On a server DB that is a lost-update race. → Allocate **in the DB**:
   PG `UPDATE session_state SET next_seq=next_seq+1 WHERE session_id=? RETURNING next_seq-1`;
   MySQL `SELECT next_seq … FOR UPDATE` then `UPDATE`. `(session_id, seq)` PK is the backstop; retry
   on serialization/deadlock. This makes PG/MySQL **genuinely multi-writer** (the "one file = one
   writer" caveat goes away).
2. **Atomicity** — `with self._lock, self._conn:` is the unit of work. → An explicit
   `async with db.transaction() as tx:` per method; real BEGIN/COMMIT/ROLLBACK + row locks. Callers
   still call one method and get all-or-nothing (no transaction object leaks out).
3. **SQLite-only ops** — WAL, `backup()`, VACUUM, `wal_checkpoint`, `PRAGMA user_version`, the read
   pool. → Optional **capability** methods (real on SQLite; no-op / NotSupported on server backends,
   where backup/vacuum are operator concerns).
4. **Migrations** — `PRAGMA user_version` + monolithic `SCHEMA_SQL`. → A portable `schema_migrations`
   version table + a logical `TableSpec` list rendered to **per-dialect DDL**; same
   fresh-stamp / upgrade-ladder / refuse-newer control flow.
5. **Dialect SQL** — `ON CONFLICT … excluded`, `INSERT OR REPLACE`, `?` paramstyle, JSON-as-TEXT. →
   A small `Dialect` (paramstyle, `upsert()` renderer, `allocate_seq()`, DDL type map,
   `FOR UPDATE`/`RETURNING` support).

## Architecture — one store, two ports, N backends

The ~50 methods' *logic* is identical across backends; only the SQL *dialect* differs. So the store
is written **once** against two small ports.

```
power_loop/runtime/store/
  types.py        # dataclasses + enums (backend-neutral)
  store.py        # SessionStore: async facade, the ~50 methods, written ONCE against Database+Dialect
  db.py           # Database / Connection / Transaction protocols (the driver port)
  dialect.py      # Dialect protocol: paramstyle, upsert(), allocate_seq(), ddl types, for_update, returning
  schema.py       # logical TableSpec list → per-dialect DDL + portable schema_migrations ladder
  capabilities.py # Maintenance (backup/vacuum/checkpoint) — optional; SQLite real, others no-op
  backends/
    sqlite.py     # stdlib sqlite3 via threadpool (+ WAL/RLock/read-pool internal)  [ZERO-dep, default]
    postgres.py   # asyncpg                                                          [extra: postgres]
    mysql.py      # asyncmy / aiomysql                                               [extra: mysql]
  factory.py      # open_store(dsn) → picks backend by URL scheme
```

### The ports (sketch)

```python
class Transaction(Protocol):
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int: ...        # → affected rows
    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Mapping | None: ...
    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping]: ...

class Database(Protocol):
    dialect: "Dialect"
    def transaction(self) -> AsyncContextManager[Transaction]: ...   # BEGIN/COMMIT/ROLLBACK
    async def fetchall(self, sql, params=()) -> list[Mapping]: ...    # autocommit read (pooled)
    async def close(self) -> None: ...
    # optional: Maintenance capability (backup/vacuum/checkpoint)

class Dialect(Protocol):
    def param(self, i: int) -> str: ...                  # "?" | "%s" | "$1"
    def upsert(self, table, key_cols, val_cols, *, add_cols=()) -> str: ...
    async def allocate_seq(self, tx: Transaction, session_id: str) -> int: ...  # the atomic counter
    def ddl(self, spec: TableSpec) -> list[str]: ...     # CREATE TABLE + indexes for this dialect
    def supports_returning(self) -> bool: ...
```

### Invariants preserved verbatim (correctness-critical)

- Per-session seq is **gap-free + monotonic** (the `SQLiteSink` index↔seq invariant depends on it).
- Compaction marks an **explicit fold-seq set** (never a `BETWEEN` range) and a `compact_note`'s
  **logical position ≠ identity seq**. → Promote `meta['ord']` to a real **`ord` column** so read
  order is `ORDER BY ord, seq` in SQL on every backend (no Python re-sort).
- Each multi-statement method (append, compaction-fold, cascade-delete, import) is one transaction.
- Time stays **epoch-ms BIGINT** everywhere (dialect-uniform). `pinned` becomes a real BOOLEAN on
  server backends. JSON stays TEXT+`json` by default (portable; no querying needed yet).
- Cascade delete stays in **application code** (the DETACHED/LINKED lifecycle can't be a plain FK).

### Table prefix (isolation on shared databases)

All power-loop tables get a configurable **prefix, default `pl_`** (`pl_sessions`,
`pl_messages`, `pl_session_state`, …, plus the `pl_schema_migrations` version table). On a
server DB shared with another app (e.g. testing against DeepTalk's Postgres) this prevents
name collisions without needing a dedicated schema/database. The prefix is applied once, at
DDL-render + query-build time in the schema/dialect layer — it is **not** sprinkled through
the method bodies. It is **uniform across all backends including SQLite** (so behavior is
identical everywhere); this renames the SQLite tables, which is a breaking change for
existing `.db` files — acceptable per the no-compat mandate (delete/recreate, or a one-time
rename migration). On Postgres a dedicated `schema=` (search_path) is offered as an optional
extra layer of isolation; the prefix is the portable baseline.

### Schema initialization (auto-create vs caller-provisioned)

Power-loop's ethos is "zero infrastructure — it just works", so the default stays **auto-create
on open** (the migration ladder runs `CREATE … IF NOT EXISTS` + version-stamps), exactly the
current SQLite DX. But for managed/least-privilege server deployments where the runtime DB user
should not hold DDL rights, the caller can opt out:

- `open_store(dsn, create_schema=True)` (default) — provision/upgrade the schema on open.
- `open_store(dsn, create_schema=False)` — do **not** touch DDL; on open, verify the
  `pl_schema_migrations` row exists and equals `CURRENT_SCHEMA_VERSION`, else raise a clear
  `StoreSchemaError` ("schema missing/old — run `power-loop db upgrade`"). A
  newer-than-code version is always refused (as today).
- An explicit `await store.create_schema()` / a small `power-loop db upgrade <dsn>` entry
  point lets ops provision separately.

Auto-create must be **concurrency-safe** on a multi-writer server (two processes opening at
once): wrap the migration in a transaction guarded by a DB lock — Postgres
`pg_advisory_xact_lock`, MySQL `GET_LOCK` — so the version bump can't race. (SQLite's single
writer makes this trivially safe.)

### Async ripple

Every store method becomes `async def`; the SQLite backend wraps sync `sqlite3` in a threadpool, so
the scattered `asyncio.to_thread` discipline collapses into the backend. ~25 inline-on-loop call
sites (`TimerRunner`, `SqliteBlackboard`, pipeline pending paths, `workflow/journal`, `stateful_loop`)
become `await`; the already-offloaded sites stop double-wrapping. `StatefulAgentLoop(store=…)`
injection stays; `db_path=` becomes sugar for `sqlite:///…`; add `open_store(dsn)`.

## Phased roadmap

- **Phase 1** — async store abstraction + SQLite backend; convert callers; full suite green on SQLite.
- **Phase 2** — PostgreSQL backend (`power-loop[postgres]`); shared store **conformance** test suite
  run against a dockerized PG; document multi-writer.
- **Phase 3** — MySQL backend (`power-loop[mysql]`); `open_store` DSN factory; docs + extras +
  STABLE-API baseline updates.

## Testing

SQLite `:memory:` stays the fast default + the backend the unit suite uses. A backend-agnostic
**conformance** suite (the behaviors every backend must satisfy: seq monotonicity under concurrency,
compaction ordering, cascade, upsert accumulate, timer CAS, atomic rollback) runs against SQLite
always and against PG/MySQL when a server DSN is provided (gated like the live-LLM suite).
