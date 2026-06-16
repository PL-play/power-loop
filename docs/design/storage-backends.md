# Design: Pluggable storage backends (SQLite · PostgreSQL · MySQL)

Status: **DONE** — SQLite (default, zero-dep), PostgreSQL (`[postgres]`), and MySQL (`[mysql]`)
backends all ship, the full caller swap to the async API is complete, and the conformance suite
runs against all three. Decisions locked: **async-first**, **hand-rolled thin dialect**.

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

- **Phase 1** ✅ — async store abstraction + SQLite backend; conformance suite; full suite green.
- **Phase 2** ✅ — PostgreSQL backend (`power-loop[postgres]`); conformance run against a dockerized
  PG; `SELECT … FOR UPDATE` seq allocation → genuinely multi-writer.
- **Caller swap** ✅ — `StatefulAgentLoop` + every internal caller, the unit/real test suites, the
  bench harness, and all examples flipped to the async API (lazy-open owned store; `await`
  everywhere). Legacy `session_store.py` kept only as the parity oracle.
- **Phase 3** ✅ — MySQL backend (`power-loop[mysql]`, pure-Python `aiomysql`): `MySQLDialect`
  (`?`→`%s`, VARCHAR PK/index cols + utf8mb4, inline indexes, `ON DUPLICATE KEY UPDATE … AS new_row`
  upsert, `FOR UPDATE` seq alloc) + `MySQLDatabase` (autocommit pool, `CLIENT.FOUND_ROWS` so
  `rowcount` = matched rows for CAS parity); `open_store` `mysql://` DSN; conformance run against a
  dockerized MySQL 8. The `session_runtime_state`/`shared_state` `key` column was renamed to
  `state_key` (MySQL reserved word) — backend-neutral, invisible to callers.

## Resumable loop, schema policy & the per-session window cache

* **Loop binding (multi-backend).** `StatefulAgentLoop` is sync-constructed but lazy-opens its
  owned store through `open_store(dsn, table_prefix=, schema=)` on first async use, so it binds
  to SQLite/PG/MySQL by DSN + prefix. Pass EITHER a pre-opened `store=` (shared across loops,
  the DeepTalk path) OR a store config (`dsn=`/`db_path=`/`table_prefix=`/`schema=`) — passing
  both raises. The loop holds no authoritative session state, so it is freely re-creatable:
  `send(user_input, session_id=sid)` resumes any session by id (a fresh cold loop reproduces
  identical behavior). `prewarm(sid)` optionally pre-loads a session's window so the first send
  skips its reload. (Resuming PENDING tool-calls is the separate `resume()` method.)

* **`SchemaPolicy`** governs provisioning at open: `AUTO_CREATE` (default) probes the version
  table and, if absent, creates every table/index + stamps; a DDL failure (no CREATE rights)
  raises `StoreSchemaError` carrying the **complete** provisioning script (migrations table +
  all CREATEs + the version stamp) so an operator can run it by hand. `VERIFY` only checks and
  raises (with the same script) if missing/stale — for roles with no DDL rights, provision
  out-of-band then open VERIFY. `create_schema: bool` is a deprecated alias (True→AUTO_CREATE,
  False→VERIFY). The version stamp is an idempotent `ON CONFLICT DO NOTHING` / `INSERT IGNORE`.

* **Per-session window cache.** The loop keeps an LRU of each session's *durable* active
  projection (the `load_active_messages` rows + the `next_seq` validity token), NOT the
  pipeline's mutated working history — so recall placeholders and microcompacted content (both
  transient, never persisted) and any session-prompt edit are re-applied/re-read fresh each
  send. A send reuses the window iff the live `next_seq` still matches (and extends it with a
  cheap delta read of its own appends); a fold or any other writer's change forces a reload. So
  the cache is a rebuildable accelerator that never changes what the LLM sees — proven by a
  warm-vs-cold conformance suite (`test_loop_resume_cache.py`) asserting byte-identical prompts
  under recall, microcompaction, double-fold compaction, and a between-send prompt edit. Only
  the plain-send path is cached (resume/submit_input keep their own pre-primed sinks).

### Preconditions (documented, not silently assumed)

* **Single-writer-per-session.** The per-session lock is in-memory/per-loop-instance; it gives
  no cross-process mutual exclusion. The window cache stays *data-safe* across processes (a
  stale token forces a reload, never serves wrong-accepted data) but the pending-state machine
  assumes one writer per session at a time. Driving one session from two loops/processes
  concurrently needs a DB-level guard (advisory lock / session-state epoch CAS) — a follow-up.
* **Concurrent first-boot.** `AUTO_CREATE` is idempotent and self-heals on retry but is NOT
  atomic on MySQL (DDL auto-commits) and takes no cross-process lock; N instances booting
  against a *fresh* server DB simultaneously should provision out-of-band and open `VERIFY`. A
  `pg_advisory_xact_lock` / `GET_LOCK` guard for true concurrent first-boot is a follow-up.
* **Cache scope (non-goal).** The cache is per-loop, so spawn / workflow child loops (one send,
  then discarded) never hit it; only long-lived multi-session front-door loops benefit. Entry
  COUNT is LRU-bounded (`session_cache_size`, default 256, 0 disables); total bytes are not, so
  a few large pre-compaction sessions can dominate RAM. `cache_stats` exposes hits/misses/
  evictions for visibility.

## Testing

SQLite `:memory:` stays the fast default + the backend the unit suite uses. A backend-agnostic
**conformance** suite (the behaviors every backend must satisfy: seq monotonicity under concurrency,
compaction ordering, cascade, upsert accumulate, timer CAS, atomic rollback) runs against SQLite
always and against PG/MySQL when a server DSN is provided (gated like the live-LLM suite). The
resumable-loop + window-cache invariants (warm == cold prompts under recall / microcompact /
double-fold / prompt-edit; schema policy; next_seq monotonicity; delta load) are pinned by
`test_loop_resume_cache.py` + `test_store_schema_policy.py`, with gated PG/MySQL loop-binding smokes.
