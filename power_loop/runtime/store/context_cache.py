"""Durable summary jobs/cache and an independent, non-evictable attempt ledger."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any


def context_cache_ddl(name: str, prefix: str) -> list[str]:
    text = 'LONGTEXT' if name == 'mysql' else 'TEXT'
    statements = [
        f'''CREATE TABLE IF NOT EXISTS {prefix}context_cache (
            cache_key VARCHAR(64) PRIMARY KEY, executor_key VARCHAR(64) NOT NULL,
            owner_session_id VARCHAR(200) NOT NULL, status VARCHAR(20) NOT NULL,
            payload_json {text} NOT NULL, summary {text}, result_hash VARCHAR(64),
            lease_token VARCHAR(64), lease_until BIGINT NOT NULL DEFAULT 0,
            created_at BIGINT NOT NULL, accessed_at BIGINT NOT NULL, expires_at BIGINT NOT NULL,
            size_bytes BIGINT NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
            retry_at BIGINT NOT NULL DEFAULT 0)''',
        f'''CREATE TABLE IF NOT EXISTS {prefix}context_attempts (
            attempt_id VARCHAR(64) PRIMARY KEY, cache_key VARCHAR(64) NOT NULL,
            owner_session_id VARCHAR(200) NOT NULL, outcome VARCHAR(20) NOT NULL,
            record_json {text} NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
            accounted INTEGER NOT NULL DEFAULT 0, created_at BIGINT NOT NULL)''',
    ]
    indexes = (
        ('context_cache', 'context_due', 'executor_key,status,retry_at,created_at'),
        ('context_cache', 'context_running', 'status,lease_until'),
        ('context_cache', 'context_lru', 'status,accessed_at'),
        ('context_cache', 'idx_context_capacity', 'size_bytes'),
        ('context_attempts', 'context_outbox', 'accounted,delivered,created_at'),
        ('context_attempts', 'context_attempt_key', 'cache_key,outcome'),
        ('context_attempts', 'context_attempt_owner', 'owner_session_id,created_at'),
    )
    for table, index, columns in indexes:
        if name == 'mysql':
            i = 0 if table == 'context_cache' else 1
            statements[i] = statements[i].rstrip()[:-1] + f', INDEX {prefix}{index} ({columns}))'
        else:
            statements.append(f'CREATE INDEX IF NOT EXISTS {prefix}{index} ON {prefix}{table} ({columns})')
    return statements


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def now_ms() -> int:
    return int(time.time() * 1000)


class ContextCache:
    def __init__(self, store: Any):
        self.store, self.db = store, store._db
        prefix = store.t.sessions.removesuffix('sessions')
        self.cache, self.attempts = prefix + 'context_cache', prefix + 'context_attempts'

    async def _lock(self, tx: Any) -> None:
        table = self.store.t.shared_state
        sql = self.db.dialect.upsert(table, ('owner', 'state_key'), ('value_json', 'updated_at'))
        # A single cheap row serializes admission/eviction across processes, not LLM calls.
        await tx.execute(sql, ('context_cache', 'scheduler', '{}', now_ms()))

    async def enqueue(self, key: str, executor: str, payload: dict, ttl_s: float, *, max_entries: int = 10000, max_bytes: int = 67108864) -> bool:
        now = now_ms()
        data = dumps(payload)
        async with self.db.transaction() as tx:
            await self._lock(tx)
            row = await tx.fetchone(f'SELECT status, expires_at FROM {self.cache} WHERE cache_key=?', (key,))
            if row and (row['status'] in ('pending', 'running') or int(row['expires_at']) > now):
                return False
            uncertain = await tx.fetchone(f"SELECT 1 FROM {self.attempts} WHERE cache_key=? AND outcome IN ('started','interrupted','cancelled') LIMIT 1", (key,))
            if uncertain is not None:
                return False
            if row:
                await tx.execute(f'DELETE FROM {self.cache} WHERE cache_key=?', (key,))
            total = await tx.fetchone(f'SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS bytes FROM {self.cache}')
            if int(total['n']) >= max_entries or int(total['bytes']) + len(data.encode()) > max_bytes:
                return False  # admission backpressure: reader keeps rule/full, no model expense
            await tx.execute(f'''INSERT INTO {self.cache}
                (cache_key,executor_key,owner_session_id,status,payload_json,created_at,accessed_at,expires_at,size_bytes)
                VALUES (?,?,?,'pending',?,?,?,?,?)''',
                (key, executor, payload['owner_session_id'], data, now, now, now + int(ttl_s * 1000), len(data.encode())))
            return True

    async def get(self, key: str) -> str | None:
        now = now_ms()
        row = await self.db.fetchone(f'SELECT summary, result_hash, accessed_at FROM {self.cache} '
                                    "WHERE cache_key=? AND status='ready' AND expires_at>?", (key, now))
        if row is None:
            return None
        summary = str(row['summary'])
        if hashlib.sha256(summary.encode()).hexdigest() != row['result_hash']:
            return None
        # Approximate LRU: at most one timestamp update per minute per key.
        if now - int(row['accessed_at']) >= 60000:
            await self.db.execute(f'UPDATE {self.cache} SET accessed_at=? WHERE cache_key=? AND accessed_at<?',
                                  (now, key, now - 60000))
        return summary

    async def claim(self, executor: str, *, limit: int = 2, per_session: int = 1) -> dict | None:
        now = now_ms()
        async with self.db.transaction() as tx:
            await self._lock(tx)
            # An expired lease may have reached the provider: never silently reissue it.
            await tx.execute(f"UPDATE {self.cache} SET status='uncertain' WHERE status='running' AND lease_until<?", (now,))
            count = await tx.fetchone(f"SELECT COUNT(*) AS n FROM {self.cache} WHERE status='running'")
            if int(count['n']) >= limit or per_session <= 0:
                return None
            blocked = []
            if int(count['n']) >= per_session:
                owners = await tx.fetchall(f"SELECT owner_session_id,COUNT(*) AS n FROM {self.cache} "
                    "WHERE status='running' GROUP BY owner_session_id")
                blocked = [owner['owner_session_id'] for owner in owners if int(owner['n']) >= per_session]
            predicate, params = '', [executor, now, now]
            if blocked:
                predicate = ' AND owner_session_id NOT IN (' + ','.join('?' for _ in blocked) + ')'
                params.extend(blocked)
            row = await tx.fetchone(f"SELECT * FROM {self.cache} WHERE executor_key=? "
                "AND status='pending' AND retry_at<=? AND expires_at>? " + predicate +
                ' ORDER BY created_at LIMIT 1', params)
            if row is None:
                return None
            payload = json.loads(row['payload_json'])
            token = uuid.uuid4().hex
            # Model timeout is bounded; lease also covers accounting and slow DB cleanup.
            lease_until = now + int((payload['timeout_s'] + 60) * 1000)
            await tx.execute(f"UPDATE {self.cache} SET status='running', lease_token=?, lease_until=?, attempts=attempts+1 WHERE cache_key=?",
                             (token, lease_until, row['cache_key']))
            return {**dict(row), 'payload': payload, 'lease_token': token, 'lease_until': lease_until}

    async def begin_attempt(self, job: dict, record: dict) -> bool:
        async with self.db.transaction() as tx:
            await self._lock(tx)
            live = await tx.fetchone(f"SELECT 1 FROM {self.cache} WHERE cache_key=? AND status='running' "
                                    "AND lease_token=? AND lease_until>=?",
                                    (job['cache_key'], job['lease_token'], now_ms()))
            if live is None:
                return False  # stale worker must not issue an unowned provider request
            await tx.execute(f'''INSERT INTO {self.attempts}
                (attempt_id,cache_key,owner_session_id,outcome,record_json,created_at) VALUES (?,?,?,'started',?,?)''',
                (record['attempt_id'], job['cache_key'], record['owner_session_id'], dumps(record), now_ms()))
            return True

    async def finish(self, job: dict, record: dict, summary: str | None, result_hash: str | None) -> None:
        async with self.db.transaction() as tx:
            await self._lock(tx)
            previous = await tx.fetchone(f'SELECT record_json FROM {self.attempts} WHERE attempt_id=?',
                                         (record['attempt_id'],))
            old = json.loads(previous['record_json'])
            record['revision'] = int(old.get('revision', 0)) + 1
            await tx.execute(f'UPDATE {self.attempts} SET outcome=?, record_json=?, accounted=0, delivered=0 WHERE attempt_id=?',
                             (record['outcome'], dumps(record), record['attempt_id']))
            status = 'ready' if summary is not None else 'failed'
            if record['outcome'] in ('cancelled', 'interrupted'):
                status = 'uncertain'
            retry_at = 0
            if record.get('error_http_status') in (429, 503) and int(job['attempts']) < 1:
                status, retry_at = 'pending', now_ms() + 30000
            await tx.execute(f'UPDATE {self.cache} SET retry_at=? WHERE cache_key=? AND lease_token=?',
                             (retry_at, job['cache_key'], job['lease_token']))
            await tx.execute(f'''UPDATE {self.cache} SET status=?,summary=?,result_hash=?,size_bytes=size_bytes+?,lease_until=0
                WHERE cache_key=? AND lease_token=? AND status='running' ''',
                (status, summary, result_hash, len((summary or '').encode()), job['cache_key'], job['lease_token']))

    async def reconcile(self, receiver: Any = None, *, delivery_timeout: float = 2) -> int:
        # Finished receipts are independently replayable even after cache eviction or send end.
        pending = 'accounted=0 OR delivered=0' if receiver is not None else 'accounted=0'
        rows = await self.db.fetchall(f'SELECT * FROM {self.attempts} WHERE {pending} ORDER BY created_at LIMIT 100')
        done = 0
        for row in rows:
            # Re-read under the same short scheduler lock as finish(). A stale
            # scan must never overwrite a completed provider receipt.
            async with self.db.transaction() as tx:
                await self._lock(tx)
                row = await tx.fetchone(f'SELECT * FROM {self.attempts} WHERE attempt_id=?', (row['attempt_id'],))
                record = json.loads(row['record_json'])
                if row['outcome'] == 'started':
                    job = await tx.fetchone(f'SELECT status,lease_until FROM {self.cache} WHERE cache_key=?', (row['cache_key'],))
                    if job and job['status'] == 'running' and int(job['lease_until']) >= now_ms():
                        continue
                    record.update(outcome='interrupted', success=False, usage_unknown=True,
                                  finished_at=time.time(), revision=int(record.get('revision', 0)) + 1)
                    await tx.execute(f'UPDATE {self.attempts} SET outcome=?,record_json=? WHERE attempt_id=?',
                                     ('interrupted', dumps(record), row['attempt_id']))
            receipt = dumps(record)
            if not row['accounted']:
                # Session deletion does not delete the request ledger.
                if await self.store.get_session(record['owner_session_id']) is not None:
                    usage = dict(record.get('usage') or {})
                    usage['total_tokens'] = usage.get('total_tokens') or (usage.get('prompt_tokens') or 0) + (usage.get('completion_tokens') or 0)
                    await self.store.record_auxiliary_usage(record['owner_session_id'], record['attempt_id'], usage, revision=int(record.get('revision', 0)))
                await self.db.execute(f'UPDATE {self.attempts} SET accounted=1 WHERE attempt_id=? AND record_json=?', (row['attempt_id'], receipt))
            if receiver is not None and not row['delivered']:
                import asyncio
                try:
                    await asyncio.wait_for(receiver(record), delivery_timeout)
                except Exception:
                    break  # outage: keep the outbox, do not hammer API for every row
                await self.db.execute(f'UPDATE {self.attempts} SET delivered=1 WHERE attempt_id=? AND record_json=?', (row['attempt_id'], receipt))
            done += 1
        return done

    async def stats(self) -> dict[str, Any]:
        rows = await self.db.fetchall(
            f'SELECT status,COUNT(*) AS entries,COALESCE(SUM(size_bytes),0) AS bytes FROM {self.cache} GROUP BY status')
        return {str(r['status']): {'entries': int(r['entries']), 'bytes': int(r['bytes'])} for r in rows}

    async def prune(self, *, max_entries: int, max_bytes: int, batch: int = 1000) -> int:
        removed = 0
        async with self.db.transaction() as tx:
            await self._lock(tx)
            await tx.execute(f"DELETE FROM {self.cache} WHERE status='pending' AND expires_at<=?", (now_ms(),))
            await tx.execute(f"UPDATE {self.cache} SET status='uncertain' WHERE status='running' AND lease_until<?", (now_ms(),))
            # Active jobs are never evicted; their bounded payload is included in capacity.
            rows = await tx.fetchall(f"SELECT cache_key,size_bytes,expires_at FROM {self.cache} WHERE status NOT IN ('running','pending') ORDER BY expires_at,accessed_at LIMIT ?", (batch,))
            for row in rows:
                if int(row['expires_at']) <= now_ms():
                    removed += int(await tx.execute(f'DELETE FROM {self.cache} WHERE cache_key=?', (row['cache_key'],)))
            total = await tx.fetchone(f'SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS bytes FROM {self.cache}')
            count, size = int(total['n']), int(total['bytes'])
            rows = await tx.fetchall(f"SELECT cache_key,size_bytes FROM {self.cache} WHERE status NOT IN ('running','pending') ORDER BY accessed_at LIMIT ?", (batch,))
            for row in rows:
                if count <= max_entries and size <= max_bytes:
                    break
                removed += int(await tx.execute(f'DELETE FROM {self.cache} WHERE cache_key=?', (row['cache_key'],)))
                count -= 1
                size -= int(row['size_bytes'])
        return removed
