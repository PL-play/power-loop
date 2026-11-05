"""P0 bounded maintenance: 100 expired inbox rows, 0/10,000 recent closed jobs.

History is a query dataset cloned from a real closed send row, not a claim that
10,000 full conversations were executed. Fixture insertion is outside timing.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES
from tests.design203_benchmark import percentile, rss, simple
from tests.unit.test_send_sql_profile import Profiled, _counts
from tests.write_meter import WriteMeter


async def insert_many(db, table, columns, rows):
    width = len(columns)
    for start in range(0, len(rows), 100):
        part = rows[start:start + 100]
        marks = ','.join('(' + ','.join('?' for _ in range(width)) + ')' for _ in part)
        await db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES {marks}",
                         tuple(value for row in part for value in row))


async def run(args):
    if '://' in args.dsn and not urlparse(args.dsn).path.endswith('_test'):
        raise ValueError('isolated _test database required')
    prefix = 'p0m_' + os.urandom(4).hex() + '_'
    store = await open_store(args.dsn, table_prefix=prefix, schema='tasks')
    db, now = store._db, int(time.time() * 1000)
    report = {'db': db.dialect.name, 'history_closed_jobs': args.history, 'expired_per_call': 100,
              'warmup': 10, 'pairs': []}
    try:
        async with simple(store, 0, 'send') as operation:
            await operation()
        template = await db.fetchone(f"SELECT * FROM {prefix}jobs WHERE supervision_status='closed' LIMIT 1")
        assert template is not None
        await store.close_session(template['session_id'])
        copies = []
        for n in range(args.history):
            row = {**template, 'job_id': f'history{n:08d}', 'root_job_id': f'history{n:08d}',
                   'completion_scope_job_id': f'history{n:08d}'}
            copies.append(tuple(row.values()))
        await insert_many(db, prefix + 'jobs', tuple(template), copies)
        assert (await db.fetchone(f'SELECT COUNT(*) AS n FROM {prefix}jobs'))['n'] == args.history
        sid = await store.create_session()
        sequence = 0

        async def seed():
            nonlocal sequence
            sequence += 1
            await insert_many(db, store.t.follow_up_queue,
                ('session_id', 'item_id', 'status', 'content', 'created_at'),
                [(sid, f'expired{sequence}-{i}', 'void', 'ok', 1) for i in range(100)])

        async def prune():
            result = await store.purge_expired(now, limit=100)
            assert result['follow_up_queue'] == 100, result

        for pair in range(3):
            for arm in (['B', 'repeat'] if pair % 2 == 0 else ['repeat', 'B']):
                for _ in range(10):
                    await seed()
                    await prune()
                times, cpu = [], 0
                before, peak = rss(), rss()
                for _ in range(34):
                    await seed()
                    start, used = time.perf_counter(), time.process_time()
                    await prune()
                    times.append((time.perf_counter() - start) * 1000)
                    cpu += time.process_time() - used
                    peak = max(peak, rss())
                report['pairs'].append({'pair': pair, 'arm': arm, 'completed': 34, 'errors': 0,
                    'p50_ms': statistics.median(times), 'p95_ms': percentile(times, .95),
                    'cpu_seconds': cpu, 'rss_before': before, 'rss_peak': peak, 'latencies_ms': times})
        # One separately instrumented call for cumulative/maximum transaction work.
        await seed()
        meter = await WriteMeter.install(db, prefix)
        try:
            store._db = profiled = Profiled(db, meter)
            await prune()
            report['counts'] = {'sql': _counts(profiled.records, profiled.tx_sites),
                'total_sql': len(profiled.records), 'transactions': profiled.transactions,
                'writes': await meter.report()}
        finally:
            store._db = db
            await meter.close()
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    finally:
        if db.dialect.name != 'sqlite':
            await db.execute(f'DROP VIEW IF EXISTS {prefix}usage_rounds')
            for table in (*TASK_TABLES, *_STORE_TABLES, 'schema_migrations'):
                await db.execute(f'DROP TABLE IF EXISTS {prefix}{table}')
        await store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--history', type=int, choices=(0, 10000), required=True)
    parser.add_argument('--out', required=True)
    asyncio.run(run(parser.parse_args()))
