"""Rebuild a stopped fixture using the actual deployed protocol-21 DDL."""
from __future__ import annotations

import json
import re
from pathlib import Path

from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES


async def retain_protocol21_format(store):
    db, prefix = store._db, store.table_prefix
    fixture = json.loads((Path(__file__).parents[1]/'fixtures/protocol21-schema.json').read_text())
    statements = fixture['ddl'][db.dialect.name]
    tables = [re.search(r'CREATE TABLE IF NOT EXISTS fixture_(\w+)', sql)[1]
              for sql in statements if sql.startswith('CREATE TABLE') and 'fixture_schema_migrations ' not in sql]
    rows = {}
    for table in tables:
        source = 'usage_observations' if table == 'usage_rounds' else table
        rows[table] = await db.fetchall(f'SELECT * FROM {prefix}{source}')
    await db.execute(f'DROP VIEW {prefix}usage_rounds')
    for table in (*TASK_TABLES, *_STORE_TABLES):
        await db.execute(f'DROP TABLE IF EXISTS {prefix}{table}')
    indexes = set()
    for sql in statements:
        if sql.startswith('CREATE INDEX'):
            name = sql.split()[2]
            if name in indexes:
                continue
            indexes.add(name)
        else:
            indexes.update(re.findall(r'\b(?:KEY|INDEX)\s+(fixture_\w+)', sql))
        await db.execute(sql.replace('fixture_', prefix))
    for table, originals in rows.items():
        if not originals:
            continue
        if db.dialect.name == 'sqlite':
            cols = {r['name'] for r in await db.fetchall(f'PRAGMA table_info({prefix}{table})')}
        else:
            namespace = 'current_schema()' if db.dialect.name == 'postgres' else 'DATABASE()'
            cols = {r['column_name'] for r in await db.fetchall('SELECT column_name AS column_name '
                f'FROM information_schema.columns WHERE table_schema={namespace} AND table_name=?', (prefix+table,))}
        for original in originals:
            row = {key:value for key,value in original.items() if key in cols}
            if table == 'task_store':
                row['protocol_version'] = 21
            await db.execute(f'INSERT INTO {prefix}{table} ('+','.join(row)+') VALUES ('+
                ','.join('?' for _ in row)+')', tuple(row.values()))
    await db.execute(f'UPDATE {prefix}schema_migrations SET version=10 WHERE id=1')
    if db.dialect.name == 'postgres' and rows.get('messages'):
        await db.fetchone('SELECT setval(pg_get_serial_sequence(?,?),?,true)',
            (prefix+'messages', 'id', max(row['id'] for row in rows['messages'])))
