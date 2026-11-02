"""Offline protocol 21 -> final upgrade. Dry-run unless --apply --writers-stopped.

Run from the repository root: .venv/bin/python -m scripts.upgrade_protocol21.
The operator must stop writers and take a native database backup before applying.
This command never starts an executor, calls a provider, or reprices history.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from power_loop.runtime.store.factory import _make_database
from power_loop.runtime.store.schema import StoreSchemaError, validate_table_prefix
from power_loop.runtime.store.task_schema import (
    DEPLOYED_TASK_PROTOCOL_VERSION,
    DEPLOYED_TASK_SCHEMA_VERSION,
    TASK_PROTOCOL_VERSION,
    TASK_SCHEMA_VERSION,
    migrate_task_schema,
    task_schema_preflight,
)


async def upgrade(dsn, prefix, *, apply=False, writers_stopped=False):
    validate_table_prefix(prefix)
    if apply and not writers_stopped:
        raise StoreSchemaError('maintenance_required: --apply requires --writers-stopped')
    db = await _make_database(dsn)
    try:
        before = await task_schema_preflight(db, prefix)
        identity = await db.fetchone(f'SELECT store_id,protocol_version FROM {prefix}task_store WHERE id=1')
        source = (before['version'], identity['protocol_version'])
        if source not in ((DEPLOYED_TASK_SCHEMA_VERSION, DEPLOYED_TASK_PROTOCOL_VERSION),
                          (TASK_SCHEMA_VERSION, TASK_PROTOCOL_VERSION)):
            raise StoreSchemaError('unsupported_migration_source: only deployed protocol 21 -> final is supported')
        report = {'source': source, 'destination': (TASK_SCHEMA_VERSION, TASK_PROTOCOL_VERSION),
                  'store_id': identity['store_id'], 'before': before, 'applied': False}
        if apply:
            await migrate_task_schema(db, prefix, maintenance=True)
            report.update(applied=True, after=await task_schema_preflight(db, prefix))
        return report
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--prefix', default='pl_')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--writers-stopped', action='store_true')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(upgrade(args.dsn, args.prefix, apply=args.apply,
        writers_stopped=args.writers_stopped)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
