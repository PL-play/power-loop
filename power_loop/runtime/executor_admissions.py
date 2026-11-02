"""Derived logical-executor gates for the first registration/claim race.

Registered boots retain their existing per-boot shared admission. Only an
absent boot needs this shared group gate;registration holds it exclusively.
Registration events supply the registered fact during offline reconstruction.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from .jobs import _hash


def key(manager, identity):
    return 'exec_group_'+_hash([manager.store.store_id,
        identity.tenant_id, identity.principal_id, identity.executor_id])[:48]


async def lock(manager, tx, identity, *, registration=False):
    group, table = key(manager, identity), manager.prefix+'task_executor_admissions'
    known = tx._executor_group_locks.get(group)
    if known is not None and (not registration or known[0]):
        return known[1]
    suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR UPDATE' if registration else ' FOR SHARE'
    sql = f'SELECT registered FROM {table} WHERE group_key=?'+suffix
    row = await tx.fetchone(sql, (group,))
    if row is None:
        # The first absence claim creates a lockable derived row in this same
        # writer. A racing registration must wait for the actual claim commit.
        await tx.write_executor_group(group,
            manager.db.dialect.insert_ignore(table, ('group_key', 'registered')),
            (group, int(registration)))
        row = await tx.fetchone(sql, (group,))
    registered = bool(row['registered'])
    if registration and not registered:
        await tx.write_executor_group(group,
            f'UPDATE {table} SET registered=1 WHERE group_key=?', (group,))
        registered = True
    tx._executor_group_locks[group] = registration, registered
    return registered


async def rebuild(tx, prefix, store_id):
    """Offline only:reconstruct registered gates from original registrations."""
    from . import executor_registry as registry
    from .store.schema import StoreSchemaError

    manager = SimpleNamespace(store=SimpleNamespace(store_id=store_id), prefix=prefix)
    originals = await tx.fetchall(f"SELECT * FROM {prefix}task_executor_events WHERE action='register' ORDER BY event_id")
    expected = set()
    for event in originals:
        try:
            identity = registry.ExecutorIdentity(**json.loads(event['payload_json'])['identity'])
            saved = registry._decode(manager, identity, event)
            if saved['request']['action'] != 'register' or saved['response']['state'] != 'starting' or saved['response']['version'] != 1:
                raise ValueError()
            expected.add(key(manager, identity))
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreSchemaError('executor admission registration original invalid') from exc
    table = prefix+'task_executor_admissions'
    rows = await tx.fetchall(f'SELECT * FROM {table}')
    if any(row['registered'] and row['group_key'] not in expected for row in rows):
        raise StoreSchemaError('executor admission registration original missing')
    actual = {row['group_key']: row['registered'] for row in rows}
    for group in sorted(expected):
        if group not in actual:
            await tx.execute(f'INSERT INTO {table} (group_key,registered) VALUES (?,1)', (group,))
        elif not actual[group]:
            await tx.execute(f'UPDATE {table} SET registered=1 WHERE group_key=?', (group,))
