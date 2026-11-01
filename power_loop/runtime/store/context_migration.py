"""Offline import of former host-control originals into task-context events.

Only explicit maintenance calls this importer. Runtime reads one event system.
"""
from __future__ import annotations

import json

from ..jobs import _hash
from .schema import StoreSchemaError


async def import_context_events(tx, db, prefix):
    rows = await tx.fetchall(f"SELECT session_id,state_key,value_json,updated_at FROM {prefix}session_runtime_state "
        "WHERE state_key LIKE ? ORDER BY session_id,state_key", ('power-loop.root-stop%',))
    prepared = []
    for row in rows:
        try:
            value = json.loads(row['value_json'])
            shapes = ({'version', 'actor', 'document', 'first_request_hash', 'evidence_hash'},
                      {'version', 'actor', 'request', 'response', 'evidence_hash'})
            actor = value['actor']
            if (set(value) not in shapes or type(value['version']) is not int or value['version'] != 1
                    or set(actor) != {'tenant_id', 'principal_id', 'session_id'}
                    or actor['session_id'] != row['session_id']
                    or value['evidence_hash'] != _hash({k: v for k, v in value.items() if k != 'evidence_hash'})):
                raise ValueError()
            identity = 'rootctl_' + _hash([actor, row['state_key']])[:48]
        except (ValueError, KeyError, TypeError) as error:
            raise StoreSchemaError('context migration original is invalid') from error
        event = {'event_id': identity, 'session_id': row['session_id'], 'state_key': row['state_key'],
                 'payload_json': row['value_json'], 'created_at': row['updated_at']}
        old = await tx.fetchone(f'SELECT * FROM {prefix}task_context_events WHERE event_id=?', (identity,))
        if old is not None and old != event:
            raise StoreSchemaError('context migration original conflicts')
        if old is None:
            prepared.append(event)
    for event in prepared:
        columns = tuple(event)
        await tx.execute(f"INSERT INTO {prefix}task_context_events ({','.join(columns)}) VALUES ("
            + ','.join('?' for _ in columns) + ')', tuple(event.values()))
