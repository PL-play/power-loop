"""One offline normalization of deployed lookup keys; never a runtime fallback."""
from __future__ import annotations

import json
import re

from ..jobs import _hash, _json
from .schema import StoreSchemaError


async def rekey(tx, prefix):
    changed = {'event_ids': await rekey_events(tx, prefix), 'operation_keys': 0, 'workflow_keys': 0}
    identity = await tx.fetchone(f'SELECT store_id FROM {prefix}task_store WHERE id=1')
    operations = await tx.fetchall(f'SELECT e.*,j.tenant_id,j.principal_id FROM {prefix}job_events e '
        f'JOIN {prefix}jobs j ON j.job_id=e.job_id WHERE e.event_type=\'operation\' AND e.operation_key IS NOT NULL')
    for row in operations:
        if row['call_ref_json'] is None:
            if not re.fullmatch('[0-9a-f]{64}', row['operation_key']):
                raise StoreSchemaError('protocol21 system operation requires its original system key for offline attribution')
            continue
        ref = json.loads(row['call_ref_json'])
        origin = ['model', ref['store_id'], row['caller_session_id'], int(ref['seq']), row['tool_call_id']]
        key = _hash([identity['store_id'], row['tenant_id'], row['principal_id'], row['action'], origin])
        if key != row['operation_key']:
            await tx.execute(f'UPDATE {prefix}job_events SET operation_key=? WHERE event_id=?', (key, row['event_id']))
            changed['operation_keys'] += 1
    rows = await tx.fetchall(f'SELECT session_id,state_key,value_json FROM {prefix}session_runtime_state '
        "WHERE state_key LIKE 'workflow:run:%' OR state_key LIKE 'workflow:history:%'")
    from ...workflow.journal import _run_token
    for row in rows:
        value = json.loads(row['value_json'])
        run_id = value['run_id']
        namespace = 'workflow:run:' if row['state_key'].startswith('workflow:run:') else 'workflow:history:'
        key = namespace + _run_token(run_id)
        if key == row['state_key']:
            continue
        existing = await tx.fetchone(f'SELECT value_json FROM {prefix}session_runtime_state '
            'WHERE session_id=? AND state_key=?', (row['session_id'], key))
        if existing is not None:
            raise StoreSchemaError('protocol21 workflow key collision; inspect both originals before upgrade')
        await tx.execute(f'UPDATE {prefix}session_runtime_state SET state_key=? WHERE session_id=? AND state_key=?',
            (key, row['session_id'], row['state_key']))
        changed['workflow_keys'] += 1
    return changed


def _rewrite(value, identities, hashes, updated):
    if isinstance(value, str):
        return identities.get(value, hashes.get(value, value))
    if isinstance(value, list):
        result = [_rewrite(item, identities, hashes, updated) for item in value]
    elif isinstance(value, dict):
        result = {key:_rewrite(item, identities, hashes, updated) for key,item in value.items()}
        if 'evidence_hash' in value:
            original = {key:item for key,item in value.items() if key != 'evidence_hash'}
            if value['evidence_hash'] == _hash(original):
                result['evidence_hash'] = _hash({key:item for key,item in result.items() if key != 'evidence_hash'})
                if result['evidence_hash'] != value['evidence_hash']:
                    updated[value['evidence_hash']] = result['evidence_hash']
    else:
        return value
    if result != value:
        updated[_hash(value)] = _hash(result)
    return result


async def rekey_events(tx, prefix):
    """Rename deployed long IDs and rewrite their structured reference graph.

    Raw messages and operation IDs remain original. Hashes follow transformed
    JSON originals; an opaque fingerprint is never replaced with a guessed hash.
    All data changes share the migration transaction, after the final MySQL DDL.
    """
    rows = await tx.fetchall(f'SELECT event_id,payload_json,request_hash FROM {prefix}job_events')
    identities = {}
    for row in rows:
        old = row['event_id']
        new = old
        for source,target in (('completion_window_', 'cwin_'), ('controller_retired_', 'ctl_retire_'),
                              ('controller_review_', 'ctl_review_')):
            if re.fullmatch(re.escape(source)+'[0-9a-f]{48}', old):
                new = target + old[len(source):]
                break
        match = re.fullmatch(r'(mt_.+_)([0-9a-f]{48})', old)
        if match and len(old) > 64:
            new = match[1] + match[2][:64-len(match[1])]
        if new != old:
            identities[old] = new
    if not identities:
        return 0
    targets = [identities.get(row['event_id'], row['event_id']) for row in rows]
    if len(set(targets)) != len(targets):
        raise StoreSchemaError('protocol21 event ID collision; inspect both originals before upgrade')
    originals = {row['event_id']:json.loads(row['payload_json']) for row in rows}
    hashes = {}
    for _ in range(len(rows)+32):
        updated = {}
        payloads = {key:_rewrite(value, identities, hashes, updated) for key,value in originals.items()}
        if updated == hashes:
            break
        hashes = updated
    else:
        raise StoreSchemaError('protocol21 event reference graph cannot be normalized')
    for row in rows:
        old, payload = row['event_id'], payloads[row['event_id']]
        fingerprint = hashes.get(row['request_hash'], row['request_hash'])
        if payload != originals[old] and row['request_hash'] is not None and row['request_hash'] not in hashes:
            raise StoreSchemaError('protocol21 original fingerprint requires offline attribution')
        if old in identities or payload != originals[old] or fingerprint != row['request_hash']:
            await tx.execute(f'UPDATE {prefix}job_events SET event_id=?,payload_json=?,request_hash=? WHERE event_id=?',
                (identities.get(old, old), _json(payload), fingerprint, old))
    for table,key_columns,column,hash_column in (
        ('jobs', ('job_id',), 'result_json', None),
        ('session_runtime_state', ('session_id', 'state_key'), 'value_json', None),
        ('task_budget_attempts', ('attempt_id',), 'state_json', 'state_hash'),
    ):
        columns = (*key_columns, column, *((hash_column,) if hash_column else ()))
        records = await tx.fetchall('SELECT '+','.join(columns)+f' FROM {prefix}{table} WHERE {column} IS NOT NULL')
        for row in records:
            value = json.loads(row[column])
            transformed = _rewrite(value, identities, hashes, {})
            if transformed == value:
                continue
            sets, values = [column+'=?'], [_json(transformed)]
            if hash_column:
                if row[hash_column] != _hash(value):
                    raise StoreSchemaError('protocol21 accounting original hash mismatch')
                sets.append(hash_column+'=?')
                values.append(_hash(transformed))
            await tx.execute(f'UPDATE {prefix}{table} SET '+','.join(sets)+' WHERE '+
                ' AND '.join(key+'=?' for key in key_columns), (*values, *(row[key] for key in key_columns)))
    return len(identities)
