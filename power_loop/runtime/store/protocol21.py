"""One offline normalization of deployed lookup keys; never a runtime fallback."""
from __future__ import annotations

import json
import re

from ..jobs import _hash
from .schema import StoreSchemaError


async def rekey(tx, prefix):
    changed = {'operation_keys': 0, 'workflow_keys': 0}
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
