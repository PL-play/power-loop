"""Cold maintenance and audit of operation records' original permissions.

An older opaque record needs caller attribution in an explicit offline bridge;
opening or stamping the current protocol must never invent that permission.
"""
from __future__ import annotations

import json

from ..jobs import _hash
from ..model_exit import content_reference
from .schema import StoreSchemaError


async def verify_operation_records(tx, prefix, *, records=None):
    table = prefix+'job_events'
    rows = records if records is not None else await tx.fetchall(
        f"SELECT * FROM {table} WHERE event_type='operation' AND "
        "(action='steer' OR (action='nudge' AND call_ref_json IS NOT NULL))")
    for row in rows:
        # Host wait/workflow continuations have no raw tool call. Their own
        # original decision/effect validators supply permission,not this port.
        if row['action'] == 'nudge' and row['call_ref_json'] is None:
            continue
        try:
            key = row['operation_key']
            source = await tx.fetchone(f'SELECT * FROM {table} WHERE event_id=?', ('record_req_'+key[:48],))
            if source is None:
                # Other current nudge domains have their explicit source link
                # in their structured outcome. This check is a cold boundary.
                if row['action'] != 'nudge':
                    raise ValueError()
                payload = json.loads(row['payload_json'])
                reference = payload['request_ref']
                source = await tx.fetchone(f'SELECT * FROM {table} WHERE event_id=?', (reference['event_id'],))
                if source is None or content_reference(source) != reference:
                    raise ValueError()
                data = json.loads(source['payload_json'])
                if data['action'] not in ('nudge', 'resume', 'resume_node', 'restore'):
                    raise ValueError()
            else:
                data = json.loads(source['payload_json'])
                request = data['request']
                fingerprint = _hash([request['job_id'], request['request'], request['call_ref']])
                if (data['action'] != 'record' or request['operation_action'] != row['action']
                        or request['job_id'] != row['job_id']
                        or request['call_ref'] != json.loads(row['call_ref_json'])
                        or request['tool_call_id'] != row['tool_call_id']
                        or request['call_ref']['session_id'] != row['caller_session_id']
                        or request['result'] != json.loads(row['payload_json'])
                        or row['request_hash'] != _hash([fingerprint, content_reference(source)])):
                    raise ValueError()
            if (source['event_type'] != 'task_control_request' or data['phase'] != 'accepted'
                    or type(data['version']) is not int or data['version'] != 1
                    or source['request_hash'] != _hash(data)):
                raise ValueError()
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreSchemaError('operation record requires valid original caller permission; use the offline bridge') from exc
