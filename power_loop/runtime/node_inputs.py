"""Read the node's frozen start input; the node job owns one real session."""
import json

from .jobs import JobError
from .result_revisions import result_id


async def descriptor(manager, reader, job, sid, *, historical=False):
    if job['kind'] != 'workflow_node' or job['session_id'] != sid:
        raise JobError('node_input_binding_mismatch')
    creation=await reader.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'",(job['job_id'],))
    ref=json.loads(creation['payload_json'])['request']['input_ref']
    start=await reader.fetchone(f'SELECT meta_json FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
        (ref['session_id'],ref['message_id']))
    route=json.loads(start['meta_json'])['descriptor']['config']['workflow_node']
    return {**route,'session_id':sid,'generation':job['generation']}


async def is_closed(manager, reader, job, sid):
    event=await reader.fetchone(f'SELECT event_id FROM {manager.events} WHERE event_id=?',(result_id(job['job_id'],job['generation']),))
    return event is not None


async def input_target(manager, reader, actor, send, event):
    job=await manager._authorize(reader, actor,event['job_id'])
    if job['kind']=='workflow_node' and (job['job_id']!=send['control_owner_job_id'] or job['session_id']!=actor.session_id):
        raise JobError('processing_input_mismatch')
    return job
