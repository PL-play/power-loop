"""Original child-owner evidence and its retained SDK controller, never promotion."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, replace

from .jobs import JobError, _hash
from .model_exit import content_reference

PROTOCOL = 'owner_abort_review_protocol'


def admission_id(manager,owner):
    return 'owner_review_' + _hash([manager.store.store_id,owner['job_id']])[:48]


async def _declaration(manager,tx,owner):
    from .completion_windows import _creation
    creation, _ = await _creation(manager,tx,owner)
    value = json.loads(creation['payload_json']).get('request',{}).get(PROTOCOL)
    if value is not None and (type(value) is not int or value != 1):
        raise JobError('original_owner_review_contract_invalid')
    return creation,value


async def admit(manager,tx,owner):
    creation,value = await _declaration(manager,tx,owner)
    if owner['generation'] != 1:
        await enabled(manager,tx,owner)
        return
    old = await tx.fetchone(f'SELECT event_id FROM {manager.events} WHERE event_id=?',(admission_id(manager,owner),))
    if value is None:
        if old is not None:
            raise JobError('original_owner_review_contract_invalid')
        return
    from .execution_membership import scope_id
    from .tool_call_plans import _event
    from .workflow_controller import proof
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',(scope_id(manager,owner['job_id'],1),))
    data = {'version':1,'origin':proof(owner),'creation':content_reference(creation),'scope':content_reference(event)}
    await _event(manager,tx,admission_id(manager,owner),owner,'owner_review_admission',data)
    await enabled(manager,tx,owner)


async def enabled(manager,tx,owner):
    from .execution_membership import _lease, scope_id
    from .pending_calls import _object
    creation,value = await _declaration(manager,tx,owner)
    key = admission_id(manager,owner)
    data = await _object(manager,tx,manager.events,'payload_json','event_id=?',(key,),32768)
    if value is None and data is None:
        return None
    if value != 1 or data is None or set(data) != {'version','origin','creation','scope'} or type(data['version']) is not int or data['version'] != 1:
        raise JobError('original_owner_review_admission_required')
    origin = _lease(data['origin'])
    if origin['job_id'] != owner['job_id'] or origin['generation'] != 1:
        raise JobError('original_owner_review_admission_invalid')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',(key,))
    frozen = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',(scope_id(manager,owner['job_id'],1),))
    if (event['event_type'] != 'owner_review_admission' or event['job_id'] != owner['job_id'] or event['delivery_status'] != 'not_applicable'
            or event['request_hash'] != _hash(data) or frozen is None
            or _hash(json.loads(frozen['payload_json'])['lease']) != _hash(origin)
            or _hash(data['scope']) != _hash(content_reference(frozen))
            or _hash(data['creation']) != _hash(content_reference(creation))):
        raise JobError('original_owner_review_admission_invalid')
    return creation


async def read(manager, tx, source):
    from .execution_membership import member_id, members, scope, scope_id
    from .managed_completion import _scope
    from .model_exit import historical_row
    from .pending_calls import _object
    if source['execution_role'] == 'main':
        return None
    if source['kind'] != 'send' or source['execution_role'] not in ('workflow_node','subagent'):
        raise JobError('original_owner_review_role_invalid')
    original = await _scope(manager, tx, source)
    lease = original['owner']
    if lease is None or source['parent_job_id'] != lease['job_id']:
        raise JobError('original_owner_review_binding_invalid')
    current = await manager._row(tx, lease['job_id'])
    if current is None:
        raise JobError('original_owner_review_binding_invalid')
    frozen = await scope(manager, tx, lease['job_id'], lease['generation'])
    if (_hash(frozen['lease']) != _hash(lease) or frozen['kind'] != source['execution_role']
            or frozen['execution_role'] != source['execution_role']
            or (frozen['tenant_id'],frozen['principal_id']) != (source['tenant_id'],source['principal_id'])):
        raise JobError('original_owner_review_binding_invalid')
    owner = historical_row(current,frozen)
    creation = await enabled(manager,tx,owner)
    if creation is None:
        raise JobError('original_owner_review_adapter_required')
    links = [x for x in await members(manager,tx,frozen) if _hash(x['payload']['member']) == _hash(original['lease'])]
    if len(links) != 1:
        raise JobError('original_owner_review_member_missing')
    meta = await _object(manager,tx,manager.store.t.sessions,'metadata_json','session_id=?',(source['session_id'],),32768)
    if meta is None or meta.get('execution_role') != source['execution_role'] or meta.get('kind') == 'wf_driver':
        raise JobError('original_owner_review_session_invalid')
    source_creation = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE job_id=? AND action='create'",(source['job_id'],))
    if source_creation is None:
        raise JobError('original_owner_review_source_missing')
    source_request = json.loads(source_creation['payload_json']).get('request',{})
    limit = source_request.get('original_round_limit')
    if type(source_request.get(PROTOCOL)) is not int or source_request[PROTOCOL] != 1:
        raise JobError('original_owner_review_source_missing')
    if type(limit) is not int or not 0 <= limit <= 1000000:
        raise JobError('original_owner_round_limit_invalid')
    refs = [source_creation['event_id']]
    if owner['kind'] == 'workflow_node':
        from .node_inputs import descriptor
        await descriptor(manager,tx,owner,source['session_id'],historical=True)
        refs.append(manager._node_session_event_id(owner['job_id'],source['session_id'],lease['generation']))
    elif frozen['session_id'] != source['session_id'] or not meta.get('standalone_subagent'):
        raise JobError('original_owner_review_session_invalid')
    refs.extend([scope_id(manager,source['job_id'],source['generation']),scope_id(manager,owner['job_id'],lease['generation']),
                 member_id(manager,source['job_id'],source['generation']),creation['event_id'],admission_id(manager,owner)])
    evidence = []
    for key in refs:
        event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',(key,))
        if event is None:
            raise JobError('original_owner_review_member_missing')
        evidence.append(content_reference(event))
    return {'version':1,'owner':lease,'session_id':source['session_id'],'execution_role':source['execution_role'],'round_limit':limit,'evidence':evidence}


async def require_handler(manager,tx,source,handler):
    from .managed_completion import _scope
    expected = await read(manager,tx,source)
    if handler['session_id'] != source['session_id'] or handler['kind'] != 'send':
        raise JobError('original_owner_review_handler_invalid')
    if expected is None:
        from .managed_delivery import _main
        await _main(manager,tx,handler)
        return
    actual = await _scope(manager,tx,handler)
    if (actual['execution_role'] != expected['execution_role'] or _hash(actual['owner']) != _hash(expected['owner'])
            or actual['parent_job_id'] != expected['owner']['job_id']):
        raise JobError('original_owner_review_handler_invalid')


@dataclass(eq=False)
class _Failure:
    source: object
    owner: object
    error: BaseException
    round_limit: int
    attempted: bool = False


def remember(source,owner,error,round_limit):
    if owner is not None and source.runtime is owner.runtime:
        ticket = _Failure(source,owner,error,round_limit)
        source.resource_evidence['original_owner_failure'] = ticket
        error._power_loop_owner_failure = ticket


async def require_live(manager,tx,actor,source,ticket):
    from .execution_membership import read as membership
    from .task_runtime import _OWNER, _SEND
    if (type(ticket) is not _Failure or ticket.source.resource_evidence.get('original_owner_failure') is not ticket
            or ticket.source.runtime.manager is not manager or ticket.source.actor != actor
            or ticket.source.lease.job_id != source['job_id'] or _OWNER.get() is not ticket.owner
            or ticket.owner.stop_renewal.is_set()):
        raise JobError('original_owner_controller_required')
    runtime = ticket.source.runtime
    if not any(local['binding'] is ticket.owner and local['task'] is asyncio.current_task()
               for local in runtime._local_executions.values()):
        raise JobError('original_owner_controller_required')
    active = _SEND.get()
    if active is not None and active.actor.session_id == actor.session_id:
        raise JobError('original_owner_model_still_active')
    binding = await read(manager,tx,source)
    if (binding is None or _hash(binding['owner']) != _hash(asdict(ticket.owner.lease))
            or type(ticket.round_limit) is not int or binding['round_limit'] != ticket.round_limit):
        raise JobError('original_owner_controller_required')
    owner = await manager._fence(tx,ticket.owner.actor,ticket.owner.lease)
    if (await membership(manager,tx,owner,owner['generation']))['sealed']:
        raise JobError('original_owner_controller_sealed')
    from .completion_windows import require_time
    await require_time(manager,tx,owner)


async def try_handle(loop,sid,error,stop_event):
    from .cancellation import CancellationToken
    ticket = getattr(error,'_power_loop_owner_failure',None)
    if (type(ticket) is not _Failure or ticket.error is not error or ticket.attempted
            or isinstance(error,asyncio.CancelledError) or CancellationToken.from_any(stop_event).is_cancelled()):
        return None
    ticket.attempted = True
    runtime,source = ticket.source.runtime,ticket.source.lease.job_id
    if getattr(loop,'task_runtime',None) is not runtime or sid != ticket.source.actor.session_id:
        return None
    manager,actor = runtime.manager,ticket.source.actor
    try:
        from .abort_outcomes import _original, _reconcile, handled_references
        async with manager._write() as tx:
            row = await manager._authorize(tx,actor,source)
            await require_live(manager,tx,actor,row,ticket)
            closed,_,_,_,_ = await _original(manager,tx,row)
        from functools import partial

        from .owner_waits import _confirm
        await _confirm(partial(_reconcile,permit=ticket),loop,actor,source)
        ref = closed['evidence'].get('last_assistant_ref',{})
        spent = ref.get('round_index',0)+1
        if type(spent) is not int or spent >= ticket.round_limit:
            return None
        from .completion import WINDOW
        window = WINDOW.get()
        allowance = min(1,ticket.round_limit-spent)
        if window is not None:
            window.require_time()
            allowance = min(allowance,window.allowance({'rounds':spent}))
        if allowance <= 0:
            return None
        # Two sends in this newly-created child: projection keeps original calls
        # as data, without a fabricated missing tool reply or a migration model.
        from .representation import ProjectedRepresentation
        original_config = loop.config
        loop.config = replace(original_config,representation=ProjectedRepresentation(),migrate_history_on_switch=False)
        try:
            result = await loop.flush_follow_ups(sid,stop_event=stop_event,tools=[],max_rounds=allowance)
        finally:
            loop.config = original_config
        if window is not None and result is not None:
            if not window.consume(result,allowance):
                return None
        if result is None or result.status != 'completed':
            return None
        async with manager._write() as tx:
            row = await manager._authorize(tx,actor,source)
            await handled_references(manager,tx,row)
        from power_loop.agent.stateful_loop import StatefulResult
        usage = dict(ticket.source.resource_evidence.get('model_usage_totals',{}))
        for key,value in result.usage.items():
            usage[key] = usage.get(key,0)+value
        return StatefulResult(session_id=sid,status='failed',final_text=result.final_text,rounds=spent+result.rounds,usage=usage)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('Original owner outcome remains unresolved: %s',type(exc).__name__)
        return None


async def require_ready(manager,tx,owner):
    if owner['kind'] not in ('workflow_node','subagent'):
        return
    declared = await enabled(manager,tx,owner)
    from .pending_calls import ERROR, SNAPSHOT, _closed, negative_review_references, require_disposition
    rows = await tx.fetchall(f"SELECT j.* FROM {manager.jobs} j WHERE j.kind='send' AND j.control_owner_job_id=? AND j.control_owner_generation=?",
                             (owner['job_id'],owner['generation']))
    if declared is None:
        for source in rows:
            creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'",(source['job_id'],))
            request = json.loads(creation['payload_json']).get('request',{}) if creation else {}
            if PROTOCOL in request:
                raise JobError('original_owner_review_admission_required')
        return  # Explicit historical owners keep their original protocol.
    for source in rows:
        event = await tx.fetchone(f'SELECT event_id FROM {manager.events} WHERE event_id=?',('model_closed_'+source['job_id'],))
        if event is None:
            continue
        closed = await _closed(manager,tx,source)
        if SNAPSHOT in closed['evidence'] or ERROR in closed['evidence']:
            await require_disposition(manager,tx,source)
            await negative_review_references(manager,tx,source)
