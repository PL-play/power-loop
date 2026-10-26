"""Explicit original-main review of an exited child's v2 pending calls.

This transfers only result review. It never grants execution, claims that all
owner resources stopped, or creates a processing or billing acknowledgement.
"""
from __future__ import annotations

from .jobs import JobError, JobLease, _hash
from .model_exit import content_reference


def identity(manager, source):
    return 'pc_owner_' + _hash([manager.store.store_id, source['job_id'], source['generation']])[:48]


async def _expected(manager, tx, source):
    from .controller_retirement import read_retirement
    from .controller_task_exit import identity as exit_id
    from .controller_task_exit import read as task_exit
    from .execution_membership import read as membership
    from .execution_membership import seal_id
    from .managed_delivery import _main
    from .model_exit import historical_row
    from .original_owner_review import read as original_owner
    from .pending_calls import SNAPSHOT, model

    closed = await model(manager, tx, source)
    if closed['evidence'].get(SNAPSHOT, {}).get('version') != 2:
        raise JobError('pending_owner_requires_v2_snapshot')
    binding = await original_owner(manager, tx, source)
    if binding is None:
        raise JobError('pending_owner_original_child_required')
    main = await _main(manager, tx, source)
    current = await manager._row(tx, binding['owner']['job_id'])
    frozen = await membership(manager, tx, current, binding['owner']['generation'])
    if not frozen['sealed'] or _hash(frozen['scope']['lease']) != _hash(binding['owner']):
        raise JobError('pending_owner_members_unsealed')
    owner = historical_row(current, frozen['scope'])
    lease = JobLease(**binding['owner'])
    await task_exit(manager, tx, owner, lease)
    retirement, data = await read_retirement(manager, tx, owner, lease)
    expected = {'version': 1, 'lease': binding['owner'], 'kind': owner['kind'],
        'execution_role': owner['execution_role'], 'session_id': owner['session_id'],
        'reason': 'sdk_controller_context_exited', 'execution_authority': False,
        'resource_stop_confirmed': False, 'result_handled': False}
    if _hash(data) != _hash(expected):
        raise JobError('pending_owner_retirement_changed')
    events = [retirement]
    for key in (exit_id(manager, lease), seal_id(manager, owner['job_id'], lease.generation)):
        events.append(await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,)))
    from .workflow_controller import proof
    return {'version': 1, 'source': proof(source), 'review_owner': binding,
        'main_session_id': main['session_id'], 'evidence': [content_reference(event) for event in events],
        'execution_authority': False, 'resource_stop_confirmed': False,
        'acknowledgement_created': False, 'billing_settled': False}


async def read(manager, tx, source, *, optional=False):
    from .tool_call_plans import _read
    data = await _read(manager, tx, identity(manager, source), source, 'pending_owner_transfer', optional=optional)
    if data is None:
        return None
    if _hash(data) != _hash(await _expected(manager, tx, source)):
        raise JobError('pending_owner_transfer_changed')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (identity(manager, source),))
    if event['caller_session_id'] != data['main_session_id']:
        raise JobError('pending_owner_transfer_changed')
    return event, data


async def transfer(manager, tx, actor, source):
    from .execution_membership import seal
    from .managed_delivery import _main
    from .original_owner_review import read as original_owner
    from .tool_call_plans import _event
    from .workflow_controller import proof

    main = await _main(manager, tx, source)
    if actor.session_id != main['session_id']:
        raise JobError('pending_call_requires_original_main')
    found = await read(manager, tx, source, optional=True)
    if found is not None:
        return found
    binding = await original_owner(manager, tx, source)
    if binding is None:
        raise JobError('pending_owner_original_child_required')
    owner = await manager._authorize(tx, actor, binding['owner']['job_id'])
    if _hash(proof(owner)) != _hash(binding['owner']) or owner['intake'] != 'closed':
        raise JobError('pending_owner_controller_not_retired')
    # A stopped original Task and closed intake fence all future admissions.
    # Sealing membership does not claim sibling resources are already stopped.
    await seal(manager, tx, owner)
    data = await _expected(manager, tx, source)
    await _event(manager, tx, identity(manager, source), source, 'pending_owner_transfer', data)
    await tx.execute(f'UPDATE {manager.events} SET caller_session_id=? WHERE event_id=?',
                     (actor.session_id, identity(manager, source)))
    return await read(manager, tx, source)
