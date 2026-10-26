"""Transfer only original negative-result review, never a child execution lease."""
from __future__ import annotations

import json

from .jobs import JobError, _hash
from .model_exit import content_reference


def identity(manager, source, *, legacy=False):
    return ('controller_review_' if legacy else 'ctl_review_') + _hash([manager.store.store_id, source['job_id'], source['generation']])[:48]


def _public(original, key, owner):
    keys = ('version', 'outcome_scope', 'source_job_id', 'generation', 'outcome', 'resolution', 'result',
            'calls', 'original_call', 'checkpoint', 'model_exit', 'new_effect_requests',
            'source_model_usage', 'execution_authority', 'acknowledgement_created', 'billing_settled')
    return {**{k: original[k] for k in keys if k in original},
            'original_owner_job_id': owner['job_id'], 'review_transfer_event_id': key}


async def read(manager, tx, source, result, *, optional=True):
    from .abort_outcomes import notice
    from .managed_delivery import _main
    from .original_owner_review import read as original_owner
    from .tool_call_plans import _read

    candidates = await tx.fetchall(f'SELECT event_id FROM {manager.events} WHERE event_id IN (?,?)',
        (identity(manager, source), identity(manager, source, legacy=True)))
    if not candidates:
        if optional:
            return None
        raise JobError('controller_handoff_required')
    if len(candidates) != 1:
        raise JobError('controller_handoff_ambiguous')
    key = candidates[0]['event_id']
    data = await _read(manager, tx, key, source, 'controller_review_handoff')
    binding = await original_owner(manager, tx, source)
    if binding is None:
        raise JobError('controller_handoff_original_owner_required')
    main = await _main(manager, tx, source)
    old = await notice(manager, tx, source, result, original=True)
    required = {'version', 'source_job_id', 'main_session_id', 'review_owner', 'result', 'original_notice',
                'evidence', 'public', 'message_hash', 'execution_authority', 'resource_stop_confirmed', 'billing_settled'}
    from .abort_outcomes import render
    from .controller_retirement import read_retirement
    from .execution_membership import seal_id
    from .pending_calls import identity as disposition_id
    owner = await manager._row(tx, binding['owner']['job_id'])
    retirement = (await read_retirement(manager, tx, owner, binding['owner']))[0]['event_id']
    required_ids = {retirement, disposition_id(manager, source), seal_id(manager, binding['owner']['job_id'],
                    binding['owner']['generation']), 'model_closed_' + source['job_id']}
    if (set(data) != required or type(data['version']) is not int or data['version'] != 1
            or data['source_job_id'] != source['job_id'] or data['main_session_id'] != main['session_id']
            or data['review_owner'] != binding or data['result'] != content_reference(result)
            or data['original_notice'] != content_reference(old[0])
            or any(data[k] is not False for k in ('execution_authority', 'resource_stop_confirmed', 'billing_settled'))
            or data['public'] != _public(old[1]['public'], key, binding['owner'])
            or data['message_hash'] != _hash(render(data['public']))
            or type(data['evidence']) is not list or not 1 <= len(data['evidence']) <= 2000):
        raise JobError('controller_handoff_corrupt')
    ids = set()
    for ref in data['evidence']:
        if type(ref) is not dict or type(ref.get('event_id')) is not str or ref['event_id'] in ids:
            raise JobError('controller_handoff_corrupt')
        ids.add(ref['event_id'])
        event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (ref['event_id'],))
        if event is None or content_reference(event) != ref:
            raise JobError('controller_handoff_evidence_changed')
    if not required_ids.issubset(ids):
        raise JobError('controller_handoff_evidence_missing')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    if event['caller_session_id'] != main['session_id']:
        raise JobError('controller_handoff_corrupt')
    return event, data


async def _quiescent(manager, tx, actor, owner, source, binding):
    """Retirement fences new work; every existing descendant needs exit evidence."""
    from .execution_membership import read as membership
    from .execution_membership import seal, seal_id
    from .model_exit import require_stopped
    from .pending_calls import _closed
    from .workflow_controller import proof

    if proof(owner) != binding['owner'] or owner['intake'] != 'closed':
        raise JobError('controller_handoff_owner_not_retired')
    from .controller_retirement import read_retirement
    retirement, data = await read_retirement(manager, tx, owner, binding['owner'])
    key = retirement['event_id']
    expected = {'version': 1, 'lease': binding['owner'], 'kind': owner['kind'],
        'execution_role': owner['execution_role'], 'session_id': owner['session_id'],
        'reason': 'sdk_controller_context_exited', 'execution_authority': False,
        'resource_stop_confirmed': False, 'result_handled': False}
    if data != expected or owner['kind'] not in ('subagent', 'workflow_node'):
        raise JobError('controller_handoff_owner_not_retired')
    evidence = [await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))]
    frontier, seen = [owner['job_id']], {owner['job_id']}
    for _ in range(64):
        if not frontier:
            break
        following = []
        for parent in frontier:
            children = await tx.fetchall(f'SELECT * FROM {manager.jobs} WHERE parent_job_id=? ORDER BY job_id LIMIT 1001', (parent,))
            for row in children:
                if row['job_id'] in seen or len(seen) >= 1000:
                    raise JobError('controller_handoff_scope_too_large')
                seen.add(row['job_id'])
                await manager._authorize(tx, actor, row['job_id'])
                if row['intake'] != 'closed' or row['resource_state'] not in ('stopped', 'not_started'):
                    raise JobError('controller_handoff_resource_unconfirmed')
                if row['kind'] == 'send':
                    closed = await _closed(manager, tx, row)
                    await require_stopped(manager, tx, row, closed)
                    event_id = 'model_closed_' + row['job_id']
                else:
                    from .result_revisions import result_id
                    event_id = result_id(row['job_id'], row['generation'])
                    result = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (event_id,))
                    if result is None or json.loads(result['payload_json']).get('resource_state') not in ('stopped', 'not_started'):
                        raise JobError('controller_handoff_resource_unconfirmed')
                evidence.append(await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (event_id,)))
                following.append(row['job_id'])
        frontier = following
    else:
        raise JobError('controller_handoff_scope_too_large')
    if source['job_id'] not in seen:
        raise JobError('controller_handoff_source_missing')
    # No additional execution can join this retired scope. Seal only its actual
    # existing members; this creates no result, processing ACK or new generation.
    await seal(manager, tx, owner)
    frozen = await membership(manager, tx, owner, owner['generation'])
    if not frozen['sealed']:
        raise JobError('controller_handoff_members_unsealed')
    evidence.append(await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?',
                                    (seal_id(manager, owner['job_id'], owner['generation']),)))
    return evidence


async def transfer_review(loop, actor, source_job_id):
    """Host mechanical port after actual exit; enqueue original-main review only."""
    from .managed_completion import _host_only

    _host_only()
    runtime = getattr(loop, 'task_runtime', None)
    if runtime is None or runtime.actor(actor.session_id) != actor or await loop.ensure_store() is not runtime.manager.store:
        raise JobError('task_runtime_required')
    response = await transfer_review_runtime(runtime, actor, source_job_id)
    try:
        loop._note_committed_inbox(actor.session_id)
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Original review handoff persisted; local inbox hint failed')
    return response


async def transfer_review_runtime(runtime, actor, source_job_id):
    """Host-runtime form for mechanical recovery without constructing a model."""
    from .abort_outcomes import _inbox, notice, render, result_event
    from .managed_completion import _host_only
    from .managed_delivery import _main
    from .original_owner_review import read as original_owner
    from .pending_calls import require_disposition
    from .task_runtime import TaskRuntime
    from .tool_call_plans import _event

    _host_only()
    if not isinstance(runtime, TaskRuntime) or runtime.actor(actor.session_id) != actor:
        raise JobError('task_runtime_required')
    runtime._managed_completions.check()
    manager = runtime.manager
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, source_job_id)
        binding = await original_owner(manager, tx, source)
        main = await _main(manager, tx, source)
        if binding is None or actor.session_id != main['session_id']:
            raise JobError('controller_handoff_requires_original_main')
        result = await result_event(manager, tx, source)
        old = await read(manager, tx, source, result)
        if old is None:
            if source['result_status'] == 'handled':
                raise JobError('controller_handoff_already_handled')
            owner = await manager._authorize(tx, actor, binding['owner']['job_id'])
            evidence = await _quiescent(manager, tx, actor, owner, source, binding)
            disposition = await require_disposition(manager, tx, source)
            evidence.append(await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (disposition['event_id'],)))
            original = await notice(manager, tx, source, result, original=True)
            inbox = await _inbox(manager, tx, source, original)
            if inbox['status'] not in ('pending', 'delivered'):
                raise JobError('controller_handoff_original_notice_claimed')
            key = identity(manager, source)
            public = _public(original[1]['public'], key, binding['owner'])
            data = {'version': 1, 'source_job_id': source_job_id, 'main_session_id': main['session_id'],
                'review_owner': binding, 'result': content_reference(result), 'original_notice': content_reference(original[0]),
                'evidence': [content_reference(e) for e in evidence], 'public': public,
                'message_hash': _hash(render(public)), 'execution_authority': False,
                'resource_stop_confirmed': False, 'billing_settled': False}
            await _event(manager, tx, key, source, 'controller_review_handoff', data)
            await tx.execute(f'UPDATE {manager.events} SET caller_session_id=? WHERE event_id=?', (actor.session_id, key))
            columns = ('session_id', 'item_id', 'kind', 'mode', 'status', 'content', 'created_at')
            await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, columns),
                (actor.session_id, key, 'original_call_outcome', 'queue', 'pending', render(public), manager.now()))
            await tx.execute(f"UPDATE {manager.store.t.follow_up_queue} SET status='transferred' WHERE session_id=? AND item_id=? AND status='pending'",
                             (source['session_id'], original[0]['event_id']))
        found = await read(manager, tx, source, result, optional=False)
        await _inbox(manager, tx, source, found)
    return {'source_job_id': source_job_id, 'notification_id': found[0]['event_id'],
            'main_session_id': actor.session_id, 'execution_authority': False,
            'acknowledgement_created': False, 'billing_settled': False}


async def handoff_retired_reviews(runtime, actor, owner_job_id, *, limit=100):
    """Bounded host recovery for an original owner's known negative batches."""
    from .managed_completion import _host_only
    from .managed_delivery import _main
    from .task_runtime import TaskRuntime

    _host_only()
    if not isinstance(runtime, TaskRuntime) or type(limit) is not int or not 1 <= limit <= 100:
        raise JobError('invalid_controller_handoff_query')
    runtime._managed_completions.check()
    manager = runtime.manager
    async with manager._write() as tx:
        owner = await manager._authorize(tx, actor, owner_job_id)
        main = await _main(manager, tx, owner)
        if main['session_id'] != actor.session_id:
            raise JobError('controller_handoff_requires_original_main')
        rows = await tx.fetchall(f"SELECT j.job_id FROM {manager.jobs} j WHERE j.kind='send' "
            "AND j.control_owner_job_id=? AND j.control_owner_generation=? AND j.result_status NOT IN ('none','handled') "
            f"AND EXISTS (SELECT 1 FROM {manager.events} e WHERE e.job_id=j.job_id AND e.event_type='tool_call_abort') "
            "ORDER BY j.job_id LIMIT ?", (owner_job_id, owner['generation'], limit+1))
    if len(rows) > limit:
        raise JobError('controller_handoff_scope_too_large')
    results = []
    for row in rows:
        try:
            results.append(await transfer_review_runtime(runtime, actor, row['job_id']))
        except JobError as error:
            results.append({'source_job_id': row['job_id'], 'state': 'unconfirmed', 'error': error.code,
                            'execution_authority': False})
    return results
