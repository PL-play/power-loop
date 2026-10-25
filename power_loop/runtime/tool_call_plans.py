"""Complete native batches frozen before any callback, using actual tool contracts."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from .jobs import JobError, _hash, _json

FIELD = 'tool_call_plan_protocol'


@dataclass(frozen=True)
class _BatchBindings:
    plan_id: str
    plan_hash: str
    source_scope: str
    call_ref: str
    registrations: tuple


def _retained(pipeline, manager, source, ref, plan):
    from .workflow_controller import proof
    original = getattr(pipeline, '_native_tool_batch_bindings', None)
    if (type(original) is not _BatchBindings or original.plan_id != identity(manager, source, ref)
            or original.plan_hash != _hash(plan) or original.source_scope != _json(proof(source))
            or original.call_ref != _json(ref.to_dict())):
        # A persisted declaration alone is not the original process's actual
        # registered binding. Mechanical recovery never recreates this grant.
        raise JobError('tool_call_plan_original_requires_reconciliation')
    return original


def identity(manager, source, ref):
    return 'tcp_' + _hash([manager.store.store_id, source['job_id'], source['generation'], ref.to_dict()])[:48]


async def required(manager, tx, source):
    from .pending_calls import _object
    creates = await tx.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create' LIMIT 2", (source['job_id'],))
    if len(creates) != 1:
        raise JobError('tool_call_plan_source_missing')
    data = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (creates[0]['event_id'],), 262144)
    flag = data.get('request', {}).get(FIELD)
    if flag is None:
        return False
    if type(flag) is not int or flag != 1:
        raise JobError('tool_call_plan_protocol_invalid')
    return True


def registration_value(registered):
    if registered is None:
        return None
    from .tool_contracts import resolve
    implementation = resolve(registered)
    d = registered.definition
    return json.loads(_json({'schema': d.to_openai_tool(), 'required': list(d.required_params), 'interrupt': d.interrupt,
        'async_capable': d.async_capable if type(d.async_capable) is bool else sorted(d.async_capable),
        'usage_hint': d.usage_hint, 'background_hint': d.background_hint,
        'runtime_contract': asdict(implementation.runtime_contract) if implementation is not None else None}))


async def _event(manager, tx, key, source, kind, value):
    document = _json(value)
    if len(document.encode('utf-8')) > 262144:
        raise JobError('tool_call_plan_too_large')
    await tx.execute(f'INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)',
        (key, source['job_id'], kind, source['version'], _hash(value), document, 'not_applicable', manager.now()))


async def _read(manager, tx, key, source, kind, *, optional=False):
    from .pending_calls import _object
    value = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (key,), 262144)
    if value is None:
        if optional:
            return None
        raise JobError('tool_call_plan_missing')
    event = await tx.fetchone(f'SELECT job_id,event_type,request_hash,delivery_status FROM {manager.events} WHERE event_id=?', (key,))
    if (event['job_id'] != source['job_id'] or event['event_type'] != kind
            or event['request_hash'] != _hash(value) or event['delivery_status'] != 'not_applicable'):
        raise JobError('tool_call_plan_corrupt')
    return value


async def read(manager, tx, source, ref, *, optional=False):
    from .jobs import JobActor
    from .managed_completion import _scope
    from .managed_effects import _verify_call
    from .workflow_controller import proof

    key = identity(manager, source, ref)
    value = await _read(manager, tx, key, source, 'tool_call_plan', optional=optional)
    if value is None:
        return None
    anchor = await _read(manager, tx, key + '_seal', source, 'tool_call_plan_seal')
    await _scope(manager, tx, source)
    if (set(value) != {'version', 'scope', 'call_ref', 'entries'} or type(value['version']) is not int
            or value['version'] != 1 or _hash(value['scope']) != _hash(proof(source)) or _hash(value['call_ref']) != _hash(ref.to_dict())
            or type(value['entries']) is not list or not 0 < len(value['entries']) <= 64
            or _hash(anchor) != _hash({'version': 1, 'scope': proof(source), 'plan_id': key, 'plan_hash': _hash(value)})):
        raise JobError('tool_call_plan_corrupt')
    actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
    # Bound the original raw row before decoding it, including when a plan is
    # read mechanically after its original registry/process has disappeared.
    length = 'length(CAST(tool_calls_json AS BLOB))' if manager.db.dialect.name == 'sqlite' else 'octet_length(tool_calls_json)'
    size = await tx.fetchone(f'SELECT {length} AS n FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (ref.session_id, ref.seq))
    if size is None or size['n'] is None or size['n'] > 131072:
        raise JobError('tool_call_plan_raw_unavailable')
    raw = await tx.fetchone(f'SELECT tool_calls_json FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (ref.session_id, ref.seq))
    calls = json.loads(raw['tool_calls_json'])
    fields = {'call_id', 'tool_name', 'arguments_hash', 'registration', 'recovery', 'job_id', 'operation_id'}
    if (type(calls) is not list or not 0 < len(calls) <= 64
            or any(type(c) is not dict or type(c.get('id')) is not str for c in calls)
            or any(type(e) is not dict or set(e) != fields for e in value['entries'])):
        raise JobError('tool_call_plan_corrupt')
    if [e['call_id'] for e in value['entries']] != [c['id'] for c in calls]:
        raise JobError('tool_call_plan_call_set_changed')
    for entry in value['entries']:
        if set(entry) != {'call_id', 'tool_name', 'arguments_hash', 'registration', 'recovery', 'job_id', 'operation_id'}:
            raise JobError('tool_call_plan_corrupt')
        await _verify_call(manager, tx, actor, source, ref, entry['call_id'], entry['tool_name'], entry['arguments_hash'])
        if entry['recovery'] is not None:
            if (type(entry['recovery']) is not dict or set(entry['recovery']) != {'kind', 'request'}
                    or entry['recovery']['kind'] != 'managed_effect' or type(entry['recovery']['request']) is not dict):
                raise JobError('tool_call_plan_recovery_unsupported')
            from .managed_effects import descriptor, job_id
            row = await manager._authorize(tx, actor, entry['job_id'])
            found = await descriptor(manager, tx, row)
            if (found is None or row['job_id'] != job_id(manager, actor, ref, entry['call_id'])
                    or found[0]['event_id'] != entry['operation_id']
                    or json.loads(found[0]['payload_json'])['request'] != entry['recovery']['request']):
                raise JobError('tool_call_plan_registration_changed')
        elif entry['job_id'] is not None or entry['operation_id'] is not None:
            raise JobError('tool_call_plan_corrupt')
    return value


async def _prepare(pipeline, attempt):
    from .managed_effects import _document, _verify_call
    from .task_runtime import _SEND
    from .tool_contracts import policy, resolve, validate_registration

    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return
    runtime, actor = binding.runtime, binding.actor
    manager = runtime.manager
    seq = getattr(pipeline.sink, 'persisted_assistant_seq', None)
    if type(seq) is not int:
        raise JobError('persisted_tool_call_required')
    ref = await manager.store.message_ref(actor.session_id, seq)
    # Read only after a bounded metadata check; raw calls, not a rewritten hook
    # list, define the complete ordered set of original intentions.
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, binding.lease.job_id)
        manager._check_proof(source, binding.lease)
        if not await required(manager, tx, source):
            return
        from .pending_calls import current
        pending = await current(manager, tx, actor.session_id)
        if pending is None or pending.get('assistant_seq') != seq:
            raise JobError('tool_call_plan_pending_required')
        length = 'length(CAST(tool_calls_json AS BLOB))' if manager.db.dialect.name == 'sqlite' else 'octet_length(tool_calls_json)'
        size = await tx.fetchone(f'SELECT {length} AS n FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (actor.session_id, seq))
        if size is None or size['n'] is None or size['n'] > 131072:
            raise JobError('tool_call_plan_raw_unavailable')
        row = await tx.fetchone(f'SELECT tool_calls_json FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (actor.session_id, seq))
        calls = json.loads(row['tool_calls_json'])
        if (type(calls) is not list or not 0 < len(calls) <= 64
                or any(type(c) is not dict or type(c.get('id')) is not str or not c['id'] for c in calls)
                or len({c['id'] for c in calls}) != len(calls)):
            raise JobError('tool_call_plan_invalid_call_set')
        entries, originals, control_calls = [], [], []
        for call in calls:
            name = call['function']['name']
            args = _document(json.loads(call['function']['arguments']), 131072)
            registered = pipeline.tool_registry.get(name) if pipeline.tool_registry is not None else None
            implementation = resolve(registered)
            spec = implementation.pending_call_spec(args, runtime=runtime, definition=registered.definition) if implementation is not None else None
            if spec is not None:
                spec = _document(spec, 65536)
                if set(spec) != {'kind', 'request'} or spec['kind'] != 'managed_effect':
                    raise JobError('tool_call_plan_recovery_unsupported')
            entry = {'call_id': call['id'], 'tool_name': name, 'arguments_hash': _hash(args),
                'registration': registration_value(registered), 'recovery': spec, 'job_id': None, 'operation_id': None}
            await _verify_call(manager, tx, actor, source, ref, call['id'], name, _hash(args))
            entries.append(entry)
            originals.append(registered)
            control_calls.append(spec is None and policy(registered, args, runtime=runtime).control_only)
        if all(control_calls):
            # A complete batch of actual control implementations may inspect or
            # stop its original execution after lease expiry. This grants no
            # new managed intent, callback, model request or generation. A mixed
            # batch still requires the ordinary execution fence.
            from .task_control import _source
            for entry in entries:
                await _source(manager, tx, actor, binding.lease, ref, entry['call_id'])
        else:
            await manager._fence(tx, actor, binding.lease)
        from .tool_call_abort import _freeze
        from .tool_call_abort import read as read_abort
        attempt['value'] = _freeze(pipeline, binding, source, ref, entries)
        pipeline._native_abort_attempt = attempt['value']
        if await read_abort(manager, tx, source, ref) is not None:
            raise JobError('tool_call_batch_aborted')
        existing = await read(manager, tx, source, ref, optional=True)
        if existing is not None:
            comparable = [{**e, 'job_id': None, 'operation_id': None} for e in existing['entries']]
            if comparable != entries:
                raise JobError('tool_call_plan_binding_changed')
            retained = _retained(pipeline, manager, source, ref, existing)
            if any(before is not after for (_call_id, before), after in zip(retained.registrations, originals, strict=True)):
                raise JobError('tool_call_plan_binding_changed')
            return
        for entry in entries:
            if entry['recovery'] is not None:
                prepared = manager._registration_spec(actor, kind='background_tool', title=entry['tool_name'],
                    parent=binding.lease, call_ref=ref, tool_call_id=entry['call_id'], request=entry['recovery']['request'])
                created = await manager._register_job_tx(tx, actor, **prepared)
                if created.replayed:
                    raise JobError('tool_call_plan_registration_conflict')
                entry.update(job_id=created.job.job_id, operation_id=created.operation.operation_id)
        for entry, original in zip(entries, originals, strict=True):
            if original is not None:
                validate_registration(pipeline.tool_registry, entry['tool_name'], original)
            elif pipeline.tool_registry is not None and pipeline.tool_registry.get(entry['tool_name']) is not None:
                raise JobError('tool_call_plan_binding_changed')
        from .workflow_controller import proof
        key = identity(manager, source, ref)
        value = {'version': 1, 'scope': proof(source), 'call_ref': ref.to_dict(), 'entries': entries}
        await _event(manager, tx, key, source, 'tool_call_plan', value)
        await _event(manager, tx, key + '_seal', source, 'tool_call_plan_seal',
                     {'version': 1, 'scope': proof(source), 'plan_id': key, 'plan_hash': _hash(value)})
    # Publish the process-local binding only after the whole transaction's ACK.
    # If commit loses its ACK, no callback receives execution authority.
    pipeline._native_tool_batch_bindings = _BatchBindings(
        key, _hash(value), _json(proof(source)), _json(ref.to_dict()),
        tuple((entry['call_id'], original) for entry, original in zip(entries, originals, strict=True)))


async def prepare(pipeline):
    attempt = {}
    try:
        await _prepare(pipeline, attempt)
    except Exception:
        if attempt.get('value') is not None:
            from .tool_call_abort import _record
            try:
                await _record(attempt['value'])
            except Exception as exc:
                # Preserve the original failure. Without this positive durable
                # checkpoint, source exit/pending recovery remains unknown.
                logging.getLogger(__name__).warning(
                    'Pre-admission abort checkpoint unavailable; original pending remains unresolved',
                    extra={'source_job_id': attempt['value'].execution.lease.job_id,
                           'assistant_seq': attempt['value'].ref.seq,
                           'failure_type': type(exc).__name__[:128]},
                )
        raise
    finally:
        pipeline._native_abort_attempt = None


async def dispatch(pipeline, name, arguments, registered):
    from .invocation import current_invocation
    from .task_runtime import _SEND
    from .tool_contracts import policy

    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return
    manager, actor = binding.runtime.manager, binding.actor
    seq, call_id = current_invocation(actor.session_id)
    ref = await manager.store.message_ref(actor.session_id, seq)
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, binding.lease.job_id)
        if not await required(manager, tx, source):
            return
        declared = policy(registered, arguments, runtime=binding.runtime)
        if declared.control_only:
            from .task_control import _source
            await _source(manager, tx, actor, binding.lease, ref, call_id)
        else:
            await manager._fence(tx, actor, binding.lease)
        from .tool_call_abort import read as read_abort
        if await read_abort(manager, tx, source, ref) is not None:
            raise JobError('tool_call_batch_aborted')
        plan = await read(manager, tx, source, ref)
        retained = _retained(pipeline, manager, source, ref, plan)
        original_bindings = dict(retained.registrations)
        if call_id not in original_bindings or original_bindings[call_id] is not registered:
            raise JobError('tool_call_plan_binding_changed')
        matches = [e for e in plan['entries'] if e['call_id'] == call_id]
        if (len(matches) != 1 or matches[0]['tool_name'] != name or matches[0]['arguments_hash'] != _hash(arguments)
                or matches[0]['registration'] != registration_value(registered)):
            raise JobError('tool_call_plan_binding_changed')
        key = identity(manager, source, ref) + '_start_' + _hash(call_id)[:24]
        if await _read(manager, tx, key, source, 'tool_call_dispatch_started', optional=True) is not None:
            raise JobError('tool_call_already_dispatched')
        await _event(manager, tx, key, source, 'tool_call_dispatch_started',
            {'version': 1, 'scope': plan['scope'], 'plan_id': identity(manager, source, ref),
             'plan_hash': _hash(plan), 'entry_hash': _hash(matches[0])})


async def queued_for_call(manager, tx, row, source, invocation):
    if (invocation is None or row['generation'] != 0 or row['resource_state'] != 'not_started'
            or row['status'] != 'queued' or row['cancel_requested_at'] is not None):
        return False
    from .managed_effects import _never_claimed
    await _never_claimed(manager, tx, row)
    seq, call_id = invocation
    from .message_ref import MessageRef
    raw_ref = await tx.fetchone(f'SELECT id,seq,role,tool_call_id,send_index,round_index FROM {manager.store.t.messages} WHERE session_id=? AND seq=?', (source['session_id'], seq))
    if raw_ref is None:
        raise JobError('tool_call_plan_raw_unavailable')
    ref = MessageRef(manager.store.store_id, source['session_id'], raw_ref['id'], raw_ref['seq'], raw_ref['role'], raw_ref['tool_call_id'], raw_ref['send_index'], raw_ref['round_index'])
    plan = await read(manager, tx, source, ref, optional=True)
    return plan is not None and any(e['call_id'] == call_id for e in plan['entries']) and any(e['job_id'] == row['job_id'] for e in plan['entries'])


async def adopt(manager, tx, source, ref, call_id, request):
    if not await required(manager, tx, source):
        return None
    plan = await read(manager, tx, source, ref)
    matches = [e for e in plan['entries'] if e['call_id'] == call_id]
    if len(matches) != 1 or matches[0]['recovery'] != {'kind': 'managed_effect', 'request': request}:
        raise JobError('tool_call_plan_binding_changed')
    entry = matches[0]
    key = identity(manager, source, ref) + '_start_' + _hash(call_id)[:24]
    started = await _read(manager, tx, key, source, 'tool_call_dispatch_started')
    if started != {'version': 1, 'scope': plan['scope'], 'plan_id': identity(manager, source, ref), 'plan_hash': _hash(plan), 'entry_hash': _hash(entry)}:
        raise JobError('tool_call_plan_corrupt')
    row = await manager._row(tx, entry['job_id'])
    if row['generation'] != 0 or row['status'] != 'queued' or row['resource_state'] != 'not_started':
        raise JobError('managed_effect_original_requires_reconciliation')
    from .jobs import JobRegistration, _operation, _record
    operation = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (entry['operation_id'],))
    return JobRegistration(_record(row), _operation(operation), False)


async def finish_receipt(manager, actor, ref, call_id):
    operation = await manager.operation_for_call(actor, ref, call_id)
    if operation is None:
        return
    async with manager._write() as tx:
        source = await manager._row(tx, operation.job_id)
        if source is None or source['kind'] != 'background_tool' or source['generation'] != 0:
            return
        parent = await manager._row(tx, source['parent_job_id'])
        if not await required(manager, tx, parent):
            return
        plan = await read(manager, tx, parent, ref)
        if not any(e['job_id'] == source['job_id'] for e in plan['entries']):
            raise JobError('tool_call_plan_registration_changed')
    from .managed_effects import terminate_unstarted
    await terminate_unstarted(manager, actor, source['job_id'])


async def reconcile_pending(loop, actor, source_job_id):
    """Original-main host stages known outcomes/negative proofs; never executes."""
    from .managed_completion import _host_only
    from .managed_delivery import enqueue, resolved_review, result_event
    from .managed_effects import _never_claimed, terminate_unstarted
    from .pending_calls import SNAPSHOT, _verify, model

    _host_only()
    runtime = getattr(loop, 'task_runtime', None)
    if runtime is None or runtime.actor(actor.session_id) != actor or await loop.ensure_store() is not runtime.manager.store:
        raise JobError('task_runtime_required')
    manager = runtime.manager
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, source_job_id)
        closed = await model(manager, tx, source)
        if source['root_job_id'] != source_job_id or source['session_id'] != actor.session_id:
            raise JobError('pending_call_requires_original_main')
        if not await required(manager, tx, source):
            raise JobError('tool_call_plan_required')
        snapshot = closed['evidence'].get(SNAPSHOT)
        if snapshot is None:
            raise JobError('pending_call_snapshot_required')
        _pending, calls, ref = await _verify(manager, tx, source, snapshot)
        plan = await read(manager, tx, source, ref)
        entries = {entry['call_id']: entry for entry in plan['entries']}
        targets = []
        # Validate the complete pending set before recording any negative result.
        # Unsupported calls remain obligations; absence is not non-dispatch.
        for call in calls:
            entry = entries.get(call['id'])
            if entry is None or entry['recovery'] is None:
                raise JobError('pending_call_adapter_required')
            row = await manager._authorize(tx, actor, entry['job_id'])
            if row['generation'] == 0:
                await _never_claimed(manager, tx, row)
            else:
                await resolved_review(manager, tx, row, await result_event(manager, tx, row))
            targets.append((row['job_id'], row['generation']))
    for job_id, generation in targets:
        if generation == 0:
            await terminate_unstarted(manager, actor, job_id)
    notices = [await enqueue(loop, actor, job_id) for job_id, _generation in targets]
    return {'source_job_id': source_job_id, 'notices': notices, 'execution_authority': False,
            'acknowledgement_created': False, 'billing_settled': False}
