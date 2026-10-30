"""Original tool invocation identity and frozen settlement in the budget journal.

Tool observations are distinct from model attempts. Accounting counters and the
host submission outbox use the existing budget engine. Recovery never invokes a
handler or recalculates a persisted charge using a current registration.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field

from .jobs import JobError, _hash
from .model_exit import content_reference
from .pricing import BillingReceipt, Charge, PricedCall, canonical


@dataclass
class ToolAttempt:
    binding: object
    attempt_id: str
    prepared: dict
    call: PricedCall
    pricing: object
    task: object
    observation: dict | None = field(default=None, repr=False)


def event_id(manager, phase, attempt_id):
    if phase not in ('prepared', 'dispatched', 'finished', 'receipt'):
        raise JobError('invalid_tool_attempt_phase')
    return 'ta_' + phase + '_' + _hash([manager.store.store_id, attempt_id])[:48]


def context(manager, job, prepared):
    return dict(version=1, store_id=manager.store.store_id, tenant_id=job['tenant_id'],
        principal_id=job['principal_id'], root_job_id=job['root_job_id'], job_id=job['job_id'],
        generation=prepared['scope']['generation'], session_id=prepared['session_id'],
        attempt_id=prepared['attempt_id'], prepared_event_id=event_id(manager, 'prepared', prepared['attempt_id']),
        prepared_hash=_hash(prepared), request_hash=prepared['request_hash'], kind='tool',
        tool_name=prepared['tool_name'], pricing_rule=prepared['pricing_rule'])


async def _read(manager, tx, job, phase, attempt_id, *, optional=False):
    from .tool_call_plans import _read as read_event
    key = event_id(manager, phase, attempt_id)
    value = await read_event(manager, tx, key, job, 'tool_attempt_' + phase, optional=optional)
    anchor = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key + '_a',))
    if value is None:
        if anchor is not None:
            raise JobError('tool_attempt_evidence_missing')
        return None
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    proof = await read_event(manager, tx, key + '_a', job, 'tool_attempt_anchor')
    if proof != {'version': 1, 'source': content_reference(event)}:
        raise JobError('tool_attempt_evidence_changed')
    return event, value


async def _save(manager, tx, job, phase, attempt_id, value):
    from .tool_call_plans import _event
    previous = await _read(manager, tx, job, phase, attempt_id, optional=True)
    if previous is not None:
        if previous[1] != value:
            raise JobError('tool_attempt_observation_conflict')
        return previous
    key = event_id(manager, phase, attempt_id)
    await _event(manager, tx, key, job, 'tool_attempt_' + phase, value)
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    await _event(manager, tx, key + '_a', job, 'tool_attempt_anchor',
                 {'version': 1, 'source': content_reference(event)})
    return event, value


async def _original(manager, tx, job, prepared, *, depth=0):
    from .jobs import JobActor
    from .managed_effects import _ref, _verify_call
    ref = _ref(prepared['call_ref'])
    scope = prepared['scope']
    if depth > 16:
        raise JobError('tool_attempt_nesting_limit')
    if prepared.get('version') == 2:
        _, parent = await _read(manager, tx, job, 'prepared', prepared['parent_attempt_id'])
        expected = 'toolattempt_' + _hash(['nested', manager.store.store_id, scope,
            prepared['parent_attempt_id'], prepared['invocation_id']])[:48]
        if (prepared['attempt_id'] != expected or parent['attempt_id'] == expected
                or prepared['parent_prepared_hash'] != _hash(parent)
                or any(prepared[k] != parent[k] for k in ('scope', 'session_id', 'call_ref', 'call_id'))):
            raise JobError('tool_attempt_original_mismatch')
        await _original(manager, tx, job, parent, depth=depth + 1)
        await _read(manager, tx, job, 'dispatched', parent['attempt_id'])
        return
    expected = 'toolattempt_' + _hash([manager.store.store_id, scope, ref.to_dict(), prepared['call_id']])[:48]
    if (prepared['attempt_id'] != expected or scope['job_id'] != job['job_id']
            or prepared['session_id'] != job['session_id'] or ref.session_id != job['session_id']):
        raise JobError('tool_attempt_original_mismatch')
    actor = JobActor(job['tenant_id'], job['principal_id'], job['session_id'])
    await _verify_call(manager, tx, actor, job, ref, prepared['call_id'],
                       prepared['tool_name'], prepared['request_hash'])


async def prepare(binding, ref, call_id, registered, arguments):
    from power_loop.tools.registry import validate_pricing

    from .task_runtime import _SEND
    if _SEND.get() is not binding:
        raise JobError('tool_attempt_original_required')
    manager = binding.runtime.manager
    pricing = validate_pricing(registered)
    async with manager._write() as tx:
        job = await manager._fence(tx, binding.actor, binding.lease)
        scope = asdict(binding.lease)
        key = 'toolattempt_' + _hash([manager.store.store_id, scope, ref.to_dict(), call_id])[:48]
        value = dict(version=1, attempt_id=key, scope=scope, session_id=ref.session_id,
            call_ref=ref.to_dict(), call_id=call_id, request_hash=_hash(arguments),
            tool_name=registered.definition.name,
            pricing_rule={'rule_id': pricing.rule_id, 'rule_version': registered.pricing_version})
        await _original(manager, tx, job, value)
        if await _read(manager, tx, job, 'prepared', key, optional=True) is not None:
            raise JobError('tool_attempt_already_prepared')
        await _save(manager, tx, job, 'prepared', key, value)
    call = PricedCall(registered.definition.name, arguments, job['execution_role'])
    return ToolAttempt(binding, key, value, call, pricing, asyncio.current_task())


async def prepare_nested(parent, registered, arguments):
    """An internal call belongs to a dispatched original callback, with its own identity."""
    import uuid

    from power_loop.tools.registry import validate_pricing

    from .nested_tools import require_parent
    require_parent(parent)
    binding, original = parent.binding, parent.budget_attempt
    pricing = validate_pricing(registered)
    manager = binding.runtime.manager
    async with manager._write() as tx:
        job = await manager._fence(tx, binding.actor, binding.lease)
        source = await _read(manager, tx, job, 'prepared', original.attempt_id)
        if source[1] != original.prepared:
            raise JobError('tool_attempt_original_mismatch')
        nonce = uuid.uuid4().hex
        scope = asdict(binding.lease)
        key = 'toolattempt_' + _hash(['nested', manager.store.store_id, scope, original.attempt_id, nonce])[:48]
        value = {k: original.prepared[k] for k in ('call_ref', 'call_id', 'session_id')}
        value.update(version=2, attempt_id=key, scope=scope, invocation_id=nonce,
            parent_attempt_id=original.attempt_id, parent_prepared_hash=_hash(original.prepared),
            request_hash=_hash(arguments), tool_name=registered.definition.name,
            pricing_rule={'rule_id': pricing.rule_id, 'rule_version': registered.pricing_version})
        await _original(manager, tx, job, value)
        await _save(manager, tx, job, 'prepared', key, value)
    return ToolAttempt(binding, key, value,
        PricedCall(registered.definition.name, arguments, job['execution_role']), pricing, asyncio.current_task())


def _original_owner(attempt):
    from .task_runtime import _SEND
    if _SEND.get() is not attempt.binding or asyncio.current_task() is not attempt.task:
        raise JobError('tool_attempt_original_required')
    if (attempt.pricing.rule_id != attempt.prepared['pricing_rule']['rule_id']
            or attempt.pricing.version() != attempt.prepared['pricing_rule']['rule_version']):
        raise JobError('tool_pricing_binding_changed')


async def reserve(attempt):
    from . import model_budgets as budgets
    _original_owner(attempt)
    return await budgets.reserve(attempt, attempt.call, pricing=attempt.pricing)


async def authorize_dispatch(attempt, tx, job):
    """Run in the same transaction as the original tool dispatch marker."""
    from . import model_budgets as budgets
    _original_owner(attempt)
    manager = attempt.binding.runtime.manager
    saved = await _read(manager, tx, job, 'prepared', attempt.attempt_id)
    if saved[1] != attempt.prepared:
        raise JobError('tool_attempt_original_mismatch')
    await _original(manager, tx, job, saved[1])
    if await _read(manager, tx, job, 'dispatched', attempt.attempt_id, optional=True):
        raise JobError('tool_attempt_already_dispatched')
    await budgets.authorize_dispatch(attempt, tx, job)
    await _save(manager, tx, job, 'dispatched', attempt.attempt_id,
                dict(version=1, scope=attempt.prepared['scope'], attempt_id=attempt.attempt_id,
                     prepared_hash=_hash(attempt.prepared)))


def capture(attempt, receipt: BillingReceipt | None, error: BaseException | None):
    """Freeze synchronously before any cancellable persistence/host await."""
    _original_owner(attempt)
    if attempt.observation is not None:
        raise JobError('tool_attempt_already_observed')
    charge = attempt.pricing.settle(attempt.call, receipt, error)
    if (not isinstance(charge, Charge) or charge.rule_id != attempt.pricing.rule_id
            or charge.rule_version != attempt.prepared['pricing_rule']['rule_version']):
        raise JobError('tool_settlement_rule_mismatch')
    cancelled = isinstance(error, asyncio.CancelledError)
    attempt.observation = dict(version=1, scope=attempt.prepared['scope'], attempt_id=attempt.attempt_id,
        prepared_hash=_hash(attempt.prepared), status='unknown' if cancelled else 'returned',
        usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, estimated=False,
        outcome='cancelled' if cancelled else 'failed' if error is not None else 'returned',
        error_type=type(error).__name__ if error is not None else None,
        provider_response_id=receipt.provider_request_id if receipt is not None else None,
        charge=dict(amount=format(charge.amount.amount, 'f'), currency=charge.amount.currency,
            quality=charge.quality, units=json.loads(canonical(charge.units)),
            rule_id=charge.rule_id, rule_version=charge.rule_version))


async def verify_state(manager, tx, job, state, *, finished=False):
    _, prepared = await _read(manager, tx, job, 'prepared', state['attempt_id'])
    await _original(manager, tx, job, prepared)
    if state['context'] != context(manager, job, prepared):
        raise JobError('tool_attempt_context_mismatch')
    dispatched = await _read(manager, tx, job, 'dispatched', state['attempt_id'], optional=not finished)
    if dispatched is not None and dispatched[1] != dict(version=1, scope=prepared['scope'],
            attempt_id=state['attempt_id'], prepared_hash=_hash(prepared)):
        raise JobError('tool_attempt_dispatch_changed')
    if not finished:
        return prepared
    original = await _read(manager, tx, job, 'finished', state['attempt_id'])
    observation = original[1]
    rule = prepared['pricing_rule']
    if (observation['prepared_hash'] != _hash(prepared) or observation['scope'] != prepared['scope']
            or observation['attempt_id'] != state['attempt_id']
            or observation['charge']['rule_id'] != rule['rule_id']
            or observation['charge']['rule_version'] != rule['rule_version']
            or state['observation'] is not None and state['observation'] != observation):
        raise JobError('tool_attempt_observation_changed')
    return original


async def finish(attempt):
    """Persist frozen facts, project counters once, and deliver the original charge."""
    from . import model_budgets as budgets
    if attempt.observation is None:
        raise JobError('tool_attempt_observation_required')
    runtime, manager = attempt.binding.runtime, attempt.binding.runtime.manager
    async with manager._write() as tx:
        job = await manager._authorize(tx, attempt.binding.actor, attempt.binding.lease.job_id)
        manager._check_proof(job, attempt.binding.lease)
        state = await budgets._read(manager, tx, attempt.attempt_id)
        await verify_state(manager, tx, job, state)
        if state['phase'] not in ('dispatch_authorized', 'settling'):
            raise JobError('tool_attempt_dispatch_required')
        source = await _save(manager, tx, job, 'finished', attempt.attempt_id, attempt.observation)
        policy = await budgets._policy(runtime, tx)
        await budgets._project_observation(manager, tx, policy, state, job, attempt.observation, source)
    return await budgets._deliver(runtime, attempt.binding.actor, attempt.binding.lease.job_id, attempt.attempt_id)


async def observe_managed_free(attempt, tx, job):
    """Freeze known zero fees atomically with the original managed dispatch.

    A finite managed effect has its own outcome/recovery protocol. A free fee
    is already known even if that callback never returns. This observation
    deliberately makes no assertion about its execution or business outcome.
    """
    from . import model_budgets as budgets
    from .pricing import Free
    _original_owner(attempt)
    if type(attempt.pricing) is not Free or attempt.observation is not None:
        raise JobError('managed_free_original_required')
    manager = attempt.binding.runtime.manager
    state = await budgets._read(manager, tx, attempt.attempt_id)
    await verify_state(manager, tx, job, state)
    if state['phase'] != 'dispatch_authorized':
        raise JobError('tool_attempt_dispatch_required')
    charge = attempt.pricing.settle(attempt.call, None, None)
    if (charge.rule_id != attempt.prepared['pricing_rule']['rule_id']
            or charge.rule_version != attempt.prepared['pricing_rule']['rule_version']):
        raise JobError('tool_settlement_rule_mismatch')
    # Pricing itself has returned an exact zero, so it owns no vendor inflight
    # slot. The independent managed child still retains the business outcome.
    value = dict(version=1, scope=attempt.prepared['scope'], attempt_id=attempt.attempt_id,
        prepared_hash=_hash(attempt.prepared), status='returned',
        usage={'prompt_tokens':0, 'completion_tokens':0, 'total_tokens':0}, estimated=False,
        outcome='unknown', error_type=None, provider_response_id=None,
        charge=dict(amount=format(charge.amount.amount,'f'), currency=charge.amount.currency,
            quality=charge.quality, units={}, rule_id=charge.rule_id, rule_version=charge.rule_version))
    source = await _save(manager, tx, job, 'finished', attempt.attempt_id, value)
    policy = await budgets._policy(attempt.binding.runtime, tx)
    await budgets._project_observation(manager, tx, policy, state, job, value, source)
    attempt.observation = value


async def reconcile(runtime, actor, job_id, attempt_id):
    from . import model_budgets as budgets
    manager = runtime.manager
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        state = await budgets._read(manager, tx, attempt_id)
        policy = await budgets._policy(runtime, tx)
        if policy is None or state['job_id'] != job_id or state['policy_key'] != policy['policy_key']:
            raise JobError('budget_reconciliation_scope_mismatch')
        source = await verify_state(manager, tx, job, state, finished=True)
        await budgets._project_observation(manager, tx, policy, state, job, source[1], source)
    ack = await budgets._deliver(runtime, actor, job_id, attempt_id)
    return dict(attempt_id=attempt_id, job_id=job_id, revision=ack['revision'],
                request_hash=ack['request_hash'], usage_acknowledged=True, execution_authority=False)


async def begin_pipeline(pipeline, registered, arguments):
    from .invocation import current_invocation
    from .pricing import Free
    from .task_runtime import _SEND
    from .tool_call_plans import read as read_plan
    from .tool_call_plans import required
    from .tool_contracts import policy
    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return None
    runtime, manager = binding.runtime, binding.runtime.manager
    if runtime.budget_config is None or not callable(getattr(runtime.budget_config.provider, 'prepare_tool', None)):
        return None
    if policy(registered, arguments, runtime=runtime).control_only:
        return None  # Cancellation/query must remain available after a spent lease.
    seq, call_id = current_invocation(pipeline.session_id)
    ref = await manager.store.message_ref(pipeline.session_id, seq)
    async with manager._write() as tx:
        job = await manager._fence(tx, binding.actor, binding.lease)
        if not await required(manager, tx, job):
            raise JobError('tool_budget_original_plan_required')
        plan = await read_plan(manager, tx, job, ref)
        entries = [e for e in plan['entries'] if e['call_id'] == call_id]
        if len(entries) != 1:
            raise JobError('tool_budget_original_plan_required')
        if entries[0]['recovery'] is not None:
            if not isinstance(registered.pricing, Free):
                raise JobError('budget_managed_tool_adapter_unavailable')
    attempt = await prepare(binding, ref, call_id, registered, arguments)
    await reserve(attempt)
    return attempt


async def release_unstarted(runtime, actor, job_id, attempt_id, *, require_cancelled=False):
    """Seal original dispatch before refunding a reservation; never replay tools."""
    from . import model_budgets as budgets
    from .budget_contracts import BudgetAck, BudgetGrant, BudgetPlan, BudgetSubmission
    manager = runtime.manager
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        policy = await budgets._policy(runtime, tx)
        state = await budgets._read(manager, tx, attempt_id)
        if policy is None or state['job_id'] != job_id or state['policy_key'] != policy['policy_key']:
            raise JobError('budget_reconciliation_scope_mismatch')
        await verify_state(manager, tx, job, state)
        if await _read(manager, tx, job, 'dispatched', attempt_id, optional=True) is not None:
            raise JobError('budget_release_not_permitted')
        if state['phase'] == 'released':
            return state['ack']
        if state['phase'] not in ('local_reserved', 'host_reserving', 'host_reserved', 'dispatch_closed'):
            raise JobError('budget_release_not_permitted')
        if require_cancelled:
            await budgets._require_cancelled_lineage(manager, tx, job)
        if state['phase'] != 'dispatch_closed':
            state.update(phase='dispatch_closed', pending=True)
            await budgets._save(manager, tx, state, job)
        if not state['host_intent']:
            await budgets._refund_local(manager, tx, policy, state)
            state.update(phase='released', pending=False, ack={'local_only': True, 'dispatch_not_authorized': True})
            await budgets._save(manager, tx, state, job)
            return state['ack']
    provider = runtime.budget_config.provider
    if state['submission'] is not None:
        grant = None  # Replay the frozen release, including after a lost ACK.
    elif state['grant'] is None:
        plan = BudgetPlan(**state['plan'])
        grant = await budgets._host_call(runtime, job_id, attempt_id, 'lookup', lambda: provider.lookup(plan))
        if grant is not None and (type(grant) is not BudgetGrant
                or (grant.attempt_id, grant.request_hash, grant.state) != (attempt_id, state['plan_hash'], 'reserved')):
            raise JobError('budget_host_reservation_unconfirmed')
    else:
        grant = BudgetGrant(**state['grant'])
    proof = dict(proof='dispatch_not_authorized', reservation_request_hash=state['plan_hash'],
        event_id='mb_' + _hash([attempt_id, state['version']])[:48], evidence_hash=_hash(state))
    if state['submission'] is None:
        submission = await budgets._host_call(runtime, job_id, attempt_id, 'prepare_release',
            lambda: provider.prepare_release(state['context'], grant, proof))
        if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (attempt_id, 1):
            raise JobError('invalid_budget_submission')
        async with manager._write() as tx:
            job = await manager._authorize(tx, actor, job_id)
            state = await budgets._read(manager, tx, attempt_id)
            if state['phase'] == 'released':
                return state['ack']
            if state['submission'] is not None and state['submission'] != asdict(submission):
                raise JobError('budget_submission_conflict')
            if state['submission'] is None:
                state['submission'] = asdict(submission)
                await budgets._save(manager, tx, state, job)
    submission = BudgetSubmission(**state['submission'])
    ack = await budgets._host_call(runtime, job_id, attempt_id, 'release', lambda: provider.release(submission))
    if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
            attempt_id, submission.revision, submission.request_hash):
        raise JobError('budget_ack_mismatch')
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        state = await budgets._read(manager, tx, attempt_id)
        policy = await budgets._policy(runtime, tx)
        if state['phase'] == 'released':
            return state['ack']
        if state['phase'] != 'dispatch_closed' or state['submission'] != asdict(submission):
            raise JobError('budget_release_conflict')
        await budgets._refund_local(manager, tx, policy, state)
        state.update(phase='released', pending=False, ack=asdict(ack))
        await budgets._save(manager, tx, state, job)
    return asdict(ack)
