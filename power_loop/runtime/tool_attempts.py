"""Original tool invocation identity, distinct from model attempts."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .jobs import JobError, _hash
from .pricing import PricedCall


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


async def prepare(binding, ref, call_id, registered, arguments):
    from .execution_tools import prepare_attempt
    return await prepare_attempt(binding, ref, call_id, registered, arguments)


def _original_owner(attempt):
    from .task_runtime import _SEND
    if _SEND.get() is not attempt.binding or asyncio.current_task() is not attempt.task:
        raise JobError('tool_attempt_original_required')
    if (attempt.pricing.rule_id != attempt.prepared['pricing_rule']['rule_id']
            or attempt.pricing.version() != attempt.prepared['pricing_rule']['rule_version']):
        raise JobError('tool_pricing_binding_changed')


async def authorize_dispatch(attempt, tx, job):
    from .control_runtime import current
    _original_owner(attempt)
    control = current(attempt.binding)
    saved = control.get("attempt", attempt.attempt_id)
    if saved is None or saved["prepared"] != attempt.prepared:
        raise JobError("tool_attempt_original_mismatch")
    if saved["phase"] != "prepared":
        raise JobError("tool_attempt_already_dispatched")
    control.dispatch(attempt.attempt_id)


def capture(attempt, error: BaseException | None):
    """Freeze synchronously before any cancellable persistence/host await."""
    _original_owner(attempt)
    if attempt.observation is not None:
        raise JobError('tool_attempt_already_observed')
    cancelled = isinstance(error, asyncio.CancelledError)
    attempt.observation = dict(status="unknown" if cancelled else "returned",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, estimated=False,
        outcome="cancelled" if cancelled else "failed" if error else "returned",
        error_type=type(error).__name__ if error is not None else None)


async def finish(attempt):
    from .control_runtime import current
    if attempt.observation is None:
        raise JobError("tool_attempt_observation_required")
    control = current(attempt.binding)
    saved = control.get("attempt", attempt.attempt_id)
    if saved is None or saved["prepared"] != attempt.prepared:
        raise JobError("tool_attempt_original_mismatch")
    control.observe(attempt.attempt_id, attempt.observation)
    await control.checkpoint()


async def begin_pipeline(pipeline, registered, arguments):
    from .execution_tools import entry
    from .invocation import current_invocation
    from .task_runtime import _SEND
    from .tool_contracts import policy
    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return None
    if policy(registered, arguments, runtime=binding.runtime).control_only:
        return None
    if not binding.resource_evidence["_control_tool_plan_required"]:
        return None
    seq, call_id = current_invocation(pipeline.session_id)
    owned, _selected = entry(binding, seq, call_id)
    return await prepare(binding, owned["ref"], call_id, registered, arguments)
