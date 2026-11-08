"""Track physical model calls in the active execution and report their usage."""

from __future__ import annotations

import logging
import secrets
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation

from .jobs import JobError, _hash

_LIMIT = 10000
_PHASES = ("tracking", "prepared", "dispatched", "finished")
_TOKENS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "prompt_cached_tokens",
    "prompt_cache_miss_tokens",
    "prompt_image_tokens",
    "prompt_audio_tokens",
    "prompt_text_tokens",
    "completion_reasoning_tokens",
    "completion_audio_tokens",
    "completion_text_tokens",
    "completion_image_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
    "cached_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "accepted_tokens",
    "rejected_tokens",
)
log = logging.getLogger(__name__)


def is_managed(session_id):
    from .task_runtime import _SEND

    binding = _SEND.get()
    return binding is not None and binding.actor.session_id == session_id


@dataclass(frozen=True)
class Attempt:
    binding: object
    attempt_id: str
    prepared: dict
    local_deadline: float | None = field(default=None, compare=False, repr=False)
    capture_binding: object | None = field(default=None, compare=False, repr=False)


def event_identity(attempt):
    if attempt is None:
        return {}
    return {
        "model_attempt_id": attempt.attempt_id,
        "model_attempt_store_id": attempt.binding.runtime.manager.store.store_id,
        "model_attempt_job_id": attempt.prepared["scope"]["job_id"],
        "model_attempt_generation": attempt.prepared["scope"]["generation"],
        "budget_managed": attempt.binding.runtime._original_budget_config is not None,
    }


def event_id(manager, phase, identity):
    if phase not in _PHASES:
        raise JobError("invalid_model_attempt_phase")
    return "ma_" + phase + "_" + _hash([manager.store.store_id, identity])[:48]


def _scope(lease):
    from .execution_membership import _lease

    return _lease(asdict(lease))


def _cost(value):
    if value is None:
        return True
    try:
        return isinstance(value, str) and 0 < len(value) <= 128 and Decimal(value).is_finite() and Decimal(value) >= 0
    except InvalidOperation:
        return False


def _response_id(value):
    return value is None or isinstance(value, str) and 0 < len(value) <= 255 and all(ord(c) >= 32 for c in value)


def response_observation(usage, response):
    raw = getattr(response, "raw_completion", None)
    response_id = raw.get("id") if isinstance(raw, dict) else getattr(raw, "id", None)
    cost = getattr(response, "provider_cost", None)
    return {
        "usage": {key: getattr(usage, key) for key in _TOKENS if getattr(usage, key, None) is not None},
        "cost_amount": cost.amount if cost is not None else None,
        "cost_currency": cost.currency if cost is not None else None,
        "response_metrics": getattr(response, 'debug', {}).get('response_extraction', {}),
        "response_id": response_id,
    }


async def enable(binding):
    from .budget_policy_sources import PolicySourceChanged
    from .control_runtime import open_control

    joined = binding.runtime.manager._joined_write.get() is not None
    policy_source = None
    while True:
        try:
            await open_control(binding, policy_source=policy_source)
            break
        except PolicySourceChanged as changed:
            if joined:
                raise
            policy_source = changed.source
    binding.resource_evidence["model_attempt_protocol"] = 1


async def _no_unknown_sends(manager, tx, row):
    # Re-entry in a fresh physical retry session cannot evade an unresolved
    # earlier send of the same original owner execution.
    if row["control_owner_job_id"] is None:
        # A new main send is a new authorized root, including control/result
        # handling. Same-SID history is not a business action key. Its budget
        # admission must retain old risk independently of execution ownership.
        return
    blocked = await tx.fetchone(
        f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND resource_state='unknown' AND job_id<>? "
        "AND tenant_id=? AND principal_id=? AND control_owner_job_id=? LIMIT 1",
        (
            row["job_id"],
            row["tenant_id"],
            row["principal_id"],
            row["control_owner_job_id"],
        ),
    )
    if blocked:
        raise JobError("unresolved_prior_attempt")


async def prepare(binding, *, nonce, round_index, retry_index, request_hash, model, max_tokens, purpose=None):
    if (
        purpose not in (None, "fold")
        or 
        not isinstance(nonce, str)
        or not 1 <= len(nonce) <= 64
        or type(round_index) is not int
        or not 0 <= round_index < 2**63
        or type(retry_index) is not int
        or not 1 <= retry_index < 2**63
        or not isinstance(request_hash, str)
        or len(request_hash) != 64
        or any(c not in "0123456789abcdef" for c in request_hash)
        or not isinstance(model, str)
        or len(model) > 255
        or (max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens < 2**63))
    ):
        raise JobError("invalid_model_attempt")
    manager = binding.runtime.manager
    scope = _scope(binding.lease)
    identity = "attempt_" + _hash([manager.store.store_id, scope, nonce])[:48]
    data = {
        "version": 1,
        "attempt_id": identity,
        "scope": scope,
        "session_id": binding.actor.session_id,
        "nonce": nonce,
        "round_index": round_index,
        "retry_index": retry_index,
        "request_hash": request_hash,
        "model": model,
        "max_tokens": max_tokens,
    }
    if purpose is not None:
        data["purpose"] = purpose
    from .control_runtime import current
    control = current(binding)
    old = control.get("attempt", identity)
    if old is not None:
        if old["prepared"] != data:
            raise JobError("model_attempt_conflict")
        return Attempt(binding, identity, data)
    if control.models >= _LIMIT:
        raise JobError("model_attempt_limit")
    control.begin_attempt(identity, "model", data)
    return Attempt(binding, identity, data)


async def dispatch(attempt):
    from .control_runtime import current
    binding, data = attempt.binding, attempt.prepared
    control = current(binding)
    control.require_admission()
    saved = control.get("attempt", attempt.attempt_id)
    if saved is None or saved["prepared"] != data or data["scope"] != _scope(binding.lease):
        raise JobError("model_attempt_owner_mismatch")
    if saved["phase"] != "prepared":
        raise JobError("model_attempt_dispatch_already_admitted")
    if binding.runtime._original_budget_config is not None:
        from .execution_budget import authorize
        authorize(attempt)
    control.dispatch(attempt.attempt_id)
    return None


async def _recall_dispatch(attempt):
    return (await _recall_dispatch_authority(attempt))[0]


async def _recall_dispatch_authority(attempt):
    """Recall only this live owner's original admission; never restart an effect."""
    from .control_runtime import current
    control = current(attempt.binding)
    control.require_admission()
    saved = control.get("attempt", attempt.attempt_id)
    if saved is None or saved["prepared"] != attempt.prepared:
        raise JobError("model_attempt_owner_mismatch")
    if saved["phase"] == "prepared":
        return False, None
    if saved["phase"] != "dispatched":
        raise JobError("model_attempt_dispatch_conflict")
    return True, None


async def finish(attempt, *, status, usage, estimated, error_type=None, cost_amount=None, response_id=None,
                 cost_currency=None, response_metrics=None):
    if status not in ("returned", "unknown") or type(estimated) is not bool or not isinstance(usage, dict):
        raise JobError("invalid_model_attempt_observation")
    if set(usage) - set(_TOKENS) or any(type(v) is not int or not 0 <= v < 2**63 for v in usage.values()):
        raise JobError("invalid_model_attempt_usage")
    if error_type is not None and (not isinstance(error_type, str) or not 0 < len(error_type) <= 255):
        raise JobError("invalid_model_attempt_observation")
    if not _cost(cost_amount) or not _response_id(response_id):
        raise JobError("invalid_model_attempt_observation")
    binding = attempt.binding
    from .control_runtime import current
    control = current(binding)
    saved = control.get("attempt", attempt.attempt_id)
    if (saved is None or saved["prepared"] != attempt.prepared
            or saved["prepared"]["scope"] != _scope(binding.lease)
            or saved["prepared"]["session_id"] != binding.actor.session_id):
        raise JobError("model_attempt_owner_mismatch")
    data = {
        "status": status, "usage": dict(usage), "estimated": estimated, "error_type": error_type,
        "provider_cost": cost_amount, "provider_cost_currency": cost_currency,
        "provider_response_id": response_id,
    }
    if saved["phase"] == "finished" and saved["observation"] != data:
        raise JobError("model_attempt_conflict")
    control.observe(attempt.attempt_id, data)
    if control.closed:
        await control.checkpoint(force=True)
    return data


async def require_quiescent(manager, tx, row, *, include_budget=True):
    """Check the live execution and outstanding accounting before closing."""
    binding = manager._execution_controls.get((row["job_id"], row["generation"]))
    if binding is not None:
        from .control_runtime import current
        state = current(binding)
        if any(attempt["kind"] == "model" and attempt["phase"] == "dispatched"
               for attempt in state.active.values()):
            raise JobError("model_attempt_unsettled")
    if include_budget:
        from .model_budgets import require_quiescent as budget_quiescent
        return await budget_quiescent(manager, tx, row)
    return []


async def start_call(pipeline, request, *, round_index, retry_index, model, purpose=None):
    from .task_runtime import _SEND

    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return None
    # Hash the exact logical request, never persist its sensitive contents.
    values = asdict(request)
    args = dict(
        nonce=secrets.token_hex(16),
        round_index=round_index,
        retry_index=retry_index,
        request_hash=_hash(values),
        model=model,
        max_tokens=request.max_tokens,
        purpose=purpose,
    )
    try:
        attempt = await prepare(binding, **args)
    except Exception:
        # Same key recall only; prepare never sends a provider request.
        attempt = await prepare(binding, **args)
    from .model_budgets import close_unreserved, reserve

    try:
        await reserve(attempt, request, service=getattr(pipeline, "llm", None))
    except Exception:
        try:
            await close_unreserved(attempt)
        except Exception:
            # Preserve the original admission error. Failed proof leaves the
            # attempt unresolved; it never authorizes a retry or zero charge.
            binding.resource_evidence["budget_pending"] = True
            log.exception("Unreserved attempt closure remains pending: %s", attempt.attempt_id)
        raise
    from .response_capture import bind
    attempt = replace(attempt, capture_binding=await bind(attempt))
    try:
        deadline = await dispatch(attempt)
    except JobError:
        raise
    except Exception:
        try:
            admitted, deadline = await _recall_dispatch_authority(attempt)
        except BaseException:
            binding.resource_evidence["resource_state"] = "unknown"
            raise
        if not admitted:
            raise JobError("model_attempt_dispatch_unconfirmed") from None
    except BaseException:
        binding.resource_evidence["resource_state"] = "unknown"
        raise
    binding.resource_evidence.setdefault("model_attempts_inflight", set()).add(attempt.attempt_id)
    return replace(attempt, local_deadline=deadline)


async def provider_deadline(attempt):
    """Use the original validated dispatch's bound; never a fresh grant/read.

    Captured before the dispatch transaction commits, so commit/ACK delay
    consumes the original window. An ACK recall revalidates the same policy.
    This local field is not persisted and never authorizes another request.
    """
    return None if attempt is None else attempt.local_deadline


async def observe(attempt, *, usage, estimated, error=None, cost_amount=None, response_id=None,
                  cost_currency=None, response_metrics=None, rejected=False):
    if attempt is None:
        return
    if error is not None and not rejected:
        attempt.binding.resource_evidence["resource_state"] = "unknown"
    args = dict(
        status="unknown" if error is not None and not rejected else "returned",
        usage=usage,
        estimated=estimated,
        error_type=type(error).__name__ if error is not None else None,
        cost_amount=cost_amount,
        cost_currency=cost_currency, response_metrics=response_metrics,
        response_id=response_id,
    )
    if error is None and any(e.get('interface') == 'cost' and e.get('on_error') == 'raise'
                             for e in (response_metrics or {}).get('errors', [])):
        args['error_type'] = 'response_cost_extraction_failed'
    try:
        observation = await finish(attempt, **args)
    except Exception:
        try:
            observation = await finish(attempt, **args)
        except Exception:
            # Retain the actual response/usage in the normal pipeline. Failed
            # evidence persistence must block further dispatch, not charge or
            # execute the same request again.
            attempt.binding.resource_evidence["resource_state"] = "unknown"
            log.exception("model attempt observation remains unresolved: %s", attempt.attempt_id)
            return
    try:
        from .control_runtime import current
        from .model_budgets import observe as observe_budget

        control = current(attempt.binding)
        await observe_budget(attempt, observation)
        if not control.closed:
            await control.checkpoint()
    except Exception:
        # The actual model response must survive budget-delivery failure. The
        # durable pending record blocks further paid work, not jobs/read/stop.
        attempt.binding.resource_evidence["budget_pending"] = True
        log.exception("budget handoff remains pending: %s", attempt.attempt_id)
    if error is None or rejected:
        attempt.binding.resource_evidence.get("model_attempts_inflight", set()).discard(attempt.attempt_id)
