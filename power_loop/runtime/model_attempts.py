"""Durable observations of physical model calls, not budget reservations.

No request/response body, credential or provider control capability is stored.
An exception is uncertain remote execution; only a complete response proves
this model call returned. Late observations never grant dispatch authority.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation

from .jobs import JobError, JobLease, _hash, _json

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


def _object(raw):
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError()
        _json(data)
        return data
    except (ValueError, TypeError) as exc:
        raise JobError("model_attempt_evidence_corrupt") from exc


def _hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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


def _validate_data(manager, phase, identity, data):
    from .execution_membership import _lease

    try:
        scope = _lease(data.get("scope"))
        keys = {"version", "scope"}
        if phase == "tracking":
            keys |= {"session_id"}
            valid = scope == identity
        elif phase == "prepared":
            keys |= {
                "attempt_id",
                "session_id",
                "nonce",
                "round_index",
                "retry_index",
                "request_hash",
                "model",
                "max_tokens",
            }
            if "purpose" in data:
                keys |= {"purpose"}
            valid = (
                data.get("purpose", "fold") == "fold"
                and isinstance(data.get("nonce"), str)
                and 1 <= len(data["nonce"]) <= 64
                and data.get("attempt_id")
                == identity
                == "attempt_" + _hash([manager.store.store_id, scope, data["nonce"]])[:48]
                and type(data.get("round_index")) is int
                and 0 <= data["round_index"] < 2**63
                and type(data.get("retry_index")) is int
                and 1 <= data["retry_index"] < 2**63
                and _hash_value(data.get("request_hash"))
                and isinstance(data.get("model"), str)
                and len(data["model"]) <= 255
                and (
                    data.get("max_tokens") is None
                    or type(data["max_tokens"]) is int
                    and 1 <= data["max_tokens"] < 2**63
                )
            )
        else:
            keys |= {"attempt_id", "prepared_hash"}
            valid = data.get("attempt_id") == identity and _hash_value(data.get("prepared_hash"))
            if phase == "finished":
                keys |= {
                    "dispatch_hash",
                    "status",
                    "usage",
                    "estimated",
                    "error_type",
                    "provider_cost",
                    "provider_cost_currency",
                    "provider_response_id",
                }
                usage = data.get("usage")
                valid = (
                    valid
                    and _hash_value(data.get("dispatch_hash"))
                    and data.get("status") in ("returned", "unknown")
                    and type(data.get("estimated")) is bool
                    and isinstance(usage, dict)
                    and not set(usage) - set(_TOKENS)
                    and all(type(v) is int and 0 <= v < 2**63 for v in usage.values())
                    and _cost(data.get("provider_cost"))
                    and (data.get("provider_cost_currency") is None or
                         isinstance(data['provider_cost_currency'], str) and len(data['provider_cost_currency']) == 3
                         and data['provider_cost_currency'].isalpha() and data['provider_cost_currency'].isupper())
                    and _response_id(data.get("provider_response_id"))
                    and (
                        data.get("error_type") is None
                        or isinstance(data["error_type"], str)
                        and 0 < len(data["error_type"]) <= 255
                    )
                )
        if (
            set(data) != keys
            or not valid
            or phase in ("tracking", "prepared")
            and (not isinstance(data.get("session_id"), str) or not 0 < len(data["session_id"]) <= 255)
        ):
            raise JobError("model_attempt_evidence_corrupt")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise JobError("model_attempt_evidence_corrupt") from exc


async def _read(manager, tx, phase, identity, *, required=True):
    key = event_id(manager, phase, identity)
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if row is None:
        if required:
            raise JobError("model_attempt_evidence_missing")
        return None
    data = _object(row["payload_json"])
    _validate_data(manager, phase, identity, data)
    if (
        row["event_type"] != "model_attempt_" + phase
        or row["delivery_status"] != "not_applicable"
        or row["request_hash"] != _hash(data)
        or data.get("scope", {}).get("job_id") != row["job_id"]
    ):
        raise JobError("model_attempt_evidence_corrupt")
    return row, data


async def _all(manager, tx, job_id):
    rows = await tx.fetchall(
        f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type IN "
        "('model_attempt_prepared','model_attempt_dispatched','model_attempt_finished') ORDER BY event_id LIMIT ?",
        (job_id, 3 * _LIMIT + 1),
    )
    if len(rows) > 3 * _LIMIT:
        raise JobError("model_attempt_limit")
    phases = {phase: {} for phase in _PHASES if phase != "tracking"}
    for row in rows:
        phase = row["event_type"].removeprefix("model_attempt_")
        data = _object(row["payload_json"])
        identity = data.get("attempt_id")
        _validate_data(manager, phase, identity, data)
        if (
            row["job_id"] != job_id
            or data["scope"]["job_id"] != job_id
            or row["event_id"] != event_id(manager, phase, identity)
            or row["delivery_status"] != "not_applicable"
            or row["request_hash"] != _hash(data)
            or identity in phases[phase]
        ):
            raise JobError("model_attempt_evidence_corrupt")
        phases[phase][identity] = (row, data)
    prepared, dispatched, finished = (phases[p] for p in ("prepared", "dispatched", "finished"))
    if len(prepared) > _LIMIT:
        raise JobError("model_attempt_limit")
    if set(dispatched) - set(prepared) or set(finished) - set(dispatched):
        raise JobError("model_attempt_evidence_corrupt")
    for identity, (_, data) in dispatched.items():
        source = prepared[identity][1]
        if data != {"version": 1, "scope": source["scope"], "attempt_id": identity, "prepared_hash": _hash(source)}:
            raise JobError("model_attempt_evidence_corrupt")
    for identity, (_, data) in finished.items():
        if (
            data["scope"] != prepared[identity][1]["scope"]
            or data["prepared_hash"] != _hash(prepared[identity][1])
            or data["dispatch_hash"] != _hash(dispatched[identity][1])
        ):
            raise JobError("model_attempt_evidence_corrupt")
    return phases


async def enable(binding, *, review_sources=None):
    from .control_runtime import ReviewSourcesChanged, needs_review_sources, open_control, prepare_review_sources

    manager = binding.runtime.manager
    joined = manager._joined_write.get() is not None
    if (binding.resource_evidence.get('_control_state') is None and not joined and review_sources is None
            and needs_review_sources(manager,binding.actor)):
        review_sources = await prepare_review_sources(manager,binding.actor)
    while True:
        try:
            await open_control(binding,review_sources=review_sources)
            break
        except ReviewSourcesChanged:
            if joined:
                raise
            review_sources = await prepare_review_sources(manager,binding.actor)
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
        "AND tenant_id=? AND principal_id=? AND control_owner_job_id=? AND control_owner_generation=? LIMIT 1",
        (
            row["job_id"],
            row["tenant_id"],
            row["principal_id"],
            row["control_owner_job_id"],
            row["control_owner_generation"],
        ),
    )
    if blocked:
        raise JobError("unresolved_prior_attempt")


async def _no_inflight(manager, tx, row, *, except_attempt=None):
    phases = await _all(manager, tx, row["job_id"])
    for identity in phases["prepared"]:
        if identity == except_attempt:
            continue
        exited = phases["finished"].get(identity)
        if exited is None or exited[1].get("status") != "returned":
            if identity not in phases["dispatched"]:
                from .model_budgets import is_closed_unstarted

                if await is_closed_unstarted(manager, tx, phases["prepared"][identity][1]):
                    continue
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
    from .control_runtime import current, source
    control = current(binding)
    old = control.get("attempt", identity)
    if old is not None:
        if old["prepared"] != data:
            raise JobError("model_attempt_conflict")
        return Attempt(binding, identity, data)
    if control.models >= _LIMIT:
        raise JobError("model_attempt_limit")
    value = control.begin_attempt(identity, "model", data)
    value["event_metadata"] = {"job_version": source(binding)["version"], "created_at": manager.now()}
    control.record("attempt", identity, value)
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
    if "_control_wait_deadline" not in binding.resource_evidence:
        # A failed/unknown wait-registration transaction invalidates this hint.
        # Resolve its durable policy before any provider dispatch; successful
        # registrations publish their bound without another database round trip.
        from .owner_waits import require_dispatch
        manager = binding.runtime.manager
        async with manager._write() as tx:
            row = await manager._fence(tx, binding.actor, binding.lease)
            deadline = await require_dispatch(manager, tx, row)
        binding.resource_evidence["_control_wait_deadline"] = deadline
    if binding.runtime._original_budget_config is not None:
        from .execution_budget import authorize
        authorize(attempt)
    control.dispatch(attempt.attempt_id)
    bound = binding.resource_evidence.get("_control_wait_deadline")
    return None if bound is None else asyncio.get_running_loop().time() + max(
        0, (bound - binding.runtime.manager.now()) / 1000)


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
    bound = attempt.binding.resource_evidence.get("_control_wait_deadline")
    deadline = None if bound is None else asyncio.get_running_loop().time() + max(
        0, (bound - attempt.binding.runtime.manager.now()) / 1000)
    return True, deadline


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
    from .control_runtime import current, retained_owner
    control = current(binding)
    saved = control.get("attempt", attempt.attempt_id)
    if (saved is None or saved["prepared"] != attempt.prepared
            or saved["prepared"]["scope"] != _scope(binding.lease)
            or saved["prepared"]["session_id"] != binding.actor.session_id):
        raise JobError("model_attempt_owner_mismatch")
    prepared = saved["prepared"]
    dispatched = {"version": 1, "scope": prepared["scope"], "attempt_id": attempt.attempt_id,
                  "prepared_hash": _hash(prepared)}
    data = {
        "version": 1, "scope": prepared["scope"], "attempt_id": attempt.attempt_id,
        "prepared_hash": _hash(prepared), "dispatch_hash": _hash(dispatched),
        "status": status, "usage": dict(usage), "estimated": estimated, "error_type": error_type,
        "provider_cost": cost_amount, "provider_cost_currency": cost_currency,
        "provider_response_id": response_id,
    }
    if saved["phase"] == "finished" and saved["observation"] != data:
        raise JobError("model_attempt_conflict")
    if control.closed:
        if retained_owner(binding):
            # Finish any original failed/unknown close commit first. A clean
            # sealed owner has no new RAM batch to publish after another
            # process advances its journal for financial reconciliation.
            while control._pending is not None or control._dirty:
                await control.checkpoint(force=True)
        await _finish_closed(attempt, {**saved, "phase": "finished", "observation": data})
        return data
    control.observe(attempt.attempt_id, data)
    return data


async def _finish_closed(attempt, value):
    """Append an original late return using the current sealed journal head."""
    from .control_runtime import append_correction
    binding = attempt.binding
    manager = binding.runtime.manager
    async with manager._write() as tx:
        job = await manager._authorize(tx, binding.actor, binding.lease.job_id)
        manager._control_retirements.get().add((binding.lease.job_id, binding.lease.generation))
        phases = await _all(manager, tx, job["job_id"])
        prepared = phases["prepared"].get(attempt.attempt_id)
        dispatched = phases["dispatched"].get(attempt.attempt_id)
        if prepared is None or dispatched is None:
            raise JobError("model_attempt_evidence_missing")
        if prepared[1] != attempt.prepared:
            raise JobError("model_attempt_owner_mismatch")
        finished = phases["finished"].get(attempt.attempt_id)
        if finished is not None:
            if finished[1] != value["observation"]:
                raise JobError("model_attempt_conflict")
            return
        await append_correction(manager, tx, job, binding.lease.generation,
            [{"kind": "attempt", "key": attempt.attempt_id, "value": value}],
            summary_delta={"completed": 1, "inflight": -1,
                           "unknown": int(value["observation"]["status"] == "unknown")})


async def require_quiescent(manager, tx, row, *, include_budget=True, loaded=None):
    """Return immutable event IDs; absent tracking is legacy, not zero calls.

    ``loaded`` is ``(tracking, phases)`` already read in this transaction.
    """
    if type(include_budget) is not bool:
        raise JobError("invalid_model_quiescence_options")
    scope = _scope(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"]))
    tracking = loaded[0] if loaded else await _read(manager, tx, "tracking", scope, required=False)
    if tracking is None:
        creation = await tx.fetchone(
            f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create' AND event_type='operation'",
            (row["job_id"],),
        )
        if (
            creation
            and json.loads(creation["payload_json"]).get("request", {}).get("model_attempt_protocol") is not None
        ):
            raise JobError("model_attempt_evidence_missing")
        if await tx.fetchone(
            f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type IN "
            "('model_attempt_prepared','model_attempt_dispatched','model_attempt_finished') LIMIT 1",
            (row["job_id"],),
        ):
            raise JobError("model_attempt_evidence_missing")
        return []
    if tracking[1] != {"version": 1, "scope": scope, "session_id": row["session_id"]}:
        raise JobError("model_attempt_evidence_corrupt")
    refs = [tracking[0]["event_id"]]
    phases = loaded[1] if loaded else await _all(manager, tx, row["job_id"])
    for identity, prepared in phases["prepared"].items():
        raw, data = prepared
        if prepared[1]["scope"] != scope or prepared[1]["session_id"] != row["session_id"]:
            raise JobError("model_attempt_evidence_corrupt")
        refs.append(raw["event_id"])
        dispatched = phases["dispatched"].get(identity)
        finished = phases["finished"].get(identity)
        if dispatched is None:
            if finished is not None or row["intake"] != "closed":
                raise JobError("model_attempt_unsettled")
            continue
        if (
            dispatched[1]
            != {"version": 1, "scope": scope, "attempt_id": data["attempt_id"], "prepared_hash": _hash(data)}
            or finished is None
            or finished[1].get("prepared_hash") != _hash(data)
            or finished[1].get("dispatch_hash") != _hash(dispatched[1])
            or finished[1].get("status") != "returned"
        ):
            raise JobError("model_attempt_unsettled")
        refs.extend([dispatched[0]["event_id"], finished[0]["event_id"]])
    if include_budget:
        from .model_budgets import require_quiescent as budget_quiescent
        refs.extend(await budget_quiescent(manager, tx, row))
    return refs


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
                  cost_currency=None, response_metrics=None):
    if attempt is None:
        return
    if error is not None:
        attempt.binding.resource_evidence["resource_state"] = "unknown"
    args = dict(
        status="unknown" if error is not None else "returned",
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
    if error is None:
        attempt.binding.resource_evidence.get("model_attempts_inflight", set()).discard(attempt.attempt_id)
