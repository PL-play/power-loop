"""Durable observations of physical model calls, not budget reservations.

No request/response body, credential or provider control capability is stored.
An exception is uncertain remote execution; only a complete response proves
this model call returned. Late observations never grant dispatch authority.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass
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
    cost = getattr(usage, "cost", None)
    return {
        "usage": {key: getattr(usage, key) for key in _TOKENS if getattr(usage, key, None) is not None},
        "cost_amount": str(cost) if cost is not None else None,
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
            valid = (
                isinstance(data.get("nonce"), str)
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
                    and data.get("provider_cost_currency") is None
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


async def _put(manager, tx, row, phase, identity, data):
    old = await _read(manager, tx, phase, identity, required=False)
    if old:
        if _hash(old[1]) != _hash(data):
            raise JobError("model_attempt_conflict")
        return old[0]["event_id"]
    key = event_id(manager, phase, identity)
    await tx.execute(
        f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            key,
            row["job_id"],
            "model_attempt_" + phase,
            row["version"],
            _hash(data),
            _json(data),
            "not_applicable",
            manager.now(),
        ),
    )
    return key


async def enable(binding):
    from .model_budgets import ensure_policy

    await ensure_policy(binding)
    manager = binding.runtime.manager
    scope = _scope(binding.lease)
    data = {"version": 1, "scope": scope, "session_id": binding.actor.session_id}
    identity = scope
    try:
        async with manager._write() as tx:
            row = await manager._fence(tx, binding.actor, binding.lease)
            if row["kind"] != "send" or row["session_id"] != binding.actor.session_id:
                raise JobError("model_attempt_owner_mismatch")
            await _put(manager, tx, row, "tracking", identity, data)
    except Exception:
        async with manager._write() as tx:
            await manager._authorize(tx, binding.actor, binding.lease.job_id)
            old = await _read(manager, tx, "tracking", identity, required=False)
            if old is None or old[1] != data:
                raise
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


async def prepare(binding, *, nonce, round_index, retry_index, request_hash, model, max_tokens):
    if (
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
    async with manager._write() as tx:
        row = await manager._authorize(tx, binding.actor, scope["job_id"])
        old = await _read(manager, tx, "prepared", identity, required=False)
        if old:
            if old[1] != data:
                raise JobError("model_attempt_conflict")
            return Attempt(binding, identity, data)
        row = await manager._fence(tx, binding.actor, binding.lease)
        tracking = await _read(manager, tx, "tracking", scope)
        if (
            row["kind"] != "send"
            or row["session_id"] != binding.actor.session_id
            or tracking[1] != {"version": 1, "scope": scope, "session_id": binding.actor.session_id}
        ):
            raise JobError("model_attempt_owner_mismatch")
        await _no_unknown_sends(manager, tx, row)
        await _no_inflight(manager, tx, row)
        count = await tx.fetchone(
            f"SELECT COUNT(*) AS count FROM {manager.events} WHERE job_id=? AND event_type='model_attempt_prepared'",
            (row["job_id"],),
        )
        if count["count"] >= _LIMIT:
            raise JobError("model_attempt_limit")
        await _put(manager, tx, row, "prepared", identity, data)
    return Attempt(binding, identity, data)


async def dispatch(attempt):
    binding, data = attempt.binding, attempt.prepared
    manager = binding.runtime.manager
    async with manager._write() as tx:
        # Even exact dispatch replays must not revive a closed/cancelled lease.
        row = await manager._fence(tx, binding.actor, binding.lease)
        await _no_unknown_sends(manager, tx, row)
        saved = await _read(manager, tx, "prepared", attempt.attempt_id)
        if saved[1] != data or data["scope"] != _scope(binding.lease) or data["session_id"] != binding.actor.session_id:
            raise JobError("model_attempt_owner_mismatch")
        tracking = await _read(manager, tx, "tracking", data["scope"])
        if tracking[1] != {"version": 1, "scope": data["scope"], "session_id": binding.actor.session_id}:
            raise JobError("model_attempt_owner_mismatch")
        await _no_inflight(manager, tx, row, except_attempt=attempt.attempt_id)
        if await _read(manager, tx, "finished", attempt.attempt_id, required=False):
            raise JobError("model_attempt_already_finished")
        payload = {"version": 1, "scope": data["scope"], "attempt_id": attempt.attempt_id, "prepared_hash": _hash(data)}
        if await _read(manager, tx, "dispatched", attempt.attempt_id, required=False):
            # This acknowledgement proves a previous dispatch admission; a
            # caller cannot treat it as authority for another provider call.
            raise JobError("model_attempt_dispatch_already_admitted")
        from .model_budgets import authorize_dispatch

        await authorize_dispatch(attempt, tx, row)
        await _put(manager, tx, row, "dispatched", attempt.attempt_id, payload)


async def _recall_dispatch(attempt):
    """Same in-flight caller recalls a lost ACK; not a restart dispatch port."""
    binding = attempt.binding
    manager = binding.runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        await _no_unknown_sends(manager, tx, row)
        await _no_inflight(manager, tx, row, except_attempt=attempt.attempt_id)
        prepared = await _read(manager, tx, "prepared", attempt.attempt_id)
        tracking = await _read(manager, tx, "tracking", _scope(binding.lease))
        if (
            prepared[1].get("session_id") != binding.actor.session_id
            or prepared[1].get("scope") != _scope(binding.lease)
            or tracking[1] != {"version": 1, "scope": _scope(binding.lease), "session_id": binding.actor.session_id}
        ):
            raise JobError("model_attempt_owner_mismatch")
        saved = await _read(manager, tx, "dispatched", attempt.attempt_id, required=False)
        if saved is None:
            return False
        if (
            prepared[1] != attempt.prepared
            or saved[1]
            != {
                "version": 1,
                "scope": _scope(binding.lease),
                "attempt_id": attempt.attempt_id,
                "prepared_hash": _hash(attempt.prepared),
            }
            or await _read(manager, tx, "finished", attempt.attempt_id, required=False)
        ):
            raise JobError("model_attempt_dispatch_conflict")
        from .model_budgets import require_dispatch

        await require_dispatch(attempt, tx)
        return True


async def finish(attempt, *, status, usage, estimated, error_type=None, cost_amount=None, response_id=None):
    if status not in ("returned", "unknown") or type(estimated) is not bool or not isinstance(usage, dict):
        raise JobError("invalid_model_attempt_observation")
    if set(usage) - set(_TOKENS) or any(type(v) is not int or not 0 <= v < 2**63 for v in usage.values()):
        raise JobError("invalid_model_attempt_usage")
    if error_type is not None and (not isinstance(error_type, str) or not 0 < len(error_type) <= 255):
        raise JobError("invalid_model_attempt_observation")
    if not _cost(cost_amount) or not _response_id(response_id):
        raise JobError("invalid_model_attempt_observation")
    binding = attempt.binding
    manager = binding.runtime.manager
    async with manager._write() as tx:
        # Observation is allowed after cancellation/expiry. Original immutable
        # attempt identity is authority only to preserve evidence, not execute.
        row = await manager._authorize(tx, binding.actor, binding.lease.job_id)
        saved = await _read(manager, tx, "prepared", attempt.attempt_id)
        if (
            saved[1] != attempt.prepared
            or saved[1]["scope"] != _scope(binding.lease)
            or saved[1]["session_id"] != binding.actor.session_id
        ):
            raise JobError("model_attempt_owner_mismatch")
        dispatched = await _read(manager, tx, "dispatched", attempt.attempt_id)
        data = {
            "version": 1,
            "scope": saved[1]["scope"],
            "attempt_id": attempt.attempt_id,
            "prepared_hash": _hash(saved[1]),
            "dispatch_hash": _hash(dispatched[1]),
            "status": status,
            "usage": dict(usage),
            "estimated": estimated,
            "error_type": error_type,
            "provider_cost": cost_amount,
            "provider_cost_currency": None,
            "provider_response_id": response_id,
        }
        await _put(manager, tx, row, "finished", attempt.attempt_id, data)


async def require_quiescent(manager, tx, row):
    """Return immutable event IDs; absent tracking is legacy, not zero calls."""
    scope = _scope(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"]))
    tracking = await _read(manager, tx, "tracking", scope, required=False)
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
    phases = await _all(manager, tx, row["job_id"])
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
    from .model_budgets import require_quiescent as budget_quiescent

    refs.extend(await budget_quiescent(manager, tx, row))
    return refs


async def start_call(pipeline, request, *, round_index, retry_index, model):
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
    try:
        await dispatch(attempt)
    except JobError:
        raise
    except Exception:
        try:
            admitted = await _recall_dispatch(attempt)
        except BaseException:
            binding.resource_evidence["resource_state"] = "unknown"
            raise
        if not admitted:
            raise JobError("model_attempt_dispatch_unconfirmed") from None
    except BaseException:
        binding.resource_evidence["resource_state"] = "unknown"
        raise
    binding.resource_evidence.setdefault("model_attempts_inflight", set()).add(attempt.attempt_id)
    return attempt


async def observe(attempt, *, usage, estimated, error=None, cost_amount=None, response_id=None):
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
        response_id=response_id,
    )
    try:
        await finish(attempt, **args)
    except Exception:
        try:
            await finish(attempt, **args)
        except Exception:
            # Retain the actual response/usage in the normal pipeline. Failed
            # evidence persistence must block further dispatch, not charge or
            # execute the same request again.
            attempt.binding.resource_evidence["resource_state"] = "unknown"
            log.exception("model attempt observation remains unresolved: %s", attempt.attempt_id)
            return
    try:
        from .model_budgets import observe as observe_budget

        async with attempt.binding.runtime.manager._write() as tx:
            saved = await _read(attempt.binding.runtime.manager, tx, "finished", attempt.attempt_id)
        await observe_budget(attempt, saved[1])
    except Exception:
        # The actual model response must survive budget-delivery failure. The
        # durable pending record blocks further paid work, not jobs/read/stop.
        attempt.binding.resource_evidence["budget_pending"] = True
        log.exception("budget handoff remains pending: %s", attempt.attempt_id)
    if error is None:
        attempt.binding.resource_evidence.get("model_attempts_inflight", set()).discard(attempt.attempt_id)
