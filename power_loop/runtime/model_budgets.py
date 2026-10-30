"""Durable local admission and host-budget handoff for original model attempts.

The store control transaction serializes local scopes. No transaction spans a
host request. API grants are evidence for one original attempt, never replay
permission. All model arguments and actual prompt bodies stay outside this log.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict

from . import budget_scopes
from .budget_contracts import BudgetAck, BudgetConfig, BudgetGrant, BudgetPlan, BudgetSubmission
from .jobs import JobError, _hash, _json

_COUNTERS = budget_scopes.COUNTERS


def accounting_context(state):
    """Carry the original reserved request across hosts, recovery and revisions."""
    from .budget_contracts import payload
    return {**payload(state['context']),
        'reservation_request': payload(state['plan']['request'])}


def policy_key(runtime):
    return "bp_" + _hash([runtime.manager.store.store_id, runtime.tenant_id, runtime.principal_id])[:48]


def scope_key(key, name):
    return "bs_" + _hash([key, name])[:48]


def _object(raw, digest):
    try:
        value = json.loads(raw)
        if type(value) is not dict or _hash(value) != digest:
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise JobError("budget_evidence_corrupt") from None


async def _policy(runtime, tx, *, create=False, job=None):
    manager, config, key = runtime.manager, runtime.budget_config, policy_key(runtime)
    if config is not runtime._original_budget_config:
        raise JobError("budget_configuration_changed")
    if config is not None:
        lifecycle = runtime._lifecycle
        lifecycle._check_manager_binding()
        if runtime.manager is not lifecycle.manager or any(
            getattr(runtime, k) != getattr(lifecycle.identity, k)
            for k in ("tenant_id", "principal_id", "executor_id", "boot_id")
        ):
            raise JobError("budget_runtime_binding_changed")
    row = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_policies WHERE policy_key=?", (key,))
    marker_id = "mb_policy_" + key[3:]
    marker = await tx.fetchone(f"SELECT payload_json,request_hash FROM {manager.events} WHERE event_id=?", (marker_id,))
    if row is None:
        if marker is not None:
            raise JobError("budget_policy_missing")
        if config is None:
            return None
        if not create:
            raise JobError("budget_policy_missing")
        manifest = config.manifest()
        if job is None:
            raise JobError("budget_policy_owner_required")
        legacy = await tx.fetchone(
            f"SELECT job_id FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND kind='send' "
            "AND job_id<>? AND resource_state IN ('running','unknown') LIMIT 1",
            (runtime.tenant_id, runtime.principal_id, job["job_id"]),
        )
        if legacy:
            raise JobError("budget_legacy_execution_unsettled")
        await tx.execute(
            f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                marker_id,
                job["job_id"],
                "model_budget_policy",
                job["version"],
                _hash(manifest),
                _json(manifest),
                "not_applicable",
                manager.now(),
            ),
        )
        marker = {"payload_json": _json(manifest), "request_hash": _hash(manifest)}
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_policies "
            "(policy_key,tenant_id,principal_id,request_hash,manifest_json,enabled,created_at) VALUES (?,?,?,?,?,?,?)",
            (key, runtime.tenant_id, runtime.principal_id, _hash(manifest), _json(manifest), 1, manager.now()),
        )
        for limits in manifest["scopes"]:
            if limits.get("binding", "authority") != "authority":
                continue
            counters = dict.fromkeys(_COUNTERS, 0)
            await tx.execute(
                f"INSERT INTO {manager.prefix}task_budget_scopes "
                "(scope_key,policy_key,limits_hash,limits_json,counters_hash,counters_json) VALUES (?,?,?,?,?,?)",
                (scope_key(key, limits["key"]), key, _hash(limits), _json(limits), _hash(counters), _json(counters)),
            )
        row = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_policies WHERE policy_key=?", (key,))
    manifest = _object(row["manifest_json"], row["request_hash"])
    if marker is None or _object(marker["payload_json"], marker["request_hash"]) != manifest:
        raise JobError("budget_policy_evidence_corrupt")
    if (row["tenant_id"], row["principal_id"]) != (runtime.tenant_id, runtime.principal_id):
        raise JobError("budget_policy_identity_mismatch")
    if type(row["enabled"]) is not int or row["enabled"] not in (0, 1):
        raise JobError("budget_policy_evidence_corrupt")
    if config is None:
        raise JobError("budget_provider_required")
    if type(config) is not BudgetConfig or config.manifest() != manifest:
        raise JobError("budget_policy_conflict")
    return {**row, "manifest": manifest}


async def ensure_policy(binding):
    runtime = binding.runtime
    async with runtime.manager._write() as tx:
        job = await runtime.manager._fence(tx, binding.actor, binding.lease)
        policy = await _policy(runtime, tx, create=True, job=job)
        if policy:
            for limits in await budget_scopes.resolve(runtime, tx, policy, job, create=True):
                await _scope(runtime.manager, tx, policy["policy_key"], limits)
    return policy is not None


async def _scope(manager, tx, key, limits):
    row = await tx.fetchone(
        f"SELECT * FROM {manager.prefix}task_budget_scopes WHERE scope_key=?", (budget_scopes.scope_key(key, limits),)
    )
    if row is None or row["policy_key"] != key or _object(row["limits_json"], row["limits_hash"]) != limits:
        raise JobError("budget_scope_missing_or_changed")
    await budget_scopes.check_marker(manager, tx, key, limits)
    counters = _object(row["counters_json"], row["counters_hash"])
    if set(counters) != set(_COUNTERS) or any(type(v) is not int or v < 0 for v in counters.values()):
        raise JobError("budget_counters_corrupt")
    return row, counters


async def _save_counters(manager, tx, row, counters):
    if any(type(v) is not int or v < 0 for v in counters.values()):
        raise JobError("budget_counter_underflow")
    await tx.execute(
        f"UPDATE {manager.prefix}task_budget_scopes SET counters_json=?,counters_hash=? WHERE scope_key=?",
        (_json(counters), _hash(counters), row["scope_key"]),
    )


async def _read(manager, tx, attempt_id, *, required=True):
    row = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_attempts WHERE attempt_id=?", (attempt_id,))
    if row is None:
        marker = "mb_" + _hash([attempt_id, 1])[:48]
        if required or await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (marker,)):
            raise JobError("budget_attempt_missing")
        return None
    value = _object(row["state_json"], row["state_hash"])
    if type(row["pending"]) is not int or row["pending"] not in (0, 1) or type(value.get("pending")) is not bool:
        raise JobError("budget_evidence_corrupt")
    if (
        value.get("attempt_id"),
        value.get("job_id"),
        value.get("policy_key"),
        value.get("version"),
        value.get("phase"),
        value.get("pending"),
    ) != (row["attempt_id"], row["job_id"], row["policy_key"], row["version"], row["phase"], bool(row["pending"])):
        raise JobError("budget_evidence_corrupt")
    event_id = "mb_" + _hash([attempt_id, row["version"]])[:48]
    event = await tx.fetchone(
        f"SELECT payload_json,request_hash,job_id,event_type,delivery_status FROM {manager.events} WHERE event_id=?",
        (event_id,),
    )
    if (
        event is None
        or (event["job_id"], event["event_type"], event["delivery_status"])
        != (row["job_id"], "model_budget_state", "not_applicable")
        or _object(event["payload_json"], event["request_hash"]) != value
    ):
        raise JobError("budget_evidence_corrupt")
    return value


async def _save(manager, tx, state, job, *, new=False):
    state["version"] += 1
    digest, raw = _hash(state), _json(state)
    event_id = "mb_" + _hash([state["attempt_id"], state["version"]])[:48]
    await tx.execute(
        f"INSERT INTO {manager.events} "
        "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (event_id, job["job_id"], "model_budget_state", job["version"], digest, raw, "not_applicable", manager.now()),
    )
    if new:
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_attempts "
            "(attempt_id,job_id,policy_key,version,phase,pending,state_hash,state_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                state["attempt_id"],
                job["job_id"],
                state["policy_key"],
                state["version"],
                state["phase"],
                int(state["pending"]),
                digest,
                raw,
                manager.now(),
                manager.now(),
            ),
        )
    else:
        changed = await tx.execute(
            f"UPDATE {manager.prefix}task_budget_attempts "
            "SET version=?,phase=?,pending=?,state_hash=?,state_json=?,updated_at=? WHERE attempt_id=? AND version=?",
            (
                state["version"],
                state["phase"],
                int(state["pending"]),
                digest,
                raw,
                manager.now(),
                state["attempt_id"],
                state["version"] - 1,
            ),
        )
        if changed != 1:
            raise JobError("budget_version_conflict")


def _context(attempt, job):
    from .tool_attempts import ToolAttempt
    from .tool_attempts import context as tool_context
    if type(attempt) is ToolAttempt:
        return tool_context(attempt.binding.runtime.manager, job, attempt.prepared)
    manager = attempt.binding.runtime.manager
    return {
        "version": 1,
        "store_id": manager.store.store_id,
        "tenant_id": job["tenant_id"],
        "principal_id": job["principal_id"],
        "root_job_id": job["root_job_id"],
        "job_id": job["job_id"],
        "generation": attempt.binding.lease.generation,
        "session_id": attempt.binding.actor.session_id,
        "attempt_id": attempt.attempt_id,
        "prepared_event_id": "ma_prepared_" + _hash([manager.store.store_id, attempt.attempt_id])[:48],
        "prepared_hash": _hash(attempt.prepared),
        "request_hash": attempt.prepared["request_hash"],
        "model": attempt.prepared["model"],
        "max_tokens": attempt.prepared["max_tokens"],
    }


async def _port(attempt, operation, call):
    """Bound waits without discarding an API call that may already have committed."""
    return await _host_call(attempt.binding.runtime, attempt.binding.lease.job_id,
                            attempt.attempt_id, operation, call)


async def _host_call(runtime, job_id, attempt_id, operation, call):
    key = (attempt_id, operation)
    task = runtime._budget_handles.get(key)
    if task is None:
        coroutine = call()
        try:
            task = asyncio.create_task(coroutine, name="budget-" + operation + "-" + attempt_id)
        except BaseException:
            if hasattr(coroutine, "close"):
                coroutine.close()
            raise
        runtime._budget_handles[key] = task
        runtime.track_task(task, job_id=job_id)

        def observed(done):
            if not done.cancelled():
                done.exception()

        task.add_done_callback(observed)
    done, _ = await asyncio.wait({task}, timeout=runtime.budget_config.timeout_s)
    if not done:
        raise JobError("budget_host_timeout")
    runtime._budget_handles.pop(key, None)
    return task.result()


async def _check_pending(manager, tx, policy):
    # Include invalid nonzero projections so they cannot disappear from the
    # pending cap through truthiness normalization (e.g. SQL 2 vs JSON true).
    rows = await tx.fetchall(
        f"SELECT attempt_id FROM {manager.prefix}task_budget_attempts WHERE policy_key=? AND pending<>0 LIMIT ?",
        (policy["policy_key"], policy["manifest"]["max_pending"]),
    )
    for row in rows:
        await _read(manager, tx, row["attempt_id"])
    if len(rows) >= policy["manifest"]["max_pending"]:
        raise JobError("budget_reconciliation_required")


async def reserve(attempt, logical_request, *, service=None, pricing=None):
    runtime, manager = attempt.binding.runtime, attempt.binding.runtime.manager
    async with manager._write() as tx:
        job = await manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
        policy = await _policy(runtime, tx)
        if policy is None:
            return False
        if not policy["enabled"]:
            raise JobError("budget_policy_closed")
        await _check_pending(manager, tx, policy)
        context = _context(attempt, job)
    provider = runtime.budget_config.provider
    from .tool_attempts import ToolAttempt
    tool = type(attempt) is ToolAttempt
    if tool and not callable(getattr(provider, 'prepare_tool', None)):
        raise JobError('budget_tool_adapter_unavailable')
    supplied = await _port(
        attempt,
        "prepare",
        (lambda: provider.prepare_tool(copy.deepcopy(context), logical_request, pricing=pricing)) if tool else
        lambda: provider.prepare(copy.deepcopy(context), copy.deepcopy(logical_request), service=service,
                                 **({"purpose": attempt.prepared["purpose"]} if "purpose" in attempt.prepared else {})),
    )
    if type(supplied) is not BudgetPlan:
        raise JobError("invalid_budget_plan")
    plan = BudgetPlan(supplied.token_bound, supplied.request)
    if (tool and plan.token_bound != 0) or (not tool and (
        logical_request.max_tokens is None or plan.token_bound < logical_request.max_tokens
    )):
        raise JobError("budget_request_bound_missing")
    async with manager._write() as tx:
        job = await manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
        policy = await _policy(runtime, tx)
        if not policy["enabled"]:
            raise JobError("budget_policy_closed")
        if await _read(manager, tx, attempt.attempt_id, required=False):
            raise JobError("budget_attempt_already_reserved")
        await _check_pending(manager, tx, policy)
        limits_set = await budget_scopes.resolve(runtime, tx, policy, job, create=True)
        for limits in limits_set:
            row, counters = await _scope(manager, tx, policy["policy_key"], limits)
            for maximum, consumed in (
                ("max_calls", counters["used_calls"] + counters["held_calls"] + 1),
                ("max_tokens", counters["used_tokens"] + counters["held_tokens"] + plan.token_bound),
                ("max_inflight", counters["inflight"] + 1),
            ):
                if limits[maximum] is not None and consumed > limits[maximum]:
                    raise JobError("budget_" + maximum + "_exhausted")
            counters["held_calls"] += 1
            counters["held_tokens"] += plan.token_bound
            counters["inflight"] += 1
            await _save_counters(manager, tx, row, counters)
        state = {
            "version": 0,
            "attempt_id": attempt.attempt_id,
            "job_id": job["job_id"],
            "policy_key": policy["policy_key"],
            "context": context,
            "plan": asdict(plan),
            "plan_hash": plan.request_hash,
            "phase": "local_reserved",
            "pending": True,
            "started_at_ms": manager.now(),
            "host_intent": False,
            "grant": None,
            "observation": None,
            "submission": None,
            "ack": None,
        }
        if policy["manifest"]["version"] == 2:
            state["scopes"] = limits_set
        await _save(manager, tx, state, job, new=True)
    async with manager._write() as tx:
        job = await manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
        state = await _read(manager, tx, attempt.attempt_id)
        if state["phase"] != "local_reserved":
            raise JobError("budget_dispatch_closed")
        state.update(phase="host_reserving", host_intent=True)
        await _save(manager, tx, state, job)
    reserved = None
    try:
        reserved = await _port(attempt, "reserve", lambda: provider.reserve(plan))
    except Exception:
        pass  # exact current-state lookup, never a replacement reserve/attempt.
    if reserved is not None and (
        type(reserved) is not BudgetGrant
        or (reserved.attempt_id, reserved.request_hash, reserved.state)
        != (attempt.attempt_id, state["plan_hash"], "reserved")
    ):
        raise JobError("budget_host_reservation_unconfirmed")
    grant = await _port(attempt, "lookup", lambda: provider.lookup(plan))
    if (
        type(grant) is not BudgetGrant
        or (grant.attempt_id, grant.request_hash, grant.state) != (attempt.attempt_id, state["plan_hash"], "reserved")
        or reserved is not None
        and reserved.version != grant.version
    ):
        raise JobError("budget_host_reservation_unconfirmed")
    async with manager._write() as tx:
        # A receipt may be saved after cancellation, but cannot grant dispatch.
        job = await manager._authorize(tx, attempt.binding.actor, attempt.binding.lease.job_id)
        state = await _read(manager, tx, attempt.attempt_id)
        if state["phase"] != "host_reserving" or state["plan_hash"] != grant.request_hash:
            raise JobError("budget_reservation_conflict")
        state.update(grant=asdict(grant), phase="host_reserved", pending=False)
        await _save(manager, tx, state, job)
    return True


async def authorize_dispatch(attempt, tx, job):
    runtime, manager = attempt.binding.runtime, attempt.binding.runtime.manager
    policy = await _policy(runtime, tx)
    if policy is None:
        return
    if not policy["enabled"]:
        raise JobError("budget_policy_closed")
    state = await _read(manager, tx, attempt.attempt_id)
    if (
        state["policy_key"] != policy["policy_key"]
        or state["context"] != _context(attempt, job)
        or state["phase"] != "host_reserved"
        or state["pending"]
    ):
        raise JobError("budget_dispatch_not_authorized")
    grant = BudgetGrant(**state["grant"])
    if (grant.attempt_id, grant.request_hash, grant.state) != (attempt.attempt_id, state["plan_hash"], "reserved"):
        raise JobError("budget_grant_mismatch")
    for limits in budget_scopes.frozen(policy, state):
        _, counters = await _scope(manager, tx, state["policy_key"], limits)
        if (
            counters["held_calls"] < 1
            or counters["held_tokens"] < state["plan"]["token_bound"]
            or counters["inflight"] < 1
        ):
            raise JobError("budget_reservation_hold_missing")
    state.update(phase="dispatch_authorized")
    await _save(manager, tx, state, job)


async def require_dispatch(attempt, tx):
    manager = attempt.binding.runtime.manager
    policy = await _policy(attempt.binding.runtime, tx)
    if policy:
        state = await _read(manager, tx, attempt.attempt_id)
        if state["phase"] != "dispatch_authorized" or state["context"]["prepared_hash"] != _hash(attempt.prepared):
            raise JobError("budget_dispatch_not_authorized")


async def observe(attempt, observation):
    return await _observe(attempt.binding.runtime, attempt.binding.actor,
                          attempt.binding.lease.job_id, attempt.attempt_id, observation)


async def _observe(runtime, actor, job_id, attempt_id, observation):
    manager = runtime.manager
    async with manager._write() as tx:
        policy = await _policy(runtime, tx)
        if policy is None:
            return
        job = await manager._authorize(tx, actor, job_id)
        from .model_attempts import _read as read_model

        source = await read_model(manager, tx, "finished", attempt_id)
        if source[1] != observation:
            raise JobError("budget_observation_source_mismatch")
        state = await _read(manager, tx, attempt_id)
        if state.get("initial_usage_evidence") is not None:
            from .budget_revisions import read_initial
            await read_initial(manager, tx, state)
            if state["observation"] is not None:
                if state["observation"] == observation:
                    return
                raise JobError("budget_observation_conflict")
            # Financial evidence already consumed the call/token reservation.
            # Only an authentic original return releases execution concurrency;
            # it never overwrites or reapplies the independent financial facts.
            if observation["status"] == "returned":
                for limits in budget_scopes.frozen(policy, state):
                    row, counters = await _scope(manager, tx, state["policy_key"], limits)
                    counters["inflight"] -= 1
                    await _save_counters(manager, tx, row, counters)
            state.update(observation=observation, finished_at_ms=source[0]["created_at"])
            await _save(manager, tx, state, job)
        else:
            if not await _project_observation(manager, tx, policy, state, job, observation, source):
                return
    await _deliver(runtime, actor, job_id, attempt_id)


async def _project_observation(manager, tx, policy, state, job, observation, source):
    """First ordinary observation, when no independent financial evidence exists."""
    if state["phase"] != "dispatch_authorized":
        if state["observation"] == observation:
            return False
        raise JobError("budget_observation_conflict")
    returned = observation["status"] == "returned"
    usage = observation["usage"]
    tokens = None
    if not observation["estimated"]:
        tokens = usage.get("total_tokens")
        if tokens is None and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    bound = state["plan"]["token_bound"]
    for limits in budget_scopes.frozen(policy, state):
        row, counters = await _scope(manager, tx, state["policy_key"], limits)
        counters["held_calls"] -= 1
        counters["used_calls"] += 1
        if tokens is not None:
            counters["held_tokens"] -= bound
            counters["used_tokens"] += tokens
        if returned:
            counters["inflight"] -= 1
        await _save_counters(manager, tx, row, counters)
    if tokens is not None and tokens > bound:
        await tx.execute(
            f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?", (state["policy_key"],)
        )
    # Recovery may happen hours later; never retime the original usage to
    # the reconciler's wall clock (which could change its pricing period).
    state.update(phase="settling", pending=True, observation=observation,
                 finished_at_ms=source[0]["created_at"])
    await _save(manager, tx, state, job)
    return True


async def deliver(attempt):
    """Replay only the frozen original API submission, never the model request."""
    return await _deliver(attempt.binding.runtime, attempt.binding.actor,
                          attempt.binding.lease.job_id, attempt.attempt_id)


async def _deliver(runtime, actor, job_id, attempt_id):
    manager = runtime.manager
    async with manager._write() as tx:
        await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        state = await _read(manager, tx, attempt_id)
        if state['context'].get('kind') == 'tool':
            from .tool_attempts import verify_state
            await verify_state(manager, tx, job, state, finished=True)
        if state["phase"] != "settling":
            raise JobError("budget_usage_not_ready")
        if state.get("initial_usage_evidence") is not None:
            from .budget_revisions import read_initial
            await read_initial(manager, tx, state)
        if state.get("usage_revision") is not None:
            from .budget_revisions import read_revision
            await read_revision(manager, tx, state)
        elif state["submission"] is not None and state["submission"]["revision"] != 1:
            raise JobError("budget_revision_evidence_missing")
        if state["ack"] is not None:
            return state["ack"]
    if state["submission"] is None:
        if state.get("usage_revision") is not None:
            from .budget_revisions import prepare
            submission = await prepare(runtime, actor, state)
        else:
            from .model_attempts import event_id
            if state['context'].get('kind') == 'tool':
                from .tool_attempts import event_id
            observation = {
                **state["observation"],
                "started_at_ms": state["started_at_ms"],
                "finished_at_ms": state["finished_at_ms"],
                "evidence_event_id": event_id(manager, "finished", attempt_id),
                "evidence_hash": _hash(state["observation"]),
            }
            submission = await _host_call(
                runtime, job_id, attempt_id,
                "prepare_usage",
                lambda: runtime.budget_config.provider.prepare_usage(
                    accounting_context(state), BudgetGrant(**state["grant"]), observation
                ),
            )
        if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (
            attempt_id,
            state["usage_revision"]["revision"] if state.get("usage_revision") else 1,
        ):
            raise JobError("invalid_budget_submission")
        async with manager._write() as tx:
            job = await manager._authorize(tx, actor, job_id)
            latest = await _read(manager, tx, attempt_id)
            if latest.get("usage_revision") != state.get("usage_revision"):
                raise JobError("budget_revision_changed")
            if latest["submission"] is not None and latest["submission"] != asdict(submission):
                raise JobError("budget_submission_conflict")
            if latest["submission"] is None:
                latest["submission"] = asdict(submission)
                await _save(manager, tx, latest, job)
            state = latest
    submission = BudgetSubmission(**state["submission"])
    ack = await _host_call(
        runtime, job_id, attempt_id,
        "apply_usage:" + str(submission.revision),
        lambda: runtime.budget_config.provider.apply_usage(submission),
    )
    if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
        attempt_id,
        submission.revision,
        submission.request_hash,
    ):
        raise JobError("budget_ack_mismatch")
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        latest = await _read(manager, tx, attempt_id)
        if latest["submission"] != state["submission"]:
            raise JobError("budget_submission_conflict")
        if latest["ack"] is None:
            latest.update(ack=asdict(ack), pending=False)
            await _save(manager, tx, latest, job)
        elif latest["ack"] != asdict(ack):
            raise JobError("budget_ack_conflict")
    return asdict(ack)


async def _refund_local(manager, tx, policy, state):
    for limits in budget_scopes.frozen(policy, state):
        row, counters = await _scope(manager, tx, state["policy_key"], limits)
        counters["held_calls"] -= 1
        counters["held_tokens"] -= state["plan"]["token_bound"]
        counters["inflight"] -= 1
        await _save_counters(manager, tx, row, counters)


async def _original_preparation(attempt, tx, job):
    runtime, manager, owner = attempt.binding.runtime, attempt.binding.runtime.manager, attempt.binding.lease
    if (owner.executor_id, owner.boot_id) != (runtime.executor_id, runtime.boot_id) or (
        job["executor_id"],
        job["boot_id"],
        job["generation"],
    ) != (owner.executor_id, owner.boot_id, owner.generation):
        raise JobError("budget_release_requires_original_owner")
    from .model_attempts import _read as read_model

    saved = await read_model(manager, tx, "prepared", attempt.attempt_id)
    if (
        saved[1] != attempt.prepared
        or saved[1]["scope"] != asdict(owner)
        or saved[1]["session_id"] != attempt.binding.actor.session_id
        or saved[1]["scope"]["job_id"] != job["job_id"]
    ):
        raise JobError("budget_release_requires_original_owner")


async def close_unreserved(attempt):
    """Seal only proven absence of a reservation; never issue an API release.

    A missing mutable row alone is insufficient: its original first journal
    event must also be absent. Already reserved/uncertain work remains owned by
    close_unstarted and its original host tombstone/ACK protocol.
    """
    runtime, manager = attempt.binding.runtime, attempt.binding.runtime.manager
    async with manager._write() as tx:
        policy = await _policy(runtime, tx)
        if policy is None:
            return None
        job = await manager._authorize(tx, attempt.binding.actor, attempt.binding.lease.job_id)
        await _original_preparation(attempt, tx, job)
        state = await _read(manager, tx, attempt.attempt_id, required=False)
        if state is not None:
            if state["job_id"] != job["job_id"] or state["context"] != _context(attempt, job):
                raise JobError("budget_release_requires_original_owner")
            return (
                state["ack"]
                if state["phase"] == "released"
                and isinstance(state["ack"], dict)
                and state["ack"].get("no_reservation") is True
                else None
            )
        from .model_attempts import _read as read_model

        if await read_model(manager, tx, "dispatched", attempt.attempt_id, required=False):
            raise JobError("budget_release_not_permitted")
        state = {
            "version": 0,
            "attempt_id": attempt.attempt_id,
            "job_id": job["job_id"],
            "policy_key": policy["policy_key"],
            "context": _context(attempt, job),
            "plan": None,
            "plan_hash": None,
            "phase": "released",
            "pending": False,
            "started_at_ms": manager.now(),
            "host_intent": False,
            "grant": None,
            "observation": None,
            "submission": None,
            "ack": {"local_only": True, "dispatch_not_authorized": True, "no_reservation": True},
        }
        if policy["manifest"]["version"] == 2:
            state["scopes"] = []  # No scope was reserved, so none may be refunded.
        await _save(manager, tx, state, job, new=True)
        return state["ack"]


async def is_closed_unstarted(manager, tx, prepared):
    """Read original release evidence without granting another dispatch."""
    state = await _read(manager, tx, prepared["attempt_id"], required=False)
    if state is None or state["phase"] != "released" or state["pending"] or state["ack"] is None:
        return False
    if (
        state["context"]["prepared_hash"] != _hash(prepared)
        or state["context"]["session_id"] != prepared["session_id"]
        or state["job_id"] != prepared["scope"]["job_id"]
        or state["context"]["generation"] != prepared["scope"]["generation"]
    ):
        raise JobError("budget_evidence_corrupt")
    return True


async def close_unstarted(attempt):
    """Original running owner's closure, retaining its exact execution check."""
    local = await close_unreserved(attempt)
    if local is not None:
        return local
    return await _release_unstarted(attempt.binding.runtime, attempt.binding.actor,
        attempt.binding.lease.job_id, attempt.attempt_id, original=attempt)


async def _release_source(runtime, tx, job, attempt_id, policy):
    """Validate persisted authority without impersonating its old executor."""
    from .model_attempts import _read as read_model

    manager = runtime.manager
    if policy is None:
        raise JobError("budget_provider_required")
    state = await _read(manager, tx, attempt_id)
    _, prepared = await read_model(manager, tx, "prepared", attempt_id)
    _, tracking = await read_model(manager, tx, "tracking", prepared["scope"])
    context = {
        "version": 1, "store_id": manager.store.store_id,
        "tenant_id": job["tenant_id"], "principal_id": job["principal_id"],
        "root_job_id": job["root_job_id"], "job_id": job["job_id"],
        "generation": prepared["scope"]["generation"], "session_id": job["session_id"],
        "attempt_id": attempt_id, "prepared_event_id":
            "ma_prepared_" + _hash([manager.store.store_id, attempt_id])[:48],
        "prepared_hash": _hash(prepared), "request_hash": prepared["request_hash"],
        "model": prepared["model"], "max_tokens": prepared["max_tokens"],
    }
    if (job["kind"] != "send" or state["job_id"] != job["job_id"]
            or state["policy_key"] != policy["policy_key"]
            or prepared["scope"]["job_id"] != job["job_id"]
            or prepared["session_id"] != job["session_id"] or state["context"] != context
            or tracking != {"version": 1, "scope": prepared["scope"], "session_id": job["session_id"]}):
        raise JobError("budget_reconciliation_evidence_mismatch")
    return state


async def release_unstarted(runtime, actor, job_id, attempt_id, *, require_cancelled=False):
    """Mechanical recovery: seal original dispatch, then release original risk.

    Does not claim an execution, renew an old lease, or close supervision. A
    missing reservation is not a refund proof. Dispatched attempts are refused.
    """
    if isinstance(attempt_id, str) and attempt_id.startswith('toolattempt_'):
        from .tool_attempts import release_unstarted as release_tool
        if type(require_cancelled) is not bool:
            raise JobError('invalid_budget_release_guard')
        return await release_tool(runtime, actor, job_id, attempt_id, require_cancelled=require_cancelled)
    if (type(job_id) is not str or not job_id or len(job_id) > 80
            or type(attempt_id) is not str or len(attempt_id) != 56 or not attempt_id.startswith("attempt_")
            or any(c not in "0123456789abcdef" for c in attempt_id[8:])):
        raise JobError("invalid_budget_reconciliation_identity")
    if type(require_cancelled) is not bool:
        raise JobError("invalid_budget_release_guard")
    ack = await _release_unstarted(runtime, actor, job_id, attempt_id, require_cancelled=require_cancelled)
    return {"attempt_id": attempt_id, "job_id": job_id, "released": True,
            "execution_authority": False, "ack": ack}


async def _require_cancelled_lineage(manager, tx, job):
    """Read original cancellation by indexed ancestor keys; never infer death."""
    current, seen = job, set()
    for _ in range(64):
        if current is None or current["job_id"] in seen:
            raise JobError("invalid_job_lineage")
        seen.add(current["job_id"])
        if current["cancel_requested_at"] is not None:
            return
        if current["parent_job_id"] is None:
            raise JobError("budget_release_requires_cancellation")
        current = await manager._row(tx, current["parent_job_id"])
    raise JobError("job_depth_exceeded")


async def _release_unstarted(runtime, actor, job_id, attempt_id, *, original=None, require_cancelled=False):
    manager = runtime.manager
    async with manager._write() as tx:
        policy = await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        if original is not None:
            await _original_preparation(original, tx, job)
        state = await _release_source(runtime, tx, job, attempt_id, policy)
        if original is not None and state["context"] != _context(original, job):
            raise JobError("budget_release_requires_original_owner")
        from .model_attempts import _read as read_model

        if await read_model(manager, tx, "dispatched", attempt_id, required=False):
            raise JobError("budget_release_not_permitted")
        if state["phase"] == "released":
            return state["ack"]
        if state["phase"] not in ("local_reserved", "host_reserving", "host_reserved", "dispatch_closed"):
            raise JobError("budget_release_not_permitted")
        if state["phase"] != "dispatch_closed":
            if require_cancelled:
                await _require_cancelled_lineage(manager, tx, job)
            state.update(phase="dispatch_closed", pending=True)
            await _save(manager, tx, state, job)
        if not state["host_intent"]:
            await _refund_local(manager, tx, policy, state)
            state.update(phase="released", pending=False, ack={"local_only": True, "dispatch_not_authorized": True})
            await _save(manager, tx, state, job)
            return state["ack"]
    provider = runtime.budget_config.provider
    if state["grant"] is None:
        plan = BudgetPlan(**state["plan"])
        grant = await _host_call(runtime, job_id, attempt_id, "lookup", lambda: provider.lookup(plan))
        if grant is not None and (
            type(grant) is not BudgetGrant
            or (grant.attempt_id, grant.request_hash) != (attempt_id, state["plan_hash"])
        ):
            raise JobError("budget_host_reservation_unconfirmed")
        if grant is not None and grant.state not in ("reserved", "released"):
            raise JobError("budget_release_not_permitted")
        if grant is not None:
            async with manager._write() as tx:
                job = await manager._authorize(tx, actor, job_id)
                state = await _read(manager, tx, attempt_id)
                if state["phase"] == "released":
                    return state["ack"]
                if state["phase"] != "dispatch_closed":
                    raise JobError("budget_release_not_permitted")
                if state["grant"] is None:
                    state["grant"] = asdict(grant)
                    await _save(manager, tx, state, job)
                if grant.state == "released":
                    await _refund_local(manager, tx, policy, state)
                    state.update(phase="released", pending=False, ack={"release_observation": asdict(grant)})
                    await _save(manager, tx, state, job)
                    return state["ack"]
    if state["submission"] is None:
        proof = {
            "proof": "dispatch_not_authorized",
            "event_id": "mb_" + _hash([attempt_id, state["version"]])[:48],
            "evidence_hash": _hash(state),
            "reservation_request_hash": state["plan_hash"],
        }
        submission = await _host_call(
            runtime, job_id, attempt_id,
            "prepare_release",
            lambda: provider.prepare_release(
                state["context"], BudgetGrant(**state["grant"]) if state["grant"] is not None else None, proof
            ),
        )
        if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (
            attempt_id,
            1,
        ):
            raise JobError("invalid_budget_submission")
        async with manager._write() as tx:
            job = await manager._authorize(tx, actor, job_id)
            latest = await _read(manager, tx, attempt_id)
            if latest["submission"] is not None and latest["submission"] != asdict(submission):
                raise JobError("budget_submission_conflict")
            if latest["submission"] is None:
                latest["submission"] = asdict(submission)
                await _save(manager, tx, latest, job)
            state = latest
    submission = BudgetSubmission(**state["submission"])
    ack = await _host_call(runtime, job_id, attempt_id, "release", lambda: provider.release(submission))
    if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
        attempt_id,
        submission.revision,
        submission.request_hash,
    ):
        raise JobError("budget_ack_mismatch")
    async with manager._write() as tx:
        policy = await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        latest = await _read(manager, tx, attempt_id)
        if latest["phase"] == "released":
            return latest["ack"]
        if latest["phase"] != "dispatch_closed" or latest["submission"] != asdict(submission):
            raise JobError("budget_release_conflict")
        await _refund_local(manager, tx, policy, latest)
        latest.update(phase="released", pending=False, ack=asdict(ack))
        await _save(manager, tx, latest, job)
    return asdict(ack)


async def get_budget(manager, actor, attempt_id):
    """Authorized mechanical inspection, independent of model/provider availability."""
    async with manager._write() as tx:
        state = await _read(manager, tx, attempt_id)
        await manager._authorize(tx, actor, state["job_id"])
        return state


async def recovery_config(manager, actor, job_id, attempt_id, *, provider_key, provider, timeout_s=10):
    """Bind a trusted bookkeeping provider to the original persisted policy.

    This read does not enable the policy, register a boot or create an allowance.
    The supplied provider must independently authenticate its original authority.
    """
    from .budget_contracts import LocalBudgetScope
    async with manager._write() as tx:
        await manager._authorize(tx, actor, job_id)
        state = await _read(manager, tx, attempt_id)
        key = "bp_" + _hash([manager.store.store_id, actor.tenant_id, actor.principal_id])[:48]
        if state["job_id"] != job_id or state["policy_key"] != key:
            raise JobError("budget_reconciliation_scope_mismatch")
        row = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_policies WHERE policy_key=?", (key,))
        marker = await tx.fetchone(f"SELECT payload_json,request_hash FROM {manager.events} WHERE event_id=?",
                                  ("mb_policy_" + key[3:],))
        if row is None or marker is None:
            raise JobError("budget_policy_missing")
        manifest = _object(row["manifest_json"], row["request_hash"])
        if (_object(marker["payload_json"], marker["request_hash"]) != manifest
                or (row["tenant_id"], row["principal_id"]) != (actor.tenant_id, actor.principal_id)
                or manifest.get("provider_key") != provider_key):
            raise JobError("budget_policy_conflict")
    try:
        scopes = tuple(LocalBudgetScope(**{**s, **({"owner_kinds": tuple(s["owner_kinds"])}
            if s.get("owner_kinds") is not None else {})}) for s in manifest["scopes"])
        config = BudgetConfig(provider_key, scopes, provider, timeout_s=timeout_s,
                              max_pending=manifest["max_pending"])
        if config.manifest() != manifest:
            raise ValueError()
        return config
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise JobError("budget_policy_evidence_corrupt") from exc


async def _reconciliation_state(runtime, actor, job_id, attempt_id, *, allow_missing_observation=False):
    """Read and verify original model, policy and accounting provenance."""
    if (type(job_id) is not str or not 0 < len(job_id) <= 64
            or type(attempt_id) is not str or len(attempt_id) != 56
            or not attempt_id.startswith("attempt_")
            or any(c not in "0123456789abcdef" for c in attempt_id[8:])):
        raise JobError("invalid_budget_reconciliation_identity")
    manager = runtime.manager
    from .model_attempts import _read as read_model
    async with manager._write() as tx:
        job = await manager._authorize(tx, actor, job_id)
        policy = await _policy(runtime, tx)
        if policy is None:
            raise JobError("budget_provider_required")
        state = await _read(manager, tx, attempt_id)
        if state["job_id"] != job_id or state["policy_key"] != policy["policy_key"]:
            raise JobError("budget_reconciliation_scope_mismatch")
        _, prepared = await read_model(manager, tx, "prepared", attempt_id)
        _, dispatched = await read_model(manager, tx, "dispatched", attempt_id)
        finished = await read_model(manager, tx, "finished", attempt_id,
            required=not allow_missing_observation and state.get("initial_usage_evidence") is None)
        observation = finished[1] if finished is not None else None
        if state.get("initial_usage_evidence") is not None:
            from .budget_revisions import read_initial
            await read_initial(manager, tx, state)
        _, tracking = await read_model(manager, tx, "tracking", prepared["scope"])
        expected = {
            "version": 1, "store_id": manager.store.store_id,
            "tenant_id": job["tenant_id"], "principal_id": job["principal_id"],
            "root_job_id": job["root_job_id"], "job_id": job_id,
            "generation": prepared["scope"]["generation"], "session_id": job["session_id"],
            "attempt_id": attempt_id, "prepared_event_id":
                "ma_prepared_" + _hash([manager.store.store_id, attempt_id])[:48],
            "prepared_hash": _hash(prepared), "request_hash": prepared["request_hash"],
            "model": prepared["model"], "max_tokens": prepared["max_tokens"],
        }
        if (job["kind"] != "send" or prepared["scope"]["job_id"] != job_id
                or prepared["session_id"] != job["session_id"] or state["context"] != expected
                or tracking != {"version": 1, "scope": prepared["scope"], "session_id": job["session_id"]}
                or dispatched != {"version": 1, "scope": prepared["scope"],
                    "attempt_id": attempt_id, "prepared_hash": _hash(prepared)}
                or observation is not None and (observation["scope"] != prepared["scope"]
                    or observation["prepared_hash"] != _hash(prepared)
                    or observation["dispatch_hash"] != _hash(dispatched))):
            raise JobError("budget_reconciliation_evidence_mismatch")
        if state["phase"] not in ("dispatch_authorized", "settling"):
            raise JobError("budget_usage_not_ready")
        if (state["phase"] == "settling" and state["observation"] != observation
                and not (state.get("initial_usage_evidence") is not None
                         and state["observation"] is None and observation is not None)):
            raise JobError("budget_observation_source_mismatch")
    return state, observation


async def reconcile_usage(runtime, actor, job_id, attempt_id):
    if isinstance(attempt_id, str) and attempt_id.startswith('toolattempt_'):
        from .tool_attempts import reconcile
        return await reconcile(runtime, actor, job_id, attempt_id)
    state, observation = await _reconciliation_state(runtime, actor, job_id, attempt_id)
    # A crash can occur after model observation commit but before its budget
    # projection. The shared transaction applies counters at most once.
    if state["phase"] == "dispatch_authorized" or state["observation"] is None and observation is not None:
        await _observe(runtime, actor, job_id, attempt_id, observation)
    ack = await _deliver(runtime, actor, job_id, attempt_id)
    return {"attempt_id": attempt_id, "job_id": job_id, "revision": ack["revision"],
            "request_hash": ack["request_hash"], "usage_acknowledged": True,
            "execution_authority": False}


async def require_quiescent(manager, tx, job):
    rows = await tx.fetchall(
        f"SELECT attempt_id FROM {manager.prefix}task_budget_attempts WHERE job_id=? LIMIT 10001", (job["job_id"],)
    )
    if len(rows) > 10000:
        raise JobError("budget_attempt_limit")
    refs = []
    for row in rows:
        state = await _read(manager, tx, row["attempt_id"])
        if state["pending"] or state["phase"] not in ("released", "settling") or state["ack"] is None:
            raise JobError("budget_reconciliation_required")
        refs.append("mb_" + _hash([state["attempt_id"], state["version"]])[:48])
    return refs
