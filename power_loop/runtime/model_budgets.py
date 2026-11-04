"""Durable local admission and host-budget handoff for original model attempts.

The store control transaction serializes local scopes. No transaction spans a
host request. API grants are evidence for one original attempt, never replay
permission. All model arguments and actual prompt bodies stay outside this log.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from . import budget_scopes
from .budget_contracts import BudgetAck, BudgetConfig, BudgetGrant, BudgetSubmission
from .budget_policy_sources import PolicySourceChanged
from .budget_policy_sources import current as prepared_source
from .budget_policy_sources import prepared as prepare_policy
from .jobs import JobError, _hash, _json
from .jobs import verify_stored as _verify_stored

_COUNTERS = budget_scopes.COUNTERS


def accounting_context(state):
    """Keep pricing inputs with the live owner until its submission is frozen."""
    from .budget_contracts import payload
    context = payload(state['context'])
    if state['plan'] is not None and 'request' in state['plan']:
        context['reservation_request'] = payload(state['plan']['request'])
    return context


def stored_state(state):
    """Persist accounting and the frozen outbox, without pricing or phase proofs."""
    value = {key: state[key] for key in (
        'allocation_id', 'plan_hash', 'host_intent', 'observation', 'observation_hash',
        'usage_revision', 'initial_usage_evidence') if key in state}
    if state['submission'] is not None and state['observation'] is not None:
        value['observation'] = {key: state['observation'][key] for key in ('status', 'usage', 'estimated')}
    value['generation'] = state['context']['generation']
    value['kind'] = state['context'].get('kind', 'model')
    value['token_bound'] = state['plan']['token_bound'] if state['plan'] is not None else None
    grant = state['grant']
    value['grant'] = {key: grant[key] for key in ('version', 'state')} if grant else None
    submission = state['submission']
    value['submission'] = {key: submission[key] for key in ('revision', 'request')} if submission else None
    ack = state['ack']
    value['ack'] = ({key: ack[key] for key in ('revision', 'request_hash')}
                    if ack and 'request_hash' in ack else ack)
    return value


def policy_key(runtime):
    from .budget_admissions import key
    return key(runtime.manager, runtime.tenant_id, runtime.principal_id)


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
    admissions = getattr(tx, '_budget_policy_admissions', {})
    row = admissions[key] if key in admissions else await tx.fetchone(
        f"SELECT * FROM {manager.prefix}task_budget_policies WHERE policy_key=?", (key,))
    marker_id = "mb_policy_" + key[3:]
    from .execution_membership import _remember_after_commit, _trusted
    known = row is not None and _trusted(manager, "_budget_markers", key)
    if known and known == row["request_hash"]:
        marker = {"payload_json": row["manifest_json"], "request_hash": row["request_hash"]}  # Immutable, verified.
    else:
        source = prepared_source(runtime)
        if source is None:
            if manager._control_intents.get() is not None:
                raise JobError('budget_policy_source_required_before_writer')
            marker = await tx.fetchone(f"SELECT payload_json,request_hash FROM {manager.events} WHERE event_id=?", (marker_id,))
        else:
            marker = source.marker
            if row is not None and marker is None:
                raise PolicySourceChanged(row['request_hash'])
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
        await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'request_hash', 'payload_json', 'delivery_status', 'created_at'), (
                marker_id,
                job["job_id"],
                "model_budget_policy",
                job["version"],
                _hash(manifest),
                _json(manifest),
                "not_applicable",
                manager.now(),
            ))
        marker = {"payload_json": _json(manifest), "request_hash": _hash(manifest)}
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_policies "
            "(policy_key,tenant_id,principal_id,request_hash,manifest_json,enabled,created_at) VALUES (?,?,?,?,?,?,?)",
            (key, runtime.tenant_id, runtime.principal_id, _hash(manifest), _json(manifest), 1, manager.now()),
        )
        from .budget_admissions import activated
        await activated(runtime, tx, _hash(manifest))
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
        admissions[key] = row
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
    _remember_after_commit(manager, None, "_budget_markers", key, row["request_hash"])
    return {**row, "manifest": manifest}


async def flush_observed_controls(binding):
    runtime = binding.runtime
    if runtime.manager._joined_write.get() is not None:
        return  # The joined admission flushed before taking the store lock.
    from .control_runtime import current, flush_binding
    for active in tuple(runtime.manager._execution_controls.values()):
        control = current(active)
        if (active.runtime is runtime and not control.active and not control.pending_obligations
                and (control._dirty or control._pending is not None)):
            await flush_binding(active)


async def ensure_policy(binding, *, transaction=None, job=None):
    runtime = binding.runtime
    if transaction is not None:
        return await _ensure_policy(binding, transaction, job)
    await flush_observed_controls(binding)
    source = None
    while True:
        try:
            async with prepare_policy(runtime, source=source), runtime.manager._write() as tx:
                from .budget_admissions import enter
                await enter(runtime, tx)
                job = await runtime.manager._fence(tx, binding.actor, binding.lease)
                return await _ensure_policy(binding, tx, job)
        except PolicySourceChanged as changed:
            source = changed.source


async def _ensure_policy(binding, tx, job):
    runtime = binding.runtime
    policy = await _policy(runtime, tx, create=True, job=job)
    if policy:
        scopes = await budget_scopes.resolve(runtime, tx, policy, job, create=True)
        for limits in scopes:
            await _scope(runtime.manager, tx, policy["policy_key"], limits)
        await _check_pending(runtime.manager, tx, policy)
        binding.resource_evidence["_control_budget_policy"] = policy
        binding.resource_evidence["_control_budget_scopes"] = scopes
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
    row = await tx.fetchone(
        f"SELECT b.*,j.tenant_id,j.principal_id,j.session_id,j.root_job_id "
        f"FROM {manager.prefix}task_budget_attempts b JOIN {manager.jobs} j ON j.job_id=b.job_id "
        "WHERE b.attempt_id=?", (attempt_id,))
    if row is None:
        if required:
            raise JobError("budget_attempt_missing")
        return None
    return _budget_row(manager, row)


def _budget_row(manager, row):
    value = _object(row["state_json"], row["state_hash"])
    value.update({key: row[key] for key in ('attempt_id', 'job_id', 'policy_key', 'version', 'phase')})
    value['pending'] = bool(row['pending'])
    value['started_at_ms'], value['finished_at_ms'] = row['created_at'], row['updated_at']
    value['context'] = {key: row[key] for key in ('tenant_id', 'principal_id', 'session_id', 'root_job_id', 'job_id', 'attempt_id')}
    value['context'].update(store_id=manager.store.store_id, generation=value.pop('generation'))
    if value.pop('kind') == 'tool':
        value['context']['kind'] = 'tool'
    bound = value.pop('token_bound')
    value['plan'] = {'token_bound': bound} if bound is not None else None
    if value['grant'] is not None:
        value['grant'].update(attempt_id=value['attempt_id'], request_hash=value['plan_hash'], receipt={})
    if value['submission'] is not None:
        value['submission']['attempt_id'] = value['attempt_id']
    if value['ack'] is not None and 'request_hash' in value['ack']:
        value['ack'].update(attempt_id=value['attempt_id'], receipt={})
    return value


async def _save(manager, tx, state, job, *, new=False):
    from .control_runtime import append_correction
    state["version"] += 1
    await append_correction(manager, tx, job, state["context"]["generation"],
                            [{"kind": "budget", "key": state["attempt_id"], "value": state}])


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
    # Admission only needs the pending count. Reconciliation verifies each
    # attempt's original chain in its own tree;do not borrow those trees here.
    # Every nonzero flag still occupies capacity.
    row = await tx.fetchone(
        f"SELECT COUNT(*) AS n FROM (SELECT 1 FROM {manager.prefix}task_budget_attempts "
        "WHERE policy_key=? AND pending<>0 LIMIT ?) AS pending_attempts",
        (policy["policy_key"], policy["manifest"]["max_pending"]),
    )
    if row['n'] >= policy["manifest"]["max_pending"]:
        raise JobError("budget_reconciliation_required")


async def reserve(attempt, logical_request, *, service=None, pricing=None):
    from .execution_budget import reserve as reserve_owned
    return await reserve_owned(attempt, logical_request, service=service, pricing=pricing)


async def observe(attempt, observation):
    from .control_runtime import current
    from .execution_budget import observe as observe_owned
    if current(attempt.binding).closed:
        return await _observe(attempt.binding.runtime, attempt.binding.actor,
                              attempt.binding.lease.job_id, attempt.attempt_id, observation)
    return await observe_owned(attempt, observation)


async def _observe(runtime, actor, job_id, attempt_id, observation):
    manager = runtime.manager
    async with prepare_policy(runtime), manager._write() as tx:
        policy = await _policy(runtime, tx)
        if policy is None:
            return
        job = await manager._authorize(tx, actor, job_id)
        state = await _read(manager, tx, attempt_id)
        if state.get("initial_usage_evidence") is not None:
            from .budget_revisions import read_initial
            await read_initial(manager, tx, state)
            if state["observation"] is not None:
                if state["observation_hash"] == _hash(observation):
                    return
                raise JobError("budget_observation_conflict")
            # Financial evidence already consumed the call/token reservation.
            # Only an authentic original return releases execution concurrency;
            # it never overwrites or reapplies the independent financial facts.
            if observation["status"] == "returned":
                from .execution_budget import adjust_allocation
                await adjust_allocation(manager, tx, state, {"inflight": -1})
            state.update(observation=observation, observation_hash=_hash(observation), finished_at_ms=manager.now())
            await _save(manager, tx, state, job)
        else:
            if not await _project_observation(manager, tx, policy, state, job, observation, manager.now()):
                return
    await _deliver(runtime, actor, job_id, attempt_id)


async def _project_observation(manager, tx, policy, state, job, observation, finished_at_ms):
    """First ordinary observation, when no independent financial evidence exists."""
    if state["phase"] != "dispatch_authorized":
        if state["observation_hash"] == _hash(observation):
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
    from .execution_budget import adjust_allocation
    changes = {"held_calls": -1, "used_calls": 1}
    if tokens is not None:
        changes.update(held_tokens=-bound, used_tokens=tokens)
    if returned:
        changes["inflight"] = -1
    await adjust_allocation(manager, tx, state, changes)
    if tokens is not None and tokens > bound:
        await tx.execute(
            f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?", (state["policy_key"],)
        )
    state.update(phase="settling", pending=True, observation=observation, observation_hash=_hash(observation),
                 finished_at_ms=finished_at_ms)
    await _save(manager, tx, state, job)
    return True


async def deliver(attempt):
    """Replay the original live owner's frozen usage submission only."""
    from .execution_budget import get, observe
    state = get(attempt)
    if state["observation"] is None:
        raise JobError("budget_usage_not_ready")
    return await observe(attempt, state["observation"])


async def _deliver(runtime, actor, job_id, attempt_id):
    manager = runtime.manager
    async with prepare_policy(runtime), manager._write() as tx:
        await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        state = await _read(manager, tx, attempt_id)
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
        elif any(latest['ack'][key] != getattr(ack, key) for key in ('attempt_id', 'revision', 'request_hash')):
            raise JobError("budget_ack_conflict")
    return asdict(ack)


async def _refund_local(manager, tx, policy, state):
    from .execution_budget import adjust_allocation
    changes = {"held_calls": -1, "held_tokens": -state["plan"]["token_bound"], "inflight": -1}
    observation = state['observation']
    if observation is not None:
        changes.update(held_calls=0, used_calls=-1)
        usage = observation['usage']
        tokens = None if observation['estimated'] else usage.get('total_tokens')
        if tokens is None and not observation['estimated'] and all(k in usage for k in ('prompt_tokens', 'completion_tokens')):
            tokens = usage['prompt_tokens'] + usage['completion_tokens']
        if tokens is not None:
            changes.update(held_tokens=0, used_tokens=-tokens)
        if observation['status'] == 'returned':
            changes['inflight'] = 0
    await adjust_allocation(manager, tx, state, changes)


async def close_unreserved(attempt):
    from .execution_budget import close_unreserved as close_owned
    return await close_owned(attempt)


async def close_unstarted(attempt):
    from .execution_budget import release_unstarted as release_owned
    return await release_owned(attempt)


async def _release_source(runtime, tx, job, attempt_id, policy):
    manager = runtime.manager
    if policy is None:
        raise JobError("budget_provider_required")
    state = await _read(manager, tx, attempt_id)
    if state["job_id"] != job["job_id"] or state["policy_key"] != policy["policy_key"]:
        raise JobError("budget_reconciliation_scope_mismatch")
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


async def _release_unstarted(runtime, actor, job_id, attempt_id, *, require_cancelled=False):
    manager = runtime.manager
    async with prepare_policy(runtime), manager._write() as tx:
        policy = await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        state = await _release_source(runtime, tx, job, attempt_id, policy)
        if state["phase"] == "released":
            return state["ack"]
        if state["phase"] != "dispatch_closed":
            # An open checkpoint has an unknown suffix. Missing dispatch
            # evidence cannot prove an external request was never made.
            raise JobError("budget_unstarted_execution_unconfirmed")
        from .control_runtime import require_closed
        await require_closed(manager, tx, job_id, state["context"]["generation"])
        if not state["host_intent"]:
            await _refund_local(manager, tx, policy, state)
            state.update(phase="released", pending=False, ack={"local_only": True, "dispatch_not_authorized": True})
            await _save(manager, tx, state, job)
            return state["ack"]
    provider = runtime.budget_config.provider
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
    async with prepare_policy(runtime), manager._write() as tx:
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
    return {**asdict(ack), "receipt": {}}


async def get_budget(manager, actor, attempt_id):
    """Authorized mechanical inspection, independent of model/provider availability."""
    binding = manager._live_budget_bindings.get(attempt_id)
    if binding is not None:
        # Explicit inspection is a consistency boundary, never hot admission.
        await manager._authorize(manager.db, actor, binding.lease.job_id)
        from .control_runtime import current, flush_binding
        control = current(binding)
        head = await control.journal.inspect(control.identity)
        if head["batch_no"] > control.cursor and control._pending is None:
            # Another trusted reconciler appended to the original journal.
            # The old owner cannot overwrite that fact or gain new admission.
            control.needs_reconciliation = True
        else:
            await flush_binding(binding)
            from .budget_contracts import payload
            live = control.get('budget', attempt_id)
            if live is not None:
                return payload(live)
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
    """Use the stored accounting state for settlement and resubmission."""
    manager = runtime.manager
    async with prepare_policy(runtime), manager._write() as tx:
        await manager._authorize(tx, actor, job_id)
        policy = await _policy(runtime, tx)
        if policy is None:
            raise JobError("budget_provider_required")
        state = await _read(manager, tx, attempt_id)
        if state["job_id"] != job_id or state["policy_key"] != policy["policy_key"]:
            raise JobError("budget_reconciliation_scope_mismatch")
        if state["phase"] not in ("dispatch_authorized", "settling"):
            raise JobError("budget_usage_not_ready")
        observation = state["observation"]
        if observation is None and not allow_missing_observation and state.get("initial_usage_evidence") is None:
            raise JobError("budget_usage_not_ready")
        if state.get("initial_usage_evidence") is not None:
            from .budget_revisions import read_initial
            await read_initial(manager, tx, state)
        from .control_runtime import require_closed
        await require_closed(manager, tx, job_id, state["context"]["generation"])
    return state, observation


async def _prepare_unfrozen_release(runtime, actor, job_id, attempt_id):
    """An expired send loses its unknown usage; retained reservations can close."""
    from .execution_budget import project_allocation
    from .execution_journal import ExecutionJournal
    from .execution_state import ExecutionIdentity
    from .jobs import JobLease

    manager = runtime.manager
    async with prepare_policy(runtime), manager.joined_write() as tx:
        policy = await _policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        state = await _release_source(runtime, tx, job, attempt_id, policy)
        if state['submission'] is not None or state['phase'] == 'released':
            return False
        generation = state['context']['generation']
        if job['generation'] != generation:
            raise JobError('stale_execution')
        head = await tx.fetchone(f'SELECT * FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?',
                                (job_id, generation))
        summary = json.loads(head['summary_json'])
        if not summary['closed']:
            # A live lease is never interpreted as a crashed owner. Fence the
            # old job and checkpoint in this transaction before any API release.
            changed = await tx.write_job(job_id, f"UPDATE {manager.jobs} SET intake='closed',version=version+1 "
                "WHERE job_id=? AND generation=? AND version=? AND lease_until_ms<=?",
                (job_id, generation, job['version'], manager.now()))
            if changed != 1:
                raise JobError('execution_reconciliation_requires_quiescence')
            identity = ExecutionIdentity(manager.store.store_id, job_id, generation,
                job['executor_id'], job['boot_id'], job['tenant_id'], job['principal_id'], job['session_id'])
            summary.update(closed=True, unknown=summary['unknown'] + summary['inflight'], inflight=0)
            rows = await tx.fetchall(f'SELECT state_json,state_hash FROM {manager.prefix}task_budget_attempts WHERE job_id=?',
                                     (job_id,))
            attempts = [_object(row['state_json'], row['state_hash']) for row in rows]
            for allocation in summary['allocations']:
                row = await tx.fetchone(f'SELECT state_json,state_hash FROM {manager.prefix}execution_allocations WHERE allocation_id=?',
                                        (allocation['allocation_id'],))
                saved = _object(row['state_json'], row['state_hash'])
                await project_allocation(manager, tx, identity, allocation['allocation_id'],
                    {**allocation, 'closed': True, 'observed_calls': sum(
                        a.get('allocation_id') == allocation['allocation_id'] and a['observation'] is not None
                        for a in attempts),
                     'report_version': saved['report_version'] + 1})
            await ExecutionJournal(manager.store).append(
                identity, head['batch_no'] + 1, [], summary, transaction=tx)
            await manager.close_model_execution(actor, JobLease(job_id, job['executor_id'], job['boot_id'], generation),
                evidence={'exit': 'interrupted'})
        state.update(phase='dispatch_closed', pending=True)
        await _save(manager, tx, state, job)
    return True


@_verify_stored
async def reconcile_usage(runtime, actor, job_id, attempt_id):
    if await _prepare_unfrozen_release(runtime, actor, job_id, attempt_id):
        ack = await _release_unstarted(runtime, actor, job_id, attempt_id)
        return {'attempt_id': attempt_id, 'job_id': job_id, 'released': True,
                'execution_authority': False, 'ack': ack}
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
    from .execution_membership import _trusted
    summary = _trusted(manager, "_closed_summaries", (job["job_id"], job["generation"]))
    if (summary is not None and job["generation"] == 1 and _trusted(manager, "_first_claims", job["job_id"])
            and summary.get("budget_attempts") == 0 and not summary.get("budget_unsettled")):
        return []  # This process wrote every budget attempt of this execution: none.
    rows = await tx.fetchall(
        f"SELECT pending,phase FROM {manager.prefix}task_budget_attempts WHERE job_id=? LIMIT 10001", (job["job_id"],))
    if len(rows) > 10000:
        raise JobError("budget_attempt_limit")
    head = await tx.fetchone(f"SELECT summary_json FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?",
                             (job["job_id"], job["generation"]))
    if head is not None:
        summary = json.loads(head["summary_json"])
        if summary["budget_attempts"] != len(rows):
            raise JobError("budget_attempt_missing")
        if summary["budget_unsettled"]:
            raise JobError("budget_reconciliation_required")
    for row in rows:
        if row["pending"] or row["phase"] not in ("released", "settling"):
            raise JobError("budget_reconciliation_required")
    return []
