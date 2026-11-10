"""One reservation per ordinary send, keyed by execution id.

The hold is written before the model call and settled after it. A crash in
between leaves the row pending; recovery applies the stored estimate. Later
model calls in the same send only talk to the host, so they add no SQL.
"""
from __future__ import annotations

from .budget_scopes import COUNTERS
from .jobs import JobError, _hash, _json
from .model_budgets import _object, policy_key, scope_key

_CHUNK = 64


def execution_id(session_id, send_index):
    return f"send:{session_id}:{send_index}"


def install_local_policy(binding):
    """Let the in-memory check pass. The database row is still read by hold."""
    config = binding.runtime.budget_config
    manifest = config.manifest()
    binding.resource_evidence["_control_budget_policy"] = {
        "policy_key": policy_key(binding.runtime),
        "manifest": manifest,
        "enabled": 1,
        "request_hash": _hash(manifest),
    }


def _attempt_id(manager, execution):
    return "attempt_" + _hash([manager.store.store_id, execution])[:48]


def _room(limit, used, held):
    if limit is None:
        return None
    return max(0, limit - used - held)


async def hold(binding, token_bound):
    """Reserve shared capacity once. A second call in this send does not write."""
    existing = binding.resource_evidence.get("_lightweight_hold")
    if existing is not None:
        return existing["attempt_id"]
    if type(token_bound) is not int or token_bound < 0:
        raise JobError("invalid_budget_token_bound")
    runtime = binding.runtime
    manager = runtime.manager
    send_index = binding.resource_evidence.get("_runtime_send_index")
    if type(send_index) is not int:
        raise JobError("budget_execution_missing")
    execution = execution_id(binding.actor.session_id, send_index)
    key = policy_key(runtime)
    manifest = runtime.budget_config.manifest()
    from .root_context import for_send
    provenance = for_send(runtime, binding.actor.session_id)
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    async with manager._write() as tx:
        rows = await _scopes(tx, manager, key, suffix)
        if not rows:
            await _create_policy(tx, runtime, key, manifest)
            rows = await _scopes(tx, manager, key, suffix)
        if not rows or not rows[0]["enabled"] or rows[0]["request_hash"] != _hash(manifest):
            raise JobError("budget_policy_closed")
        scopes = []
        held_calls = held_tokens = None
        for row in rows:
            limits = _object(row["limits_json"], row["limits_hash"])
            counters = _object(row["counters_json"], row["counters_hash"])
            if set(counters) != set(COUNTERS):
                raise JobError("budget_counters_corrupt")
            calls = _room(limits.get("max_calls"), counters["used_calls"], counters["held_calls"])
            tokens = _room(limits.get("max_tokens"), counters["used_tokens"], counters["held_tokens"])
            slots = _room(limits.get("max_inflight"), 0, counters["inflight"])
            want_calls = _CHUNK if calls is None else min(_CHUNK, calls)
            want_tokens = token_bound * _CHUNK if tokens is None else min(token_bound * _CHUNK, tokens)
            want_slots = 1 if slots is None else min(1, slots)
            if held_calls is None:
                held_calls, held_tokens = want_calls, want_tokens
            held_calls = min(held_calls, want_calls)
            held_tokens = min(held_tokens, want_tokens)
            if want_slots < 1:
                held_calls = 0
            scopes.append((row, counters))
        if not scopes or held_calls < 1:
            raise JobError("budget_max_calls_exhausted")
        if token_bound and held_tokens < token_bound:
            raise JobError("budget_max_tokens_exhausted")
        known = []
        for row, counters in scopes:
            counters["held_calls"] += held_calls
            counters["held_tokens"] += held_tokens
            counters["inflight"] += 1
            await _write_counters(tx, manager, row, counters)
            known.append({"scope_key": row["scope_key"], "counters": counters, "counters_hash": _hash(counters)})
        state = {
            "execution_id": execution,
            "held_calls": held_calls,
            "held_tokens": held_tokens,
            "inflight": 1,
            "estimate_calls": 1,
            "estimate_tokens": token_bound,
            "scopes": known,
        }
        attempt_id = _attempt_id(manager, execution)
        now = manager.now()
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_attempts "
            "(attempt_id,job_id,execution_id,tenant_id,principal_id,session_id,policy_key,"
            "version,phase,pending,state_hash,state_json,root_context_json,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (attempt_id, None, execution, binding.actor.tenant_id, binding.actor.principal_id,
             binding.actor.session_id, key, 1, "reserved", 1, _hash(state), _json(state),
             None if provenance is None else _json(provenance), now, now),
        )
    binding.resource_evidence["_lightweight_hold"] = {**state, "attempt_id": attempt_id}
    binding.resource_evidence["_control_budget_scopes"] = []
    return attempt_id


async def record_usage(attempt, observation):
    """Remember the real usage and tell the host. The scope row waits for settle."""
    binding = attempt.binding
    seen = binding.resource_evidence.setdefault("_lightweight_billed", set())
    if attempt.attempt_id in seen:
        return None
    from .budget_contracts import BudgetAck, BudgetGrant, BudgetSubmission
    from .execution_budget import get, save
    from .model_budgets import _port, accounting_context

    state = get(attempt)
    usage = observation["usage"]
    tokens = None if observation["estimated"] else usage.get("total_tokens")
    if (tokens is None and not observation["estimated"] and usage.get("prompt_tokens") is not None
            and usage.get("completion_tokens") is not None):
        tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    stats = binding.resource_evidence.setdefault("_lightweight_usage", {"calls": 0, "tokens": 0})
    stats["calls"] += 1
    stats["tokens"] += 0 if tokens is None else tokens
    facts = {
        **observation,
        "started_at_ms": state["started_at_ms"],
        "finished_at_ms": binding.runtime.manager.now(),
        "evidence_event_id": "lightweight",
        "evidence_hash": _hash(observation),
    }
    submission = await _port(
        attempt, "prepare_usage",
        lambda: binding.runtime.budget_config.provider.prepare_usage(
            accounting_context(state), BudgetGrant(**state["grant"]), facts),
    )
    if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (
            attempt.attempt_id, 1):
        raise JobError("invalid_budget_submission")
    ack = await _port(
        attempt, "apply_usage:1",
        lambda: binding.runtime.budget_config.provider.apply_usage(submission),
    )
    if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
            attempt.attempt_id, submission.revision, submission.request_hash):
        raise JobError("budget_ack_mismatch")
    state.update(ack={"attempt_id": ack.attempt_id, "revision": ack.revision, "request_hash": ack.request_hash},
                 pending=False, phase="settling", observation=observation)
    save(binding, state)
    seen.add(attempt.attempt_id)
    return state["ack"]


async def settle(binding):
    hold_state = binding.resource_evidence.get("_lightweight_hold")
    if hold_state is None or hold_state.get("done"):
        return
    usage = binding.resource_evidence.get("_lightweight_usage") or {"calls": 0, "tokens": 0}
    await _finish(binding.runtime.manager, hold_state, usage["calls"], usage["tokens"])
    hold_state["done"] = True


async def recover(manager):
    """Settle every ordinary reservation that never reached its second transaction."""
    rows = await manager.db.fetchall(
        f"SELECT attempt_id,state_json,state_hash FROM {manager.prefix}task_budget_attempts "
        "WHERE pending<>0 AND execution_id IS NOT NULL")
    for row in rows:
        state = _object(row["state_json"], row["state_hash"])
        state["attempt_id"] = row["attempt_id"]
        await _finish(manager, state, state["estimate_calls"], state["estimate_tokens"])


async def _finish(manager, hold_state, used_calls, used_tokens):
    async with manager._write() as tx:
        current = dict(hold_state)
        current.update(used_calls=used_calls, used_tokens=used_tokens, phase="settled")
        changed = await tx.execute(
            f"UPDATE {manager.prefix}task_budget_attempts SET pending=0,phase='settled',version=version+1,"
            "state_json=?,state_hash=?,updated_at=? WHERE attempt_id=? AND pending<>0",
            (_json(current), _hash(current), manager.now(), hold_state["attempt_id"]),
        )
        if changed != 1:
            return
        delta = {
            "held_calls": -hold_state["held_calls"],
            "held_tokens": -hold_state["held_tokens"],
            "inflight": -hold_state["inflight"],
            "used_calls": used_calls,
            "used_tokens": used_tokens,
        }
        for scope in hold_state["scopes"]:
            await _add(tx, manager, scope, delta)


async def _scopes(tx, manager, key, suffix):
    return await tx.fetchall(
        f"SELECT policy.enabled AS enabled,policy.request_hash AS request_hash,scope.* "
        f"FROM {manager.prefix}task_budget_policies policy "
        f"JOIN {manager.prefix}task_budget_scopes scope ON scope.policy_key=policy.policy_key "
        "WHERE policy.policy_key=?" + suffix, (key,))


async def _create_policy(tx, runtime, key, manifest):
    manager = runtime.manager
    await tx.execute(
        f"INSERT INTO {manager.prefix}task_budget_policies "
        "(policy_key,tenant_id,principal_id,request_hash,manifest_json,enabled,created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (key, runtime.tenant_id, runtime.principal_id, _hash(manifest), _json(manifest), 1, manager.now()),
    )
    counters = dict.fromkeys(COUNTERS, 0)
    for limits in manifest["scopes"]:
        if limits.get("binding", "authority") != "authority":
            continue
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_scopes "
            "(scope_key,policy_key,limits_hash,limits_json,counters_hash,counters_json) VALUES (?,?,?,?,?,?)",
            (scope_key(key, limits["key"]), key, _hash(limits), _json(limits), _hash(counters), _json(counters)),
        )


async def _write_counters(tx, manager, row, counters):
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise JobError("budget_counter_underflow")
    await tx.execute(
        f"UPDATE {manager.prefix}task_budget_scopes SET counters_json=?,counters_hash=? WHERE scope_key=?",
        (_json(counters), _hash(counters), row["scope_key"]),
    )


async def _add(tx, manager, scope, delta):
    counters = {name: scope["counters"][name] + delta.get(name, 0) for name in COUNTERS}
    if any(value < 0 for value in counters.values()):
        raise JobError("budget_counter_underflow")
    changed = await tx.execute(
        f"UPDATE {manager.prefix}task_budget_scopes SET counters_json=?,counters_hash=? "
        "WHERE scope_key=? AND counters_hash=?",
        (_json(counters), _hash(counters), scope["scope_key"], scope["counters_hash"]),
    )
    if changed == 1:
        return
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    row = await tx.fetchone(
        f"SELECT * FROM {manager.prefix}task_budget_scopes WHERE scope_key=?" + suffix, (scope["scope_key"],))
    if row is None:
        raise JobError("budget_scope_missing_or_changed")
    current = _object(row["counters_json"], row["counters_hash"])
    current = {name: current[name] + delta.get(name, 0) for name in COUNTERS}
    await _write_counters(tx, manager, row, current)
