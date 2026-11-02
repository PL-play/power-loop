"""Bounded shared admission with one live owner's in-memory consumption.

Allocation rows reserve capacity, not money. The host still owns financial
reservations and actual charges. A restart never recycles an open allocation:
its uncheckpointed suffix may already have reached an external service.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict

from . import budget_scopes
from .budget_contracts import BudgetAck, BudgetGrant, BudgetPlan, BudgetSubmission
from .control_runtime import current, source
from .execution_state import LocalAllocation
from .jobs import JobError, _hash, _json

_CHUNK = 64


def check(binding):
    runtime = binding.runtime
    if runtime.budget_config is not runtime._original_budget_config:
        raise JobError("budget_configuration_changed")
    if runtime._original_budget_config is None:
        return None
    runtime._lifecycle._check_manager_binding()
    if runtime.manager is not runtime._lifecycle.manager or any(
        getattr(runtime, key) != getattr(runtime._lifecycle.identity, key)
        for key in ("tenant_id", "principal_id", "executor_id", "boot_id")
    ):
        raise JobError("budget_runtime_binding_changed")
    config = runtime._original_budget_config
    policy = binding.resource_evidence.get("_control_budget_policy")
    if policy is None or config.manifest() != policy["manifest"]:
        raise JobError("budget_policy_conflict")
    return policy


def get(attempt):
    check(attempt.binding)
    state = current(attempt.binding).get("budget", attempt.attempt_id)
    if state is None:
        raise JobError("budget_attempt_missing")
    from .model_budgets import _context
    if state["context"] != _context(attempt, source(attempt.binding)):
        raise JobError("budget_attempt_original_mismatch")
    return state


def save(binding, state):
    state["version"] += 1
    current(binding).record("budget", state["attempt_id"], state)
    binding.runtime.manager._live_budget_bindings[state["attempt_id"]] = binding


def allocation_record(binding):
    control = current(binding)
    allocation = binding.resource_evidence.get("_control_allocation")
    if allocation is None:
        return
    allocation.report_version += 1
    control.record("budget_allocation", allocation.allocation_id,
                   {**allocation.summary(), "closed": control.closed or control.cancel_requested,
                    "observed_calls": len(allocation.observed), "report_version": allocation.report_version})


async def allocate(binding, token_bound, attempt_id):
    """Refill only at the edge of a bounded grant; never query per consumption."""
    from . import model_budgets as budgets
    manager = binding.runtime.manager
    control = current(binding)
    lock = binding.resource_evidence.setdefault("_allocation_lock", asyncio.Lock())
    async with lock:
        control.require_admission()
        policy = check(binding)
        allocation = binding.resource_evidence.get("_control_allocation")
        enough = allocation is not None and (
            allocation.used_calls < allocation.calls
            and allocation.used_tokens + allocation.held_tokens + token_bound <= allocation.tokens
            and allocation.inflight < allocation.max_inflight)
        if not enough:
            # Do not change a grant beneath an unacknowledged checkpoint.
            await control.checkpoint(force=True)
            scopes = binding.resource_evidence["_control_budget_scopes"]
            key = "allocation_" + _hash(asdict(control.identity))[:48]
            table = manager.prefix + "execution_allocations"
            try:
                # Scope rows, in deterministic order, serialize competing
                # owners. Unrelated scopes never share the task-store lock.
                async with manager._write() as tx:
                    await tx.require_job(binding.lease.job_id)
                    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
                    p = await tx.fetchone(f"SELECT enabled,request_hash FROM {manager.prefix}task_budget_policies "
                                          "WHERE policy_key=?" + suffix, (policy["policy_key"],))
                    if p is not None and (type(p["enabled"]) is not int or p["enabled"] not in (0, 1)):
                        raise JobError("budget_policy_evidence_corrupt")
                    if p is None or not p["enabled"] or p["request_hash"] != policy["request_hash"]:
                        raise JobError("budget_policy_closed")
                    await manager._fence(tx, binding.actor, binding.lease)
                    old = await tx.fetchone(f"SELECT * FROM {table} WHERE allocation_id=?" + suffix, (key,))
                    if (old is None) != (allocation is None):
                        raise JobError("execution_allocation_requires_reconciliation")
                    state = budgets._object(old["state_json"], old["state_hash"]) if old else {
                        "identity": asdict(control.identity), "policy_key": policy["policy_key"],
                        "scopes": scopes, "calls": 0, "tokens": 0, "max_inflight": 0,
                        "used_calls": 0, "used_tokens": 0, "closed": False,
                        "report_version": 0, "booked": dict.fromkeys(budget_scopes.COUNTERS, 0),
                    }
                    if (state["identity"] != asdict(control.identity) or state["scopes"] != scopes
                            or state["closed"]):
                        raise JobError("execution_allocation_changed")
                    rows = []
                    extra_calls = _CHUNK if allocation is None or allocation.used_calls >= allocation.calls else 0
                    extra_tokens = max(token_bound, token_bound * _CHUNK - (
                        0 if allocation is None else allocation.tokens - allocation.used_tokens - allocation.held_tokens))
                    if allocation is not None and allocation.tokens - allocation.used_tokens - allocation.held_tokens >= token_bound:
                        extra_tokens = 0
                    extra_slots = 1 if allocation is None or allocation.inflight >= allocation.max_inflight else 0
                    for limits in scopes:
                        scope_id = budget_scopes.scope_key(policy["policy_key"], limits)
                        row = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_scopes "
                                                "WHERE scope_key=?" + suffix, (scope_id,))
                        if row is None or budgets._object(row["limits_json"], row["limits_hash"]) != limits:
                            raise JobError("budget_scope_missing_or_changed")
                        counters = budgets._object(row["counters_json"], row["counters_hash"])
                        if set(counters) != set(budget_scopes.COUNTERS) or any(type(v) is not int or v < 0 for v in counters.values()):
                            raise JobError("budget_counters_corrupt")
                        if limits["max_calls"] is not None:
                            available = max(0, limits["max_calls"] - counters["used_calls"] - counters["held_calls"])
                            extra_calls = min(extra_calls, max(1, available // 2) if available else 0)
                        if limits["max_tokens"] is not None:
                            available = max(0, limits["max_tokens"] - counters["used_tokens"] - counters["held_tokens"])
                            extra_tokens = min(extra_tokens, min(available, max(token_bound, available // 2)))
                        if limits["max_inflight"] is not None:
                            extra_slots = min(extra_slots, max(0, limits["max_inflight"] - counters["inflight"]))
                        rows.append((row, counters))
                    available_calls = state["calls"] - (allocation.used_calls if allocation else 0) + extra_calls
                    available_tokens = state["tokens"] - (allocation.used_tokens + allocation.held_tokens if allocation else 0) + extra_tokens
                    available_slots = state["max_inflight"] - (allocation.inflight if allocation else 0) + extra_slots
                    for field, available, required in (("calls", available_calls, 1), ("tokens", available_tokens, token_bound),
                                                       ("inflight", available_slots, 1)):
                        if available < required:
                            raise JobError("budget_max_" + field + "_exhausted")
                    for row, counters in rows:
                        counters["held_calls"] += extra_calls
                        counters["held_tokens"] += extra_tokens
                        counters["inflight"] += extra_slots
                        await budgets._save_counters(manager, tx, row, counters)
                    state["calls"] += extra_calls
                    state["tokens"] += extra_tokens
                    state["max_inflight"] += extra_slots
                    state["booked"]["held_calls"] += extra_calls
                    state["booked"]["held_tokens"] += extra_tokens
                    state["booked"]["inflight"] += extra_slots
                    if old is None:
                        await tx.execute(f"INSERT INTO {table} (allocation_id,job_id,generation,state_json,state_hash) VALUES (?,?,?,?,?)",
                                         (key, binding.lease.job_id, binding.lease.generation, _json(state), _hash(state)))
                    else:
                        await tx.execute(f"UPDATE {table} SET state_json=?,state_hash=? WHERE allocation_id=?",
                                         (_json(state), _hash(state), key))
            except BaseException as exc:
                # A transport failure may have committed a grant. Never mint a
                # replacement after an uncertain ACK. Ordinary rejected limits
                # have no grant and can still be closed without consuming it.
                if not isinstance(exc, JobError):
                    control.needs_reconciliation = True
                raise
            if allocation is None:
                allocation = LocalAllocation(key, state["calls"], state["tokens"], state["max_inflight"])
                binding.resource_evidence["_control_allocation"] = allocation
                control.allocations[key] = allocation
            else:
                allocation.calls, allocation.tokens, allocation.max_inflight = state["calls"], state["tokens"], state["max_inflight"]
        allocation.reserve(attempt_id, token_bound)
        allocation_record(binding)
        return allocation.allocation_id


async def _locked_allocation(manager, tx, identity, allocation_id):
    from .budget_admissions import key

    suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR UPDATE'
    # Lock policy before any shared scope. Allocation rows belong to this
    # already locked root,so another root cannot hold the same allocation.
    return await tx.fetchone(f'SELECT allocation.* FROM {manager.prefix}execution_allocations allocation '
        f'JOIN {manager.prefix}task_budget_policies policy ON policy.policy_key=? '
        'WHERE allocation.allocation_id=?'+suffix,
        (key(manager, identity.tenant_id, identity.principal_id), allocation_id))


async def project_allocation(manager, tx, identity, key, value):
    """Apply a cumulative checkpoint once, under the original scope locks."""
    from . import model_budgets as budgets
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    table = manager.prefix + "execution_allocations"
    row = await _locked_allocation(manager, tx, identity, key)
    if row is None:
        raise JobError("execution_allocation_missing")
    state = budgets._object(row["state_json"], row["state_hash"])
    if state["identity"] != asdict(identity):
        raise JobError("execution_allocation_changed")
    # An old repair must not roll back a later checkpoint or refund twice.
    if value["report_version"] <= state["report_version"]:
        return
    closed = state["closed"] or value["closed"]
    booked = dict(used_calls=value["observed_calls"], used_tokens=value["used_tokens"],
        held_calls=max(0, (value["used_calls"] if closed else state["calls"]) - value["observed_calls"]),
        held_tokens=value["held_tokens"] if closed else max(0, state["tokens"] - value["used_tokens"]),
        inflight=value["inflight"] if closed else state["max_inflight"])
    differences = {k: booked[k] - state["booked"][k] for k in budget_scopes.COUNTERS}
    for limits in state["scopes"]:
        scope_id = budget_scopes.scope_key(state["policy_key"], limits)
        scope = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_scopes WHERE scope_key=?" + suffix, (scope_id,))
        if scope is None:
            raise JobError("budget_scope_missing_or_changed")
        counters = budgets._object(scope["counters_json"], scope["counters_hash"])
        counters = {k: counters[k] + differences[k] for k in budget_scopes.COUNTERS}
        await budgets._save_counters(manager, tx, scope, counters)
    if value["used_tokens"] > state["tokens"]:
        await tx.execute(f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?", (state["policy_key"],))
    state.update(used_calls=value["observed_calls"], used_tokens=value["used_tokens"], closed=closed,
                 booked=booked, report_version=value["report_version"])
    await tx.execute(f"UPDATE {table} SET state_json=?,state_hash=? WHERE allocation_id=?", (_json(state), _hash(state), key))


async def adjust_allocation(manager, tx, state, changes):
    """A cold, proven correction changes the original capacity accounting."""
    from . import model_budgets as budgets
    from .control_runtime import append_correction, require_closed
    await require_closed(manager, tx, state["job_id"], state["context"]["generation"])
    key = state.get("allocation_id")
    if key is None:
        # A no-reservation closure owns no allocation and cannot refund one.
        if state.get("plan") is None:
            raise JobError("execution_allocation_missing")
        raise JobError("execution_allocation_migration_required")
    table = manager.prefix + "execution_allocations"
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    row = await tx.fetchone(f"SELECT * FROM {table} WHERE allocation_id=?" + suffix, (key,))
    if row is None:
        raise JobError("execution_allocation_missing")
    allocation = budgets._object(row["state_json"], row["state_hash"])
    if (allocation["identity"]["job_id"], allocation["identity"]["generation"], allocation["policy_key"]) != (
            state["job_id"], state["context"]["generation"], state["policy_key"]):
        raise JobError("execution_allocation_changed")
    allocation["booked"] = {k: allocation["booked"][k] + changes.get(k, 0) for k in budget_scopes.COUNTERS}
    if any(v < 0 for v in allocation["booked"].values()):
        raise JobError("budget_counter_underflow")
    allocation["report_version"] += 1
    allocation["used_calls"] = allocation["booked"]["used_calls"]
    allocation["used_tokens"] = allocation["booked"]["used_tokens"]
    job = await manager._row(tx, state["job_id"])
    await append_correction(manager, tx, job, state["context"]["generation"], [
        {"kind": "allocation_adjustment_immutable", "key": key + ":" + str(allocation["report_version"]),
         "value": {"allocation_id": key, "state": allocation}}])


async def project_adjustment(manager, tx, identity, value):
    from . import model_budgets as budgets
    table = manager.prefix + "execution_allocations"
    key, allocation = value["allocation_id"], value["state"]
    if allocation["identity"] != asdict(identity):
        raise JobError("execution_allocation_changed")
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    row = await _locked_allocation(manager, tx, identity, key)
    if row is None:
        raise JobError("execution_allocation_missing")
    previous = budgets._object(row["state_json"], row["state_hash"])
    if allocation["report_version"] <= previous["report_version"]:
        return
    if allocation["scopes"] != previous["scopes"] or allocation["identity"] != previous["identity"]:
        raise JobError("execution_allocation_changed")
    changes = {k: allocation["booked"][k] - previous["booked"][k] for k in budget_scopes.COUNTERS}
    for limits in allocation["scopes"]:
        scope_id = budget_scopes.scope_key(allocation["policy_key"], limits)
        scope = await tx.fetchone(f"SELECT * FROM {manager.prefix}task_budget_scopes WHERE scope_key=?" + suffix, (scope_id,))
        if scope is None:
            raise JobError("budget_scope_missing_or_changed")
        counters = budgets._object(scope["counters_json"], scope["counters_hash"])
        await budgets._save_counters(manager, tx, scope, {k: counters[k] + changes[k] for k in budget_scopes.COUNTERS})
    await tx.execute(f"UPDATE {table} SET state_json=?,state_hash=? WHERE allocation_id=?", (_json(allocation), _hash(allocation), key))


async def reserve(attempt, logical_request, *, service=None, pricing=None):
    from . import model_budgets as budgets
    from .tool_attempts import ToolAttempt
    binding = attempt.binding
    control = current(binding)
    control.require_admission()
    policy = check(binding)
    if policy is None:
        return False
    if not policy["enabled"]:
        raise JobError("budget_policy_closed")
    if control.get("budget", attempt.attempt_id) is not None:
        raise JobError("budget_attempt_already_reserved")
    provider = binding.runtime.budget_config.provider
    context = budgets._context(attempt, source(binding))
    tool = type(attempt) is ToolAttempt
    if tool and not callable(getattr(provider, "prepare_tool", None)):
        raise JobError("budget_tool_adapter_unavailable")
    supplied = await budgets._port(attempt, "prepare",
        (lambda: provider.prepare_tool(copy.deepcopy(context), logical_request, pricing=pricing)) if tool else
        lambda: provider.prepare(copy.deepcopy(context), copy.deepcopy(logical_request), service=service,
                                 **({"purpose": attempt.prepared["purpose"]} if "purpose" in attempt.prepared else {})))
    if type(supplied) is not BudgetPlan:
        raise JobError("invalid_budget_plan")
    plan = BudgetPlan(supplied.token_bound, supplied.request)
    if (tool and plan.token_bound != 0) or (not tool and (logical_request.max_tokens is None or plan.token_bound < logical_request.max_tokens)):
        raise JobError("budget_request_bound_missing")
    async with binding.resource_evidence.setdefault("_budget_prepare_lock", asyncio.Lock()):
        # Metadata preparation yielded to other local tasks. A non-dispatch
        # proof may have won while it was in flight. Allocation and publishing
        # host intent must then form one local choice against that closer.
        if control.get("budget", attempt.attempt_id) is not None:
            raise JobError("budget_attempt_already_reserved")
        allocation_id = await allocate(binding, plan.token_bound, attempt.attempt_id)
        state = dict(version=0, attempt_id=attempt.attempt_id, job_id=binding.lease.job_id,
            policy_key=policy["policy_key"], context=context, plan=asdict(plan), plan_hash=plan.request_hash,
            phase="host_reserving", pending=True, started_at_ms=binding.runtime.manager.now(),
            host_intent=True, grant=None, observation=None, submission=None, ack=None,
            allocation_id=allocation_id, scopes=binding.resource_evidence["_control_budget_scopes"])
        save(binding, state)
    try:
        try:
            grant = await budgets._port(attempt, "reserve", lambda: provider.reserve(plan))
        except Exception:
            grant = await budgets._port(attempt, "lookup", lambda: provider.lookup(plan))
        # A successful exact ACK is sufficient. Lookup is only for uncertainty.
        if (type(grant) is not BudgetGrant or (grant.attempt_id, grant.request_hash, grant.state)
                != (attempt.attempt_id, plan.request_hash, "reserved")):
            raise JobError("budget_host_reservation_unconfirmed")
        state.update(grant=asdict(grant), phase="host_reserved", pending=False)
        save(binding, state)
        return True
    except BaseException:
        control.pending_obligations.add(attempt.attempt_id)
        await control.checkpoint(force=True)
        raise


def authorize(attempt):
    if check(attempt.binding) is None:
        return
    state = get(attempt)
    if state["phase"] != "host_reserved" or state["pending"]:
        raise JobError("budget_dispatch_not_authorized")
    grant = BudgetGrant(**state["grant"])
    if (grant.attempt_id, grant.request_hash, grant.state) != (attempt.attempt_id, state["plan_hash"], "reserved"):
        raise JobError("budget_grant_mismatch")
    state["phase"] = "dispatch_authorized"
    save(attempt.binding, state)


async def observe(attempt, observation):
    from . import model_budgets as budgets
    if check(attempt.binding) is None:
        return
    binding, control = attempt.binding, current(attempt.binding)
    state = get(attempt)
    if state["observation"] is not None:
        if state["observation"] != observation:
            raise JobError("budget_observation_conflict")
    else:
        if state["phase"] != "dispatch_authorized":
            raise JobError("budget_observation_conflict")
        usage = observation["usage"]
        tokens = None if observation["estimated"] else usage.get("total_tokens")
        if not observation["estimated"] and tokens is None and all(k in usage for k in ("prompt_tokens", "completion_tokens")):
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        allocation = binding.resource_evidence["_control_allocation"]
        allocation.observe(attempt.attempt_id, tokens, returned=observation["status"] == "returned")
        if observation["status"] != "returned" or tokens is not None and tokens > state["plan"]["token_bound"]:
            control.needs_reconciliation = True
        if tokens is not None and tokens > state["plan"]["token_bound"]:
            binding.resource_evidence["_control_budget_policy"]["enabled"] = 0
            control.record("budget_policy_closed", state["policy_key"], {"enabled": False})
        allocation_record(binding)
        state.update(phase="settling", pending=True, observation=observation,
                     finished_at_ms=binding.runtime.manager.now())
        save(binding, state)
    if state["ack"] is not None:
        return state["ack"]
    try:
        if state["submission"] is None:
            from .model_attempts import event_id
            if state["context"].get("kind") == "tool":
                from .tool_attempts import event_id
            facts = {**observation, "started_at_ms": state["started_at_ms"], "finished_at_ms": state["finished_at_ms"],
                     "evidence_event_id": event_id(binding.runtime.manager, "finished", attempt.attempt_id),
                     "evidence_hash": _hash(observation)}
            submission = await budgets._port(attempt, "prepare_usage", lambda: binding.runtime.budget_config.provider.prepare_usage(
                budgets.accounting_context(state), BudgetGrant(**state["grant"]), facts))
            if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (attempt.attempt_id, 1):
                raise JobError("invalid_budget_submission")
            state["submission"] = asdict(submission)
            save(binding, state)
        submission = BudgetSubmission(**state["submission"])
        # A durable response capture can own this obligation independently.
        # Otherwise seal the frozen fee outbox before any remote apply. This is
        # required financial I/O, not repeated execution authorization.
        charge = observation.get("charge", {})
        known_free = state["context"].get("kind") == "tool" and charge.get("amount") == "0" and charge.get("quality") == "free"
        if not known_free and not binding.resource_evidence.get("_captured_" + attempt.attempt_id):
            await control.checkpoint(force=True)
        ack = await budgets._port(attempt, "apply_usage:1", lambda: binding.runtime.budget_config.provider.apply_usage(submission))
        if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
                attempt.attempt_id, submission.revision, submission.request_hash):
            raise JobError("budget_ack_mismatch")
        state.update(ack=asdict(ack), pending=False)
        control.pending_obligations.discard(attempt.attempt_id)
        save(binding, state)
        return asdict(ack)
    except BaseException:
        control.pending_obligations.add(attempt.attempt_id)
        await control.checkpoint(force=True)
        # A durable outbox does not itself prove remote acceptance. Block new
        # admission until this original submission has a valid ACK.
        raise


async def close_unreserved(attempt):
    """The original stack can prove it never even requested a host grant."""
    from .model_budgets import _context
    binding, control = attempt.binding, current(attempt.binding)
    async with binding.resource_evidence.setdefault("_budget_prepare_lock", asyncio.Lock()):
        policy = check(binding)
        saved = control.get("attempt", attempt.attempt_id)
        if saved is None or saved["prepared"] != attempt.prepared:
            raise JobError("budget_release_requires_original_owner")
        state = control.get("budget", attempt.attempt_id)
        if state is not None:
            return state["ack"] if state["phase"] == "released" else None
        if saved["phase"] != "prepared":
            raise JobError("budget_release_not_permitted")
        ack = {"local_only": True, "dispatch_not_authorized": True, "no_reservation": True}
        if policy is not None:
            state = dict(version=0, attempt_id=attempt.attempt_id, job_id=binding.lease.job_id,
                policy_key=policy["policy_key"], context=_context(attempt, source(binding)),
                plan=None, plan_hash=None, phase="released", pending=False, started_at_ms=binding.runtime.manager.now(),
                host_intent=False, grant=None, observation=None, submission=None, ack=ack, scopes=[])
            save(binding, state)
        control.abort_prepared(attempt.attempt_id)
        return ack


async def release_unstarted(attempt):
    """Close the original local admission, then acknowledge the host tombstone."""
    from . import model_budgets as budgets
    local = await close_unreserved(attempt)
    if local is not None:
        return local
    binding, control = attempt.binding, current(attempt.binding)
    async with binding.resource_evidence.setdefault("_budget_release_lock", asyncio.Lock()):
        state = get(attempt)
        saved = control.get("attempt", attempt.attempt_id)
        if saved["phase"] not in ("prepared", "aborted") or state["phase"] not in ("host_reserving", "host_reserved", "dispatch_closed", "released"):
            raise JobError("budget_release_not_permitted")
        if state["phase"] == "released":
            return state["ack"]
        control.abort_prepared(attempt.attempt_id)
        state.update(phase="dispatch_closed", pending=True)
        save(binding, state)
        await control.checkpoint(force=True)
        provider = binding.runtime.budget_config.provider
        if state["grant"] is None:
            plan = BudgetPlan(**state["plan"])
            grant = await budgets._port(attempt, "lookup", lambda: provider.lookup(plan))
            if grant is not None:
                if type(grant) is not BudgetGrant or (grant.attempt_id, grant.request_hash) != (attempt.attempt_id, plan.request_hash):
                    raise JobError("budget_host_reservation_unconfirmed")
                state["grant"] = asdict(grant)
                save(binding, state)
        if state["submission"] is None:
            proof = {"proof": "dispatch_not_authorized", "event_id": "mb_" + _hash([attempt.attempt_id, state["version"]])[:48],
                     "evidence_hash": _hash(state), "reservation_request_hash": state["plan_hash"]}
            submission = await budgets._port(attempt, "prepare_release", lambda: provider.prepare_release(
                copy.deepcopy(state["context"]), BudgetGrant(**state["grant"]) if state["grant"] else None, proof))
            if type(submission) is not BudgetSubmission or (submission.attempt_id, submission.revision) != (attempt.attempt_id, 1):
                raise JobError("invalid_budget_submission")
            state["submission"] = asdict(submission)
            save(binding, state)
            await control.checkpoint(force=True)
        submission = BudgetSubmission(**state["submission"])
        ack = await budgets._port(attempt, "release", lambda: provider.release(submission))
        if type(ack) is not BudgetAck or (ack.attempt_id, ack.revision, ack.request_hash) != (
                attempt.attempt_id, 1, submission.request_hash):
            raise JobError("budget_ack_mismatch")
        allocation = binding.resource_evidence["_control_allocation"]
        allocation.release_unstarted(attempt.attempt_id)
        allocation_record(binding)
        state.update(phase="released", pending=False, ack=asdict(ack))
        control.pending_obligations.discard(attempt.attempt_id)
        save(binding, state)
        await control.checkpoint(force=True)
        return asdict(ack)
