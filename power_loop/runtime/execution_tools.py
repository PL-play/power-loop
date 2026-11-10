"""Freeze a tool batch once and advance each original callback in memory."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from .control_runtime import current, source
from .jobs import JobError, _hash, _json


def batch(binding, seq):
    current(binding)
    result = binding.resource_evidence.get("_control_batches", {}).get(seq)
    if result is None:
        raise JobError("tool_call_plan_original_requires_reconciliation")
    return result


def entry(binding, seq, call_id):
    value = batch(binding, seq)
    result = value["by_id"].get(call_id)
    if result is None:
        raise JobError("tool_call_plan_binding_changed")
    return value, result


async def prepare(pipeline, attempt):
    from . import tool_call_plans as plans
    from .execution_membership import proof
    from .managed_effects import _document
    from .message_ref import MessageRef
    from .task_runtime import _SEND
    from .tool_call_abort import _freeze
    from .tool_contracts import policy, resolve, validate_registration
    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return
    control, job = current(binding), source(binding)
    if control.closed:
        raise JobError("execution_control_closed")
    if not binding.resource_evidence["_control_tool_plan_required"]:
        return
    seq = getattr(pipeline.sink, "persisted_assistant_seq", None)
    if type(seq) is not int:
        raise JobError("persisted_tool_call_required")
    retained = binding.resource_evidence.setdefault("_control_batches", {})
    if seq in retained:
        value = retained[seq]
        for key, registration in value["registrations"].items():
            if pipeline.tool_registry.get(value["by_id"][key]["tool_name"]) is not registration:
                raise JobError("tool_call_plan_binding_changed")
        return
    manager = binding.runtime.manager
    appended = manager.store.appended_tool_calls(binding.actor.session_id, seq)
    if appended is not None and len(appended.encode()) <= 131072:
        # This process committed the row; its identity and calls cannot change.
        ref = await manager.store.message_ref(binding.actor.session_id, seq)
        raw = {"role": ref.role, "calls": appended}
    else:
        length = "length(CAST(tool_calls_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(tool_calls_json)"
        # Fetch an oversized body as NULL so even malformed stored input cannot
        # force an unbounded transfer.
        raw = await manager.db.fetchone(
            f"SELECT id,role,send_index,round_index,tool_call_id,CASE WHEN {length}<=131072 THEN tool_calls_json ELSE NULL END AS calls "
            f"FROM {manager.store.t.messages} WHERE session_id=? AND seq=?", (binding.actor.session_id, seq))
        if raw is not None:
            ref = MessageRef(manager.store.store_id, binding.actor.session_id, raw["id"], seq, raw["role"],
                             raw["tool_call_id"], raw["send_index"], raw["round_index"])
    if raw is None or raw["role"] != "assistant" or raw["calls"] is None:
        raise JobError("tool_call_plan_raw_unavailable")
    if binding.resource_evidence["_runtime_send_index"] != ref.send_index:
        raise JobError("tool_call_plan_original_send_mismatch")
    try:
        calls = json.loads(raw["calls"])
        if (type(calls) is not list or not 0 < len(calls) <= 64
                or any(type(c) is not dict or type(c.get("id")) is not str or not c["id"] for c in calls)
                or len({c["id"] for c in calls}) != len(calls)):
            raise ValueError()
        entries, originals, controls = [], {}, []
        for call in calls:
            name = call["function"]["name"]
            arguments = _document(json.loads(call["function"]["arguments"]), 131072)
            registered = pipeline.tool_registry.get(name) if pipeline.tool_registry is not None else None
            implementation = resolve(registered)
            recovery = implementation.pending_call_spec(arguments, runtime=binding.runtime, definition=registered.definition) if implementation else None
            if recovery is not None:
                recovery = _document(recovery, 65536)
                if set(recovery) != {"kind", "request"} or recovery["kind"] != "managed_effect":
                    raise JobError("tool_call_plan_recovery_unsupported")
            entries.append(dict(call_id=call["id"], tool_name=name, arguments_hash=_hash(arguments),
                                registration=plans.registration_value(registered), recovery=recovery, job_id=None, operation_id=None))
            originals[call["id"]] = registered
            controls.append(recovery is None and policy(registered, arguments, runtime=binding.runtime).control_only)
    except (ValueError, TypeError, KeyError) as exc:
        raise JobError("tool_call_plan_invalid_call_set") from exc
    attempt["value"] = _freeze(pipeline, binding, job, ref, entries)
    pipeline._native_abort_attempt = attempt["value"]
    if not all(controls):
        control.require_admission()
    if any(e["recovery"] is not None for e in entries):
        from .lightweight_send import ensure
        await ensure(binding)
        # Independent effects need their own durable jobs. Their registration is
        # an explicit control-plane event, not one job for each ordinary call.
        await control.checkpoint(force=True)
        await plans._prepare_children(pipeline, attempt)
        async with manager.db.transaction() as tx:
            value = await plans.read(manager, tx, job, ref)
        entries = value["entries"]
    else:
        key = plans.identity(manager, job, ref)
        value = {"version": 1, "scope": proof(job), "call_ref": ref.to_dict(), "entries": entries}
        pipeline._native_tool_batch_bindings = plans._BatchBindings(
            key, _hash(value), _json(proof(job)), _json(ref.to_dict()), tuple(originals.items()))
    for e in entries:
        original = originals[e["call_id"]]
        if original is not None:
            validate_registration(pipeline.tool_registry, e["tool_name"], original)
    retained[seq] = {"ref": ref, "plan": value, "hash": _hash(value), "entries": entries,
                     "by_id": {e["call_id"]: e for e in entries}, "registrations": originals}


async def dispatch(pipeline, name, arguments, registered, *, budget_attempt=None):
    from .invocation import current_invocation
    from .task_runtime import _SEND
    from .tool_call_plans import identity, registration_value, start_id
    from .tool_contracts import policy
    binding = _SEND.get()
    if binding is None or binding.actor.session_id != pipeline.session_id:
        return
    if not binding.resource_evidence["_control_tool_plan_required"]:
        return
    seq, call_id = current_invocation(binding.actor.session_id)
    owned, selected = entry(binding, seq, call_id)
    control, manager, job = current(binding), binding.runtime.manager, source(binding)
    if control.closed:
        raise JobError("execution_control_closed")
    declared = policy(registered, arguments, runtime=binding.runtime)
    if not declared.control_only:
        control.require_admission()
    if (owned["registrations"][call_id] is not registered or selected["tool_name"] != name
            or (selected["recovery"] is not None and selected["arguments_hash"] != _hash(arguments))
            or selected["registration"] != registration_value(registered)):
        raise JobError("tool_call_plan_binding_changed")
    plan, ref = owned["plan"], owned["ref"]
    key = start_id(identity(manager, job, ref), call_id)
    dispatched = binding.resource_evidence.setdefault("_control_dispatched_tools", set())
    if key in dispatched:
        raise JobError("tool_call_already_dispatched")
    if budget_attempt is not None:
        from .tool_attempts import authorize_dispatch
        await authorize_dispatch(budget_attempt, None, job)
    dispatched.add(key)
    if selected["recovery"] is not None:
        # A managed child must see the start before it can claim its own lease.
        if budget_attempt is not None:
            from .tool_attempts import observe_managed_free
            await observe_managed_free(budget_attempt, None, job)
        await control.checkpoint(force=True)
        return
    slot = binding.runtime._native_returns.reserve(binding, ref, plan, selected, plan_hash=owned["hash"])
    slot.operation_revision = getattr(manager, "_control_operation_revision", 0)
    return slot


async def prepare_attempt(binding, ref, call_id, registered, arguments):
    from power_loop.tools.registry import validate_pricing

    from .pricing import Free, PricedCall
    from .task_runtime import _SEND
    from .tool_attempts import ToolAttempt
    from .tool_contracts import policy
    if _SEND.get() is not binding:
        raise JobError("tool_attempt_original_required")
    control, job = current(binding), source(binding)
    owned, selected = entry(binding, ref.seq, call_id)
    # TOOL_BEFORE may rewrite ordinary call arguments. Keep the original
    # assistant identity, but price and dispatch the actual hook output.
    # Managed effects still use their separately frozen request contract.
    if (ref != owned["ref"] or owned["registrations"][call_id] is not registered
            or (selected["recovery"] is not None and selected["arguments_hash"] != _hash(arguments))):
        raise JobError("tool_call_plan_binding_changed")
    pricing = validate_pricing(registered)
    control_only = policy(registered, arguments, runtime=binding.runtime).control_only
    if control_only and not isinstance(pricing, Free):
        raise JobError("control_tool_must_be_free")
    scope = asdict(binding.lease)
    key = "toolattempt_" + _hash([binding.runtime.manager.store.store_id, scope, ref.to_dict(), call_id])[:48]
    prepared = dict(version=1, attempt_id=key, scope=scope, session_id=ref.session_id,
        call_ref=ref.to_dict(), call_id=call_id, request_hash=_hash(arguments), tool_name=registered.definition.name,
        pricing_rule={"rule_id": pricing.rule_id, "rule_version": registered.pricing_version})
    if control.get("attempt", key) is not None:
        raise JobError("tool_attempt_already_prepared")
    value = control.begin_attempt(key, "tool", prepared, control_only=control_only)
    config = binding.runtime._original_budget_config
    value["budget_required"] = not control_only and config is not None and callable(getattr(config.provider, "prepare_tool", None))
    control.record("attempt", key, value)
    return ToolAttempt(binding, key, prepared, PricedCall(registered.definition.name, arguments, job["execution_role"]), pricing, asyncio.current_task())


async def persist_return(runtime, slot):
    from .model_exit import content_reference
    cache, manager = runtime._native_returns, runtime.manager
    try:
        async with cache.lock:
            cache.check()
            if slot.saved and slot.binding.runtime is runtime:
                return
            cache.require(slot)
            if slot.text is None:
                raise JobError("native_return_unconfirmed")
            _, selected = entry(slot.binding, slot.ref.seq, slot.call_id)
            operations = []
            if (getattr(slot, "operation_revision", None) != getattr(manager, "_control_operation_revision", 0)
                    or (selected["registration"] or {}).get("runtime_contract") is not None):
                rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE caller_session_id=? AND tool_call_id=? "
                    "AND event_type='operation' ORDER BY event_id LIMIT 1001", (slot.ref.session_id, slot.call_id))
                if len(rows) > 1000:
                    raise JobError("native_return_operations_too_large")
                for row in rows:
                    if json.loads(row["call_ref_json"] or "null") == slot.ref.to_dict():
                        await manager._authorize(manager.db, slot.binding.actor, row["job_id"])
                        operations.append(content_reference(row))
            key = cache.key(slot)
            slot.binding.resource_evidence.setdefault("_control_tool_operations", {})[(slot.ref.seq, slot.call_id)] = operations
            del cache.slots[key]
            slot.saved = True
    except BaseException:
        slot.failed = True
        raise


async def persist_billing_receipt(runtime, slot):
    cache = runtime._native_returns
    cache.require(slot)
    if slot.billing_receipt is None:
        raise JobError("tool_billing_receipt_missing")
    slot.billing_saved = True
