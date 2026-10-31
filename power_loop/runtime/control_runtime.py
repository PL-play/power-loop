"""Bind the memory reducer to the task owner and indexed inspection views."""
from __future__ import annotations

from dataclasses import asdict

from .execution_journal import ExecutionJournal
from .execution_state import ExecutionIdentity, ExecutionState
from .jobs import JobError, _hash, _json


def current(binding):
    state = binding.resource_evidence.get("_control_state")
    if state is None:
        raise JobError("execution_control_not_started")
    from .root_context import _authority
    authority = binding.resource_evidence.get("_control_authority")
    actual = _authority(binding.runtime)
    if (authority is None or any(a is not b for a, b in zip(actual[:3], authority[:3], strict=True))
            or actual[3:] != authority[3:]):
        raise JobError("execution_control_binding_changed")
    expected = ExecutionIdentity(
        binding.runtime.manager.store.store_id, binding.lease.job_id, binding.lease.generation,
        binding.lease.executor_id, binding.lease.boot_id, binding.actor.tenant_id,
        binding.actor.principal_id, binding.actor.session_id,
    )
    if type(state) is not ExecutionState or state.identity != expected:
        raise JobError("execution_control_binding_changed")
    return state


async def project_events(manager, tx, identity, records):
    """Rebuildable observation views, atomically advanced with the journal.

    The journal remains authoritative. Inspection keeps stable event identities,
    so existing task-result references do not acquire another authority ledger.
    """
    from .model_attempts import event_id

    rows, budget_rows = [], []
    def event(key, kind, payload, metadata):
        row = (key, identity.job_id, kind, metadata["job_version"], _hash(payload),
               _json(payload), "not_applicable", metadata["created_at"])
        rows.append(row)
        return dict(zip(("event_id", "job_id", "event_type", "job_version", "request_hash",
                         "payload_json", "delivery_status", "created_at"), row, strict=True))

    for record in records:
        value = record["value"]
        phases = {}
        if record["kind"] == "allocation_adjustment_immutable":
            from .execution_budget import project_adjustment
            await project_adjustment(manager, tx, identity, value)
            continue
        if record["kind"] == "budget_policy_closed":
            await tx.execute(f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?", (record["key"],))
            continue
        if record["kind"] == "budget_allocation":
            from .execution_budget import project_allocation
            await project_allocation(manager, tx, identity, record["key"], value)
            continue
        if record["kind"] == "event_immutable":
            event(record["key"], value["event_type"], value["payload"], value["event_metadata"])
            continue
        if record["kind"] == "budget":
            metadata = {"job_version": 1, "created_at": value["started_at_ms"]}
            event("mb_" + _hash([record["key"], value["version"]])[:48], "model_budget_state", value, metadata)
            budget_rows.append((value["attempt_id"], identity.job_id, value["policy_key"], value["version"],
                                value["phase"], int(value["pending"]), _hash(value), _json(value),
                                value["started_at_ms"], value.get("finished_at_ms", value["started_at_ms"])))
            continue
        if record["kind"] == "model_tracking_immutable":
            phases["tracking"] = value["data"]
            original = value["data"]["scope"]
        elif record["kind"] == "attempt":
            prepared = value["prepared"]
            original = record["key"]
            phases["prepared"] = prepared
            if value["phase"] in ("dispatched", "finished"):
                phases["dispatched"] = {"version": 1, "scope": prepared["scope"],
                                        "attempt_id": original, "prepared_hash": _hash(prepared)}
            if value["phase"] == "finished":
                phases["finished"] = value["observation"]
        else:
            continue
        metadata = value.get("event_metadata")
        if (type(metadata) is not dict or set(metadata) != {"job_version", "created_at"}
                or type(metadata["job_version"]) is not int or type(metadata["created_at"]) is not int):
            raise JobError("execution_event_metadata_missing")
        for phase, payload in phases.items():
            if record["kind"] == "attempt" and value["kind"] == "tool":
                from .model_exit import content_reference
                from .tool_attempts import event_id as tool_event_id
                key = tool_event_id(manager, phase, original)
                original_event = event(key, "tool_attempt_" + phase, payload, metadata)
                event(key + "_a", "tool_attempt_anchor", {"version": 1, "source": content_reference(original_event)}, metadata)
            else:
                event(event_id(manager, phase, original), "model_attempt_" + phase, payload, metadata)
    if budget_rows:
        await upsert(manager, tx, manager.prefix + "task_budget_attempts",
                     ("attempt_id", "job_id", "policy_key", "version", "phase", "pending", "state_hash",
                      "state_json", "created_at", "updated_at"), "attempt_id", budget_rows)
    if not rows:
        return
    columns = ("event_id", "job_id", "event_type", "job_version", "request_hash", "payload_json",
               "delivery_status", "created_at")
    await upsert(manager, tx, manager.events, columns, "event_id", rows)
    attempts = {row[0]: dict(zip(columns, row, strict=True)) for row in rows if row[2].startswith("model_attempt_")}
    if attempts:
        from .execution_membership import remember_merge
        remember_merge(manager, "_projected_attempts", identity.job_id, attempts)


async def upsert(manager, tx, table, columns, primary, rows):
    # Bound statement size independently of action count; actual rows and bytes
    # are still counted by benchmarks, not hidden behind a single driver call.
    for offset in range(0, len(rows), 100):
        part = rows[offset:offset + 100]
        values = ",".join("(" + ",".join("?" for _ in columns) + ")" for _ in part)
        if manager.db.dialect.name == "mysql":
            conflict = " ON DUPLICATE KEY UPDATE " + ",".join(
                f"{column}=VALUES({column})" for column in columns[1:])
        else:
            conflict = f" ON CONFLICT({primary}) DO UPDATE SET " + ",".join(
                f"{column}=excluded.{column}" for column in columns[1:])
        await tx.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES {values}" + conflict,
                         tuple(value for row in part for value in row))


def queue_event(binding, key, kind, value):
    """Freeze one trusted fact for indexed inspection at the next checkpoint."""
    control = current(binding)
    old = control.get("event_immutable", key)
    if old is not None:
        if old["event_type"] != kind or old["payload"] != value:
            raise JobError("execution_record_conflict")
        return old
    record = {"event_type": kind, "payload": value,
              "event_metadata": {"job_version": source(binding)["version"], "created_at": binding.runtime.manager.now()}}
    control.record("event_immutable", key, record)
    return record


async def append_correction(manager, tx, job, generation, records):
    """Cold reconciliation appends to the same original journal authority.

    It grants no lease. A running owner must use its own memory reducer; a
    reconciler cannot race that owner's next checkpoint or silently overwrite it.
    """
    import json
    binding = manager._execution_controls.get((job["job_id"], generation))
    if binding is not None and not current(binding).closed:
        raise JobError("execution_reconciliation_requires_quiescence")
    if binding is not None:
        control = current(binding)
        if control._dirty or control._pending is not None or control._flush_lock.locked():
            raise JobError("execution_reconciliation_requires_quiescence")
        retirements = manager._control_retirements.get()
        if retirements is None:
            raise JobError("execution_reconciliation_transaction_required")
        retirements.add((job["job_id"], generation))
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    head = await tx.fetchone(f"SELECT * FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?" + suffix,
                             (job["job_id"], generation))
    if head is None:
        raise JobError("execution_checkpoint_missing")
    identity = ExecutionIdentity(**json.loads(head["identity_json"]))
    if (identity.job_id, identity.generation, identity.tenant_id, identity.principal_id) != (
            job["job_id"], generation, job["tenant_id"], job["principal_id"]):
        raise JobError("execution_checkpoint_corrupt")
    async def projector(tx, identity, records):
        await project_events(manager, tx, identity, records)
    journal = ExecutionJournal(manager.store, projector=projector)
    summary = json.loads(head["summary_json"])
    if not summary.get("closed"):
        raise JobError("execution_reconciliation_requires_quiescence")
    for record in records:
        if record["kind"] == "budget":
            previous = await tx.fetchone(f"SELECT state_json,state_hash FROM {manager.prefix}task_budget_attempts WHERE attempt_id=?",
                                         (record["key"],))
            from .model_budgets import _object
            def unsettled(value):
                return value is not None and (value["pending"] or value["phase"] not in ("released", "settling") or value["ack"] is None)
            old = _object(previous["state_json"], previous["state_hash"]) if previous else None
            summary["budget_attempts"] = summary.get("budget_attempts", 0) + int(old is None)
            summary["budget_unsettled"] = summary.get("budget_unsettled", 0) + int(unsettled(record["value"])) - int(unsettled(old))
    digest = await journal.append(identity, head["batch_no"] + 1, head["chain_hash"], records, summary, transaction=tx)
    return head["batch_no"] + 1, digest


async def require_closed(manager, tx, job_id, generation):
    """A cold writer cannot supersede an owner with an unflushed suffix."""
    from .model_budgets import _object
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    head = await tx.fetchone(
        f"SELECT summary_json,summary_hash FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?" + suffix,
        (job_id, generation))
    if head is None:
        raise JobError("execution_checkpoint_missing")
    if not _object(head["summary_json"], head["summary_hash"]).get("closed"):
        raise JobError("execution_reconciliation_requires_quiescence")


async def open_control(binding):
    runtime, manager = binding.runtime, binding.runtime.manager
    if binding.resource_evidence.get("_control_state") is not None:
        return current(binding)
    if (binding.lease.job_id, binding.lease.generation) in manager._execution_controls:
        raise JobError("execution_already_owned")
    from .model_attempts import _no_unknown_sends
    from .model_budgets import ensure_policy, flush_observed_controls
    from .owner_waits import require_dispatch

    await flush_observed_controls(binding)
    async with manager._write() as tx:
        source = dict(await manager._fence(tx, binding.actor, binding.lease))
        from .workflow_controller import validate_dispatch
        await validate_dispatch(manager, tx, source)
        if source["kind"] != "send" or source["session_id"] != binding.actor.session_id:
            raise JobError("model_attempt_owner_mismatch")
        await ensure_policy(binding, transaction=tx, job=source)
        await _no_unknown_sends(manager, tx, source)
        wait_deadline = await require_dispatch(manager, tx, source)
        import json

        from .abort_outcomes import pending_reviews
        length = "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
        registered = binding.resource_evidence.pop("_control_registered_payload", None)
        fresh = registered is not None and registered.get("job_id") == source["job_id"]
        if fresh:
            creation_request = registered["request"]  # Inserted earlier in this same transaction.
        else:
            creations = await tx.fetchall(f"SELECT CASE WHEN {length}<=262144 THEN payload_json ELSE NULL END AS payload_json "
                f"FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create' LIMIT 2", (source["job_id"],))
            if len(creations) != 1 or creations[0]["payload_json"] is None:
                raise JobError("execution_creation_missing")
            creation_request = json.loads(creations[0]["payload_json"])["request"]
        flag = creation_request.get("tool_call_plan_protocol")
        if flag is not None and (type(flag) is not int or flag != 1):
            raise JobError("tool_call_plan_protocol_invalid")
        plan_required = flag == 1
        from . import session_controls
        seen = await session_controls.version(manager, tx, binding.actor.session_id)
        review_pending = (False if session_controls.is_clean(manager, binding.actor.session_id, seen)
                          else bool(await pending_reviews(manager, tx, binding.actor)))
        # A job registered in this same transaction cannot have children yet.
        managed_check = not fresh and await tx.fetchone(
            f"SELECT job_id FROM {manager.jobs} WHERE parent_job_id=? AND kind='background_tool' LIMIT 1",
            (source["job_id"],)) is not None
        ancestors, ancestor = [], source
        while ancestor is not None:
            ancestors.append(ancestor["job_id"])
            if len(ancestors) > 64:
                raise JobError("invalid_job_lineage")
            ancestor = await manager._row(tx, ancestor["parent_job_id"]) if ancestor["parent_job_id"] else None
        identity = ExecutionIdentity(manager.store.store_id, source["job_id"], source["generation"],
                                     source["executor_id"], source["boot_id"], source["tenant_id"],
                                     source["principal_id"], source["session_id"])

        async def projector(tx, identity, records):
            await project_events(manager, tx, identity, records)

        journal = binding.resource_evidence.get("_control_journal_opening")
        if journal is None:
            journal = ExecutionJournal(manager.store, projector=projector)
            binding.resource_evidence["_control_journal_opening"] = journal
        state = ExecutionState(identity, journal, lease_until_ms=source["lease_until_ms"],
                               policy=runtime.checkpoint_policy, now_ms=lambda: manager.now())
        policy = binding.resource_evidence.get("_control_budget_policy")
        if policy is not None:
            state.max_pending_obligations = policy["manifest"]["max_pending"]
        state.managed_check = managed_check
        await journal.create(identity, state.summary(), transaction=tx)
    binding.resource_evidence.pop("_control_journal_opening", None)
    binding.resource_evidence["_control_source"] = source
    binding.resource_evidence["_control_ancestors"] = tuple(ancestors)
    binding.resource_evidence["_control_create_request"] = creation_request
    binding.resource_evidence["_control_tool_plan_required"] = plan_required
    binding.resource_evidence["_control_review_pending"] = review_pending
    binding.resource_evidence["_control_wait_deadline"] = wait_deadline
    binding.resource_evidence["_control_state"] = state
    from .root_context import _authority
    binding.resource_evidence["_control_authority"] = _authority(runtime)
    async def refresh(transaction=None):
        if state.closed:
            return  # Only original observations/fees may arrive after closure.
        reader = manager.db if transaction is None else transaction
        rows = await reader.fetchall(
            f"SELECT job_id,generation,executor_id,boot_id,cancel_requested_at,intake,resource_state,lease_until_ms "
            f"FROM {manager.jobs} WHERE job_id IN (" + ",".join("?" for _ in ancestors) + ")", tuple(ancestors))
        found = {row["job_id"]: row for row in rows}
        if len(found) != len(ancestors):
            state.needs_reconciliation = True
            raise JobError("invalid_job_lineage")
        actual = found[identity.job_id]
        if (actual["generation"], actual["executor_id"], actual["boot_id"]) != (
                identity.generation, identity.executor_id, identity.boot_id):
            state.needs_reconciliation = True
            state.cancel_requested = True
        if any(row["cancel_requested_at"] is not None for row in rows) or actual["intake"] != "open":
            state.cancel_requested = True
        if actual["lease_until_ms"] is not None:
            state.lease_until_ms = min(state.lease_until_ms, actual["lease_until_ms"])
        if actual["resource_state"] == "unknown":
            state.needs_reconciliation = True
        if state.cancel_requested:
            state.record("lifecycle", "cancel", {"requested": True})
            from .execution_budget import allocation_record
            allocation_record(binding)
    state.before_checkpoint = refresh
    manager._execution_controls[(source["job_id"], source["generation"])] = binding
    scope = asdict(binding.lease)
    state.record("model_tracking_immutable", "tracking", {
        "data": {"version": 1, "scope": scope, "session_id": binding.actor.session_id},
        "event_metadata": {"job_version": source["version"], "created_at": manager.now()},
    })
    return state


def source(binding):
    current(binding)
    return binding.resource_evidence["_control_source"]


async def flush_binding(binding, *, close=False, force=True, transaction=None):
    if binding.resource_evidence.get("_control_state") is None:
        return
    state = current(binding)
    if close:
        from .execution_budget import allocation_record
        state.closed = True
        allocation_record(binding)
        if transaction is not None:
            # Joins the close transaction; a rollback keeps the batch for retry.
            manager = binding.runtime.manager
            await state.close(transaction=transaction, defer=manager.after_outcome)
            if state._staged:
                from .execution_membership import _remember_after_commit
                _remember_after_commit(manager, None, "_closed_summaries",
                                       (state.identity.job_id, state.identity.generation), state._pending[3])
            return
        try:
            await state.close()
        except Exception:
            # Retry only the retained checkpoint, never a model/tool action.
            await state.close()
    else:
        await state.checkpoint(force=force)


def forget_completed(binding):
    """Release completed execution caches; retain outstanding evidence/retries."""
    state = current(binding)
    if (not state.closed or state.active or state.unknown or state.pending_obligations
            or state._dirty or state._pending is not None or state._flush_lock.locked()):
        return
    manager = binding.runtime.manager
    key = (state.identity.job_id, state.identity.generation)
    if manager._execution_controls.get(key) is binding:
        manager._execution_controls.pop(key)
    forget_budget_bindings(binding)


def forget_budget_bindings(binding):
    manager = binding.runtime.manager
    for kind, key in current(binding)._records:
        if kind == "budget" and manager._live_budget_bindings.get(key) is binding:
            manager._live_budget_bindings.pop(key)
