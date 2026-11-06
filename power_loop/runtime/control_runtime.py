"""Bind the memory reducer to the task owner and indexed inspection views."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

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
    """Persist control projections in the checkpoint transaction."""
    rows, budget_rows = [], []
    def event(key, kind, payload, metadata):
        row = (key, identity.job_id, kind, metadata["job_version"], _hash(payload),
               _json(payload), "not_applicable", metadata["created_at"])
        rows.append(row)
        return dict(zip(("event_id", "job_id", "event_type", "job_version", "request_hash",
                         "payload_json", "delivery_status", "created_at"), row, strict=True))

    for record in records:
        value = record["value"]
        if record["kind"] == "allocation_adjustment_immutable":
            from .execution_budget import project_adjustment
            await project_adjustment(manager, tx, identity, value)
            continue
        if record["kind"] == "budget_policy_closed":
            await tx.execute(f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?", (record["key"],))
            continue
        if record["kind"] == "budget_allocation":
            from .execution_budget import project_allocation
            # The full grant remains reserved while its owner is open. Its
            # actual consumption is already in this checkpoint's summary and
            # records; closing applies the cumulative difference exactly once.
            # An overrun must still disable admission and book the excess now.
            if value["closed"] or value["used_tokens"] > value["tokens"]:
                await project_allocation(manager, tx, identity, record["key"], value)
            continue
        if record["kind"] == "event_immutable":
            event(record["key"], value["event_type"], value["payload"], value["event_metadata"])
            continue
        if record["kind"] == "budget":
            from .model_budgets import stored_state
            stored = stored_state(value)
            budget_rows.append((value["attempt_id"], identity.job_id, value["policy_key"], value["version"],
                                value["phase"], int(value["pending"]), _hash(stored), _json(stored),
                                value["started_at_ms"], value.get("finished_at_ms", value["started_at_ms"])))
            continue
    if budget_rows:
        await upsert(manager, tx, manager.prefix + "task_budget_attempts",
                     ("attempt_id", "job_id", "policy_key", "version", "phase", "pending", "state_hash",
                      "state_json", "created_at", "updated_at"), "attempt_id", budget_rows)
    if not rows:
        return
    columns = ("event_id", "job_id", "event_type", "job_version", "request_hash", "payload_json",
               "delivery_status", "created_at")
    await upsert(manager, tx, manager.events, columns, "event_id", rows, event_projection=True)


async def upsert(manager, tx, table, columns, primary, rows, *, event_projection=False):
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
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES {values}" + conflict
        params = tuple(value for row in part for value in row)
        intents = manager._control_intents.get() if event_projection else None
        if intents is not None:
            await intents.project_events(columns, part, sql, params)
        else:
            # Standalone owner checkpoints already fence their original journal
            # append. Their immutable inspection events create no new work.
            await tx.execute(sql, params)


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


async def append_correction(manager, tx, job, generation, records, *, summary_delta=None):
    """Advance a closed execution's projections for accounting reconciliation."""
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
    identity = ExecutionIdentity(manager.store.store_id, job['job_id'], generation,
        job['executor_id'], job['boot_id'], job['tenant_id'], job['principal_id'], job['session_id'])
    async def projector(tx, identity, records):
        await project_events(manager, tx, identity, records)
    journal = ExecutionJournal(manager.store, projector=projector)
    summary = json.loads(head["summary_json"])
    if not summary.get("closed"):
        raise JobError("execution_reconciliation_requires_quiescence")
    for key, delta in (summary_delta or {}).items():
        summary[key] += delta
    for record in records:
        if record["kind"] == "budget":
            previous = await tx.fetchone(f"SELECT pending,phase FROM {manager.prefix}task_budget_attempts WHERE attempt_id=?",
                                         (record["key"],))
            def unsettled(value):
                return value is not None and (value["pending"] or value["phase"] not in ("released", "settling"))
            summary["budget_attempts"] = summary.get("budget_attempts", 0) + int(previous is None)
            summary["budget_unsettled"] = summary.get("budget_unsettled", 0) + int(unsettled(record["value"])) - int(unsettled(previous))
    await journal.append(identity, head["batch_no"] + 1, records, summary, transaction=tx)
    return head["batch_no"] + 1


async def require_closed(manager, tx, job_id, generation):
    """A cold writer cannot supersede an owner with an unflushed suffix."""
    import json
    suffix = "" if manager.db.dialect.name == "sqlite" else " FOR UPDATE"
    head = await tx.fetchone(
        f"SELECT summary_json FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?" + suffix,
        (job_id, generation))
    if head is None:
        raise JobError("execution_checkpoint_missing")
    if not json.loads(head["summary_json"]).get("closed"):
        raise JobError("execution_reconciliation_requires_quiescence")


class ReviewSourcesChanged(JobError):
    def __init__(self):
        super().__init__('pending_review_sources_changed')


@dataclass(frozen=True)
class ReviewSources:
    version: tuple
    pending: bool


def needs_review_sources(manager, actor):
    return not (manager._trust_own_writes.get() and actor.session_id in manager.__dict__.get('_clean_sessions',{}))


async def prepare_review_sources(manager, actor):
    from . import session_controls
    from .abort_outcomes import pending_reviews

    async with manager._read() as reader:
        seen = await session_controls.version(manager,reader,actor.session_id)
        pending = bool(await pending_reviews(manager,reader,actor))
        return ReviewSources(seen,pending)


async def open_control(binding, *, review_sources=None, policy_source=None):
    runtime, manager = binding.runtime, binding.runtime.manager
    if binding.resource_evidence.get("_control_state") is not None:
        return current(binding)
    if (binding.lease.job_id, binding.lease.generation) in manager._execution_controls:
        raise JobError("execution_already_owned")
    from .budget_policy_sources import prepared as prepare_policy
    from .model_attempts import _no_unknown_sends
    from .model_budgets import ensure_policy, flush_observed_controls
    from .owner_waits import dispatch_policy

    await flush_observed_controls(binding)
    async with prepare_policy(runtime, source=policy_source), manager._write() as tx:
        from .budget_admissions import enter
        await enter(runtime, tx)
        source = dict(await manager._fence(tx, binding.actor, binding.lease))
        if source["kind"] not in {"send", "background_tool", "background_command"} or source["session_id"] != binding.actor.session_id:
            raise JobError("model_attempt_owner_mismatch")
        from . import session_controls
        seen = await session_controls.version(manager,tx,binding.actor.session_id)
        if session_controls.is_clean(manager,binding.actor.session_id,seen):
            review_pending = False
        elif review_sources is None or review_sources.version!=seen:
            raise ReviewSourcesChanged()
        else:
            review_pending = review_sources.pending
        await ensure_policy(binding, transaction=tx, job=source)
        if not binding.resource_evidence.get("explicit_resume"):
            await _no_unknown_sends(manager, tx, source)
        wait_deadline, wait_owners = await dispatch_policy(manager, tx, source)
        import json

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
        plan_required = source["kind"] == "send" and "input_ref" in creation_request
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

        @asynccontextmanager
        async def owned_writer():
            async with manager._write() as writer:
                await writer.require_job(identity.job_id)
                yield writer

        journal = binding.resource_evidence.get("_control_journal_opening")
        if journal is None:
            journal = ExecutionJournal(manager.store, projector=projector, writer=owned_writer)
            binding.resource_evidence["_control_journal_opening"] = journal
        from .session_ownership import current_ownership
        state = ExecutionState(identity, journal, ownership=current_ownership(),
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
    binding.resource_evidence["_control_wait_owners"] = wait_owners
    binding.resource_evidence["_control_state"] = state
    from .root_context import _authority
    binding.resource_evidence["_control_authority"] = _authority(runtime)
    async def refresh(transaction=None):
        if state.closed:
            return  # Only original observations/fees may arrive after closure.
        reader = manager.db if transaction is None else transaction
        if len(ancestors) == 1:
            rows = [await manager._row(reader, identity.job_id)]
        else:
            rows = await reader.fetchall(
                f"SELECT job_id,generation,executor_id,boot_id,cancel_requested_at,intake,resource_state "
                f"FROM {manager.jobs} WHERE job_id IN (" + ",".join("?" for _ in ancestors) + ")", tuple(ancestors))
        found = {row["job_id"]: row for row in rows}
        if len(found) != len(ancestors):
            state.needs_reconciliation = True
            raise JobError("invalid_job_lineage")
        actual = found[identity.job_id]
        if tuple(actual[k] for k in ('generation', 'executor_id', 'boot_id')) != (identity.generation, identity.executor_id, identity.boot_id):
            state.needs_reconciliation = True
            state.cancel_requested = True
        if any(row["cancel_requested_at"] is not None for row in rows) or actual["intake"] != "open":
            state.cancel_requested = True
        if actual["resource_state"] == "unknown":
            state.needs_reconciliation = True
        if state.cancel_requested:
            state.record("lifecycle", "cancel", {"requested": True})
            from .execution_budget import allocation_record
            allocation_record(binding)
    state.before_checkpoint = refresh
    manager._execution_controls[(source["job_id"], source["generation"])] = binding
    return state


def source(binding):
    current(binding)
    return binding.resource_evidence["_control_source"]


async def flush_binding(binding, *, close=False, force=True, transaction=None):
    if binding.resource_evidence.get("_control_state") is None:
        return
    state = current(binding)
    if state.closed and not retained_owner(binding):
        return  # A cold writer already advanced this sealed owner's journal.
    if transaction is not None:
        await transaction.require_job(binding.lease.job_id)
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
                                       (state.identity.job_id, state.identity.generation), state._pending_batches[-1][2])
            return
        try:
            await state.close()
        except Exception:
            # Retry only the retained checkpoint, never a model/tool action.
            await state.close()
    else:
        await state.checkpoint(force=force, transaction=transaction,
                               defer=binding.runtime.manager.after_outcome if transaction is not None else None)


def retained_owner(binding):
    return binding.runtime.manager._execution_controls.get((binding.lease.job_id, binding.lease.generation)) is binding


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
