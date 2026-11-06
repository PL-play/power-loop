"""One-time import of protocol 21 accounting into closed execution journals.

Only the explicit maintenance migration calls this module. It never grants an
execution or changes a shared balance, and never infers an external outcome.
"""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from .. import budget_scopes, model_budgets
from ..control_runtime import project_events
from ..execution_journal import ExecutionJournal
from ..execution_state import ExecutionIdentity, ExecutionState
from ..jobs import JobError, JobManager, _hash, _json
from .schema import StoreSchemaError


def _booked(state):
    result = dict.fromkeys(budget_scopes.COUNTERS, 0)
    if state["phase"] == "released" or state.get("plan") is None:
        return result
    bound = state["plan"]["token_bound"]
    observation = state.get("observation")
    revision = state.get("usage_revision")
    recorded = observation is not None or state.get("initial_usage_evidence") is not None
    tokens = None
    if revision:
        from ..budget_revisions import total
        evidence = revision["evidence"]
        tokens = total(evidence["token_quality"], evidence["usage"])
    elif observation and not observation["estimated"]:
        usage = observation["usage"]
        tokens = usage.get("total_tokens")
        if tokens is None and "prompt_tokens" in usage and "completion_tokens" in usage:
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    result.update(used_calls=int(recorded), held_calls=int(not recorded),
                  used_tokens=tokens or 0, held_tokens=bound if tokens is None else 0,
                  inflight=int(not observation or observation["status"] != "returned"))
    return result


async def import_accounting(tx, db, prefix):
    """Validate every old balance, then import without refunding or charging."""
    import json
    row = await tx.fetchone(f"SELECT store_id FROM {prefix}task_store WHERE id=1")
    from .task_schema import TASK_SCHEMA_VERSION
    store = SimpleNamespace(store_id=row["store_id"], schema_version=TASK_SCHEMA_VERSION, _db=db, table_prefix=prefix)
    manager = JobManager(store)
    expected, groups = {}, {}
    rows = await tx.fetchall(f"SELECT * FROM {prefix}task_budget_attempts ORDER BY job_id,attempt_id")
    policies = {}
    try:
        for row in rows:
            state = await model_budgets._read(manager, tx, row["attempt_id"])
            if state.get("allocation_id") is not None:
                raise JobError("execution_migration_source_protocol_mismatch")
            policy_key = state["policy_key"]
            if policy_key not in policies:
                policy = await tx.fetchone(f"SELECT * FROM {prefix}task_budget_policies WHERE policy_key=?", (policy_key,))
                if policy is None:
                    raise JobError("budget_policy_missing")
                policies[policy_key] = {**policy, "manifest": model_budgets._object(policy["manifest_json"], policy["request_hash"])}
            scopes = ([] if state.get("plan") is None and state["phase"] == "released"
                      else budget_scopes.frozen(policies[policy_key], state))
            booked = _booked(state)
            for scope in scopes:
                key = budget_scopes.scope_key(policy_key, scope)
                total = expected.setdefault(key, dict.fromkeys(budget_scopes.COUNTERS, 0))
                for field, value in booked.items():
                    total[field] += value
            context = state["context"]
            original = await tx.fetchone(f"SELECT payload_json,request_hash FROM {prefix}job_events WHERE event_id=?",
                                        (context["prepared_event_id"],))
            if original is None:
                raise JobError("execution_migration_original_missing")
            prepared = model_budgets._object(original["payload_json"], original["request_hash"])
            if _hash(prepared) != context["prepared_hash"]:
                raise JobError("execution_migration_original_changed")
            scope = prepared["scope"]
            identity = ExecutionIdentity(store.store_id, state["job_id"], context["generation"],
                scope["executor_id"], scope["boot_id"], context["tenant_id"], context["principal_id"], context["session_id"])
            if scope["job_id"] != identity.job_id or scope["generation"] != identity.generation:
                raise JobError("execution_migration_original_changed")
            group = groups.setdefault(identity, [])
            group.append((state, scopes, booked))
        actual = await tx.fetchall(f"SELECT * FROM {prefix}task_budget_scopes")
        for row in actual:
            found = model_budgets._object(row["counters_json"], row["counters_hash"])
            if found != expected.pop(row["scope_key"], dict.fromkeys(budget_scopes.COUNTERS, 0)):
                raise JobError("execution_migration_balance_mismatch")
        if expected:
            raise JobError("budget_scope_missing_or_changed")
        for identity, attempts in groups.items():
            async def projector(tx, identity, records):
                await project_events(manager, tx, identity, records)
            journal = ExecutionJournal(store, projector=projector)
            state = ExecutionState(identity, journal)
            state.closed = True
            records = []
            for original, scopes, booked in attempts:
                allocation_id = "allocation_" + _hash(["protocol21", identity.digest, original["attempt_id"]])[:48]
                allocation = dict(identity=asdict(identity), policy_key=original["policy_key"], scopes=scopes,
                    calls=booked["used_calls"] + booked["held_calls"], tokens=booked["used_tokens"] + booked["held_tokens"],
                    max_inflight=booked["inflight"], used_calls=booked["used_calls"], used_tokens=booked["used_tokens"],
                    closed=True, report_version=1, booked=booked)
                await tx.execute(f"INSERT INTO {prefix}execution_allocations "
                    "(allocation_id,job_id,generation,state_json,state_hash) VALUES (?,?,?,?,?)",
                    (allocation_id, identity.job_id, identity.generation, _json(allocation), _hash(allocation)))
                records.append(dict(kind="allocation_adjustment_immutable", key=allocation_id + ":1",
                    value=dict(allocation_id=allocation_id, state=allocation)))
                value = {**original, "version": original["version"] + 1, "allocation_id": allocation_id}
                state.record("budget", original["attempt_id"], value)
                records.append(dict(kind="budget", key=original["attempt_id"], value=value))
                state.needs_reconciliation |= bool(booked["held_calls"] or booked["held_tokens"] or booked["inflight"] or value["pending"])
            records.append(dict(kind="lifecycle", key="protocol21_import", value=dict(
                source_protocol=21, closed=True, execution_authority=False, replay_allowed=False, capacity_released=False)))
            summary = state.summary()
            await tx.execute(f"INSERT INTO {prefix}execution_checkpoints "
                "(job_id,generation,identity_json,identity_hash,admission_nonce,batch_no,chain_hash,summary_json,summary_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (identity.job_id, identity.generation, _json(asdict(identity)), identity.digest,
                journal.admission_nonce, 0, identity.digest, _json(summary), _hash(summary)))
            await journal.append(identity, 1, identity.digest, records, summary, transaction=tx)
        # Old owners were stopped by the maintenance operator. A remaining
        # running row proves neither completion nor safe replay after restart.
        await tx.execute(f"UPDATE {prefix}jobs SET intake='closed',resource_state='unknown',version=version+1 "
                         "WHERE resource_state='running'")
    except (JobError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StoreSchemaError("execution migration requires reconciliation: " + str(exc)) from exc
