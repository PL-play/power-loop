"""Atomic original-attempt revisions; provider I/O happens only after staging.

Competing snapshots are explicitly rejected, never acknowledged and discarded.
The trusted collector retains its source record until staging is confirmed.
"""
from __future__ import annotations

from dataclasses import asdict

from . import budget_scopes
from . import model_budgets as budgets
from .budget_contracts import (
    BudgetAck,
    BudgetGrant,
    BudgetRevisionProvider,
    BudgetSubmission,
    BudgetUsageEvidence,
    uint,
)
from .jobs import JobError, _hash, _json


def total(quality, usage):
    if quality != "actual":
        return None
    if "total_tokens" in usage:
        return usage["total_tokens"]
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        return usage["prompt_tokens"] + usage["completion_tokens"]
    return None


def previous_fact(state):
    revision = state.get("usage_revision")
    if revision is not None:
        value = BudgetUsageEvidence(**revision["evidence"])
        return value.token_quality, value.usage, value.fingerprint
    observation = state["observation"]
    quality = "unknown" if not observation["usage"] else "estimated" if observation["estimated"] else "actual"
    return quality, observation["usage"], _hash(observation)


def validate_transition(state, evidence):
    quality, usage, previous_hash = previous_fact(state)
    priority = {"unknown": 0, "estimated": 1, "actual": 2}
    if priority[evidence.token_quality] < priority[quality]:
        raise JobError("budget_evidence_downgrade")
    old_total, new_total = total(quality, usage), total(evidence.token_quality, evidence.usage)
    if quality == "actual" and any(k not in evidence.usage for k in usage):
        raise JobError("budget_token_fact_missing")
    changed = quality == "actual" and (any(evidence.usage[k] != v for k, v in usage.items())
        or old_total is not None and old_total != new_total)
    if evidence.correction_of is not None and evidence.correction_of != previous_hash:
        raise JobError("budget_correction_source_mismatch")
    if changed and evidence.correction_of != previous_hash:
        raise JobError("budget_actual_correction_required")
    if new_total is not None and not uint(new_total):
        raise JobError("invalid_budget_usage_evidence")
    return old_total, new_total


async def stage(runtime, actor, job_id, attempt_id, *, expected_revision, evidence):
    if not uint(expected_revision) or not 0 < expected_revision < 2**63 - 1:
        raise JobError("invalid_budget_revision")
    if type(evidence) is not BudgetUsageEvidence:
        raise JobError("invalid_budget_usage_evidence")
    evidence = BudgetUsageEvidence(**asdict(evidence))  # Copy/validate nested mutable data.
    await budgets._reconciliation_state(runtime, actor, job_id, attempt_id)
    if (not isinstance(runtime.budget_config.provider, BudgetRevisionProvider)
            or not callable(runtime.budget_config.provider.prepare_usage_revision)):
        raise JobError("budget_revision_provider_required")
    manager = runtime.manager
    event_id = "br_" + _hash([manager.store.store_id, attempt_id, evidence.evidence_id])[:48]
    request = {"attempt_id": attempt_id, "job_id": job_id,
               "expected_revision": expected_revision, "evidence": asdict(evidence)}
    async with manager._write() as tx:
        runtime._lifecycle.check_admission()  # Closed local hosts cannot acquire new control work.
        policy = await budgets._policy(runtime, tx)
        job = await manager._authorize(tx, actor, job_id)
        state = await budgets._read(manager, tx, attempt_id)
        if state["job_id"] != job_id or state["policy_key"] != policy["policy_key"]:
            raise JobError("budget_reconciliation_scope_mismatch")
        if state.get("usage_revision") is not None:
            await read_revision(manager, tx, state)
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
        if old is not None:
            saved = budgets._object(old["payload_json"], old["request_hash"])
            if (old["job_id"] != job_id or old["event_type"] != "budget_usage_revision"
                    or saved["request"] != request):
                raise JobError("budget_revision_evidence_conflict")
            return saved["receipt"]
        if state["phase"] != "settling" or state["pending"] or state["ack"] is None:
            raise JobError("budget_revision_pending")
        previous_submission = BudgetSubmission(**state["submission"])
        previous_ack = BudgetAck(**state["ack"])
        if (previous_submission.revision != expected_revision or
                (previous_ack.attempt_id, previous_ack.revision, previous_ack.request_hash) !=
                (attempt_id, expected_revision, previous_submission.request_hash)):
            raise JobError("budget_revision_changed")
        old_total, new_total = validate_transition(state, evidence)
        for limits in budget_scopes.frozen(policy, state):
            row, counters = await budgets._scope(manager, tx, state["policy_key"], limits)
            if new_total is not None:
                if old_total is None:
                    counters["held_tokens"] -= state["plan"]["token_bound"]
                counters["used_tokens"] += new_total - (old_total or 0)
                await budgets._save_counters(manager, tx, row, counters)
        if new_total is not None and new_total > state["plan"]["token_bound"]:
            await tx.execute(f"UPDATE {manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?",
                             (state["policy_key"],))
        revision = expected_revision + 1
        receipt = {"attempt_id": attempt_id, "job_id": job_id, "revision": revision,
                   "event_id": event_id, "evidence_hash": evidence.fingerprint, "execution_authority": False}
        snapshot = {"request": request, "receipt": receipt,
                    "previous_submission": asdict(previous_submission), "previous_ack": asdict(previous_ack)}
        await tx.execute(f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)", (event_id, job_id, "budget_usage_revision", job["version"],
                _hash(snapshot), _json(snapshot), "not_applicable", manager.now()))
        state.update(usage_revision={"event_id": event_id, "revision": revision, "evidence": asdict(evidence)},
                     submission=None, ack=None, pending=True)
        await budgets._save(manager, tx, state, job)
        return receipt


async def read_revision(manager, tx, state):
    try:
        revision = state["usage_revision"]
        row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (revision["event_id"],))
        if (row is None or row["event_type"] != "budget_usage_revision" or row["job_id"] != state["job_id"]
                or row["delivery_status"] != "not_applicable"):
            raise JobError("budget_revision_evidence_missing")
        saved = budgets._object(row["payload_json"], row["request_hash"])
        evidence = BudgetUsageEvidence(**revision["evidence"])
        previous = BudgetSubmission(**saved["previous_submission"])
        ack = BudgetAck(**saved["previous_ack"])
        if (saved["request"] != {"attempt_id": state["attempt_id"], "job_id": state["job_id"],
                "expected_revision": revision["revision"] - 1, "evidence": asdict(evidence)}
                or saved["receipt"] != {"attempt_id": state["attempt_id"], "job_id": state["job_id"],
                    "revision": revision["revision"], "event_id": revision["event_id"],
                    "evidence_hash": evidence.fingerprint, "execution_authority": False}
                or (previous.attempt_id, previous.revision) != (state["attempt_id"], revision["revision"] - 1)
                or (ack.attempt_id, ack.revision, ack.request_hash) !=
                    (previous.attempt_id, previous.revision, previous.request_hash)):
            raise JobError("budget_revision_evidence_conflict")
        return saved, evidence
    except (TypeError, KeyError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("budget_revision_evidence_corrupt") from exc


async def prepare(runtime, actor, state):
    manager, revision = runtime.manager, state["usage_revision"]
    if (not isinstance(runtime.budget_config.provider, BudgetRevisionProvider)
            or not callable(runtime.budget_config.provider.prepare_usage_revision)):
        raise JobError("budget_revision_provider_required")
    async with manager._write() as tx:
        await manager._authorize(tx, actor, state["job_id"])
        saved, evidence = await read_revision(manager, tx, state)
    return await budgets._host_call(runtime, state["job_id"], state["attempt_id"],
        "prepare_usage_revision:" + str(revision["revision"]),
        lambda: runtime.budget_config.provider.prepare_usage_revision(
            dict(state["context"]), BudgetGrant(**state["grant"]),
            BudgetSubmission(**saved["previous_submission"]), BudgetAck(**saved["previous_ack"]), evidence))
