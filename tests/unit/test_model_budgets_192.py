"""Durable original-attempt budget admission, handoff and mechanical recovery."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import LLMRequest
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.budget_contracts import (
    BudgetAck,
    BudgetConfig,
    BudgetGrant,
    BudgetPlan,
    BudgetSubmission,
    LocalBudgetScope,
)
from power_loop.runtime.jobs import JobError, JobManager, _hash
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, migrate_task_schema, verify_task_schema
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Host:
    def __init__(self):
        self.grants, self.applied, self.released = {}, {}, {}
        self.reserve_calls, self.apply_calls = 0, 0
        self.reserve_lost_ack = False
        self.apply_lost_ack = False
        self.apply_unavailable = False
        self.service = object()

    async def prepare(self, context, logical_request, *, service):
        assert service is self.service
        return BudgetPlan(100, {"attempt_id": context["attempt_id"], "origin": context})

    async def reserve(self, plan):
        self.reserve_calls += 1
        key = plan.request["attempt_id"]
        if key in self.released:
            raise JobError("budget_dispatch_closed")
        self.grants.setdefault(key, BudgetGrant(key, plan.request_hash, 1, "reserved", {"original": key}))
        if self.reserve_lost_ack:
            raise OSError("committed reservation ACK lost")
        return self.grants[key]

    async def lookup(self, plan):
        if plan.request["attempt_id"] in self.released:
            return BudgetGrant(
                plan.request["attempt_id"], plan.request_hash, 2, "released", {"dispatch_tombstone": True}
            )
        return self.grants.get(plan.request["attempt_id"])

    async def prepare_usage(self, context, grant, observation):
        return BudgetSubmission(
            context["attempt_id"], 1, {"attempt_id": context["attempt_id"], "evidence": observation}
        )

    async def apply_usage(self, submission):
        self.apply_calls += 1
        if self.apply_unavailable:
            raise OSError("usage endpoint unavailable")
        self.applied.setdefault(
            submission.attempt_id,
            BudgetAck(submission.attempt_id, submission.revision, submission.request_hash, {"recorded": True}),
        )
        if self.apply_lost_ack:
            raise OSError("committed usage ACK lost")
        return self.applied[submission.attempt_id]

    async def prepare_release(self, context, grant, proof):
        return BudgetSubmission(context["attempt_id"], 1, {"attempt_id": context["attempt_id"], **proof})

    async def release(self, submission):
        self.released.setdefault(
            submission.attempt_id,
            BudgetAck(submission.attempt_id, submission.revision, submission.request_hash, {"released": True}),
        )
        return self.released[submission.attempt_id]


@pytest.fixture
async def budget_case(registry_store, request):
    store, connect = registry_store
    manager, host = JobManager(store), Host()
    config = BudgetConfig(
        "host-budget-v1",
        (LocalBudgetScope("shared", max_calls=2, max_tokens=200, max_inflight=2),),
        host,
        max_pending=getattr(request, "param", {}).get("max_pending", 1),
        timeout_s=getattr(request, "param", {}).get("timeout_s", 10),
    )
    runtime = TaskRuntime(
        manager,
        tenant_id="budget-tenant",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=300000,
        boot_contract_hash="a" * 64,
        budget_config=config,
    )
    c = SimpleNamespace(store=store, connect=connect, manager=manager, host=host, config=config, runtime=runtime)
    await new_send(c)
    try:
        yield c
    finally:
        await runtime.shutdown(timeout_s=1)


async def new_send(c, *, runtime=None):
    runtime = runtime or c.runtime
    sid = await c.store.create_session()
    actor = runtime.actor(sid)
    reg = await c.manager.register_job(
        actor, kind="send", session_id=sid, system_key="send", request={"model_attempt_protocol": 1}
    )
    binding = await runtime.claim(actor, reg)
    await attempts.enable(binding)
    c.sid, c.actor, c.binding = sid, actor, binding
    return binding


async def start(c, *, max_tokens=20):
    token = _SEND.set(c.binding)
    try:
        return await attempts.start_call(
            SimpleNamespace(session_id=c.sid, llm=c.host.service),
            LLMRequest(messages=[{"role": "user", "content": "test"}], max_tokens=max_tokens),
            round_index=0,
            retry_index=1,
            model="test-model",
        )
    finally:
        _SEND.reset(token)


async def observed(c, attempt, *, tokens=10, error=None):
    await attempts.observe(attempt, usage={"total_tokens": tokens}, estimated=error is not None, error=error)
    return await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)


async def counters(c):
    [row] = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
    return json.loads(row["counters_json"])


async def test_admission_and_model_dispatch_commit_one_original_grant(budget_case):
    c = budget_case
    a = await start(c)
    state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert state["phase"] == "dispatch_authorized" and not state["pending"]
    assert state["context"]["prepared_hash"] == _hash(a.prepared)
    assert c.host.reserve_calls == 1
    assert await counters(c) == {"used_calls": 0, "held_calls": 1, "used_tokens": 0, "held_tokens": 100, "inflight": 1}
    [row] = await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )
    assert row["event_id"]
    with pytest.raises(JobError, match="dispatch_already_admitted"):
        await attempts.dispatch(a)
    state = await observed(c, a)
    assert state["phase"] == "settling" and not state["pending"] and state["ack"]
    assert await counters(c) == {"used_calls": 1, "held_calls": 0, "used_tokens": 10, "held_tokens": 0, "inflight": 0}


async def test_lost_reserve_ack_recalls_original_without_second_reserve(budget_case):
    c = budget_case
    c.host.reserve_lost_ack = True
    a = await start(c)
    assert c.host.reserve_calls == 1
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["grant"]["receipt"] == {
        "original": a.attempt_id
    }
    await observed(c, a)


async def test_usage_ack_loss_keeps_frozen_outbox_and_replays_without_duplicate_charge(budget_case):
    c = budget_case
    c.host.apply_lost_ack = True
    a = await start(c)
    state = await observed(c, a)
    assert state["pending"] and state["submission"] and state["ack"] is None
    frozen = state["submission"]
    assert len(c.host.applied) == 1
    await new_send(c)
    with pytest.raises(JobError, match="budget_reconciliation_required"):
        await start(c)
    c.host.apply_lost_ack = False
    await budgets.deliver(a)
    assert len(c.host.applied) == 1 and c.host.apply_calls == 2
    # An old caller can finish bookkeeping; it cannot change the frozen request.
    original = await budgets.get_budget(c.manager, a.binding.actor, a.attempt_id)
    assert original["submission"] == frozen and not original["pending"]


async def test_scope_counters_survive_new_sends_and_new_runtime_objects(budget_case):
    c = budget_case
    await observed(c, await start(c))
    await new_send(c)
    await observed(c, await start(c))
    replacement = TaskRuntime(
        c.manager,
        tenant_id="budget-tenant",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=300000,
        boot_contract_hash="a" * 64,
        budget_config=c.config,
    )
    try:
        await new_send(c, runtime=replacement)
        with pytest.raises(JobError, match="budget_max_calls_exhausted"):
            await start(c)
        assert c.host.reserve_calls == 2 and (await counters(c))["used_calls"] == 2
    finally:
        await replacement.shutdown(timeout_s=1)


@pytest.mark.parametrize("fault", ["missing", "changed"])
async def test_missing_or_changed_config_cannot_downgrade_persisted_policy(budget_case, fault):
    c = budget_case
    c.runtime.budget_config = None if fault == "missing" else replace(c.config, provider_key="different-binding")
    with pytest.raises(JobError, match="budget_configuration_changed"):
        await start(c)
    assert c.host.reserve_calls == 0


@pytest.mark.parametrize("field", ["policy", "scope", "counter"])
async def test_missing_or_corrupt_control_rows_do_not_recreate_an_allowance(budget_case, field):
    c = budget_case
    if field == "policy":
        await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}task_budget_policies")
    elif field == "scope":
        await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}task_budget_scopes")
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_scopes SET counters_json='{{}}'")
    with pytest.raises(JobError, match="budget_"):
        await attempts.enable(c.binding)
    assert c.host.reserve_calls == 0


async def test_cancel_after_host_grant_never_authorizes_provider_and_can_release_unused(budget_case):
    c = budget_case
    original = c.host.reserve

    async def cancel_after_grant(plan):
        value = await original(plan)
        await c.manager.request_cancel(
            c.actor, c.binding.lease.job_id, system_key="stop-before-dispatch", reason="before paid work"
        )
        return value

    c.host.reserve = cancel_after_grant
    with pytest.raises(JobError, match="cancel_requested"):
        await start(c)
    [row] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"
    )
    prepared = json.loads(row["payload_json"])
    a = attempts.Attempt(c.binding, prepared["attempt_id"], prepared)
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )
    proof = await budgets.close_unstarted(a)
    assert proof and (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["phase"] == "released"
    assert all(value == 0 for value in (await counters(c)).values())
    assert await budgets.close_unstarted(a) == proof and len(c.host.released) == 1


async def test_dispatched_unknown_preserves_token_concurrency_and_api_risk(budget_case):
    c = budget_case
    a = await start(c)
    state = await observed(c, a, tokens=3, error=asyncio.CancelledError())
    assert not state["pending"] and state["observation"]["status"] == "unknown"
    assert (await counters(c))["held_tokens"] == 100 and (await counters(c))["inflight"] == 1
    with pytest.raises(JobError, match="budget_release_not_permitted"):
        await budgets.close_unstarted(a)


async def test_actual_over_bound_is_preserved_and_closes_new_admission(budget_case):
    c = budget_case
    state = await observed(c, await start(c), tokens=101)
    assert not state["pending"] and (await counters(c))["used_tokens"] == 101
    await new_send(c)
    with pytest.raises(JobError, match="budget_policy_closed"):
        await start(c)


@pytest.mark.parametrize(
    "usage,estimated,total",
    [
        ({"prompt_tokens": 7, "completion_tokens": 3}, False, 10),
        ({"prompt_tokens": 0, "completion_tokens": 0}, False, 0),
        ({"prompt_tokens": 7, "completion_tokens": 3, "cached_tokens": 4, "reasoning_tokens": 2}, False, 10),
        ({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 12}, False, 12),
        ({"prompt_tokens": 7}, False, None),
        ({"completion_tokens": 3}, False, None),
        ({"cached_tokens": 4, "reasoning_tokens": 2}, False, None),
        ({"prompt_tokens": 7, "completion_tokens": 3}, True, None),
    ],
)
async def test_actual_component_totals_match_host_settlement_without_counting_token_subsets(
    budget_case, usage, estimated, total
):
    c = budget_case
    a = await start(c)
    await attempts.observe(a, usage=usage, estimated=estimated)
    current = await counters(c)
    assert current["used_tokens"] == (0 if total is None else total)
    assert current["held_tokens"] == (100 if total is None else 0)
    assert current["used_calls"] == 1 and current["held_calls"] == 0 and current["inflight"] == 0
    state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert not state["pending"] and state["observation"]["usage"] == usage
    await budgets.deliver(a)
    assert await counters(c) == current and len(c.host.applied) == 1


async def test_actual_component_over_bound_closes_policy_and_preserves_the_whole_charge(budget_case):
    c = budget_case
    a = await start(c)
    await attempts.observe(a, usage={"prompt_tokens": 80, "completion_tokens": 21}, estimated=False)
    assert (await counters(c))["used_tokens"] == 101
    await new_send(c)
    with pytest.raises(JobError, match="budget_policy_closed"):
        await start(c)
    assert c.host.reserve_calls == 1


@pytest.mark.parametrize("invalid", [2, -1])
async def test_invalid_pending_projection_is_rejected_by_read_and_admission_without_host_work(budget_case, invalid):
    c = budget_case
    c.host.apply_unavailable = True
    a = await start(c)
    assert (await observed(c, a))["pending"]
    before = await counters(c)
    await c.manager.db.execute(
        f"UPDATE {c.manager.prefix}task_budget_attempts SET pending=? WHERE attempt_id=?", (invalid, a.attempt_id)
    )
    with pytest.raises(JobError, match="budget_evidence_corrupt"):
        await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    await new_send(c)
    with pytest.raises(JobError, match="budget_evidence_corrupt"):
        await start(c)
    assert c.host.reserve_calls == 1 and await counters(c) == before


@pytest.mark.parametrize("invalid", [2, -1])
async def test_invalid_policy_enabled_projection_cannot_enable_paid_work(budget_case, invalid):
    c = budget_case
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_policies SET enabled=?", (invalid,))
    with pytest.raises(JobError, match="budget_policy_evidence_corrupt"):
        await start(c)
    assert c.host.reserve_calls == 0 and all(v == 0 for v in (await counters(c)).values())


@pytest.mark.parametrize("failure", [False, True])
async def test_explicit_protocol_three_migration_retains_original_ids_and_rejects_old_writers(
    budget_case, monkeypatch, failure
):
    c = budget_case
    seq = await c.store.append_message(c.sid, role="user", content="retained")
    ref = await c.store.message_ref(c.sid, seq)
    for name in ("task_budget_policies", "task_budget_scopes", "task_budget_attempts"):
        await c.manager.db.execute(f"DROP TABLE {c.manager.prefix}{name}")
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=3 WHERE id=1")
    with pytest.raises(StoreSchemaError, match="incomplete|unsupported"):
        await verify_task_schema(c.manager.db, c.manager.prefix)
    with pytest.raises(JobError, match="store_identity_changed"):
        async with c.manager._write():
            pass
    with pytest.raises(StoreSchemaError, match="maintenance_required"):
        await migrate_task_schema(c.manager.db, c.manager.prefix)
    if failure:
        from power_loop.runtime.store import task_schema

        original = task_schema._run_task_ddl

        async def broken(*args):
            await original(*args)
            raise RuntimeError("budget migration failure")

        with monkeypatch.context() as patch:
            patch.setattr(task_schema, "_run_task_ddl", broken)
            with pytest.raises(RuntimeError, match="budget migration failure"):
                await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
        assert (await c.manager.db.fetchone(f"SELECT protocol_version FROM {c.manager.prefix}task_store WHERE id=1"))[
            "protocol_version"
        ] == 3
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    assert TASK_PROTOCOL_VERSION == 7 and await c.store.message_ref(c.sid, seq) == ref
    # A bridge compiled against protocol3 cannot write the migrated control row.
    import power_loop.runtime.jobs as jobs

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "TASK_PROTOCOL_VERSION", 3)
        with pytest.raises(JobError, match="store_identity_changed"):
            async with c.manager._write():
                pass


@pytest.mark.parametrize("budget_case", [{"max_pending": 10}], indirect=True)
async def test_two_independent_requests_can_reserve_concurrently_without_oversubscription(budget_case):
    c = budget_case
    first = SimpleNamespace(**vars(c))
    await new_send(c)
    second = SimpleNamespace(**vars(c))
    entered, release = asyncio.Event(), asyncio.Event()
    original, arrivals = c.host.reserve, []

    async def held(plan):
        arrivals.append(plan.request["attempt_id"])
        if len(arrivals) == 2:
            entered.set()
        await release.wait()
        return await original(plan)

    c.host.reserve = held
    tasks = [asyncio.create_task(start(case)) for case in (first, second)]
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert (await counters(c))["held_calls"] == 2 and (await counters(c))["inflight"] == 2
        release.set()
        values = await asyncio.gather(*tasks)
        for case, attempt in zip((first, second), values, strict=True):
            await observed(case, attempt)
        assert (await counters(c))["used_calls"] == 2 and (await counters(c))["used_tokens"] == 20
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_unconfirmed_reservation_closes_dispatch_then_uses_host_tombstone_proof(budget_case):
    c = budget_case

    async def unavailable(plan):
        raise OSError("reserve endpoint unavailable")

    c.host.reserve = unavailable
    with pytest.raises(JobError, match="budget_host_reservation_unconfirmed"):
        await start(c)
    [raw] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"
    )
    prepared = json.loads(raw["payload_json"])
    original = attempts.Attempt(c.binding, prepared["attempt_id"], prepared)
    proof = await budgets.close_unstarted(original)
    assert proof and (await counters(c))["held_tokens"] == 0
    assert (await budgets.get_budget(c.manager, c.actor, original.attempt_id))["phase"] == "released"
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )


@pytest.mark.parametrize("budget_case", [{"timeout_s": 0.1}], indirect=True)
async def test_timed_out_api_task_is_retained_and_late_grant_only_allows_mechanical_release(budget_case):
    c = budget_case
    release, original = asyncio.Event(), c.host.reserve

    async def held(plan):
        await release.wait()
        return await original(plan)

    c.host.reserve = held
    with pytest.raises(JobError, match="budget_host_reservation_unconfirmed"):
        await start(c)
    assert len(c.runtime._budget_handles) == 1
    [task] = c.runtime._budget_handles.values()
    assert not task.done() and not task.cancelling()
    release.set()
    await task
    [raw] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"
    )
    prepared = json.loads(raw["payload_json"])
    original = attempts.Attempt(c.binding, prepared["attempt_id"], prepared)
    await budgets.close_unstarted(original)
    assert (await budgets.get_budget(c.manager, c.actor, original.attempt_id))["phase"] == "released"
    assert all(value == 0 for value in (await counters(c)).values())
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )


@pytest.mark.parametrize("field,value", [("attempt_id", "foreign"), ("request_hash", "c" * 64), ("state", "released")])
async def test_wrong_api_grant_is_not_authority_for_a_model_call(budget_case, field, value):
    c = budget_case
    original = c.host.reserve

    async def wrong(plan):
        return replace(await original(plan), **{field: value})

    c.host.reserve = wrong
    with pytest.raises(JobError, match="budget_host_reservation_unconfirmed"):
        await start(c)
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )


async def test_wrong_usage_ack_cannot_clear_a_frozen_outbox_or_apply_local_tokens_twice(budget_case):
    c = budget_case
    original = c.host.apply_usage

    async def wrong(submission):
        return replace(await original(submission), request_hash="d" * 64)

    c.host.apply_usage = wrong
    a = await start(c)
    state = await observed(c, a)
    assert state["pending"] and state["ack"] is None
    c.host.apply_usage = original
    await budgets.deliver(a)
    assert not (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["pending"]
    assert (await counters(c))["used_tokens"] == 10 and (await counters(c))["used_calls"] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"api_key": "forbidden"},
        {"prompt": "do not retain"},
        {"value": 1.1},
        {"value": float("nan")},
        {"value": object()},
    ],
)
async def test_host_payloads_cannot_store_credentials_model_bodies_or_inexact_numbers(body):
    with pytest.raises(JobError, match="invalid_budget_payload"):
        BudgetPlan(100, body)


async def test_duplicate_parallel_close_returns_one_original_release_and_never_refunds_twice(budget_case):
    c = budget_case
    a = await attempts.prepare(
        c.binding,
        nonce="unstarted",
        round_index=0,
        retry_index=1,
        request_hash="a" * 64,
        model="test-model",
        max_tokens=20,
    )
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    replies = await asyncio.gather(*(budgets.close_unstarted(a) for _ in range(8)))
    assert all(r == replies[0] for r in replies) and len(c.host.released) == 1
    assert all(v == 0 for v in (await counters(c)).values())
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["phase"] == "released"


@pytest.mark.parametrize("name", ["generate_image", "bash", "jobs", "workflow"])
async def test_unadapted_or_name_spoofed_tool_is_rejected_before_its_side_effect(budget_case, name):
    from power_loop import ToolDefinition
    from power_loop.runtime.task_runtime import check_tool_dispatch
    from power_loop.tools.registry import ToolRegistry

    c = budget_case
    called = []

    async def unadapted():
        called.append(True)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name=name, description="unadapted external effect", input_schema={"type": "object"}), unadapted
    )
    token = _SEND.set(c.binding)
    try:
        with pytest.raises(JobError, match="budget_tool_adapter_unavailable"):
            await check_tool_dispatch(c.sid, name, registry, {})
        assert called == [] and c.host.reserve_calls == 0
    finally:
        _SEND.reset(token)


async def test_original_builtin_pass_is_free_and_does_not_require_another_paid_request(budget_case):
    from power_loop.core.pass_turn import ensure_pass_turn
    from power_loop.runtime.task_runtime import check_tool_dispatch
    from power_loop.tools.registry import ToolRegistry

    c = budget_case
    registry = ToolRegistry()
    ensure_pass_turn(registry)
    token = _SEND.set(c.binding)
    try:
        await check_tool_dispatch(c.sid, "pass_turn", registry, {"reason": "waiting"})
        assert c.host.reserve_calls == 0
    finally:
        _SEND.reset(token)


@pytest.mark.parametrize(
    "table,column",
    [
        ("task_budget_policies", "request_hash"),
        ("task_budget_scopes", "counters_hash"),
        ("task_budget_attempts", "pending"),
    ],
)
async def test_budget_schema_rejects_a_missing_required_control_column(budget_case, table, column):
    c = budget_case
    if table == "task_budget_attempts" and column == "pending":
        suffix = f" ON {c.manager.prefix}{table}" if c.manager.db.dialect.name == "mysql" else ""
        await c.manager.db.execute(f"DROP INDEX {c.manager.prefix}idx_budget_pending{suffix}")
    await c.manager.db.execute(f"ALTER TABLE {c.manager.prefix}{table} DROP COLUMN {column}")
    with pytest.raises(StoreSchemaError, match="task budget schema incomplete"):
        await verify_task_schema(c.manager.db, c.manager.prefix)
