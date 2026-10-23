"""Only original immutable non-dispatch evidence can close a rejected attempt."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import LLMRequest
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobError
from power_loop.runtime.model_attempt_query import list_attempts

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, new_send, observed, start

__all__ = ["budget_case", "registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def prepared(c):
    return await attempts.prepare(
        c.binding,
        nonce="unstarted",
        round_index=0,
        retry_index=1,
        request_hash="a" * 64,
        model="test-model",
        max_tokens=20,
    )


async def test_another_valid_send_lease_cannot_close_the_original_jobs_reservation(budget_case):
    c = budget_case
    original = await prepared(c)
    await budgets.reserve(original, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    before = await budgets.get_budget(c.manager, original.binding.actor, original.attempt_id)
    await new_send(c)
    wrong = replace(original, binding=c.binding)
    with pytest.raises(JobError) as error:
        await budgets.close_unstarted(wrong)
    assert str(error.value) == "budget_release_requires_original_owner" and not c.host.released, (
        str(error.value),
        c.host.released,
    )
    assert await budgets.get_budget(c.manager, original.binding.actor, original.attempt_id) == before
    assert not c.host.released
    await budgets.close_unstarted(original)


async def test_prepare_failure_is_closed_locally_and_the_same_send_can_admit_its_first_real_call(budget_case):
    c = budget_case
    prepare_host = c.host.prepare

    async def unavailable(*args, **kwargs):
        raise OSError("test metadata preparation failed")

    c.host.prepare = unavailable
    with pytest.raises(OSError, match="metadata preparation failed"):
        await start(c)
    [event] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"
    )
    original = json.loads(event["payload_json"])
    state = await budgets.get_budget(c.manager, c.actor, original["attempt_id"])
    assert state["phase"] == "released" and not state["pending"]
    assert state["ack"] == {"local_only": True, "dispatch_not_authorized": True, "no_reservation": True}
    assert state["plan"] is None and state["plan_hash"] is None and not c.host.released
    [item] = (await list_attempts(c.manager, c.actor, c.binding.lease.job_id))["items"]
    assert item["status"] == "not_dispatched" and item["observed_usage"] is None and not item["response_received"]
    assert c.host.reserve_calls == 0
    c.host.prepare = prepare_host
    await observed(c, await start(c))
    assert c.host.reserve_calls == 1 and len(c.host.applied) == 1


async def test_exhausted_original_allowance_stays_exhausted_after_repeated_local_rejections(budget_case):
    c = budget_case
    await observed(c, await start(c))
    await observed(c, await start(c))
    for _ in range(3):
        with pytest.raises(JobError, match="budget_max_calls_exhausted"):
            await start(c)
    assert c.host.reserve_calls == len(c.host.applied) == 2
    [scope] = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
    assert json.loads(scope["counters_json"])["used_calls"] == 2
    assert not await c.manager.db.fetchall(
        f"SELECT attempt_id FROM {c.manager.prefix}task_budget_attempts WHERE pending<>0"
    )


async def test_deleted_reservation_row_is_not_proof_that_no_reservation_ever_existed(budget_case):
    c = budget_case
    attempt = await prepared(c)
    await budgets.reserve(attempt, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    await c.manager.db.execute(
        f"DELETE FROM {c.manager.prefix}task_budget_attempts WHERE attempt_id=?", (attempt.attempt_id,)
    )
    with pytest.raises(JobError, match="budget_attempt_missing"):
        await budgets.close_unreserved(attempt)
    with pytest.raises(JobError, match="budget_attempt_missing"):
        await list_attempts(c.manager, c.actor, c.binding.lease.job_id)
    assert c.host.reserve_calls == 1 and not c.host.released


@pytest.mark.parametrize("fault", ["prepared", "owner", "generation", "executor_id", "boot_id"])
async def test_local_closure_requires_exact_original_preparation_and_execution_identity(budget_case, fault):
    c = budget_case
    attempt = await prepared(c)
    if fault == "prepared":
        wrong = replace(attempt, prepared={**attempt.prepared, "model": "replacement"})
    elif fault == "owner":
        await new_send(c)
        wrong = replace(attempt, binding=c.binding)
    else:
        values = {fault: attempt.binding.lease.generation + 1 if fault == "generation" else "other"}
        lease = replace(attempt.binding.lease, **values)
        wrong = replace(attempt, binding=replace(attempt.binding, lease=lease))
    with pytest.raises(JobError, match="budget_release_requires_original_owner"):
        await budgets.close_unreserved(wrong)
    assert not await c.manager.db.fetchall(f"SELECT attempt_id FROM {c.manager.prefix}task_budget_attempts")


async def test_dispatched_unknown_request_cannot_be_reclassified_as_local_only(budget_case):
    c = budget_case
    attempt = await start(c)
    await attempts.observe(attempt, usage={}, estimated=True, error=OSError("unknown remote outcome"))
    assert await budgets.close_unreserved(attempt) is None
    with pytest.raises(JobError, match="budget_release_not_permitted"):
        await budgets.close_unstarted(attempt)
    with pytest.raises(JobError, match="unresolved_prior_attempt"):
        await start(c)
    assert c.host.reserve_calls == 1 and not c.host.released


async def test_repeated_closure_has_one_proof_and_the_original_attempt_can_never_dispatch(budget_case):
    c = budget_case
    attempt = await prepared(c)
    results = await asyncio.gather(*(budgets.close_unreserved(attempt) for _ in range(6)))
    assert all(r == results[0] for r in results)
    assert (
        len(
            await c.manager.db.fetchall(
                f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_budget_state'"
            )
        )
        == 1
    )
    with pytest.raises(JobError, match="budget_dispatch_not_authorized"):
        await attempts.dispatch(attempt)
    with pytest.raises(JobError, match="budget_attempt_already_reserved"):
        await budgets.reserve(attempt, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    assert c.host.reserve_calls == 0 and not c.host.applied and not c.host.released


async def test_reservation_winning_the_race_cannot_be_closed_without_original_host_release(budget_case):
    c = budget_case
    attempt = await prepared(c)
    await budgets.reserve(attempt, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    assert await budgets.close_unreserved(attempt) is None
    state = await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    assert state["phase"] == "host_reserved" and not c.host.released
    await attempts.dispatch(attempt)
    await observed(c, attempt)
    assert c.host.reserve_calls == 1 and len(c.host.applied) == 1


async def test_closure_write_failure_rolls_back_and_preserves_original_error_and_unresolved_gate(
    budget_case, monkeypatch
):
    c = budget_case
    original_prepare, save = c.host.prepare, budgets._save

    async def unavailable(*args, **kwargs):
        raise OSError("original preparation unavailable")

    async def rollback(*args, **kwargs):
        await save(*args, **kwargs)
        raise RuntimeError("test proof transaction rollback")

    c.host.prepare = unavailable
    monkeypatch.setattr(budgets, "_save", rollback)
    with pytest.raises(OSError, match="original preparation unavailable"):
        await start(c)
    assert not await c.manager.db.fetchall(f"SELECT attempt_id FROM {c.manager.prefix}task_budget_attempts")
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_budget_state'"
    )
    assert c.binding.resource_evidence["budget_pending"]
    monkeypatch.setattr(budgets, "_save", save)
    c.host.prepare = original_prepare
    with pytest.raises(JobError, match="unresolved_prior_attempt"):
        await start(c)
    assert c.host.reserve_calls == 0


async def test_no_reservation_proof_is_allowed_after_original_cancellation_without_any_refund(budget_case):
    c = budget_case
    attempt = await prepared(c)
    await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop-before-reserving", reason="test")
    proof = await budgets.close_unstarted(attempt)
    assert proof["no_reservation"] and not c.host.released and c.host.reserve_calls == 0
    with pytest.raises(JobError, match="cancel_requested"):
        await attempts.dispatch(attempt)


async def test_local_closure_can_win_while_metadata_preparation_is_still_inflight(budget_case):
    c = budget_case
    attempt = await prepared(c)
    entered, release = asyncio.Event(), asyncio.Event()
    original = c.host.prepare

    async def waiting(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    c.host.prepare = waiting
    reservation = asyncio.create_task(
        budgets.reserve(attempt, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert (await budgets.close_unreserved(attempt))["no_reservation"]
        release.set()
        with pytest.raises(JobError, match="budget_attempt_already_reserved"):
            await reservation
        assert c.host.reserve_calls == 0 and not c.host.released
    finally:
        release.set()
        await asyncio.gather(reservation, return_exceptions=True)


async def test_original_dispatch_and_host_release_cannot_both_win(budget_case):
    c = budget_case
    attempt = await prepared(c)
    await budgets.reserve(attempt, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    closed, dispatched = await asyncio.gather(
        budgets.close_unstarted(attempt), attempts.dispatch(attempt), return_exceptions=True
    )
    actual = await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )
    if actual:
        assert len(actual) == 1 and dispatched is None
        assert isinstance(closed, JobError) and str(closed) == "budget_release_not_permitted"
        assert not c.host.released
        await observed(c, attempt)
    else:
        assert not isinstance(closed, BaseException) and isinstance(dispatched, JobError)
        assert str(dispatched) == "budget_dispatch_not_authorized" and len(c.host.released) == 1
    [scope] = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
    counters = json.loads(scope["counters_json"])
    assert counters["held_calls"] == counters["held_tokens"] == counters["inflight"] == 0
    assert counters["used_calls"] == len(actual)
