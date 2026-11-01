"""A local non-dispatch winner prevents a later host reservation after awaits."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import LLMRequest
from power_loop.runtime import execution_budget, model_budgets
from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case
from .test_unstarted_budget_attempts_192 import prepared

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("window", ["metadata", "allocated"])
async def test_original_non_dispatch_proof_wins_before_late_host_reservation(budget_case, monkeypatch, window):
    c = budget_case
    attempt = await prepared(c)
    entered, resume = asyncio.Event(), asyncio.Event()
    target, method = (c.host, "prepare") if window == "metadata" else (execution_budget, "allocate")
    original = getattr(target, method)

    async def paused(*args, **kwargs):
        value = await original(*args, **kwargs)
        entered.set()
        await resume.wait()
        return value

    monkeypatch.setattr(target, method, paused)
    reservation = asyncio.create_task(model_budgets.reserve(attempt,
        LLMRequest(messages=[], max_tokens=20), service=c.host.service))
    closer = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if window == "metadata":
            proof = await model_budgets.close_unreserved(attempt)
            assert proof == {"local_only": True, "dispatch_not_authorized": True, "no_reservation": True}
            await flush_binding(c.binding)
            resume.set()
            with pytest.raises(JobError, match="budget_attempt_already_reserved"):
                await reservation
            assert c.host.reserve_calls == 0 and not c.host.released
        else:
            # Allocation and publication of host intent are one local choice.
            # A contender cannot publish "no reservation" halfway through it.
            closer = asyncio.create_task(model_budgets.close_unreserved(attempt))
            await asyncio.sleep(0)
            assert not closer.done()
            resume.set()
            assert await reservation is True and await closer is None
            assert c.host.reserve_calls == 1
            await model_budgets.close_unstarted(attempt)
            assert len(c.host.released) == 1
        await flush_binding(c.binding, close=True)
        cold = await c.connect()
        state = await model_budgets.get_budget(JobManager(cold), c.actor, attempt.attempt_id)
        assert state["phase"] == "released" and not state["pending"]
        assert (state["plan"] is None) == (window == "metadata")
        assert state["ack"].get("no_reservation", False) == (window == "metadata")
        counters = await cold._db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
        assert counters and all(json.loads(row["counters_json"]) == dict(
            used_calls=0, held_calls=0, used_tokens=0, held_tokens=0, inflight=0) for row in counters)
        assert not c.host.applied
        assert await model_budgets.get_budget(JobManager(cold), c.actor, attempt.attempt_id) == state
    finally:
        resume.set()
        await asyncio.gather(reservation, *([closer] if closer else []), return_exceptions=True)
