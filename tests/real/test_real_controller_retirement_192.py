"""Actual provider usage ends before a separate process observes retirement."""

import asyncio

import pytest

from power_loop import HookPoint
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from tests.unit.test_controller_retirement_192 import read_in_process
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_original_owner_review_192 import original_owner_case, wait_owned

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ["registry_store"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("receipt_failure", [False, True])
async def test_real_controller_exit_revokes_lease_without_faking_result_or_releasing_usage(
    registry_store, record_property, monkeypatch, receipt_failure
):
    provider = BoundedProvider(limit=5)
    c = await original_owner_case(registry_store, "subagent", allowed_calls=1, provider=provider)
    initial = set(delegation._TASKS)
    from power_loop.runtime import controller_retirement
    original_retire = controller_retirement.retire
    failed = False

    async def fail_receipt_once(runtime, binding, local):
        nonlocal failed
        if receipt_failure and not failed:
            failed = True
            raise OSError("actual provider finished; receipt write unavailable")
        return await original_retire(runtime, binding, local)

    monkeypatch.setattr(controller_retirement, "retire", fail_receipt_once)

    async def precise_action(ctx):
        if get_session_id() != c.state.root and not c.state.counts["BUILD"]:
            ctx.system_prompt += "\nCall the offered touch_once function exactly once using native function calling. Do not write XML or textual tool calls."

    c.loop.hooks.register(HookPoint.LLM_BEFORE, precise_action, name="test.retirement.actual-native-call")
    try:
        async with asyncio.timeout(150):
            await c.loop.send("Use launch_owned exactly once for the original child, then reply QUEUED.", c.state.root)
            await wait_owned(c, "subagent", initial)
            binding = c.state.owner_binding
            before = await c.manager.execution_usage(c.runtime.actor(c.state.root), binding.lease.job_id, generation=1)
            requests = provider.calls
            if receipt_failure:
                assert failed and c.runtime._controller_retirements.failed_count() == 1
                await asyncio.gather(*(c.runtime.ensure_started() for _ in range(4)))
                assert not c.runtime._controller_retirements.pending
            observed = await read_in_process(c, binding, record_property)
            assert observed.get("error") in ("job_intake_closed", "unresolved_prior_attempt")
            owner = await c.manager.get_job(c.runtime.actor(c.state.root), binding.lease.job_id)
            assert (
                owner.resource_state == "unknown"
                and owner.result_status == "none"
                and owner.supervision_status == "open"
            )
            assert await c.manager.execution_usage(c.runtime.actor(c.state.root), owner.job_id, generation=1) == before
            assert before["calls"] == 1 and provider.calls == requests and not c.state.effects
            assert c.state.counts["BUILD"] == 1 and not c.state.entered.is_set()
    finally:
        c.state.release.set()
        provider.report(record_property)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)
        await provider.close()
