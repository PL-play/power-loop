"""Live-provider later generations and same-generation completion review."""

import asyncio
import json

import pytest

from power_loop.runtime import delegation
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owner_continuation_review_192 import (
    collected_case,
    complete_initial,
    continuation_case,
    settle,
    start_collected,
)

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ["registry_store"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def wait_for_original_review(c, kind, timeout=90):
    """Watch the actual owner and review event; a stopped owner cannot enter later."""
    if c.state.entered.is_set():
        return
    if kind == "workflow_node":
        task = c.state.handles[-1].task
    else:
        run = delegation._RUNNING.get((id(c.store), c.state.child_sid))
        task = run.task if run is not None else None
    if task is None or task.done():
        raise AssertionError("Original owner stopped before entering review; inspect recorded evidence")
    waiting = asyncio.create_task(c.state.entered.wait())
    try:
        done, _ = await asyncio.wait((waiting, task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if c.state.entered.is_set():
            return
        if task in done:
            raise AssertionError("Original owner stopped before entering review; inspect recorded evidence")
        raise TimeoutError("Original owner is still active but did not enter review within the bound")
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def report_review_evidence(c, provider, record_property):
    """Bounded actual diagnostic data; no response rewriting or test retries."""
    record_property("real_provider_responses", len(provider.responses))
    if c.state.entered.is_set():
        return
    evidence = {
        "provider_calls": provider.calls,
        "provider_responses": len(provider.responses),
        "phase": c.state.phase,
        "actual_response_tail": [
            {"text": (getattr(response, "raw_text", "") or "")[:400], "tool_calls": response.get_tool_calls()}
            for response in provider.responses[-2:]
        ],
    }
    if c.state.source is not None:
        row = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        evidence["source"] = {
            key: getattr(row, key)
            for key in ("job_id", "status", "resource_state", "outcome", "result_status", "supervision_status")
        }
        messages = await c.store.load_all_messages(row.session_id)
        evidence["actual_assistant_tail"] = [
            {"seq": v.seq, "content": (v.content or "")[:400], "tool_calls": v.tool_calls}
            for v in messages
            if v.role == "assistant"
        ][-3:]
    record_property("review_entry_evidence", json.dumps(evidence, ensure_ascii=False, default=str))


@pytest.mark.parametrize("kind", ["workflow_node", "subagent"])
@pytest.mark.parametrize("allowed_calls", [3, 2])
async def test_real_original_continuation_retains_role_generation_history_and_remaining_quota(
    registry_store, kind, allowed_calls, record_property
):
    provider = BoundedProvider(limit=14)
    c = await continuation_case(registry_store, kind, allowed_calls=allowed_calls, provider=provider)
    try:
        await complete_initial(c, kind)
        if allowed_calls == 3:
            await wait_for_original_review(c, kind)
            assert c.state.counts["AFTER"] == c.state.baseline_after
            owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
            assert owner.generation == 2 and owner.result_status == "none"
            assert (await c.store.inbox_counts(c.state.root))["pending"] == 0
        c.state.release.set()
        await settle(c, kind)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert (
            source.execution_role == kind
            and source.result["aborted_call_batch"]["review_owner"]["owner"]["generation"] == 2
        )
        assert not c.state.effects
        assert (
            await c.manager.get_result_revision(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_revision
        )
        assert (
            await c.manager.get_execution_membership(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_members
        )
        if allowed_calls == 3:
            assert source.result_status == "handled" and source.supervision_status == "closed"
            assert owner.outcome == "failed" and owner.generation == 2
            assert (await c.manager.execution_usage(c.runtime.actor(c.state.root), owner.job_id, generation=2))[
                "calls"
            ] == 2
        else:
            assert c.state.counts["BUILD"] == 1 and not c.state.review_requests
            assert source.result_status == "pending_delivery" and source.supervision_status == "open"
            assert owner.result_status == "none" and owner.supervision_status == "open"
        assert not await c.manager.db.fetchall(
            f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (source.session_id,)
        )
    finally:
        c.state.release.set()
        await report_review_evidence(c, provider, record_property)
        provider.report(record_property)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)
        await provider.close()


@pytest.mark.parametrize("kind", ["workflow_node", "subagent"])
async def test_real_original_collector_processes_one_abort_without_replay_or_double_round_charge(
    registry_store, monkeypatch, kind, record_property
):
    provider = BoundedProvider(limit=10)
    c = await collected_case(registry_store, kind, monkeypatch, provider=provider, start=False)
    try:
        await start_collected(c, kind)
        await wait_for_original_review(c, kind)
        assert not c.state.effects and c.state.counts["AFTER"] == 0
        assert (await c.store.inbox_counts(c.state.root))["pending"] == 0
        c.state.release.set()
        if kind == "workflow_node":
            await asyncio.wait_for(c.state.handles[0].task, 90)
        else:
            await settle(c, kind)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert source.result_status == "handled" and source.supervision_status == "closed"
        assert owner.generation == 1 and owner.outcome == "failed"
        assert (await c.manager.execution_usage(c.runtime.actor(c.state.root), owner.job_id, generation=1))[
            "calls"
        ] == 3
        if kind == "workflow_node":
            assert c.state.window.remaining == 0
        assert not c.state.effects
    finally:
        c.state.release.set()
        await report_review_evidence(c, provider, record_property)
        provider.report(record_property)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)
        await provider.close()
