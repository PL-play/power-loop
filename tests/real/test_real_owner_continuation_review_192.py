"""Live-provider later generations and same-generation completion review."""

import asyncio
import json

import pytest

from power_loop.runtime import delegation
from tests.unit.test_job_registry_192 import registry_store

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
