"""Final schema validation follows original waiting and accepted input processing."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.tools.spawn_agent import _finalize

from .test_job_registry_192 import registry_store
from .test_owner_waits_192 import entered, original, prepare_due

__all__ = ["registry_store", "original"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["valid", "repair", "cancel"])
async def test_original_wait_is_resolved_before_final_schema_repair(original, monkeypatch, ending):
    c = original
    complete = c.model.complete

    async def provider(request, **kwargs):
        response = await complete(request, **kwargs)
        role = "main" if get_session_id() == c.state.parent else "child"
        if role == "main" and c.model.calls[role] == 1:
            args = json.loads(response.tool_calls[0]["function"]["arguments"])
            args["output_schema"] = {"name": "Answer", "schema": {"type": "object", "properties": {
                "answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}}
            response.tool_calls[0]["function"]["arguments"] = json.dumps(args)
        elif role == "child":
            if c.model.calls[role] == 2:
                response.raw_text = "WAITING-42"
            elif c.model.calls[role] == 3 and ending == "repair":
                response.raw_text = "The original answer is 42."
            else:
                response.raw_text = '{"answer":42}' if c.model.calls[role] > 2 else response.raw_text
        return response

    monkeypatch.setattr(c.model, "complete", provider)
    registration, owner = await entered(c)
    assert c.model.calls["child"] == 2, "Waiting placeholder must not trigger paid schema repair"
    [entry] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='owner_waiting'", (owner.job_id,))
    assert json.loads(entry["payload_json"])["raw"]["final_text"] == "WAITING-42"
    due = await prepare_due(c.manager, c.runtime.actor(c.state.parent), registration)
    c.state.choice, c.state.due_id = "cancel" if ending == "cancel" else "continue", due["event_id"]
    await notices.enqueue_notification(c.loop, c.runtime.actor(c.state.parent), due["event_id"])
    await c.loop.flush_follow_ups(c.state.parent)
    await effects.apply(c.manager, c.runtime.actor(c.state.parent), due["event_id"])
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    current = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
    assert current.outcome == ("cancelled" if ending == "cancel" else "succeeded"), current.result
    assert c.model.calls["child"] == {"valid": 3, "repair": 4, "cancel": 2}[ending]
    if ending != "cancel":
        assert current.result["payload"] == {"answer": 42}
        assert "error" not in current.result
    assert len(await c.store.list_children(c.state.parent)) == 1
    await c.loop.flush_follow_ups(c.state.parent)
    rows = await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}")
    assert all(row["supervision_status"] == "closed" for row in rows), rows


@pytest.mark.parametrize("status,new_input", [("failed", False), ("cancelled", False), ("hit_round_limit", False),
    ("degraded", False), ("timed_out", False), pytest.param("completed", True, id="pending_input")])
async def test_schema_shaped_repair_retains_negative_status_or_pending_original_input(original, monkeypatch, status, new_input):
    c = original
    schema = {"name": "Answer", "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}
    await c.store.set_runtime_state(c.state.parent, "workflow_execution_contract", {"spec": {"output_schema": schema}})
    calls = []

    async def send(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=status, final_text='{"answer":42}', rounds=1, usage={"total_tokens": 5}, new_input_after_pass=new_input)

    async def restore(*args):
        return SimpleNamespace(send=send)

    monkeypatch.setattr("power_loop.runtime.spec.workflow_continuation_loop", restore)
    result = await _finalize(c.loop, c.state.parent, {"status": "completed", "final_text": "not JSON", "rounds": 2, "usage": {"total_tokens": 20}})
    assert result["status"] == status and "payload" not in result, result
    assert bool(result.get("new_input_after_pass")) == new_input
    assert result["usage"]["total_tokens"] == 25 and result["rounds"] == 3
    assert len(calls) == 1 and calls[0]["tools"] == [] and calls[0]["max_rounds"] == 1
