"""Original-generation answer recovery obeys identity, limits and billing evidence."""

import asyncio
import json

import pytest

from power_loop import LLMResponse, LLMTokenUsage
from power_loop.runtime.task_runtime import _OWNER

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def verify(c, monkeypatch, record_property):
    complete = c.model.complete
    passed = False
    old = await c.manager.get_result_revision(c.actor, c.initial.job_id, 1)
    initial_nodes = await c.manager.db.fetchall(
        f"SELECT job_id,session_id FROM {c.manager.jobs} WHERE kind='workflow_node' ORDER BY job_id"
    )

    async def empty(request, **kwargs):
        nonlocal passed
        owner = _OWNER.get()
        if owner and owner.lease.generation == 2 and "ROLE-BUILD" in (request.system_prompt or "") and not passed:
            passed = True
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "restore-empty-pass",
                        "type": "function",
                        "function": {
                            "name": "pass_turn",
                            "arguments": json.dumps({"reason": "The corrected number is NUDGED-43"}),
                        },
                    }
                ],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await complete(request, **kwargs)

    monkeypatch.setattr(c.model, "complete", empty)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    record_property("actual_restore_result", json.dumps(current.result, ensure_ascii=False))
    assert passed
    assert current.outcome == "succeeded", current.result
    assert current.result["results"]["BUILD"]["text"] == "NUDGED-43"
    assert current.result["final"]["text"] == "NUDGED-43"
    assert current.result["usage"]["total_tokens"] == 30
    assert await c.manager.get_result_revision(c.actor, c.initial.job_id, 1) == old
    assert (
        await c.manager.db.fetchall(
            f"SELECT job_id,session_id FROM {c.manager.jobs} WHERE kind='workflow_node' ORDER BY job_id"
        )
        == initial_nodes
    )
