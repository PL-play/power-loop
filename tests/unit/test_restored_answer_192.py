"""Original-generation answer recovery obeys identity, limits and billing evidence."""

import asyncio
import json

import pytest

from power_loop import LLMResponse, LLMTokenUsage
from power_loop.runtime.task_runtime import _OWNER

from .test_job_registry_192 import registry_store
from .test_workflow_bound_restore_192 import bound_restore
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow", "native_restore", "bound_restore"]
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


async def test_native_empty_restore(native_restore, monkeypatch, record_property):
    await verify(native_restore, monkeypatch, record_property)


async def test_bound_empty_restore(bound_restore, monkeypatch, record_property):
    await verify(bound_restore, monkeypatch, record_property)


async def negative(c, monkeypatch, mode):
    from power_loop.workflow.resume import resume_detached

    original = c.model.complete
    phases = []
    old = await c.manager.get_result_revision(c.actor, c.initial.job_id, 1)
    before = c.pool.spent if hasattr(c, "pool") else None

    async def no_answer(request, **kwargs):
        owner = _OWNER.get()
        if owner and owner.lease.generation == 2 and "ROLE-BUILD" in (request.system_prompt or ""):
            phases.append(len(phases) + 1)
            if len(phases) > 1 and mode == "provider_error":
                raise RuntimeError("provider interrupted during bounded final answer")
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "empty-bounded-" + str(len(phases)),
                        "type": "function",
                        "function": {"name": "pass_turn", "arguments": json.dumps({"reason": "no final text"})},
                    }
                ],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(c.model, "complete", no_answer)
    if mode == "proof_missing":
        from power_loop.runtime import workflow_tasks
        from power_loop.runtime.node_resume import _PREPARED

        answer = workflow_tasks.ensure_node_answer

        async def missing_preparation(*args, **kwargs):
            token = _PREPARED.set(None)
            try:
                return await answer(*args, **kwargs)
            finally:
                _PREPARED.reset(token)

        monkeypatch.setattr(workflow_tasks, "ensure_node_answer", missing_preparation)
    handles = []

    async def handler():
        handle = await resume_detached(
            c.loop, c.parent, c.initial.run_id, continuation=("BUILD", "NUDGED-43", 1 if mode == "no_remaining" else 3)
        )
        c.state["handles"].append(handle)
        handles.append(handle)
        return {"job_id": handle.job_id, "status": "accepted"}

    await c.issue(handler=handler)
    assert len(handles) == 1
    await asyncio.wait_for(handles[0].task, 10)
    current = await c.manager.get_job(c.actor, handles[0].job_id)
    assert current.outcome != "succeeded"
    assert len(phases) == (1 if mode == "proof_missing" else 2)  # cap=1 includes the existing read-only limit finalizer
    assert c.model.counts["REVIEW"] == 1
    assert await c.manager.get_result_revision(c.actor, c.initial.job_id, 1) == old
    assert current.result["usage"] == await c.manager.execution_usage(c.actor, c.initial.job_id, generation=2)
    if before is not None:
        assert c.pool.spent == before + current.result["usage"]["total_tokens"]


@pytest.mark.parametrize("mode", ["no_remaining", "empty_again", "provider_error", "proof_missing"])
async def test_native_bounded_negative(native_restore, monkeypatch, mode):
    await negative(native_restore, monkeypatch, mode)


@pytest.mark.parametrize("mode", ["no_remaining", "empty_again", "provider_error", "proof_missing"])
async def test_bound_bounded_negative(bound_restore, monkeypatch, mode):
    await negative(bound_restore, monkeypatch, mode)
