"""Actual first-generation scope repairs empty pass output without business replay."""
import json
from contextlib import contextmanager

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime.delegation import Running

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import finish, jobs, managed

__all__ = ["registry_store", "managed"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["recover", "empty", "provider_error", "repeat_pass", "exhausted", "cancel", "cancel_pending"])
async def test_empty_initial_managed_child_gets_one_guarded_non_business_answer(managed, monkeypatch, ending):
    store, manager, runtime, loop, model, parent, _ = managed
    # This tests output recovery, not heartbeat expiry. Real PG admission and
    # result transactions must not race an unrelated one-second test lease.
    runtime.lease_ms = 30000
    original = model.complete
    requests, guarded = [], []
    @contextmanager
    def guard():
        guarded.append("enter")
        try:
            yield
        finally:
            guarded.append("exit")
    loop._child_run_guards = (("output", guard),)
    async def complete(request, **kwargs):
        if get_session_id() == parent:
            response = await original(request, **kwargs)
            if response.tool_calls and ending == "exhausted":
                args = json.loads(response.tool_calls[0]["function"]["arguments"])
                args["max_rounds"] = 1
                response.tool_calls[0]["function"]["arguments"] = json.dumps(args)
            return response
        requests.append(request)
        response = await original(request, **kwargs)
        if len(requests) == 1 or ending == "repeat_pass":
            response.raw_text = ""
            response.tool_calls = [{"id": "empty-pass-"+str(len(requests)), "type": "function", "function": {
                "name": "pass_turn", "arguments": '{"reason":"answer already delivered"}'}}]
        elif ending == "empty":
            response.raw_text, response.tool_calls = "", None
        elif ending == "provider_error":
            raise RuntimeError("output provider failed")
        if ending in ("cancel", "cancel_pending") and len(requests) == 1:
            if ending == "cancel":
                # Output cancellation with no unresolved call. The separate
                # cancel_pending case must retain the original pending proof.
                response.tool_calls = None
            from power_loop.runtime import delegation
            delegation._RUNNING[(id(store), get_session_id())].token.cancel("cancel before supplement")
        return response
    monkeypatch.setattr(model, "complete", complete)
    await loop.send("delegate", parent)
    await finish(managed)
    [child] = [j for j in await jobs(managed) if j.kind == "subagent"]
    if ending == "cancel_pending":
        # Cancellation is not proof that the unexecuted call was disposed or
        # reviewed. Never publish a premature canonical child result.
        assert child.outcome is None and child.result is None
        assert child.supervision_status == "open"
        assert len(requests) == 1
        assert (await store.get_state(child.session_id)).pending
        assert (await store.inbox_counts(parent))["pending"] == 0
        assert guarded == ["enter", "exit"]
        return
    assert child.outcome == ("succeeded" if ending == "recover" else "cancelled" if ending == "cancel" else "failed")
    assert len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1
    if ending == "recover":
        assert child.result["final_text"] == "CHILD-PROOF-42"
        assert child.result["usage"]["total_tokens"] == 20 and len(requests) == 2
    assert len(guarded) == (2 if ending in ("cancel", "exhausted") else 4)
    assert guarded == ["enter", "exit"] * (len(guarded)//2)
    # Supplement requests advertise no business tools. The core's existing
    # read-only limit finalizer may be a third actual provider request.
    if len(requests) > 1:
        assert {t["function"]["name"] for t in requests[1].tools or []} <= {"pass_turn"}
    assert (await store.inbox_counts(parent))["pending"] == 1


@pytest.mark.parametrize("mode", ["exhausted", "already_attempted", "expired_deadline"])
async def test_supplement_does_not_receive_a_second_round_allowance(monkeypatch, mode):
    import asyncio
    from types import SimpleNamespace

    from power_loop.runtime.cancellation import CancellationToken
    from power_loop.runtime.subagent_output import ensure_answer

    calls = []
    async def follow(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("no provider request should be dispatched")
    child = SimpleNamespace(config=SimpleNamespace(max_rounds=1), follow_up=follow)
    async def restore(*args):
        return child
    monkeypatch.setattr("power_loop.runtime.subagent_output.workflow_continuation_loop", restore)
    running = Running("original-test-run", task_binding=SimpleNamespace(lease=SimpleNamespace(generation=1)),
                      answer_attempted=mode == "already_attempted", token=CancellationToken())
    raw = {"status": "completed", "final_text": "", "rounds": 0 if mode == "expired_deadline" else 1}
    deadline = asyncio.get_running_loop().time()+5
    if mode == "expired_deadline":
        with pytest.raises(TimeoutError):
            await ensure_answer(SimpleNamespace(), running, "original", raw, 0)
    else:
        value = await ensure_answer(SimpleNamespace(), running, "original", raw, deadline)
        assert value is raw if mode == "already_attempted" else value["status"] == "failed"
    assert not calls


async def test_supplement_runs_original_finalizer_before_result_delivery(managed, monkeypatch):
    from power_loop.runtime import delegation

    store, _, _, loop, model, parent, _ = managed
    complete = model.complete
    calls, finalized = [], []
    async def model_complete(request, **kwargs):
        response = await complete(request, **kwargs)
        if get_session_id() != parent:
            calls.append(1)
            if len(calls) == 1:
                response.raw_text = ""
                response.tool_calls = [{"id": "empty", "type": "function", "function": {
                    "name": "pass_turn", "arguments": '{"reason":"waiting"}'}}]
        return response
    monkeypatch.setattr(model, "complete", model_complete)
    start = delegation.start
    async def finalize(sid, raw):
        finalized.append((sid, raw.get("final_text")))
        return {**raw, "final_text": "["+raw["final_text"]+"]"} if raw.get("final_text") else raw
    async def begin(*args, **kwargs):
        return await start(*args, **{**kwargs, "finalize": finalize})
    monkeypatch.setattr(delegation, "start", begin)
    await loop.send("delegate", parent)
    await finish(managed)
    [child] = [j for j in await jobs(managed) if j.kind == "subagent"]
    assert child.outcome == "succeeded" and child.result["final_text"] == "[CHILD-PROOF-42]"
    assert child.result["usage"]["total_tokens"] == 20
    assert finalized == [(child.session_id, ""), (child.session_id, "CHILD-PROOF-42")]
    assert (await store.inbox_counts(parent))["pending"] == 1


async def test_empty_completed_supplement_is_failed_without_a_second_attempt(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from power_loop.runtime.cancellation import CancellationToken
    from power_loop.runtime.subagent_output import ensure_answer

    requests = []
    async def follow(*args, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(status="completed", final_text="", usage={}, rounds=1, new_input_after_pass=False)
    async def restore(*args):
        return SimpleNamespace(config=SimpleNamespace(max_rounds=2), follow_up=follow)
    monkeypatch.setattr("power_loop.runtime.subagent_output.workflow_continuation_loop", restore)
    running = Running("original-test-run", token=CancellationToken())
    raw = {"status": "completed", "final_text": "", "rounds": 1}
    value = await ensure_answer(SimpleNamespace(), running, "original", raw, asyncio.get_running_loop().time()+5)
    assert value["status"] == "failed" and value["error"] == "subagent_output_missing"
    assert await ensure_answer(SimpleNamespace(), running, "original", raw, 0) is raw
    assert len(requests) == 1 and requests[0]["max_rounds"] == 1 and requests[0]["tools"] == []
