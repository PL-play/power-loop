"""Additional design/192 fault boundaries supported by the current 8.0 runtime.

File-backed stores are isolated beneath tmp_path. These verify observable control,
ownership, per-run usage and durable result guarantees, not the future JobManager.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
)
from power_loop.agent.follow_up import InboxItem
from power_loop.core.agent_context import get_current_loop, get_session_id, reset_session_id, set_session_id
from power_loop.core.pass_turn import request_pass
from power_loop.runtime import delegation
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.pricing import Free
from power_loop.runtime.recall import recall
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@dataclass
class ControlledModel(LLMService):
    requests: list = field(default_factory=list)
    entered: dict = field(default_factory=lambda: {k: asyncio.Event() for k in "AB"})
    releases: dict = field(default_factory=lambda: {k: asyncio.Event() for k in "AB"})
    active: int = 0
    peak: int = 0

    async def complete(self, request, **kwargs):
        key = "B" if "ROLE-B" in (request.system_prompt or "") else "A"
        self.requests.append(request)
        self.entered[key].set()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await self.releases[key].wait()
            return LLMResponse(raw_text=f"PROOF-{key}", token_usage=LLMTokenUsage(
                prompt_tokens=7, completion_tokens=3, total_tokens=10))
        finally:
            self.active -= 1

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.fixture
async def runtime(tmp_path):
    path = tmp_path / "isolated.db"
    store = await SessionStore.open(str(path))
    model = ControlledModel()
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=ToolRegistry(), config=AgentLoopConfig(
        max_rounds=5, representation=ProjectedRepresentation()))
    parent = await loop.new_session()
    context = set_session_id(parent)
    before = set(delegation._TASKS)
    try:
        yield store, loop, model, parent, path
    finally:
        for release in model.releases.values():
            release.set()
        own = set(delegation._TASKS) - before
        if own:
            await asyncio.wait_for(asyncio.gather(*own, return_exceptions=True), 5)
        reset_session_id(context)
        await store.close()


def spec(key="A"):
    return AgentSpec(name=f"child-{key}", system_prompt=f"ROLE-{key}", tools=[],
                     lifecycle="linked", max_rounds=4)


async def finish_children():
    tasks = list(delegation._TASKS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), 5)


@pytest.mark.parametrize("finalize_error", [False, True], ids=["success", "finalize-failure"])
async def test_result_is_persisted_before_completion_callback(runtime, finalize_error):
    store, loop, model, parent, _ = runtime
    observed = []

    async def finalize(sid, raw):
        if finalize_error:
            raise RuntimeError("injected finalization failure")
        return raw

    async def notified(owner, result):
        record = await store.get_runtime_state(owner, "subagent:run:" + result["run_id"])
        assert record["phase"] == "completed"
        assert record["result"] == result
        observed.append(result)

    receipt = await delegation.start(loop, spec(), "work", finalize=finalize, on_complete=notified)
    model.releases["A"].set()
    await finish_children()
    assert len(observed) == 1
    assert observed[0]["session_id"] == receipt["session_id"]
    assert observed[0]["status"] == ("failed" if finalize_error else "completed")
    assert observed[0]["usage"]["total_tokens"] == 10
    assert not delegation.running_jobs(store, parent)


@pytest.mark.parametrize("source", ["messages", "projection"])
async def test_failed_wake_result_is_readable_from_reopened_store(runtime, source, tmp_path):
    store, loop, model, parent, path = runtime

    async def failed_wake(*args):
        raise RuntimeError("injected host wake failure")

    receipt = await delegation.start(loop, spec(), "persist proof", on_complete=failed_wake)
    model.releases["A"].set()
    await finish_children()
    assert len(model.requests) == 1
    recovered = await SessionStore.open(str(path))
    try:
        record = await recovered.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
        assert record["phase"] == "completed" and not record["notified"]
        assert (await recovered.inbox_counts(parent))["pending"] == 1
        result = record["result"]
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            exported = json.loads(await recall(
                recovered, parent, sub_session=receipt["session_id"], source=source,
                ids=result[source], output_file=f"{source}.jsonl"))
        assert exported["status"] == "exported"
        rows = [json.loads(line) for line in (tmp_path / f"{source}.jsonl").read_text().splitlines()]
        assert rows and all(r["session_id"] == receipt["session_id"] for r in rows)
        assert len(model.requests) == 1, "recall/export must never execute a provider call"
    finally:
        await recovered.close()


async def test_stopping_one_child_does_not_cancel_sibling_or_misattribute_usage(runtime):
    store, loop, model, parent, _ = runtime
    first = await delegation.start(loop, spec("A"), "A")
    second = await delegation.start(loop, spec("B"), "B")
    await asyncio.wait_for(model.entered["A"].wait(), 2)
    await asyncio.wait_for(model.entered["B"].wait(), 2)
    assert len(await loop.list_jobs(parent)) == 2
    await delegation.stop(store, parent, first["run_id"], 0.05)
    assert any(j["session_id"] == second["session_id"]
               for j in delegation.running_jobs(store, parent))
    model.releases["B"].set()
    await finish_children()
    a = (await store.get_runtime_state(parent, "subagent:run:" + first["run_id"]))["result"]
    b = (await store.get_runtime_state(parent, "subagent:run:" + second["run_id"]))["result"]
    assert a["status"] == "cancelled" and b["status"] == "completed"
    assert b["final_text"] == "PROOF-B" and b["usage"]["total_tokens"] == 10
    assert a["session_id"] != b["session_id"]
    assert not delegation.running_jobs(store, parent)


async def test_cancel_before_admission_never_calls_model_and_releases_capacity(runtime):
    store, loop, model, parent, _ = runtime
    admission = asyncio.Semaphore(0)
    receipt = await delegation.start(loop, spec(), "must not run", admission=admission)
    assert not model.requests
    await delegation.stop(store, parent, receipt["run_id"], 0.01)
    await finish_children()
    result = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert result["status"] == "cancelled"
    assert not model.requests and not delegation.running_jobs(store, parent)
    admission.release()
    following = await delegation.start(loop, spec("B"), "following work", admission=admission)
    model.releases["B"].set()
    await finish_children()
    assert (await store.get_runtime_state(parent, "subagent:run:" + following["run_id"]))["result"]["status"] == "completed"
    assert len(model.requests) == 1 and admission._value == 1


async def test_repeated_stop_after_completion_does_not_rewrite_result_or_usage(runtime):
    store, loop, model, parent, _ = runtime
    receipt = await delegation.start(loop, spec(), "work")
    model.releases["A"].set()
    await finish_children()
    before = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    for _ in range(3):
        assert await delegation.stop(store, parent, receipt["run_id"], 0.01) == "finished"
    assert await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]) == before
    assert len(model.requests) == 1


@pytest.mark.parametrize("finalize_error", [False, True])
async def test_stopped_child_delivers_one_owner_result_without_external_wake(runtime, finalize_error):
    store, loop, model, parent, _ = runtime
    wakes = []

    async def notified(*args):
        wakes.append(args)

    async def finalize(sid, raw):
        if finalize_error:
            raise RuntimeError("injected failure after stop")
        return raw

    receipt = await delegation.start(loop, spec(), "wait", finalize=finalize, on_complete=notified)
    await asyncio.wait_for(model.entered["A"].wait(), 2)
    await delegation.stop(store, parent, receipt["run_id"], 0.02)
    await finish_children()
    for _ in range(2):
        await delegation.stop(store, parent, receipt["run_id"], 0.01)
    result = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert result["status"] != "completed"
    assert (await store.inbox_counts(parent))["pending"] == 1
    [item] = await store.inbox_items(parent)
    assert receipt["run_id"] in item["content"]
    assert result["status"] in item["content"]
    assert not wakes


async def test_await_children_cannot_finish_before_result_delivery(runtime, monkeypatch):
    store, loop, model, parent, _ = runtime
    entered, release = asyncio.Event(), asyncio.Event()
    deliver = loop.deliver

    async def slow_delivery(item, sid, **kwargs):
        if item.kind == "task_done":
            entered.set()
            await release.wait()
        return await deliver(item, sid, **kwargs)

    monkeypatch.setattr(loop, "deliver", slow_delivery)
    await delegation.start(loop, spec(), "work")
    model.releases["A"].set()
    await asyncio.wait_for(entered.wait(), 2)
    collector = asyncio.create_task(delegation.await_children(store, parent))
    try:
        assert delegation.running_jobs(store, parent), "delivery window must remain supervised"
        done, _ = await asyncio.wait({collector}, timeout=0.02)
        assert not done and (await store.inbox_counts(parent))["pending"] == 0
    finally:
        release.set()
        await asyncio.wait_for(collector, 2)
    assert (await store.inbox_counts(parent))["pending"] == 1
    assert not delegation.running_jobs(store, parent)


@pytest.mark.parametrize("always_pass", [False, True], ids=["consume-arrived-input", "bounded-passing"])
async def test_standalone_child_processes_pass_boundary_input_with_a_shared_budget(
        runtime, monkeypatch, always_pass):
    store, loop, model, parent, _ = runtime
    monkeypatch.setattr(delegation, "_SETTLEMENT_ROUNDS", 2)

    async def complete(request, **kwargs):
        model.requests.append(request)
        usage = LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
        if always_pass or "CORRECTION" not in str(request.messages):
            return LLMResponse(raw_text="waiting", token_usage=usage, tool_calls=[{
                "id": f"pass-{len(model.requests)}", "type": "function",
                "function": {"name": "boundary_pass", "arguments": "{}"}}])
        return LLMResponse(raw_text="CORRECTION consumed", token_usage=usage)

    async def boundary_pass():
        request_pass("wait for correction")
        await get_current_loop().deliver(InboxItem("CORRECTION"), get_session_id(), enqueue_only=True)
        return "passed"

    monkeypatch.setattr(model, "complete", complete)
    loop.tool_registry.register(ToolDefinition(name="boundary_pass", description="Test pass",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), boundary_pass)
    receipt = await delegation.start(loop, AgentSpec(name="owner", system_prompt="OWNER",
        tools=["boundary_pass"], lifecycle="linked", max_rounds=4), "work")
    await finish_children()
    saved = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    if always_pass:
        # The pipeline's existing tool-less wrap-up at hit_round_limit is one
        # additional request. It is bounded and must remain in usage, too.
        assert saved["status"] == "hit_round_limit" and len(model.requests) == 4
    else:
        assert saved["status"] == "completed" and saved["final_text"] == "CORRECTION consumed"
        assert len(model.requests) == 2 and not saved["new_input_after_pass"]
    assert saved["usage"]["total_tokens"] == len(model.requests) * 10
    assert (await store.inbox_counts(parent))["pending"] == 1


async def test_failed_finalizer_does_not_resume_and_turn_failure_into_success(runtime):
    store, loop, model, parent, _ = runtime

    async def finalize(sid, raw):
        await loop.deliver(InboxItem("must remain pending after failure"), sid, enqueue_only=True)
        return {**raw, "status": "failed", "error": "required output invalid"}

    receipt = await delegation.start(loop, spec(), "work", finalize=finalize)
    model.releases["A"].set()
    await finish_children()
    saved = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert saved["status"] == "failed" and len(model.requests) == 1
    assert (await store.inbox_counts(receipt["session_id"]))["pending"] == 1


async def test_stuck_finalizer_respects_total_deadline_and_delivers_failure(runtime, monkeypatch):
    store, loop, model, parent, _ = runtime
    monkeypatch.setattr(delegation, "_SETTLEMENT_TIMEOUT_S", 0.03)
    entered = asyncio.Event()

    async def finalize(sid, raw):
        entered.set()
        await asyncio.Event().wait()

    receipt = await delegation.start(loop, spec(), "work", finalize=finalize)
    model.releases["A"].set()
    await finish_children()
    saved = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert entered.is_set() and saved["status"] == "failed"
    assert "TimeoutError" in saved["error"] and saved["usage"]["total_tokens"] == 10
    assert (await store.inbox_counts(parent))["pending"] == 1


@pytest.mark.parametrize("mode", ["missing-result", "timeout"])
async def test_broken_subagent_continuation_cannot_publish_success(runtime, monkeypatch, mode):
    store, loop, model, parent, _ = runtime
    monkeypatch.setattr(delegation, "_SETTLEMENT_TIMEOUT_S", 0.1)
    entered = asyncio.Event()

    class Broken:
        async def flush_follow_ups(self, sid, **kwargs):
            entered.set()
            if mode == "timeout":
                await asyncio.Event().wait()
            return None

    async def resolve(*args):
        return Broken()

    async def finalize(sid, raw):
        await loop.deliver(InboxItem("required continuation input"), sid, enqueue_only=True)
        return raw

    monkeypatch.setattr(delegation, "workflow_continuation_loop", resolve)
    receipt = await delegation.start(loop, spec(), "work", finalize=finalize)
    model.releases["A"].set()
    await finish_children()
    saved = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert entered.is_set() and saved["status"] == "failed"
    assert ("TimeoutError" if mode == "timeout" else "no result") in saved["error"]
    assert saved["usage"]["total_tokens"] == 10 and len(model.requests) == 1
    assert (await store.inbox_counts(parent))["pending"] == 1


async def test_multiple_continuations_keep_ranges_and_usage_per_run(runtime):
    store, loop, model, parent, _ = runtime
    model.releases["A"].set()
    receipts = [await delegation.start(loop, spec(), "first")]
    await finish_children()
    for number in range(3):
        receipts.append(await delegation.nudge(loop, receipts[0]["session_id"], f"continue {number}"))
        await finish_children()
    results = [(await store.get_runtime_state(parent, "subagent:run:" + r["run_id"]))["result"]
               for r in receipts]
    assert len({r["session_id"] for r in results}) == 1
    assert len({r["run_id"] for r in results}) == 4
    assert all(r["usage"]["total_tokens"] == 10 for r in results)
    for left, right in zip(results, results[1:]):
        assert left["messages"][1] < right["messages"][0]
        assert left["projection"][1] < right["projection"][0]
    assert len(await store.list_children(parent)) == 1


async def test_running_nudges_share_run_and_never_start_concurrent_send(runtime):
    store, loop, model, parent, _ = runtime
    model.releases["A"].set()
    receipt = await delegation.start(loop, spec(), "first")
    await finish_children()
    model.releases["A"].clear()
    model.entered["A"].clear()
    continued = await delegation.nudge(loop, receipt["session_id"], "FIRST-CORRECTION")
    await asyncio.wait_for(model.entered["A"].wait(), 2)
    steered = await delegation.nudge(loop, receipt["session_id"], "SECOND-CORRECTION")
    assert continued["run_id"] != receipt["run_id"]
    assert steered["run_id"] == continued["run_id"] and steered["mode"] == "steer"
    model.releases["A"].set()
    await finish_children()
    assert model.peak == 1
    rows = await store.load_all_messages(receipt["session_id"])
    assert any("FIRST-CORRECTION" in str(r.content) for r in rows)
    assert any("SECOND-CORRECTION" in str(r.content) for r in rows)
    assert not delegation.running_jobs(store, parent)


async def test_foreign_parent_cannot_steer_or_stop_active_child(runtime):
    store, loop, model, parent, _ = runtime
    receipt = await delegation.start(loop, spec(), "work")
    stranger = await loop.new_session()
    context = set_session_id(stranger)
    try:
        with pytest.raises(ValueError, match="access_denied"):
            await delegation.nudge(loop, receipt["session_id"], "unauthorized")
        await delegation.stop(store, stranger, receipt["run_id"], 0.01)
    finally:
        reset_session_id(context)
    assert delegation.running_jobs(store, parent)
    model.releases["A"].set()
    await finish_children()
    result = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
    assert result["status"] == "completed"


async def test_parent_cancellation_reaches_every_waited_child(runtime):
    store, loop, model, parent, _ = runtime
    receipts = [await delegation.start(loop, spec(key), key) for key in "AB"]
    await asyncio.wait_for(model.entered["A"].wait(), 2)
    await asyncio.wait_for(model.entered["B"].wait(), 2)
    stop = CancellationToken()
    stop.cancel("test parent cancellation")
    with pytest.raises(ValueError, match="cancelled while waiting"):
        await delegation.await_children(store, parent, stop, 1)
    await finish_children()
    assert not delegation.running_jobs(store, parent)
    for receipt in receipts:
        result = (await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"]))["result"]
        assert result["status"] == "cancelled"
