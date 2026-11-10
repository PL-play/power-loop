"""Provider rejections get one repaired request; repeated failure stays explicit."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMRetryPolicy,
    LLMTokenUsage,
)
from power_loop.runtime.history_sanitize import sanitize_request_messages
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.retry import request_repair_kind
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from tests.unit.test_job_registry_192 import registry_store  # noqa: F401
from tests.unit.test_request_sanitize import StrictModel, assert_valid_request, assistant, call
from tests.unit.test_stateful_loop import _echo_registry


class Rejection(Exception):
    status_code = 400


class RepairModel(StrictModel):
    capabilities = SimpleNamespace(model="repair-model")

    def __init__(self, kind, *, fails_twice=False, summary_fails=False):
        super().__init__()
        self.kind, self.fails_twice, self.summary_fails = kind, fails_twice, summary_fails
        self.main_requests, self.summaries = [], 0

    async def complete(self, request, **kwargs):
        assert_valid_request(request.to_messages())
        usage = LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        if str(request.messages[0].get("content", "")).startswith("You are a conversation summarizer"):
            self.summaries += 1
            if self.summary_fails:
                raise Rejection("maximum context length exceeded")
            return LLMResponse(raw_text="<summary>Old tool returned known.</summary>", token_usage=usage)
        self.main_requests.append(request)
        if len(self.main_requests) == 1 or self.fails_twice:
            raise Rejection("invalid messages: tool_call format" if self.kind == "format"
                            else "maximum context length exceeded")
        return LLMResponse(raw_text="recovered", token_usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("ending", ["timeout", "cancel"])
@pytest.mark.parametrize("kind,delays,expected", [
    ("format", {"retry": 1}, ["initial", "retry"]),
    ("context", {"summary": 1}, ["initial", "summary"]),
    ("context", {"retry": 1}, ["initial", "summary", "retry"]),
    ("context", {"initial": .06, "summary": .06, "retry": .06},
     ["initial", "summary", "retry"]),
], ids=["format-retry", "context-summary", "context-retry", "shared-budget"])
async def test_recovery_stops_within_original_budget(managed, ending, kind, delays, expected,
                                                     record_property, runtime_agent):
    phases, stopped = [], []
    reached = asyncio.Event()
    started = None

    class SlowRepair(RepairModel):
        async def complete(self, request, **kwargs):
            nonlocal started
            if started is None:
                started = time.monotonic()
            summary = str(request.messages[0].get("content", "")).startswith(
                "You are a conversation summarizer")
            phase = "summary" if summary else ("retry" if self.main_requests else "initial")
            phases.append(phase)
            if phase == expected[-1]:
                reached.set()
            try:
                await asyncio.sleep(1 if ending == "cancel" and phase == expected[-1]
                                    else delays.get(phase, 0))
                return await super().complete(request, **kwargs)
            finally:
                stopped.append(phase)

    store = await open_store(":memory:", schema="tasks")
    runtime = (TaskRuntime(JobManager(store), tenant_id="eval", principal_id="host",
        executor_id="repair", boot_id="boot", boot_contract_hash="a" * 64) if managed else None)
    timeout = (1 if ending == "cancel" else .15 if len(delays) == 3 else .05)
    loop = await runtime_agent(model=SlowRepair(kind), store=store, task_runtime=runtime,
        config=AgentLoopConfig(compactor=None, delegation_policy="single_level",
            retry_policy=LLMRetryPolicy(max_attempts=5, backoff_initial=0, total_timeout=timeout)))
    try:
        sid = loop.target.session_id
        await store.append_message(sid, role="user", content="old details " * 200)
        await store.append_message(sid, role="assistant", content="done")
        from power_loop.agent.runtime_store import now_ms
        receipt = await loop.runtime.client.send(loop.target, "continue", message_id="repair", actor=loop.actor)
        task = asyncio.create_task(loop.runtime.client.wait(receipt.input_ref, actor=loop.actor,
                                                           deadline_ms=now_ms()+10000))
        if ending == "cancel":
            await asyncio.wait_for(reached.wait(), .5)
            await loop.runtime.client.cancel(receipt.input_ref, actor=loop.actor, reason="stop during repair")
        result = await asyncio.wait_for(task, .5)
        elapsed = time.monotonic() - started
        record_property("elapsed_seconds", round(elapsed, 4))
        assert result.status == ("failed" if ending == "timeout" else "cancelled")
        if ending == "timeout":
            assert "LLM timeout" in result.final_text
            assert elapsed < timeout + .1
        await asyncio.sleep(.02)
        assert phases == stopped == expected
    finally:
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["format", "context"])
@pytest.mark.parametrize("fails_twice", [False, True])
@pytest.mark.parametrize("mode", ["plain", "managed", "budgeted"])
async def test_rejected_history_recovers_once(registry_store, kind, fails_twice, mode, runtime_agent):  # noqa: F811
    store, _ = registry_store
    model = RepairModel(kind, fails_twice=fails_twice)
    runtime = (None if mode == "plain" else TaskRuntime(JobManager(store), tenant_id="eval",
        principal_id="host", executor_id="repair", boot_id="boot", boot_contract_hash="a" * 64))
    loop = await runtime_agent(model=model, store=store, task_runtime=runtime, registry=_echo_registry(),
        config=AgentLoopConfig(compactor=None, delegation_policy="single_level", max_tokens=20,
            retry_policy=LLMRetryPolicy(max_attempts=5, backoff_initial=0, give_up_on=None)))
    try:
        sid = loop.target.session_id
        for msg in [{"role": "user", "content": "old details " * 200}, assistant(call("old")),
                    {"role": "tool", "tool_call_id": "old", "content": "known"}]:
            await store.append_message(sid, **msg)
        before = await store.load_active_messages(sid)
        result = await loop.send("continue")
        if fails_twice:
            assert result.status == "failed" and result.error == "Rejection", result
        else:
            assert result.status == "completed" and result.final_text == "recovered"
        assert len(model.main_requests) == 2
        assert model.summaries == (1 if kind == "context" else 0)
        first, second = [r.messages for r in model.main_requests]
        assert any(m.get("tool_calls") for m in first)
        assert not any(m.get("tool_calls") for m in second)
        assert "known" in str(second)
        if kind == "context":
            assert len(json.dumps(second)) < len(json.dumps(first))
        assert (await store.load_active_messages(sid))[:len(before)] == before
    finally:
        await loop.aclose()


def test_strict_repair_preserves_current_tool_round():
    source = [assistant(call("old")), {"role": "tool", "tool_call_id": "old", "content": "old result"},
              {"role": "user", "content": "next"}, assistant(call("current")),
              {"role": "tool", "tool_call_id": "current", "content": "current result"}]
    out = sanitize_request_messages(source, strict=True)
    assert_valid_request(out)
    assert "old result" in out[0]["content"] and "tool_calls" not in out[0]
    assert out[-2]["tool_calls"][0]["id"] == "current"
    assert out[-1]["content"] == "current result"


@pytest.mark.parametrize("status,detail,expected", [
    (400, "context_length_exceeded", "context"), (413, "prompt is too long", "context"),
    (422, "messages invalid", "format"), (400, "reasoning_content is missing", "format"),
    (401, "messages invalid", None), (400, "invalid model", None), (500, "messages invalid", None),
])
def test_request_rejection_classifier(status, detail, expected):
    error = Rejection(detail)
    error.status_code = status
    assert request_repair_kind(error) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("old_history", [False, True])
async def test_uncompressible_request_fails_without_identical_retry(tmp_path, old_history, runtime_agent):
    model = RepairModel("context", summary_fails=True)
    loop = await runtime_agent(model=model, config=AgentLoopConfig(compactor=None))
    try:
        sid = loop.target.session_id
        if old_history:
            await loop.runtime.store.append_message(sid, role="user", content="old " * 200)
            await loop.runtime.store.append_message(sid, role="assistant", content="done")
        result = await loop.send("new")
        assert result.status == "failed" and result.error == "CompactionFailed", result
        assert len(model.main_requests) == 1
        assert model.summaries == int(old_history)
    finally:
        await loop.aclose()
