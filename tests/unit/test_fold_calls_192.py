"""History-fold model calls inside a task-managed send: booked, refused or cancelled, never unmanaged."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, CancellationToken, LLMRequest, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.errors import CancellationRequested
from power_loop.contracts.events import AgentEventType
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.fold import FoldContext, LLMSummaryFold
from power_loop.runtime.fold_calls import FoldCalls
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.representation import ProjectedRepresentation, VerbatimRepresentation
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class SummaryModel:
    capabilities = SimpleNamespace(model="summary-model")

    def __init__(self, error=None):
        self.calls, self.error = 0, error

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResponse(raw_text="<summary>folded</summary>", token_usage=LLMTokenUsage(
            prompt_tokens=40, completion_tokens=5, total_tokens=45))


class FoldHost(Host):
    def __init__(self):
        super().__init__()
        self.purposes = []

    async def prepare(self, context, logical_request, *, service, purpose=None):
        self.purposes.append(purpose)
        return await super().prepare(context, logical_request, service=service)


class Bus:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


@pytest.fixture
async def fold_case(registry_store):
    store, _ = registry_store
    manager, host = JobManager(store), FoldHost()
    config = BudgetConfig("host-budget-v1", (LocalBudgetScope("shared", max_calls=5, max_tokens=50000, max_inflight=2),),
                          host, max_pending=1, timeout_s=10)
    runtime = TaskRuntime(manager, tenant_id="fold-tenant", principal_id="host", executor_id="executor",
                          boot_id="boot", lease_ms=300000, boot_contract_hash="a" * 64, budget_config=config)
    sid = await store.create_session()
    actor = runtime.actor(sid)
    reg = await manager.register_job(actor, kind="send", session_id=sid, system_key="send",
                                     request={"model_attempt_protocol": 1})
    binding = await runtime.claim(actor, reg)
    await attempts.enable(binding)
    try:
        yield SimpleNamespace(store=store, manager=manager, host=host, runtime=runtime, sid=sid, binding=binding)
    finally:
        await runtime.shutdown(timeout_s=1)


def _request():
    return LLMRequest(messages=[{"role": "user", "content": "fold this"}], max_tokens=20)


async def _prepared(c):
    rows = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'")
    return [json.loads(row["payload_json"]) for row in rows]


async def test_fold_call_is_booked_once_as_fold_under_the_send(fold_case):
    c, bus = fold_case, Bus()
    model = SummaryModel()
    c.host.service = model
    token = _SEND.set(c.binding)
    try:
        response = await FoldCalls(session_id=c.sid, bus=bus).wrap(model).complete(_request())
    finally:
        _SEND.reset(token)
    assert response.raw_text == "<summary>folded</summary>" and model.calls == 1
    [prepared] = await _prepared(c)
    assert prepared["purpose"] == "fold" and prepared["model"] == "summary-model"
    assert c.host.purposes == ["fold"] and c.host.reserve_calls == 1
    assert len(c.host.applied) == 1 and c.host.apply_calls == 1
    started, completed = bus.events
    assert started.type == AgentEventType.LLM_CALL_STARTED and completed.type == AgentEventType.LLM_CALL_COMPLETED
    assert started.data.purpose == completed.data.purpose == "fold"
    assert completed.data.budget_managed is True and completed.data.total_tokens == 45
    assert completed.data.model_attempt_id == prepared["attempt_id"]
    assert c.binding.resource_evidence.get("resource_state") != "unknown"


async def test_fold_call_without_the_send_binding_is_refused_not_sent(fold_case):
    c = fold_case
    model = SummaryModel()
    calls = FoldCalls(session_id=c.sid)
    with pytest.raises(JobError, match="fold_call_unmanaged"):
        await calls.wrap(model).complete(_request())
    other = await c.store.create_session()
    token = _SEND.set(c.binding)
    try:
        with pytest.raises(JobError, match="fold_call_unmanaged"):
            await FoldCalls(session_id=other).wrap(model).complete(_request())
    finally:
        _SEND.reset(token)
    assert model.calls == 0 and await _prepared(c) == []


async def test_failed_fold_call_is_unknown_and_never_retried_by_a_fallback(fold_case):
    c = fold_case
    model = SummaryModel(error=RuntimeError("provider dropped"))
    c.host.service = model
    calls = FoldCalls(session_id=c.sid)
    token = _SEND.set(c.binding)
    try:
        with pytest.raises(RuntimeError):
            await calls.wrap(model).complete(_request())
        with pytest.raises(RuntimeError):
            await calls.wrap(model).complete(_request())
    finally:
        _SEND.reset(token)
    assert model.calls == 1 and len(await _prepared(c)) == 1
    assert c.binding.resource_evidence["resource_state"] == "unknown"
    [finished] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
    assert json.loads(finished["payload_json"])["status"] == "unknown"


async def test_cancelled_send_cancels_the_fold_through_a_swallowing_strategy(fold_case):
    c = fold_case
    model = SummaryModel()
    c.host.service = model
    stop = CancellationToken()
    stop.cancel("stop")
    calls = FoldCalls(session_id=c.sid, cancel_token=stop)
    context = FoldContext(session_id=c.sid, round_index=0, representation=ProjectedRepresentation(),
                          llm=calls.wrap(model), summary_llm=calls.wrap(model))
    rows = [SimpleNamespace(kind="user", send_index=1, content={"text": "hello"}, rendered_text="hello",
                            compact_from_send=None, compact_to_send=None)]
    token = _SEND.set(c.binding)
    try:
        assert await LLMSummaryFold(keep_last_sends=1).fold(rows, context=context) is None
    finally:
        _SEND.reset(token)
    with pytest.raises(CancellationRequested):
        calls.raise_if_cancelled()
    assert model.calls == 0 and await _prepared(c) == []


async def test_host_without_fold_support_refuses_the_call(fold_case):
    c = fold_case
    model = SummaryModel()
    c.host.service = model
    c.host.prepare = Host.prepare.__get__(c.host)
    token = _SEND.set(c.binding)
    try:
        with pytest.raises(TypeError):
            await FoldCalls(session_id=c.sid).wrap(model).complete(_request())
    finally:
        _SEND.reset(token)
    assert model.calls == 0 and c.host.reserve_calls == 0


async def test_prepared_purpose_is_optional_and_only_fold(fold_case):
    c = fold_case
    values = dict(round_index=0, retry_index=1, request_hash="a" * 64, model="m", max_tokens=10)
    with pytest.raises(JobError, match="invalid_model_attempt"):
        await attempts.prepare(c.binding, nonce="chat", purpose="chat", **values)
    plain = await attempts.prepare(c.binding, nonce="plain", **values)
    assert "purpose" not in plain.prepared


class SendModel:
    capabilities = SimpleNamespace(model="main-model")

    def __init__(self):
        self.requests = []

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        text = "<summary>FOLD-1</summary>" if "transcript" in str(request.messages) else f"d{len(self.requests)}"
        return LLMResponse(raw_text=text, token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


async def test_task_send_end_of_send_fold_runs_inside_the_send(registry_store):
    store, _ = registry_store
    runtime = TaskRuntime(JobManager(store), tenant_id="fold-loop", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=300000)
    model = SendModel()
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, config=AgentLoopConfig(
        delegation_policy="single_level", system_prompt="S", max_rounds=2, max_tokens=10,
        representation=ProjectedRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=2)))
    sid = await loop.new_session()
    for i in range(1, 4):
        assert (await loop.send(f"m{i}", session_id=sid)).status == "completed"
    compact = await store.latest_project_compact(sid)
    assert compact is not None and compact.content["summary"] == "FOLD-1"
    rows = await runtime.manager.db.fetchall(
        f"SELECT payload_json FROM {runtime.manager.events} WHERE event_type='model_attempt_prepared'")
    purposes = [json.loads(r["payload_json"]).get("purpose") for r in rows]
    assert purposes.count("fold") == 1 and purposes.count(None) == 3
    assert len(model.requests) == 4


async def test_task_send_verbatim_compaction_is_booked_as_fold(registry_store, monkeypatch):
    monkeypatch.delenv("CONTEXT_COMPACT_THRESHOLD", raising=False)
    store, _ = registry_store
    runtime = TaskRuntime(JobManager(store), tenant_id="fold-verbatim", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=300000)
    model = SendModel()
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, config=AgentLoopConfig(
        delegation_policy="single_level", system_prompt="S", max_rounds=2, max_tokens=10,
        representation=VerbatimRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=0.5)))
    sid = await loop.new_session()
    for i in range(1, 5):
        assert (await loop.send(f"m{i}", session_id=sid)).status == "completed"
    assert await store.list_compactions(sid)
    rows = await runtime.manager.db.fetchall(
        f"SELECT payload_json FROM {runtime.manager.events} WHERE event_type='model_attempt_prepared'")
    purposes = [json.loads(r["payload_json"]).get("purpose") for r in rows]
    folds = sum("transcript" in str(r.messages) for r in model.requests)
    assert folds >= 1 and purposes.count("fold") == folds and purposes.count(None) == len(model.requests) - folds
