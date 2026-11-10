"""Ordinary billed send: one charge, then the same reservation cannot settle twice."""

from __future__ import annotations

import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.lightweight_budget import recover
from power_loop.runtime.task_runtime import TaskRuntime
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _Model(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text="done", token_usage=LLMTokenUsage(
            prompt_tokens=3, completion_tokens=2, total_tokens=5))

    def stream(self, request):
        raise AssertionError("nonstream fixture")

    async def close(self):
        pass


async def _send(store, host):
    config = BudgetConfig(
        "send-budget",
        (LocalBudgetScope("shared", max_calls=10, max_tokens=1000, max_inflight=4),),
        host,
    )
    runtime = TaskRuntime(
        JobManager(store), tenant_id="profile", principal_id="host", executor_id="executor",
        boot_id="boot", boot_contract_hash="a" * 64, budget_config=config)
    model = _Model()
    host.service = model
    agent = await RunningAgent(
        store, model, None, AgentLoopConfig(max_rounds=2, max_tokens=20, compactor=None),
        task_runtime=runtime).start()
    try:
        result = await agent.send("hello")
    finally:
        await agent.aclose()
        await runtime.shutdown(timeout_s=1)
    return result


async def _counters(store):
    row = await store._db.fetchone(f"SELECT counters_json FROM {store.table_prefix}task_budget_scopes")
    return json.loads(row["counters_json"])


async def test_ordinary_billed_send_charges_once(registry_store):
    store, _connect = registry_store
    host = Host()
    result = await _send(store, host)
    assert result.status == "completed"
    assert host.reserve_calls == 1 and host.apply_calls == 1
    jobs = await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs")
    assert jobs["n"] == 0
    attempt = await store._db.fetchone(
        f"SELECT execution_id,pending,phase,job_id FROM {store.table_prefix}task_budget_attempts")
    assert attempt["job_id"] is None and attempt["pending"] == 0 and attempt["phase"] == "settled"
    assert attempt["execution_id"].startswith("send:")
    assert await _counters(store) == {
        "used_calls": 1, "held_calls": 0, "used_tokens": 5, "held_tokens": 0, "inflight": 0}
    await recover(JobManager(store))
    assert await _counters(store) == {
        "used_calls": 1, "held_calls": 0, "used_tokens": 5, "held_tokens": 0, "inflight": 0}


async def test_unsettled_reservation_settles_by_estimate_once(registry_store):
    store, _connect = registry_store
    host = Host()
    assert (await _send(store, host)).status == "completed"
    manager = JobManager(store)
    prefix = store.table_prefix
    scope = await store._db.fetchone(f"SELECT scope_key,counters_json,counters_hash FROM {prefix}task_budget_scopes")
    counters = json.loads(scope["counters_json"])
    counters["held_calls"] += 4
    counters["held_tokens"] += 100
    counters["inflight"] += 1
    from power_loop.runtime.jobs import _hash, _json
    await store._db.execute(
        f"UPDATE {prefix}task_budget_scopes SET counters_json=?,counters_hash=? WHERE scope_key=?",
        (_json(counters), _hash(counters), scope["scope_key"]))
    state = {
        "execution_id": "send:left:2",
        "held_calls": 4,
        "held_tokens": 100,
        "inflight": 1,
        "estimate_calls": 1,
        "estimate_tokens": 100,
        "scopes": [{"scope_key": scope["scope_key"], "counters": counters, "counters_hash": _hash(counters)}],
    }
    await store._db.execute(
        f"INSERT INTO {prefix}task_budget_attempts "
        "(attempt_id,job_id,execution_id,tenant_id,principal_id,session_id,policy_key,"
        "version,phase,pending,state_hash,state_json,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("attempt_" + "b" * 48, None, "send:left:2", "profile", "host", "left", "policy",
         1, "reserved", 1, _hash(state), _json(state), 1, 1))
    await recover(manager)
    assert await _counters(store) == {
        "used_calls": 2, "held_calls": 0, "used_tokens": 105, "held_tokens": 0, "inflight": 0}
    await recover(manager)
    assert await _counters(store) == {
        "used_calls": 2, "held_calls": 0, "used_tokens": 105, "held_tokens": 0, "inflight": 0}
