"""Whole-send behavior and control I/O, not just isolated reducers."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("models,tools_per_round", [(1, 1), (4, 1), (8, 1), (16, 1), (4, 8)])
@pytest.mark.parametrize("budgeted", [False, True])
async def test_ordinary_callbacks_and_models_complete_once(registry_store, models, tools_per_round, budgeted, record_property):
    from .test_execution_checkpoints import CountedDatabase
    store, _connect = registry_store
    original_db = store._db
    store._db = counted = CountedDatabase(original_db)
    manager = JobManager(store)
    from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
    from power_loop.runtime.response_capture import ResponseCaptureReceipt

    from .test_model_budgets_192 import Host
    host = Host()
    captures = {}
    async def capture(packet):
        assert captures.setdefault(packet.capture_id, packet) == packet
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, "fixture-original")
    async def tool_plan(context, call, *, pricing):
        return None
    host.capture_response, host.prepare_tool = capture, tool_plan
    config = BudgetConfig("send-budget", (LocalBudgetScope("shared", max_calls=10000, max_tokens=1000000, max_inflight=64),), host)
    runtime = TaskRuntime(manager, tenant_id="memory-send", principal_id="host", executor_id="executor", boot_id="boot", boot_contract_hash="a" * 64,
                          budget_config=config if budgeted else None)
    seen = []
    registry = ToolRegistry()

    async def echo(text):
        seen.append(text)
        return text
    registry.register(ToolDefinition("echo", "Return input", input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                                     pricing=Free("fixture")), echo)

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            tools = [{"id": f"call-{self.calls}-{i}", "type": "function", "function": {"name": "echo", "arguments": '{"text":"ok"}'}}
                     for i in range(tools_per_round)] if self.calls < models else None
            return LLMResponse(raw_text="done" if tools is None else "", tool_calls=tools,
                               token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        def stream(self, request):
            raise AssertionError("nonstream fixture")
        async def close(self):
            pass

    model = Model()
    host.service = model
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(max_rounds=models + 1, max_tokens=20, compactor=None, delegation_policy="single_level"))
    sid = await loop.new_session()
    try:
        counted.sql = counted.transactions = 0
        counted.read_rows = counted.read_bytes = counted.parameter_bytes = 0
        counted.statements.clear()
        result = await loop.send("run", sid)
        total_sql, total_transactions = counted.sql, counted.transactions
        checkpoint_sql = sum("execution_batches" in sql or "execution_checkpoints" in sql for sql in counted.statements)
        statement_count = len(counted.statements)
        assert total_sql == statement_count
        if models == 4 and tools_per_round == 1:
            record_property("queries", json.dumps(Counter(sql.replace(store.table_prefix, "pl_") for sql in counted.statements)))
        record_property("send_cost", json.dumps(dict(models=models, tools=(models - 1) * tools_per_round, budgeted=budgeted,
                        sql=total_sql, transactions=total_transactions, checkpoint_sql=checkpoint_sql,
                        returned_rows=counted.read_rows, returned_bytes=counted.read_bytes, parameter_bytes=counted.parameter_bytes,
                        message_queries=sum(sql.startswith("SELECT") and store.t.messages in sql for sql in counted.statements))))
        messages = await store.load_all_messages(sid)
        assert seen == ["ok"] * ((models - 1) * tools_per_round), [(m.role, m.content) for m in messages]
        assert result.status == "completed" and model.calls == models
        [row] = await manager.db.fetchall(f"SELECT summary_json FROM {manager.prefix}execution_checkpoints WHERE job_id=(SELECT job_id FROM {manager.jobs} ORDER BY created_at DESC,job_id DESC LIMIT 1)")
        summary = json.loads(row['summary_json'])
        assert summary['models'] == models and summary['tools'] == len(seen)
        if budgeted:
            assert len(captures) == models
            assert len(host.applied) == host.reserve_calls == models
            [scope] = await manager.db.fetchall(f"SELECT counters_json FROM {manager.prefix}task_budget_scopes")
            assert json.loads(scope["counters_json"]) == dict(used_calls=models, used_tokens=5 * models,
                                                             held_calls=0, held_tokens=0, inflight=0)
        assert not manager._execution_controls and not manager._live_budget_bindings
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)
        store._db = original_db


async def test_independent_managed_effects_still_run(registry_store):
    from .test_managed_effects_192 import scenario
    c = await scenario(registry_store, call_count=2)
    messages = await c.store.load_all_messages(c.sid)
    assert c.state.prepared == c.state.dispatched == 2, [(m.role, m.content) for m in messages]
    events = await c.manager.db.fetchall(f"SELECT event_id,event_type FROM {c.manager.events}")
    assert all(0 < len(row['event_id']) <= 64 for row in events), events
    assert sum(row['event_type'] == 'managed_effect_resource_exit' for row in events) == 2
