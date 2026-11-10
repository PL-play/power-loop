"""Whole-send behavior and control I/O, not just isolated reducers."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

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
    runtime = TaskRuntime(manager, tenant_id="memory-send", principal_id="host", executor_id="executor", boot_id="boot", boot_contract_hash="a" * 64)
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
    loop = await RunningAgent(store, model, registry,
        AgentLoopConfig(max_rounds=models+1, max_tokens=20, compactor=None), task_runtime=runtime).start()
    sid = loop.target.session_id
    try:
        counted.sql = counted.transactions = 0
        counted.read_rows = counted.read_bytes = counted.parameter_bytes = 0
        counted.statements.clear()
        result = await loop.send("run")
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
        assert result.usage['total_tokens'] == 5 * models
        # Persisted public results retain usage; transient checkpoints may be cleaned.
        queried = await loop.runtime.client.wait(result.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
        assert queried.usage == result.usage
        assert not manager._execution_controls
    finally:
        await loop.aclose()
        store._db = original_db


async def test_independent_managed_effects_still_run(registry_store):
    from .test_managed_effects_192 import scenario
    c = await scenario(registry_store, call_count=2)
    messages = await c.store.load_all_messages(c.sid)
    assert c.state.prepared == c.state.dispatched == 2, [(m.role, m.content) for m in messages]
    events = await c.manager.db.fetchall(f"SELECT event_id,event_type FROM {c.manager.events}")
    assert all(0 < len(row['event_id']) <= 64 for row in events), events
    assert sum(row['event_type'] == 'managed_effect_resource_exit' for row in events) == 2
