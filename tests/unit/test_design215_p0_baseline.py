"""Design 215 P0: reproducible pre-change B, not a session-native acceptance test."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, JobRef
from power_loop.runtime.pricing import Free
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent
from tests.unit._workflow_fixture_203 import Model, leaf
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_send_row_profile import _snapshot
from tests.unit.test_send_sql_profile import _LEGACY_SQL_COLUMNS, Profiled, _category, _counts
from tests.write_meter import WriteMeter

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
OUT = os.environ.get("P0_BASELINE_OUT")


def echo_definition():
    return ToolDefinition(
        "echo", "Return input",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        pricing=Free("fixture"),
    )


async def echo(text):
    return text


async def measure(store, scenario, params, operation):
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    profiled = Profiled(original, meter)
    store._db = profiled
    try:
        before = await _snapshot(original, store.table_prefix)
        profiled.records.clear()
        profiled.tx_sites.clear()
        profiled.tx_categories.clear()
        profiled.transactions = 0
        if meter:
            await meter.reset()
        started = time.monotonic()
        detail = await operation()
        elapsed_s = time.monotonic() - started
        records = list(profiled.records)
        tx_sites = list(profiled.tx_sites)
        tx_categories = list(profiled.tx_categories)
        writes = await meter.report() if meter else None
        after = await _snapshot(original, store.table_prefix)
        names = sorted(set(before) | set(after))
        delta = {
            name: [after.get(name, (0, 0))[0] - before.get(name, (0, 0))[0],
                   after.get(name, (0, 0))[1] - before.get(name, (0, 0))[1]]
            for name in names if after.get(name) != before.get(name)
        }
        legacy_column_sql = [
            {"sql": record["sql"], "category": _category(record["chain"], record["sql"])}
            for record in records
            if any(re.search(rf"(?<![a-z0-9]){column}(?![a-z0-9])", record["sql"].lower())
                   for column in _LEGACY_SQL_COLUMNS)
        ]
        assert all(item["category"] == "legacy_job_control" for item in legacy_column_sql)
        report = {
            "baseline": "design215-P0-R1-B",
            "db": original.dialect.name,
            "scenario": scenario,
            **params,
            "total_sql": len(records),
            "transactions": profiled.transactions,
            "sql": _counts(records, tx_sites, tx_categories),
            "transaction_categories": dict(sorted(Counter(tx_categories).items())),
            "legacy_column_sql": legacy_column_sql,
            "writes": writes,
            "net_residual": {
                "rows": sum(value[0] for value in delta.values()),
                "bytes": sum(value[1] for value in delta.values()),
                "tables": delta,
            },
            "elapsed_s": elapsed_s,
            "detail": detail,
            "billing_measured": False,
            "billing_note": "No real BillingChargeEvent outlet exists in this SDK fixture; budgeted/paid flags are not billing evidence.",
            "scope": "accept through execution, child/driver results, caller result handling, owner release and final activation check; excludes setup and post-measurement close",
        }
        if os.environ.get("P0_BASELINE_RECORDS"):
            report["records"] = records
            report["tx_sites"] = tx_sites
        if OUT:
            with Path(OUT).open("a") as file:
                file.write(json.dumps(report, sort_keys=True) + "\n")
        assert len(records) == sum(report["sql"][name] for name in
            ("message", "session_control", "legacy_job_control", "billing", "periodic"))
        return report
    finally:
        store._db = original
        if meter:
            await meter.close()


async def ordinary_case(store, rounds, tools_per_round):
    registry = ToolRegistry()
    registry.register(echo_definition(), echo)

    class BatchModel(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            calls = None
            if self.calls < rounds:
                calls = [{
                    "id": f"call-{self.calls}-{index}", "type": "function",
                    "function": {"name": "echo", "arguments": json.dumps({"text": "ok"})},
                } for index in range(tools_per_round)]
            return LLMResponse(raw_text="done" if calls is None else "", tool_calls=calls)

        async def close(self):
            pass

    model = BatchModel()
    runner = await RunningAgent(
        store, model, registry,
        AgentLoopConfig(max_rounds=rounds + 1, max_tokens=20, compactor=None),
    ).start()

    async def operation():
        result = await runner.send("run")
        assert result.status == "completed" and model.calls == rounds
        return {"model_calls": model.calls, "tool_calls": (rounds - 1) * tools_per_round}

    try:
        return await measure(store, "ordinary", {
            "model_rounds": rounds, "tools_per_round": tools_per_round,
        }, operation)
    finally:
        await runner.aclose()


async def subagents_case(store, children, tmp_path):
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_spawn_agent(registry)
    gate = asyncio.Event()

    class ParentAndChildren(LLMService):
        def __init__(self):
            self.spawned = False

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("CHILD"):
                await gate.wait()
                return LLMResponse(raw_text="child result")
            if not self.spawned:
                self.spawned = True
                return LLMResponse(raw_text="", tool_calls=[{
                    "id": f"spawn-{index}", "type": "function", "function": {
                        "name": "spawn_agent", "arguments": json.dumps({
                            "task": f"compute {index}", "system_prompt": f"CHILD {index}", "tools": [],
                        }),
                    },
                } for index in range(children)])
            return LLMResponse(raw_text="parent done")

        async def close(self):
            pass

    runner = await RunningAgent(store, ParentAndChildren(), registry, AgentLoopConfig(compactor=None)).start()

    async def operation():
        result = await runner.send("start")
        assert result.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        receipts = [json.loads(message.content) for message in rows
                    if (message.tool_call_id or "").startswith("spawn-")]
        assert len(receipts) == children, receipts
        gate.set()
        for receipt in receipts:
            completed = await runner.runtime.client.wait(
                JobRef(**receipt["job_ref"]), actor=runner.actor, deadline_ms=now_ms() + 15000,
            )
            assert completed.status == "completed", completed
        await asyncio.wait_for(runner.drain(), 20)
        return {"children": children, "parent_result_processing": True}

    try:
        return await measure(store, "subagents", {"children": children}, operation)
    finally:
        gate.set()
        await runner.aclose()


async def workflow_case(store, leaves):
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()

    async def operation():
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "chain", "root": {"type": "sequence", "steps": [leaf(chr(65 + i)) for i in range(leaves)]}},
            message_id="workflow", actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 20000)
        assert result.status == "completed", result
        await asyncio.wait_for(runner.drain(), 20)
        return {"leaves": leaves, "driver_and_node_results": True, "entry": "direct_client"}

    try:
        return await measure(store, "workflow_direct", {"leaves": leaves}, operation)
    finally:
        await runner.aclose()


async def batch_case(store, inputs):
    gate = asyncio.Event()
    entered = asyncio.Event()

    class BatchModel(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await gate.wait()
                return LLMResponse(raw_text="initial done")
            return LLMResponse(raw_text="batch done")

        async def close(self):
            pass

    model = BatchModel()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(max_rounds=20, compactor=None)).start()

    async def operation():
        initial = asyncio.create_task(runner.send("initial", message_id="batch-0"))
        await asyncio.wait_for(entered.wait(), 5)
        refs = [InputRef(store.store_id, runner.target.session_id, f"batch-{index}")
                for index in range(1, inputs)]
        receipts = [await runner.runtime.client.send(
            runner.target, f"follow up {index}", message_id=ref.message_id,
            actor=runner.actor, deadline_ms=now_ms() + 20000,
        ) for index, ref in enumerate(refs, 1)]
        assert all(receipt.state == "accepted" for receipt in receipts)
        gate.set()
        assert (await initial).status == "completed"
        for ref in refs:
            result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms() + 20000)
            assert result.status == "completed", result
        await asyncio.wait_for(runner.drain(), 20)
        return {"accepted_inputs": inputs, "model_calls": model.calls}

    try:
        return await measure(store, "batch_inputs", {"inputs": inputs}, operation)
    finally:
        gate.set()
        await runner.aclose()


@pytest.mark.parametrize("rounds", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tools_per_round", [1, 4])
async def test_p0_baseline_ordinary(registry_store, rounds, tools_per_round):
    store, _ = registry_store
    await ordinary_case(store, rounds, tools_per_round)


@pytest.mark.parametrize("children", [1, 2, 4])
async def test_p0_baseline_subagents(registry_store, children, tmp_path):
    store, _ = registry_store
    await subagents_case(store, children, tmp_path)


@pytest.mark.parametrize("leaves", [1, 2, 4])
async def test_p0_baseline_workflow(registry_store, leaves):
    store, _ = registry_store
    await workflow_case(store, leaves)


@pytest.mark.parametrize("inputs", [1, 20])
async def test_p0_baseline_batch_inputs(registry_store, inputs):
    store, _ = registry_store
    await batch_case(store, inputs)
