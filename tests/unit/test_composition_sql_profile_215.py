"""Whole-tree SQL profiles for one-round subagents and result arrival timing."""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import register_workflow_tool
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .test_send_sql_profile import Profiled, _category, _counts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class TreeModel(LLMService):
    def __init__(self, count, arrival):
        self.count = count
        self.arrival = arrival
        self.parent_calls = 0
        self.child_gates = [asyncio.Event() for _ in range(count)]
        self.parent_gate = asyncio.Event()
        self.handled_batches = []

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith("CHILD-"):
            index = int(request.system_prompt.splitlines()[0].split("-", 1)[1])
            await self.child_gates[index].wait()
            return LLMResponse(raw_text=f"child {index} result")
        self.parent_calls += 1
        if self.parent_calls == 1:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": f"spawn-{index}", "type": "function", "function": {
                    "name": "spawn_agent",
                    "arguments": json.dumps({
                        "task": f"task {index}",
                        "system_prompt": f"CHILD-{index}",
                        "tools": [],
                    }),
                },
            } for index in range(self.count)])
        latest = next((m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"), "")
        if "agent_result" in latest:
            self.handled_batches.append(latest)
            return LLMResponse(raw_text="parent handled results")
        if self.arrival == "concentrated":
            await self.parent_gate.wait()
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("arrival", ["concentrated", "dispersed"])
async def test_subagent_tree_sql_profile(registry_store, count, arrival):
    store, _ = registry_store
    model = TreeModel(count, arrival)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(compactor=None), capacity=5,
    ).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    store._db = profiled = Profiled(original, meter)
    started = time.monotonic()
    try:
        if arrival == "concentrated":
            parent_task = asyncio.create_task(runner.send("delegate"))
            receipts = []
            for _ in range(500):
                rows = await original.fetchall(
                    f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
                    "AND tool_call_id LIKE 'spawn-%'",
                    (runner.target.session_id,),
                )
                if len(rows) == count:
                    receipts = [json.loads(row["content"]) for row in rows]
                    break
                await asyncio.sleep(0.01)
            assert len(receipts) == count
            for gate in model.child_gates:
                gate.set()
            starts = [InputRef(**receipt["input_ref"]) for receipt in receipts]
            child_results = await asyncio.gather(*[
                runner.runtime.client.wait(start, actor=runner.actor, deadline_ms=now_ms() + 10000)
                for start in starts
            ])
            assert all(result.status == "completed" for result in child_results)
            model.parent_gate.set()
            parent = await parent_task
            assert parent.status == "completed"
            parent_refs = [
                InputRef(
                    store.store_id, runner.target.session_id,
                    result_message_id(TargetRef(**receipt["target_ref"]), start, 1, "agent_result"),
                )
                for receipt, start in zip(receipts, starts, strict=True)
            ]
            await runner.runtime.client.wait(
                parent_refs[0], actor=runner.actor, deadline_ms=now_ms() + 10000,
            )
            await runner.drain()
            rows = await original.fetchall(
                f"SELECT item_id,delivered_send_index,delivered_seq,terminal_status "
                f"FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
                (runner.target.session_id,),
            )
            assert len(rows) == count
            assert len({row["delivered_send_index"] for row in rows}) == 1
            assert len({row["delivered_seq"] for row in rows}) == 1
            assert all(row["terminal_status"] == "completed" for row in rows)
            assert len(model.handled_batches) == 1
            for start in starts:
                assert start.message_id in model.handled_batches[0] or any(
                    receipt["input_ref"]["message_id"] == start.message_id for receipt in receipts
                )
        else:
            parent = await runner.send("delegate")
            assert parent.status == "completed"
            rows = await original.fetchall(
                f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
                "AND tool_call_id LIKE 'spawn-%' ORDER BY seq",
                (runner.target.session_id,),
            )
            receipts = [json.loads(row["content"]) for row in rows]
            assert len(receipts) == count
            starts = [InputRef(**receipt["input_ref"]) for receipt in receipts]
            parent_send_indexes = set()
            for index, receipt in enumerate(receipts):
                model.child_gates[index].set()
                start = starts[index]
                child = await runner.runtime.client.wait(
                    start, actor=runner.actor, deadline_ms=now_ms() + 10000,
                )
                assert child.status == "completed"
                parent_ref = InputRef(
                    store.store_id, runner.target.session_id,
                    result_message_id(TargetRef(**receipt["target_ref"]), start, 1, "agent_result"),
                )
                handled = await runner.runtime.client.wait(
                    parent_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
                )
                await runner.finish(parent_ref)
                assert handled.status == "completed"
                assert handled.final_text == "parent handled results"
                parent_send_indexes.add(handled.send_index)
            assert len(parent_send_indexes) == count
            assert len(model.handled_batches) == count
            assert model.parent_calls == count + 2
            for index, start in enumerate(starts):
                assert sum(f"child {index} result" in batch for batch in model.handled_batches) == 1
                assert sum(start.message_id in batch for batch in model.handled_batches) == 1
        await runner.drain()
        elapsed = time.monotonic() - started
        records, tx_sites = list(profiled.records), list(profiled.tx_sites)
        counts = _counts(records, tx_sites, profiled.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        writes = await meter.report() if meter else None
        if output := os.environ.get("P215_COMPOSITION_OUT"):
            with open(output, "a") as stream:
                stream.write(json.dumps({
                    "db": original.dialect.name,
                    "scenario": f"subagents_{count}_{arrival}",
                    "children": count,
                    "arrival": arrival,
                    "total_sql": len(records),
                    "counts": counts,
                    "transactions": profiled.transactions,
                    "writes": writes,
                    "elapsed_s": elapsed,
                    "records": records,
                    "tx_sites": tx_sites,
                }) + "\n")
    finally:
        model.parent_gate.set()
        for gate in model.child_gates:
            gate.set()
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()


class WorkflowTreeModel(LLMService):
    def __init__(self, count):
        self.count = count
        self.parent_calls = 0

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith("LEAF-"):
            return LLMResponse(raw_text="done " + request.system_prompt)
        self.parent_calls += 1
        if self.parent_calls == 1:
            branches = []
            for index in range(self.count):
                branches.append({
                    "type": "agent", "id": f"N{index}", "input": f"work {index}",
                    "spec": {"name": f"N{index}", "system_prompt": f"LEAF-{index}", "tools": []},
                })
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "workflow-profile", "type": "function", "function": {
                    "name": "workflow", "arguments": json.dumps({
                        "action": "create",
                        "spec": {"name": "profile", "root": (
                            branches[0] if self.count == 1 else {"type": "parallel", "branches": branches}
                        )},
                    }),
                },
            }])
        latest = next((m.get("content", "") for m in reversed(request.messages)
                       if m.get("role") == "user"), "")
        if "workflow_result" not in latest:
            return LLMResponse(raw_text="workflow accepted")
        return LLMResponse(raw_text="caller handled workflow")

    async def close(self):
        pass


@pytest.mark.parametrize("count", [1, 2, 4])
async def test_workflow_tree_sql_profile(registry_store, count):
    store, _ = registry_store
    model = WorkflowTreeModel(count)
    registry = ToolRegistry()
    register_workflow_tool(registry)
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(compactor=None), capacity=5,
    ).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    store._db = profiled = Profiled(original, meter)
    started = time.monotonic()
    try:
        parent = await runner.send("create workflow")
        assert parent.status == "completed"
        rows = await original.fetchall(
            f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
            "AND tool_call_id='workflow-profile'",
            (runner.target.session_id,),
        )
        receipt = json.loads(rows[0]["content"])
        start = InputRef(**receipt["input_ref"])
        driver = TargetRef(**receipt["target_ref"])
        result = await runner.runtime.client.wait(start, actor=runner.actor, deadline_ms=now_ms() + 15000)
        assert result.status == "completed" and len(result.structured_output["results"]) == count
        parent_ref = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(driver, start, 1, "workflow_result"),
        )
        await runner.runtime.worker.recover()
        handled = await runner.runtime.client.wait(
            parent_ref, actor=runner.actor, deadline_ms=now_ms() + 15000,
        )
        assert handled.status == "completed" and handled.final_text == "caller handled workflow"
        await runner.drain()
        elapsed = time.monotonic() - started
        records, tx_sites = list(profiled.records), list(profiled.tx_sites)
        counts = _counts(records, tx_sites, profiled.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        journal_records = [record for record in records if "session_runtime_state" in record.get("sql", "")]
        assert journal_records and all(
            _category(record["chain"], record["sql"]) == (
                "message" if "agent.stateful_loop:_record_history_mode" in record["chain"] else "session_control"
            ) for record in journal_records
        )
        writes = await meter.report() if meter else None
        if output := os.environ.get("P215_COMPOSITION_OUT"):
            with open(output, "a") as stream:
                stream.write(json.dumps({
                    "db": original.dialect.name,
                    "scenario": f"workflow_{count}",
                    "leaves": count,
                    "total_sql": len(records),
                    "counts": counts,
                    "transactions": profiled.transactions,
                    "writes": writes,
                    "elapsed_s": elapsed,
                    "records": records,
                    "tx_sites": tx_sites,
                }) + "\n")
    finally:
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()


class CombinedModel(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith(("CHILD-", "LEAF-")):
            return LLMResponse(raw_text="child result")
        self.calls += 1
        if self.calls <= 16:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": f"echo-{self.calls}", "type": "function",
                "function": {"name": "profile_echo", "arguments": "{}"}}])
        if self.calls == 17:
            calls = [{"id": f"combined-spawn-{i}", "type": "function", "function": {
                "name": "spawn_agent", "arguments": json.dumps({
                    "task": f"task {i}", "system_prompt": f"CHILD-{i}", "tools": []})}}
                for i in range(4)]
            branches = [{"type": "agent", "id": f"N{i}", "input": f"work {i}",
                         "spec": {"name": f"N{i}", "system_prompt": f"LEAF-{i}", "tools": []}}
                        for i in range(4)]
            calls.append({"id": "combined-workflow", "type": "function", "function": {
                "name": "workflow", "arguments": json.dumps({"action": "create", "spec": {
                    "name": "combined", "root": {"type": "parallel", "branches": branches}}})}})
            return LLMResponse(raw_text="", tool_calls=calls)
        return LLMResponse(raw_text="parent handled available results")

    async def close(self):
        pass


async def test_combined_long_loop_children_workflow_sql_profile(registry_store):
    """End-to-end 16 tool rounds + four children + a four-leaf workflow, including result ingestion."""
    from power_loop import ToolDefinition
    from power_loop.runtime.pricing import Free

    store, _ = registry_store
    model = CombinedModel()
    registry = ToolRegistry()
    registry.register(ToolDefinition("profile_echo", "measured ordinary tool",
                                     input_schema={"type": "object", "properties": {}},
                                     pricing=Free("measurement")), lambda: "echo result")
    register_spawn_agent(registry)
    register_workflow_tool(registry)
    runner = await RunningAgent(store, model, registry,
                                AgentLoopConfig(compactor=None, max_rounds=32), capacity=10).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    store._db = profiled = Profiled(original, meter)
    started = time.monotonic()
    try:
        assert (await runner.send("run the combined workload")).status == "completed"
        rows = await original.fetchall(
            f"SELECT content,name FROM {store.t.messages} WHERE session_id=? AND role='tool' "
            "AND tool_call_id LIKE 'combined-%'", (runner.target.session_id,))
        assert len(rows) == 5
        for row in rows:
            receipt = json.loads(row["content"])
            start, target = InputRef(**receipt["input_ref"]), TargetRef(**receipt["target_ref"])
            result = await runner.runtime.client.wait(start, actor=runner.actor, deadline_ms=now_ms() + 20000)
            assert result.status == "completed"
            kind = "workflow_result" if row["name"] == "workflow" else "agent_result"
            if kind == "workflow_result":
                assert len(result.structured_output["results"]) == 4
            parent_ref = InputRef(store.store_id, runner.target.session_id,
                                  result_message_id(target, start, 1, kind))
            assert (await runner.runtime.client.wait(
                parent_ref, actor=runner.actor, deadline_ms=now_ms() + 20000)).status == "completed"
        await runner.drain()
        tools = await original.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.messages} WHERE session_id=? AND role='tool' "
            "AND tool_call_id LIKE 'echo-%'", (runner.target.session_id,))
        assert tools["n"] == 16
        records, tx_sites = list(profiled.records), list(profiled.tx_sites)
        counts = _counts(records, tx_sites, profiled.tx_categories)
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        writes = await meter.report() if meter else None
        if output := os.environ.get("P215_COMPOSITION_OUT"):
            with open(output, "a") as stream:
                stream.write(json.dumps({"db": original.dialect.name,
                    "scenario": "combined_16_tools_4_children_4_leaves", "total_sql": len(records),
                    "counts": counts, "transactions": profiled.transactions, "writes": writes,
                    "elapsed_s": time.monotonic() - started, "records": records, "tx_sites": tx_sites}) + "\n")
    finally:
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()
