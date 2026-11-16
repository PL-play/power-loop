"""Whole-tree SQL profiles for one-round subagents and result arrival timing."""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, LLMResponse, ProjectedRepresentation
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id
from power_loop.contracts.hooks import HookPoint
from power_loop.core.execution_context import get_execution_context
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import register_workflow_tool
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .test_projection_stream_sql_216 import StreamModel, _code_collection, _config_record
from .test_send_sql_profile import Profiled, _category, _counts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ReadWork:
    """Application row images; meter/auxiliary queries use the original DB separately."""
    def __init__(self, db, reads=None):
        self.db = db
        self.reads = [] if reads is None else reads

    def __getattr__(self, name):
        return getattr(self.db, name)

    async def read(self, method, sql, params):
        value = await getattr(self.db, method)(sql, params)
        rows = value if method == "fetchall" else ([] if value is None else [value])
        self.reads.append({"sql": " ".join(sql.split()), "params": list(params), "rows": len(rows),
                           "bytes": sum(len(json.dumps(dict(r), default=str, ensure_ascii=False).encode())
                                        for r in rows)})
        return value

    async def fetchall(self, sql, params=()):
        return await self.read("fetchall", sql, params)

    async def fetchone(self, sql, params=()):
        return await self.read("fetchone", sql, params)

    @asynccontextmanager
    async def transaction(self):
        async with self.db.transaction() as tx:
            yield ReadWork(tx, self.reads)

    @asynccontextmanager
    async def read_snapshot(self):
        async with self.db.read_snapshot() as tx:
            yield ReadWork(tx, self.reads)


class CompositionModel(StreamModel):
    async def respond(self, request, response, **kwargs):
        sid = get_execution_context().session_id
        if not hasattr(self, "streamed_calls"):
            self.streamed_calls, self.streamed_tools = Counter(), Counter()
        self.streamed_calls[sid] += 1
        self.streamed_tools[sid] += len(response.tool_calls or [])
        # A separate S01 stream emitter per call keeps parallel leaves independent.
        emitter = StreamModel(text=response.raw_text, tool_calls=response.tool_calls,
                              deltas=3, think_deltas=1, duration_s=0.01)
        return await emitter.complete(request, **kwargs)


class CompositionProfiled(Profiled):
    def note(self, sql, rows=None):
        super().note(sql, rows)
        ctx = get_execution_context(required=False)
        self.records[-1].update(
            task=asyncio.current_task().get_name(),
            execution=None if ctx is None else {"session_id": ctx.session_id, "send_index": ctx.send_index})


def observe(runner):
    receipts, condition, notifications = {}, asyncio.Condition(), Counter()

    async def after_tool(ctx):
        if ctx.tool_name in {"spawn_agent", "workflow"}:
            async with condition:
                receipts[ctx.tool_call["id"]] = (ctx.tool_name, json.loads(ctx.output))
                condition.notify_all()

    runner.hooks.register(HookPoint.TOOL_AFTER, after_tool)
    publish = runner.runtime.transport.publish

    async def counted(packet):
        notifications[packet.channel + ":" + str((packet.payload or {}).get("kind", "hint"))] += 1
        return await publish(packet)

    runner.runtime.transport.publish = counted

    async def ready(count):
        async with asyncio.timeout(15), condition:
            await condition.wait_for(lambda: len(receipts) >= count)
        return [receipts[key] for key in sorted(receipts)]

    runner._composition_receipts = ready
    runner._composition_notifications = notifications


def emit_record(output, runner, profiled, payload):
    records = payload["records"]
    reads = profiled.snapshot["reads"]
    assert [r["sql"] for r in reads] == [r["sql"] for r in records if r["rows"] is not None]
    if payload.get("writes"):
        # Triggers measure rows/bytes. We do not run extra per-transaction meter reads.
        payload["writes"].update(max_tx_rows=None, max_tx_bytes=None)
    payload.update(code=_code_collection(), config=_config_record(runner.config, runner.runtime.transport),
                   notifications=profiled.snapshot["notifications"], read_work=list(reads),
                   returned_rows=sum(r["rows"] for r in reads),
                   returned_bytes=sum(r["bytes"] for r in reads),
                   tx_categories=profiled.snapshot["tx_categories"],
                   actual_M=sum(runner.model.streamed_calls.values()),
                   actual_T=sum(runner.model.streamed_tools.values()),
                   models_per_session=dict(runner.model.streamed_calls),
                   tools_per_session=dict(runner.model.streamed_tools),
                   timing="request through worker drain; frozen before auxiliary evidence",
                   streaming={"text_deltas_per_model": 3, "think_deltas_per_model": 1,
                              "duration_per_model_s": 0.01})
    if output:
        with open(output, "a") as stream:
            stream.write(json.dumps(payload) + "\n")


def freeze_request(runner, profiled):
    # Background timer reads may continue while auxiliary evidence is collected.
    # Freeze the entire request boundary, not only elapsed_s or the SQL list.
    snapshot = {
        "records": list(profiled.records), "tx_sites": list(profiled.tx_sites),
        "tx_categories": list(profiled.tx_categories), "transactions": profiled.transactions,
        "reads": list(profiled.db.reads), "notifications": dict(runner._composition_notifications),
    }
    profiled.snapshot = snapshot
    return snapshot


async def _session_runs(store, db, sid):
    """Actual run coordinates from persisted messages, with state as cross-check.

    `send_index` on the messages of one session is the run coordinate; distinct values
    are actual runs. `session_state.send_index` is recorded separately as a
    cross-check, not used to replace the message evidence.
    """
    rows = await db.fetchall(
        f"SELECT DISTINCT send_index FROM {store.t.messages} "
        "WHERE session_id=? AND send_index IS NOT NULL ORDER BY send_index", (sid,))
    state = await db.fetchone(
        f"SELECT send_index FROM {store.t.session_state} WHERE session_id=?", (sid,))
    user_batches = await db.fetchall(
        f"SELECT send_index,COUNT(*) AS n FROM {store.t.messages} "
        "WHERE session_id=? AND role='user' GROUP BY send_index", (sid,))
    return {
        "session_id": sid,
        "send_indexes": [int(row["send_index"]) for row in rows],
        "state_send_index": int(state["send_index"]) if state is not None else None,
        "follow_up_batches": sum(max(0, int(row["n"]) - 1) for row in user_batches),
    }


async def _run_evidence(store, db, parent_sid):
    """Parent, sub-agent children, workflow drivers and leaves, with actual R."""
    sessions = await db.fetchall(
        f"SELECT session_id,session_kind,parent_session_id FROM {store.t.sessions}")
    children = [row for row in sessions
                if row["parent_session_id"] == parent_sid and row["session_kind"] == "agent"]
    drivers = [row for row in sessions
               if row["parent_session_id"] == parent_sid and row["session_kind"] == "workflow_driver"]
    driver_ids = {row["session_id"] for row in drivers}
    leaves = [row for row in sessions
              if row["parent_session_id"] in driver_ids and row["session_kind"] == "agent"]
    parent = await _session_runs(store, db, parent_sid)
    child_runs = [await _session_runs(store, db, row["session_id"]) for row in children]
    driver_runs = [await _session_runs(store, db, row["session_id"]) for row in drivers]
    leaf_runs = [await _session_runs(store, db, row["session_id"]) for row in leaves]
    return {
        "parent": parent,
        "children": child_runs,
        "drivers": driver_runs,
        "leaves": leaf_runs,
        "counts": {
            "parent_R": len(parent["send_indexes"]),
            "children_R": sum(len(item["send_indexes"]) for item in child_runs),
            "drivers_R": sum(len(item["send_indexes"]) for item in driver_runs),
            "leaves_R": sum(len(item["send_indexes"]) for item in leaf_runs),
            "children_sessions": len(child_runs),
            "driver_sessions": len(driver_runs),
            "leaf_sessions": len(leaf_runs),
        },
    }


class TreeModel(CompositionModel):
    def __init__(self, count, arrival):
        self.count = count
        self.arrival = arrival
        self.parent_calls = 0
        self.child_gates = [asyncio.Event() for _ in range(count)]
        self.parent_gate = asyncio.Event()
        self.handled_batches = []
        self.parent_ready = asyncio.Event()

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith("CHILD-"):
            index = int(request.system_prompt.splitlines()[0].split("-", 1)[1])
            await self.child_gates[index].wait()
            return await self.respond(request, LLMResponse(raw_text=f"child {index} result"), **kwargs)
        self.parent_calls += 1
        if self.parent_calls == 1:
            return await self.respond(request, LLMResponse(raw_text="", tool_calls=[{
                "id": f"spawn-{index}", "type": "function", "function": {
                    "name": "spawn_agent",
                    "arguments": json.dumps({
                        "task": f"task {index}",
                        "system_prompt": f"CHILD-{index}",
                        "tools": [],
                    }),
                },
            } for index in range(self.count)]), **kwargs)
        latest = next((m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"), "")
        if "agent_result" in latest:
            self.handled_batches.append(latest)
            return await self.respond(request, LLMResponse(raw_text="parent handled results"), **kwargs)
        self.parent_ready.set()
        if self.arrival == "concentrated":
            await self.parent_gate.wait()
        return await self.respond(request, LLMResponse(raw_text="parent done"), **kwargs)

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
        store, model, registry, AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation()), capacity=5,
    ).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    observe(runner)
    store._db = profiled = CompositionProfiled(ReadWork(original))
    started = time.monotonic()
    try:
        if arrival == "concentrated":
            parent_task = asyncio.create_task(runner.send("delegate"))
            receipts = [receipt for _, receipt in await runner._composition_receipts(count)]
            await asyncio.wait_for(model.parent_ready.wait(), 10)
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
            assert len(model.handled_batches) == 1
            for start in starts:
                assert start.message_id in model.handled_batches[0] or any(
                    receipt["input_ref"]["message_id"] == start.message_id for receipt in receipts
                )
        else:
            parent = await runner.send("delegate")
            assert parent.status == "completed"
            receipts = [receipt for _, receipt in await runner._composition_receipts(count)]
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
        snapshot = freeze_request(runner, profiled)
        if arrival == "concentrated":
            rows = await original.fetchall(
                f"SELECT item_id,delivered_send_index,delivered_seq,terminal_status "
                f"FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
                (runner.target.session_id,),
            )
            assert len(rows) == count
            assert len({row["delivered_send_index"] for row in rows}) == 1
            assert len({row["delivered_seq"] for row in rows}) == 1
            assert all(row["terminal_status"] == "completed" for row in rows)
        records, tx_sites = snapshot["records"], snapshot["tx_sites"]
        counts = _counts(records, tx_sites, snapshot["tx_categories"])
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        writes = await meter.report() if meter else None
        run_evidence = await _run_evidence(store, original, runner.target.session_id)
        follow_up_claims = sum(
            1 for record in records
            if any("claim_follow_up" in site for site in record["chain"]))
        emit_record(os.environ.get("P215_COMPOSITION_OUT"), runner, profiled, {
                    "db": original.dialect.name,
                    "scenario": f"subagents_{count}_{arrival}",
                    "children": count,
                    "arrival": arrival,
                    "total_sql": len(records),
                    "counts": counts,
                    "transactions": snapshot["transactions"],
                    "writes": writes,
                    "elapsed_s": elapsed,
                    "runs": run_evidence,
                    "follow_up_claim_sql": follow_up_claims,
                    "model_calls": {
                        "parent_M": model.parent_calls,
                        "children_M": count,
                        "total_M": model.parent_calls + count,
                        "note": "parent_M counts complete() calls, not runs",
                    },
                    "records": records,
                    "tx_sites": tx_sites,
                })
    finally:
        model.parent_gate.set()
        for gate in model.child_gates:
            gate.set()
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()


class WorkflowTreeModel(CompositionModel):
    def __init__(self, count):
        self.count = count
        self.parent_calls = 0
        self.leaf_started = [asyncio.Event() for _ in range(count)]
        self.release_leaves = asyncio.Event()

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith("LEAF-"):
            index = int(request.system_prompt.splitlines()[0].split("-", 1)[1])
            self.leaf_started[index].set()
            await self.release_leaves.wait()
            return await self.respond(request, LLMResponse(raw_text="done " + request.system_prompt), **kwargs)
        self.parent_calls += 1
        if self.parent_calls == 1:
            branches = []
            for index in range(self.count):
                branches.append({
                    "type": "agent", "id": f"N{index}", "input": f"work {index}",
                    "spec": {"name": f"N{index}", "system_prompt": f"LEAF-{index}", "tools": []},
                })
            return await self.respond(request, LLMResponse(raw_text="", tool_calls=[{
                "id": "workflow-profile", "type": "function", "function": {
                    "name": "workflow", "arguments": json.dumps({
                        "action": "create",
                        "spec": {"name": "profile", "root": (
                            branches[0] if self.count == 1 else {"type": "parallel", "branches": branches}
                        )},
                    }),
                },
            }]), **kwargs)
        latest = next((m.get("content", "") for m in reversed(request.messages)
                       if m.get("role") == "user"), "")
        if "workflow_result" not in latest:
            return await self.respond(request, LLMResponse(raw_text="workflow accepted"), **kwargs)
        return await self.respond(request, LLMResponse(raw_text="caller handled workflow"), **kwargs)

    async def close(self):
        pass


@pytest.mark.parametrize("count", [1, 2, 4])
async def test_workflow_tree_sql_profile(registry_store, count):
    store, _ = registry_store
    model = WorkflowTreeModel(count)
    registry = ToolRegistry()
    register_workflow_tool(registry)
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation()), capacity=5,
    ).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    observe(runner)
    store._db = profiled = CompositionProfiled(ReadWork(original))
    started = time.monotonic()
    try:
        parent = await runner.send("create workflow")
        assert parent.status == "completed"
        receipt = (await runner._composition_receipts(1))[0][1]
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in model.leaf_started)), 10)
        model.release_leaves.set()
        start = InputRef(**receipt["input_ref"])
        driver = TargetRef(**receipt["target_ref"])
        result = await runner.runtime.client.wait(start, actor=runner.actor, deadline_ms=now_ms() + 15000)
        assert result.status == "completed" and len(result.structured_output["results"]) == count
        parent_ref = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(driver, start, 1, "workflow_result"),
        )
        handled = await runner.runtime.client.wait(
            parent_ref, actor=runner.actor, deadline_ms=now_ms() + 15000,
        )
        assert handled.status == "completed" and handled.final_text == "caller handled workflow"
        await runner.drain()
        elapsed = time.monotonic() - started
        snapshot = freeze_request(runner, profiled)
        records, tx_sites = snapshot["records"], snapshot["tx_sites"]
        counts = _counts(records, tx_sites, snapshot["tx_categories"])
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        journal_records = [record for record in records if any(
            site.startswith("workflow.journal:") for site in record["chain"])]
        assert journal_records and all(
            _category(record["chain"], record["sql"]) == "session_control" for record in journal_records)
        writes = await meter.report() if meter else None
        run_evidence = await _run_evidence(store, original, runner.target.session_id)
        follow_up_claims = sum(
            1 for record in records
            if any("claim_follow_up" in site for site in record["chain"]))
        emit_record(os.environ.get("P215_COMPOSITION_OUT"), runner, profiled, {
                    "db": original.dialect.name,
                    "scenario": f"workflow_{count}",
                    "leaves": count,
                    "total_sql": len(records),
                    "counts": counts,
                    "transactions": snapshot["transactions"],
                    "writes": writes,
                    "elapsed_s": elapsed,
                    "runs": run_evidence,
                    "follow_up_claim_sql": follow_up_claims,
                    "model_calls": {
                        "parent_M": model.parent_calls,
                        "leaves_M": count,
                        "driver_M": 0,
                        "total_M": model.parent_calls + count,
                        "note": "driver does not call a model; parent_M is complete() calls, not runs",
                    },
                    "records": records,
                    "tx_sites": tx_sites,
                })
    finally:
        model.release_leaves.set()
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()


class CombinedModel(CompositionModel):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        if request.system_prompt.startswith(("CHILD-", "LEAF-")):
            return await self.respond(request, LLMResponse(raw_text="child result"), **kwargs)
        self.calls += 1
        if self.calls <= 16:
            return await self.respond(request, LLMResponse(raw_text="", tool_calls=[{
                "id": f"echo-{self.calls}", "type": "function",
                "function": {"name": "profile_echo", "arguments": "{}"}}]), **kwargs)
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
            return await self.respond(request, LLMResponse(raw_text="", tool_calls=calls), **kwargs)
        return await self.respond(request, LLMResponse(raw_text="parent handled available results"), **kwargs)

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
                                AgentLoopConfig(context_budget_tokens=10**9, max_rounds=32, representation=ProjectedRepresentation()), capacity=10).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    observe(runner)
    store._db = profiled = CompositionProfiled(ReadWork(original))
    started = time.monotonic()
    try:
        assert (await runner.send("run the combined workload")).status == "completed"
        receipts = await runner._composition_receipts(5)
        assert len(receipts) == 5
        for tool_name, receipt in receipts:
            start, target = InputRef(**receipt["input_ref"]), TargetRef(**receipt["target_ref"])
            result = await runner.runtime.client.wait(start, actor=runner.actor, deadline_ms=now_ms() + 20000)
            assert result.status == "completed"
            kind = "workflow_result" if tool_name == "workflow" else "agent_result"
            if kind == "workflow_result":
                assert len(result.structured_output["results"]) == 4
            parent_ref = InputRef(store.store_id, runner.target.session_id,
                                  result_message_id(target, start, 1, kind))
            assert (await runner.runtime.client.wait(
                parent_ref, actor=runner.actor, deadline_ms=now_ms() + 20000)).status == "completed"
        await runner.drain()
        elapsed = time.monotonic() - started
        snapshot = freeze_request(runner, profiled)
        tools = await original.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.messages} WHERE session_id=? AND role='tool' "
            "AND tool_call_id LIKE 'echo-%'", (runner.target.session_id,))
        assert tools["n"] == 16
        records, tx_sites = snapshot["records"], snapshot["tx_sites"]
        counts = _counts(records, tx_sites, snapshot["tx_categories"])
        assert counts["legacy_job_control"] == 0 and counts["billing"] == 0
        writes = await meter.report() if meter else None
        run_evidence = await _run_evidence(store, original, runner.target.session_id)
        follow_up_claims = sum(
            1 for record in records
            if any("claim_follow_up" in site for site in record["chain"]))
        emit_record(os.environ.get("P215_COMPOSITION_OUT"), runner, profiled, {"db": original.dialect.name,
                    "scenario": "combined_16_tools_4_children_4_leaves", "total_sql": len(records),
                    "counts": counts, "transactions": snapshot["transactions"], "writes": writes,
                    "elapsed_s": elapsed, "runs": run_evidence,
                    "follow_up_claim_sql": follow_up_claims,
                    "model_calls": {
                        "parent_M": model.calls,
                        "children_M": 4,
                        "leaves_M": 4,
                        "driver_M": 0,
                        "total_M": model.calls + 8,
                        "note": "child/leaf calls are counted as one each; parent_M is complete() calls",
                    },
                    "records": records, "tx_sites": tx_sites})
    finally:
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()


@pytest.mark.parametrize("models,tools", [(1, 0), (2, 1), (2, 4), (4, 1), (4, 4)])
async def test_ordinary_stream_matrix(registry_store, models, tools):
    from power_loop import ToolDefinition
    from power_loop.runtime.pricing import Free

    class Ordinary(CompositionModel):
        def __init__(self):
            self.calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            calls = [{"id": f"echo-{self.calls}-{i}", "type": "function", "function": {
                "name": "echo", "arguments": "{}"}} for i in range(tools)] if self.calls < models else []
            return await self.respond(request, LLMResponse(raw_text="" if calls else "done", tool_calls=calls), **kwargs)

    store, _ = registry_store
    registry = ToolRegistry()
    registry.register(ToolDefinition("echo", "ordinary echo", input_schema={"type": "object"},
                                     pricing=Free("measurement")), lambda: "echo")
    model = Ordinary()
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(
        context_budget_tokens=10**9, max_rounds=8, representation=ProjectedRepresentation())).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get("P0_WRITE_METER") else None
    observe(runner)
    store._db = profiled = CompositionProfiled(ReadWork(original))
    started = time.monotonic()
    try:
        assert (await runner.send("ordinary")).status == "completed"
        await runner.drain()
        elapsed = time.monotonic() - started
        snapshot = freeze_request(runner, profiled)
        records, tx_sites = snapshot["records"], snapshot["tx_sites"]
        writes = await meter.report() if meter else None
        evidence = await _run_evidence(store, original, runner.target.session_id)
        assert model.calls == models
        emit_record(os.environ.get("P215_COMPOSITION_OUT"), runner, profiled, {
            "db": original.dialect.name, "scenario": f"ordinary_{models}x{tools}",
            "total_sql": len(records), "counts": _counts(records, tx_sites, snapshot["tx_categories"]),
            "transactions": snapshot["transactions"], "writes": writes, "elapsed_s": elapsed,
            "runs": evidence, "records": records, "tx_sites": tx_sites,
            "follow_up_claim_sql": 0,
            "model_calls": {"parent_M": models, "total_M": models},
        })
    finally:
        store._db = original
        await runner.aclose()
        if meter:
            await meter.close()
