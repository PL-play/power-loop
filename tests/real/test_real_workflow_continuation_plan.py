"""Actual model tools repair a settled branch; stale review never reaches assembly.

This exercises the legacy owning engine, not the still-closed TaskRuntime resume
adapter. Gates control interleavings only; all model replies come from the API.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow, journal, runner

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def until(predicate):
    async with asyncio.timeout(60):
        while not predicate():
            await asyncio.sleep(0.01)


class RepairProvider(BoundedProvider):
    def __init__(self):
        super().__init__(limit=16)
        self.sibling_entered, self.release_sibling = asyncio.Event(), asyncio.Event()
        self.repair_entered, self.release_repair = asyncio.Event(), asyncio.Event()
        self.roles = Counter()
        self.node_responses = []

    async def complete(self, request, **kwargs):
        role = "MAIN" if get_session_id() == self.root_sid else next(
            r for r in ("CALCULATE", "REVIEW", "SIBLING", "ASSEMBLY") if "ROLE-"+r in (request.system_prompt or ""))
        self.roles[role] += 1
        if role == "SIBLING":
            self.sibling_entered.set()
            await self.release_sibling.wait()
        if role == "CALCULATE" and "VERSION-43" in str(request.messages):
            self.repair_entered.set()
            await self.release_repair.wait()
        response = await super().complete(request, **kwargs)
        if role != "MAIN":
            self.node_responses.append(response)
        return response


async def test_real_tool_continuation_invalidates_review_and_waits_before_assembly(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "continuation-plan.db"))
    provider = RepairProvider()
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_tokens=2048, max_rounds=4,
            system_prompt="Use exactly the tool the user requests, once. Then reply WAITING without tools or pass_turn. Never launch replacement workflows."))
    parent = await loop.new_session()
    provider.root_sid = parent
    handles, notifications, accepted = [], [], []

    def leaf(name, role, instruction, **kwargs):
        return {"type": "agent", "id": name, "spec": {"name": name, "tools": [], "max_rounds": 3,
            "system_prompt": "ROLE-"+role+". "+instruction+" Reply only with the supplied exact code, without tools or pass_turn."}, **kwargs}
    specification = {"name": "real-controlled-repair", "root": {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [
            {"type": "sequence", "steps": [
                leaf("calculate", "CALCULATE", "Report the supplied result code verbatim. A later correction replaces the original code.", input="The result code is VERSION-42. Report VERSION-42 verbatim."),
                leaf("review", "REVIEW", "Copy the exact VERSION code provided in context. Preserve every character.", input="The context below contains the current result code. Copy its VERSION-<integer> token exactly. Output only that token.", inputs_from=["calculate"])]},
            leaf("sibling", "SIBLING", "Copy the code supplied in the user message.", input="Reply exactly SIBLING-READY.")]},
        leaf("assembly", "ASSEMBLY", "Copy the exact VERSION code provided in context. Preserve every character.", input="The context below contains the reviewed code. Copy its VERSION-<integer> token exactly. Output only that token.", inputs_from=["review"])]}}

    async def on_complete(completion):
        saved = await journal.read(store, parent, completion.run_id)
        assert saved["status"] == "completed"
        notifications.append(completion)

    async def launch():
        workflow = create_workflow(specification, parent_loop=get_current_loop(), parent_session_id=get_session_id())
        handle = await workflow.start(detached=True, on_complete=on_complete)
        handles.append(handle)
        return json.dumps({"run_id": handle.run_id, "status": "queued"})

    async def repair():
        runner.queue_node_continuation(handles[0].run_id, "calculate",
            "Correction: the current result code is VERSION-43 instead of VERSION-42. Report VERSION-43 verbatim.", 3)
        accepted.append(1)
        return json.dumps({"run_id": handles[0].run_id, "status": "accepted"})

    for name, handler in (("launch_workflow", launch), ("repair_calculation", repair)):
        registry.register(ToolDefinition(name=name, description="Execute the predefined "+name+" once",
            input_schema={"type": "object"}, pricing=Free("test_fixture")), handler)
    try:
        async with asyncio.timeout(180):
            first = await loop.send("Call launch_workflow exactly once, then reply WAITING.", parent)
            assert first.status == "completed" and len(handles) == 1
            engine = runner._ENGINES[handles[0].run_id]
            branch = engine._spec.root.steps[0].branches[0]
            await provider.sibling_entered.wait()
            await until(lambda: "review" in engine._results and not engine._frames[id(branch)]["active"])
            assert engine._results["calculate"].text.strip() == "VERSION-42"
            assert engine._results["review"].text.strip() == "VERSION-42"
            calculate_sid = engine._results["calculate"].session_id
            review_sid = engine._results["review"].session_id
            original_rows = await store.load_all_messages(calculate_sid)
            second = await loop.send("Call repair_calculation exactly once, then reply WAITING. Do not launch a new workflow.", parent)
            assert second.status == "completed" and accepted == [1]
            await provider.repair_entered.wait()
            pending = await journal.read(store, parent, handles[0].run_id)
            assert next(s for s in pending["steps"] if s["node_id"] == "review")["status"] == "pending"
            provider.release_sibling.set()
            await until(lambda: bool(engine._continuation_barriers))
            assert provider.roles["REVIEW"] == 1 and provider.roles["ASSEMBLY"] == 0
            assert not handles[0].task.done() and not notifications
            provider.release_repair.set()
            await handles[0].task
            saved = await journal.read(store, parent, handles[0].run_id)
            assert saved["status"] == "completed" and len(notifications) == 1
            assert engine._results["calculate"].session_id == calculate_sid
            assert engine._results["review"].session_id != review_sid
            assert all(engine._results[n].text.strip() == "VERSION-43" for n in ("calculate", "review", "assembly"))
            assert provider.roles["SIBLING"] == 1 and provider.roles["REVIEW"] == 2
            assert provider.roles["CALCULATE"] == 2 and provider.roles["ASSEMBLY"] == 1
            rows = await store.load_all_messages(calculate_sid)
            assert rows[:len(original_rows)] == original_rows and len(rows) > len(original_rows)
            usage = sum(r.token_usage.total_tokens for r in provider.node_responses)
            assert saved["result"]["usage"]["total_tokens"] == usage > 0
            main_rows = await store.load_all_messages(parent)
            assert [r.name for r in main_rows if r.role == "tool"] == ["launch_workflow", "repair_calculation"]
            assert all(task.done() for task in engine._repairs.values())
    finally:
        provider.release_sibling.set()
        provider.release_repair.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 20)
        provider.report(record_property)
        await provider.close()
        await store.close()


async def test_real_legacy_child_empty_pass_finishes_in_original_session(tmp_path, record_property):
    from power_loop.runtime import delegation
    from power_loop.tools.spawn_agent import register_spawn_agent

    store = await SessionStore.open(str(tmp_path / "legacy-empty-pass.db"))
    provider = BoundedProvider(limit=9)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Follow the user's delegation instructions. A queued receipt is not a result. "
                          "When the child result arrives, report its exact code without new tools or pass_turn."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(180):
            first = await loop.send(
                "Call spawn_agent exactly once with task='Report result code 42', tools=[], max_rounds=2, "
                "system_prompt='The exact result code is 42. On your FIRST turn call pass_turn exactly once and do not "
                "write the answer. When later asked for a final answer, return exactly 42 in assistant text without "
                "any tools or pass_turn.'. After acceptance reply WAITING; do not call pass_turn yourself.", parent)
            assert first.status == "completed"
            await provider.child_entered.wait()
            rows = await store.load_all_messages(parent)
            [receipt] = [json.loads(r.content) for r in rows if r.role == "tool" and r.name == "spawn_agent"]
            provider.release_children.set()
            await asyncio.gather(*(set(delegation._TASKS)-before))
            saved = await store.get_runtime_state(parent, "subagent:run:"+receipt["run_id"])
            result = saved["result"]
            assert result["status"] == "completed" and result["final_text"].strip() == "42", result
            assert result["session_id"] == receipt["session_id"] and result["usage"]["calls"] >= 2
            child_rows = await store.load_all_messages(receipt["session_id"])
            assert len([r for r in child_rows if r.role == "tool" and r.name == "pass_turn"]) == 1
            assert len(await store.list_children(parent)) == 1
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final.status == "completed" and "42" in final.final_text
            assert len([r for r in await store.load_all_messages(parent) if r.role == "tool" and r.name == "spawn_agent"]) == 1
    finally:
        provider.release_children.set()
        tasks = list(set(delegation._TASKS)-before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        provider.report(record_property)
        await provider.close()
        await store.close()
