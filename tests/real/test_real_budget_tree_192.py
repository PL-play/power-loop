"""Real provider admission under tree limits; the budget authority is a test port.

No provider response is replaced. API transactions have separate real-store
acceptance; this file proves local dispatch gates, not a supplier money ceiling.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.budget_contracts import (
    BudgetAck,
    BudgetConfig,
    BudgetGrant,
    BudgetPlan,
    BudgetSubmission,
    LocalBudgetScope,
)
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


class RealBudgetHost:
    def __init__(self):
        self.service = None
        self.reserve_calls = 0
        self.grants, self.applied = {}, {}

    async def prepare(self, context, logical_request, *, service):
        assert service is self.service
        return BudgetPlan(16384, {"attempt_id": context["attempt_id"], "origin": context})

    async def reserve(self, plan):
        self.reserve_calls += 1
        key = plan.request["attempt_id"]
        assert key not in self.grants
        self.grants[key] = BudgetGrant(key, plan.request_hash, 1, "reserved", {})
        return self.grants[key]

    async def lookup(self, plan):
        return self.grants.get(plan.request["attempt_id"])

    async def prepare_usage(self, context, grant, observation):
        return BudgetSubmission(context["attempt_id"], 1, {"observation": observation})

    async def apply_usage(self, submission):
        ack = BudgetAck(submission.attempt_id, submission.revision, submission.request_hash)
        self.applied.setdefault(submission.attempt_id, ack)
        assert self.applied[submission.attempt_id] == ack
        return ack

    async def prepare_release(self, context, grant, proof):
        return BudgetSubmission(context["attempt_id"], 1, proof)

    async def release(self, submission):
        return BudgetAck(submission.attempt_id, submission.revision, submission.request_hash)


async def report(store, provider, record_property):
    provider.report(record_property)
    evidence = []
    for session in await store._db.fetchall(f"SELECT session_id FROM {store.t.sessions}"):
        evidence.append(
            {
                "sid": session["session_id"],
                "messages": [asdict(m) for m in await store.load_all_messages(session["session_id"])],
            }
        )
    record_property("original_messages", json.dumps(evidence, ensure_ascii=False, default=str))


async def test_real_new_main_roots_share_authority_cap_and_third_call_never_dispatches(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "root-cap.db"), schema="tasks")
    provider, host = BoundedProvider(limit=2), RealBudgetHost()
    host.service = provider
    runtime = TaskRuntime(
        JobManager(store),
        tenant_id="real-tree",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        boot_contract_hash="a" * 64,
        budget_config=BudgetConfig(
            "real-tree-v1",
            (
                LocalBudgetScope("authority", max_calls=2),
                LocalBudgetScope("root", max_calls=1, binding="root"),
                LocalBudgetScope("owner", max_calls=1, binding="owner"),
            ),
            host,
        ),
    )
    try:
        loop = StatefulAgentLoop(
            llm=provider,
            store=store,
            task_runtime=runtime,
            config=AgentLoopConfig(max_rounds=2, max_tokens=2048, delegation_policy="single_level"),
        )
        sid = await loop.new_session()
        async with asyncio.timeout(150):
            for _ in range(2):
                result = await loop.send("Reply exactly 42 in assistant text. Do not call tools or pass_turn.", sid)
                assert result.final_text.strip() == "42"
            rejected = await loop.send("Reply 43. This request must be rejected before the provider.", sid)
            assert rejected.status == "degraded" and "budget_max_calls_exhausted" in rejected.final_text
            assert provider.calls == host.reserve_calls == len(host.applied) == 2
            rows = await runtime.manager.db.fetchall(
                f"SELECT limits_json,counters_json FROM {runtime.manager.prefix}task_budget_scopes"
            )
            roots = [json.loads(r["counters_json"]) for r in rows if json.loads(r["limits_json"])["key"] == "root"]
            assert sorted(r["used_calls"] for r in roots) == [0, 1, 1]
            assert all(r["held_calls"] == 0 for r in roots)
    finally:
        await runtime.shutdown(timeout_s=5)
        await report(store, provider, record_property)
        await provider.close()
        await store.close()


async def test_real_workflow_owner_cap_blocks_review_before_its_provider_request(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "workflow-cap.db"), schema="tasks")
    provider, host, accepted, roles = BoundedProvider(limit=6), RealBudgetHost(), [], Counter()
    host.service = provider
    complete = provider.complete

    async def counted(request, **kwargs):
        roles[
            "BUILD"
            if "TREE-BUILD" in (request.system_prompt or "")
            else "REVIEW"
            if "TREE-REVIEW" in (request.system_prompt or "")
            else "MAIN"
        ] += 1
        return await complete(request, **kwargs)

    provider.complete = counted
    runtime = TaskRuntime(
        JobManager(store),
        tenant_id="real-tree-workflow",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=300000,
        boot_contract_hash="a" * 64,
        budget_config=BudgetConfig(
            "real-tree-v1",
            (
                LocalBudgetScope("authority", max_calls=8),
                LocalBudgetScope("root", max_calls=8, binding="root"),
                LocalBudgetScope("workflow", max_calls=1, binding="owner", owner_kinds=("workflow",)),
                LocalBudgetScope("node", max_calls=2, binding="owner", owner_kinds=("workflow_node",)),
            ),
            host,
        ),
    )

    async def on_accepted(admission):
        accepted.append(admission)

    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    spec = {
        "name": "real-tree",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "agent",
                    "id": role,
                    "spec": {
                            "name": role,
                        "system_prompt": "TREE-"
                        + role
                        + ": Calculate 19+23. Reply exactly 42 in plain assistant text, without tools or pass_turn.",
                        "tools": [],
                        "max_rounds": 2,
                        "max_tokens": 2048,
                    },
                }
                for role in ("BUILD", "REVIEW")
            ],
        },
    }
    try:
        loop = StatefulAgentLoop(
            llm=provider,
            store=store,
            task_runtime=runtime,
            tool_registry=registry,
            config=AgentLoopConfig(max_rounds=3, max_tokens=2048, delegation_policy="single_level"),
        )
        sid = await loop.new_session()
        async with asyncio.timeout(180):
            first = await loop.send(
                'Call workflow exactly once with action="create" and the following spec. After acceptance, reply WAITING in plain text. Do not poll, retry, create another task, or call pass_turn. Spec: '
                + json.dumps(spec),
                sid,
            )
            assert first.status == "completed" and len(accepted) == 1
            await accepted[0].handle.task
            workflow = await runtime.manager.get_job(runtime.actor(sid), accepted[0].handle.job_id)
            assert workflow.outcome == "failed" and "budget_max_calls_exhausted" in str(workflow.result)
            assert roles["BUILD"] == 1 and roles["REVIEW"] == 0
            assert provider.calls == host.reserve_calls == len(host.applied)
            [counter] = [
                json.loads(r["counters_json"])
                for r in await runtime.manager.db.fetchall(
                    f"SELECT limits_json,counters_json FROM {runtime.manager.prefix}task_budget_scopes"
                )
                if json.loads(r["limits_json"])["key"] == "workflow"
            ]
            assert counter["used_calls"] == 1 and counter["held_calls"] == counter["inflight"] == 0
    finally:
        if accepted:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 10)
        await runtime.shutdown(timeout_s=5)
        await report(store, provider, record_property)
        await provider.close()
        await store.close()
