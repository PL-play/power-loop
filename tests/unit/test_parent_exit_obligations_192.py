"""Actual pipeline exits retain a live child's control and original accounting."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.budget_contracts import LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_budget_tree_192 import tree_case
from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import AdapterModel

__all__ = ["registry_store", "tree_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class OwnerAborted(BaseException):
    """An unexpected exit that must traverse the runtime's BaseException path."""


@pytest.mark.parametrize("tree_case", [{"scopes": (
    LocalBudgetScope("authority", max_calls=20),
)}], indirect=True)
@pytest.mark.parametrize("exit_kind", [
    "hit_round_limit", "context_checkpoint", "budget_exceeded", "degraded", "exception",
])
async def test_actual_parent_exit_keeps_child_control_result_and_fees(tree_case, exit_kind):
    c = tree_case

    class Model(AdapterModel):
        async def complete(self, request, **kwargs):
            if get_session_id() == self.root and self.root_calls:
                await asyncio.wait_for(self.entered.wait(), 10)
                if exit_kind == "degraded":
                    raise ValueError("original parent provider failed")
                if exit_kind == "exception":
                    raise OwnerAborted("unexpected original parent exit")
            return await super().complete(request, **kwargs)

    model = Model()
    c.host.service = model
    registry = ToolRegistry()
    register_spawn_agent(registry)
    config = AgentLoopConfig(max_rounds=1 if exit_kind == "hit_round_limit" else 3,
        max_tokens=20, compactor=None, delegation_policy="single_level",
        context_checkpoint_tokens=1 if exit_kind == "context_checkpoint" else None,
        max_tokens_per_run=1 if exit_kind == "budget_exceeded" else None,
        retry_policy=LLMRetryPolicy(max_attempts=1))
    loop = StatefulAgentLoop(llm=model, store=c.store, task_runtime=c.runtime,
        tool_registry=registry, config=config)
    sid = model.root = await loop.new_session()
    actor = c.runtime.actor(sid)
    initial = set(delegation._TASKS)
    try:
        if exit_kind == "exception":
            with pytest.raises(OwnerAborted):
                await loop.send("Launch the child once.", sid)
        else:
            result = await loop.send("Launch the child once.", sid)
            assert result.status == exit_kind
        await asyncio.wait_for(model.entered.wait(), 10)
        # Reopen the persistent store; assertions must not depend on local caches.
        manager = JobManager(await c.connect())
        rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")
        jobs = [await manager.get_job(actor, row["job_id"]) for row in rows]
        parent = next(j for j in jobs if j.kind == "send" and j.session_id == sid)
        child = next(j for j in jobs if j.kind == "subagent")
        physical = next(j for j in jobs if j.kind == "send" and j.session_id != sid)
        assert parent.status == "model_closed" and parent.supervision_status == "open"
        assert child.parent_job_id == parent.job_id and child.status == "running"
        assert child.result_status == "none" and child.supervision_status == "open"
        assert physical.control_owner_job_id == child.job_id and physical.status == "running"
        [closed] = await manager.db.fetchall(
            f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='model_closed'",
            (parent.job_id,))
        evidence = json.loads(closed["payload_json"])
        assert evidence["evidence"]["exit"] == ("failed" if exit_kind == "exception" else exit_kind)
        usage = await manager.execution_usage(actor, parent.job_id)
        assert usage["total_tokens"] >= 10
        assert c.host.applied and not c.host.released
        applied = dict(c.host.applied)
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, parent.job_id, generation=parent.generation)
        # Control still works after the parent's physical send has ended.
        await manager.request_cancel(actor, child.job_id, system_key="stop-after-parent-exit",
            reason="Verify retained control after an unsuccessful parent exit")
        assert (await manager.get_job(actor, child.job_id)).cancel_requested_at is not None
        model.release.set()
        await asyncio.wait_for(asyncio.gather(*list(set(delegation._TASKS) - initial),
            return_exceptions=True), 15)
        child = await manager.get_job(actor, child.job_id)
        assert child.outcome == "cancelled" and child.supervision_status == "open"
        # execution_usage(root) includes descendants: add the child's actual
        # returned usage exactly once while preserving the original exit row.
        after = await manager.execution_usage(actor, parent.job_id)
        assert after["total_tokens"] == usage["total_tokens"] + 10
        assert after["calls"] == usage["calls"] + 1
        assert await manager.db.fetchall(
            f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='model_closed'",
            (parent.job_id,)) == [closed]
        assert all(c.host.applied[key] == value for key, value in applied.items())
        assert len(c.host.applied) == len(applied) + 1
        assert not c.host.released and model.child_calls == 1
    finally:
        model.release.set()
        await asyncio.wait_for(asyncio.gather(*list(set(delegation._TASKS) - initial),
            return_exceptions=True), 15)
        await c.runtime.shutdown(timeout_s=2)
        await loop.aclose(drain_timeout_s=.1)
