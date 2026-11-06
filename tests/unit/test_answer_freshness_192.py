"""A pre-correction answer cannot satisfy a newer user input after empty pass."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_pass_turn_builtin import _call, _Script

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("native", [False, True])
async def test_empty_pass_cannot_recover_text_from_before_the_latest_input(tmp_path, native):
    store = await SessionStore.open(str(tmp_path / "fresh-answer.db"), **({"schema": "tasks"} if native else {}))
    try:
        runtime = TaskRuntime(JobManager(store), tenant_id="freshness", principal_id="host",
            executor_id="executor", boot_id="boot") if native else None
        model = _Script(responses=[_call("old", "correct", {}, text="42"),
            _call("pass", "pass_turn", {"reason": "Computed 43"})])
        registry = ToolRegistry()
        async def correct():
            await get_current_loop().follow_up("Correction: return 43.", get_session_id())
            return "Correction queued."
        registry.register(ToolDefinition(name="correct", description="Queue correction", input_schema={"type": "object"}, pricing=Free("test_fixture")), correct)
        loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3))
        sid = await loop.new_session()
        result = await loop.send("Return 42, then apply any correction.", sid)
        assert result.status == "completed" and result.passed and model.calls == 2
        assert result.final_text == ""
        assert result.texts[0]["text"] == "42"  # retained progress is not the current answer
        rows = await store.load_all_messages(sid)
        assert any(m.name == "follow_up" and "43" in str(m.content) for m in rows)
    finally:
        await store.close()
