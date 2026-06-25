"""Real-LLM companion for the workflow control-flow unit tests
(test_workflow_fanout.py / test_workflow_halt_scope.py / test_workflow_cancel.py).

The deterministic engine mechanics (halt scope, cancel propagation, leaf ceiling) are covered with
fake executors; this exercises the REAL fan-out paths — parallel branches and a foreach — running
actual leaf agents against the live provider and rolling up usage.
"""

from __future__ import annotations

import tempfile

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop, create_llm_service_from_env
from power_loop.workflow import create_workflow

pytestmark = pytest.mark.real_llm


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=create_llm_service_from_env(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=4),
    )


async def test_real_parallel_branches_all_complete() -> None:
    spec = {
        "name": "parallel_facts", "input": "x",
        "root": {"type": "parallel", "max_concurrency": 2, "branches": [
            {"type": "agent", "id": "sun",
             "spec": {"name": "a", "system_prompt": "Reply with one short factual sentence about the Sun."},
             "input": "the Sun"},
            {"type": "agent", "id": "moon",
             "spec": {"name": "b", "system_prompt": "Reply with one short factual sentence about the Moon."},
             "input": "the Moon"},
        ]},
    }
    result = await create_workflow(spec, parent_loop=_loop()).run()
    assert result.status == "completed", f"errors: {result.errors}"
    assert result.results["sun"].ok and result.results["sun"].text.strip()
    assert result.results["moon"].ok and result.results["moon"].text.strip()
    assert result.usage.get("total_tokens", 0) > 0  # both leaves' usage rolled up


async def test_real_foreach_fans_out_one_leaf_per_item() -> None:
    spec = {
        "name": "foreach_fanout", "input": "x",
        "root": {"type": "foreach", "id": "fan", "items": ["apple", "banana", "cherry"],
                 "as": "fruit", "parallel": True, "max_concurrency": 3,
                 "body": {"type": "agent", "id": "leaf",
                          "spec": {"name": "leaf", "system_prompt": "Reply with the color of the named fruit, one word."},
                          "input": "{{fruit}}"}},
    }
    result = await create_workflow(spec, parent_loop=_loop()).run()
    assert result.status == "completed", f"errors: {result.errors}"
    items = result.results["fan"].payload["items"]
    assert len(items) == 3  # one real leaf per foreach item
