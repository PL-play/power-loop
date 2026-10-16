"""Real-LLM：workflow v2（DeepTalk design/170）——审核节点交结构化结论，下游节点用 {{节点.键}} 拿到值。"""

from __future__ import annotations

import tempfile

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop, create_llm_service_from_env
from power_loop.workflow import create_workflow

pytestmark = pytest.mark.real_llm

SPEC = {
    "name": "review_then_fix",
    "input": "A scene plan: the exit key of the pier is written as 'to_pier' (underscore). Keys must use hyphens only.",
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "review-1",
         "spec": {"name": "reviewer", "max_tokens": 2048, "system_prompt": (
             "You review a scene plan for errors. Respond ONLY with JSON: "
             '{"pass": <true if no errors>, "problems": ["one short line per error"]}.')},
         "input": "Plan: {{input}}",
         "output_schema": {"name": "Review", "schema": {"type": "object", "required": ["pass", "problems"], "properties": {
             "pass": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "agent", "id": "fix",
         "spec": {"name": "fixer", "max_tokens": 2048, "system_prompt": (
             "You receive a review verdict. If pass is false, write the corrected exit key and nothing else. "
             "If pass is true, write OK.")},
         "input": "pass={{review-1.pass}}\nproblems={{review-1.problems}}"},
    ]},
}


async def test_real_dotted_ref_carries_structured_verdict():
    loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3))
    res = await create_workflow(SPEC, parent_loop=loop).run()
    assert res.status == "completed", res.errors
    verdict = res.results["review-1"].payload
    assert verdict and verdict["pass"] is False and verdict["problems"]
    fixed = res.results["fix"].text
    assert "to-pier" in fixed and "{{" not in fixed
