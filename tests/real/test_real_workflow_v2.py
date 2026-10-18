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


LOOP_SPEC = {
    "name": "dev_review_loop",
    "input": "scene pier",
    "root": {"type": "sequence", "steps": [
        {"type": "loop", "id": "pier", "max_iterations": 3, "continue_nodes": ["dev"], "until": {"ref": "review.pass"},
         "body": {"type": "sequence", "steps": [
             {"type": "agent", "id": "dev", "spec": {"name": "dev", "max_tokens": 2048, "system_prompt": (
                 "You write the exit key for a game scene. Reply with ONLY the key. On your FIRST answer write exactly: to_pier. "
                 "If you later get review feedback, fix the key accordingly and reply with ONLY the fixed key.")},
              "input": "Write the exit key that leads to the pier."},
             {"type": "agent", "id": "review", "spec": {"name": "review", "max_tokens": 2048, "system_prompt": (
                 "You check an exit key. Rule: use to-<scene> with a hyphen preserving the word boundary; replace underscores with hyphens. Only lowercase letters, digits and hyphens are allowed. "
                 'Respond ONLY with JSON: {"pass": <true|false>, "problems": ["..."]}.')},
              "inputs_from": ["dev"], "input": "Check this key.",
              "output_schema": {"name": "Review", "schema": {"type": "object", "required": ["pass", "problems"], "properties": {
                  "pass": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}}},
         ]}},
    ]},
}


async def test_real_loop_dev_review_fix_converges():
    loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3))
    res = await create_workflow(LOOP_SPEC, parent_loop=loop).run()
    assert res.status == "completed", res.errors
    agg = res.results["pier"].payload
    assert agg["converged"] is True and agg["iterations"] >= 2, agg
    assert "to-pier" in res.results["dev"].text and "_" not in res.results["dev"].text.strip().splitlines()[-1]


MATRIX_LOOP_SPEC = {
    "name": "per_scene_loops",
    "input": "scenes",
    "root": {"type": "foreach", "id": "scenes", "items": ["pier", "tower", "market"], "as": "scene", "parallel": True,
             "body": {"type": "loop", "id": "scene-loop", "max_iterations": 3, "continue_nodes": ["dev"],
                      "until": {"ref": "review.pass"},
                      "body": {"type": "sequence", "steps": [
                          {"type": "agent", "id": "dev", "spec": {"name": "dev", "max_tokens": 2048, "system_prompt": (
                              "You write the exit key for a game scene. Reply with ONLY the key. On your FIRST answer write "
                              "exactly: to_<scene> (with an underscore). If you later get review feedback, fix the key "
                              "accordingly and reply with ONLY the fixed key.")},
                           "input": "Scene: {{scene}}. Write the exit key that leads to this scene."},
                          {"type": "agent", "id": "review", "spec": {"name": "review", "max_tokens": 2048, "system_prompt": (
                              "You check an exit key. Rule: only lowercase letters, digits and hyphens; underscores are NOT "
                              'allowed. Respond ONLY with JSON: {"pass": <true|false>, "key": "<the key you checked>", '
                              '"problems": ["..."]}.')},
                           "inputs_from": ["dev"], "input": "Check this key.",
                           "output_schema": {"name": "Review", "schema": {"type": "object", "required": ["pass", "key", "problems"],
                                                                          "properties": {"pass": {"type": "boolean"}, "key": {"type": "string"},
                                                                                         "problems": {"type": "array", "items": {"type": "string"}}}}}},
                      ]}}},
}


async def test_real_matrix_of_loops_each_scene_fixes_its_own_key():
    loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3))
    res = await create_workflow(MATRIX_LOOP_SPEC, parent_loop=loop).run()
    assert res.status == "completed", res.errors
    items = res.results["scenes"].payload["items"]
    assert len(items) == 3
    keys = sorted(it["last"]["dev"]["text"].strip().splitlines()[-1] for it in items)
    # 每一路改的是自己那个场景的 key（迭代之间不串），而且改对了（没有下划线）
    assert [k for k in keys if "market" in k] and [k for k in keys if "pier" in k] and [k for k in keys if "tower" in k], keys
    assert all("_" not in k for k in keys), keys
    assert all(it["converged"] and it["iterations"] >= 2 for it in items), items
