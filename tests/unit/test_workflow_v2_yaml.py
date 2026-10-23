"""workflow v2（design/170 §1–§2）：并发交给平台钳、就地 job 的 outputs、下游引用结构化输出。"""

from __future__ import annotations

import pytest

from power_loop.workflow.yaml import WorkflowYamlError, compile_workflow_yaml


def _compile(text: str):
    return compile_workflow_yaml(text, load_file=lambda s, p: None)


_SCENES = """
name: 场景开发
jobs:
  dev-a:
    prompt: 你开发场景 a。
  dev-b:
    prompt: 你开发场景 b。
  review-a:
    needs: [dev-a]
    prompt: 你审场景 a。
    outputs:
      pass: {type: boolean, description: 是否通过}
      problems: {type: array, description: 要修的问题}
  fix-a:
    needs: [review-a]
    prompt: 你按审核意见修场景 a。
    with:
      verdict: "通过=${{ needs.review-a.outputs.pass }} 问题=${{ needs.review-a.outputs.problems }}"
"""


def _walk(node, out):
    out.append(node)
    for k in ("branches", "steps"):
        for c in node.get(k) or []:
            _walk(c, out)
    if node.get("body"):
        _walk(node["body"], out)
    return out


def test_parallel_nodes_defer_concurrency_to_platform():
    spec = _compile(_SCENES).to_dict()
    pars = [n for n in _walk(spec["root"], []) if n.get("type") == "parallel"]
    assert pars and all(n["max_concurrency"] == 64 for n in pars)


def test_inline_outputs_become_output_schema_and_json_reply_note():
    spec = _compile(_SCENES).to_dict()
    nodes = {n.get("id"): n for n in _walk(spec["root"], [])}
    rv = nodes["review-a"]
    props = rv["output_schema"]["schema"]["properties"]
    assert props["pass"]["type"] == "boolean" and props["problems"]["type"] == "array"
    assert "最终回复是一个 JSON 对象" in rv["input"] and "`pass`（boolean）" in rv["input"]
    # 下游引用编成引擎认得的 {{节点.键}}
    assert "{{review-a.pass}}" in nodes["fix-a"]["input"] and "{{review-a.problems}}" in nodes["fix-a"]["input"]


def test_outputs_rejects_bad_type_and_unknown_key_reference():
    bad = _SCENES.replace("{type: boolean, description: 是否通过}", "{type: bool}")
    with pytest.raises(WorkflowYamlError) as exc:
        _compile(bad)
    assert any("type 'bool' 不认识" in p for p in exc.value.problems)
    bad2 = _SCENES.replace("outputs.pass }}", "outputs.passed }}")
    with pytest.raises(WorkflowYamlError) as exc2:
        _compile(bad2)
    assert any("没有声明输出 'passed'" in p for p in exc2.value.problems)


# ── 第二批：if: 条件、timeout-minutes ─────────────────────────────────────────

_IF = _SCENES.replace('''  fix-a:
    needs: [review-a]
    prompt: 你按审核意见修场景 a。''', '''  fix-a:
    needs: [review-a]
    if: ${{ needs.review-a.outputs.pass == false }}
    prompt: 你按审核意见修场景 a。
  done-a:
    needs: [review-a]
    if: ${{ needs.review-a.outputs.pass }}
    prompt: 你写场景 a 的完成说明。
  note-a:
    needs: [review-a]
    if: ${{ !needs.review-a.outputs.pass }}
    prompt: 你记一笔没通过。''')


def test_if_compiles_to_when():
    spec = _compile("timeout-minutes: 90\n" + _IF).to_dict()
    nodes = {n.get("id"): n for n in _walk(spec["root"], [])}
    assert nodes["fix-a"]["when"] == {"ref": "review-a.pass", "op": "eq", "value": False}
    assert nodes["done-a"]["when"] == {"ref": "review-a.pass", "op": "truthy"}
    assert nodes["note-a"]["when"] == {"ref": "review-a.pass", "op": "falsy"}
    assert spec["timeout_s"] == 5400


def test_if_rejects_bad_forms():
    bad = _IF.replace("needs.review-a.outputs.pass == false", "needs.dev-b.outputs.pass == false")
    with pytest.raises(WorkflowYamlError) as exc:
        _compile(bad)
    assert any("要把它写进 needs" in p for p in exc.value.problems)
    bad2 = _IF.replace("needs.review-a.outputs.pass == false", "needs.review-a.output contains x")
    with pytest.raises(WorkflowYamlError) as exc2:
        _compile(bad2)
    assert any("看不懂" in p for p in exc2.value.problems)
    with pytest.raises(WorkflowYamlError) as exc3:
        _compile("timeout-minutes: 0\n" + _IF)
    assert any("timeout-minutes" in p for p in exc3.value.problems)


# ── 第三批：loops: ────────────────────────────────────────────────────────────

_LOOP = """
name: 场景开发
jobs:
  plan:
    prompt: 你列出要开发的场景。
    outputs:
      scenes: {type: array, description: 场景 key 列表}
  dev:
    needs: [plan]
    prompt: 你开发场景 ${{ inputs.who }}。
    with:
      scene: "${{ matrix.scene }}"
      round: "第 ${{ loop.iteration }} 轮"
  review:
    needs: [dev]
    prompt: 你审场景。
    with:
      scene: "${{ matrix.scene }}"
    outputs:
      pass: {type: boolean, description: 是否通过}
      issues: {type: array, description: 要修的问题}
  assemble:
    needs: [scenes]
    prompt: 你把场景组装起来。
    with:
      result: "${{ needs.scenes.outputs.items }}"
inputs:
  who: 建造师
loops:
  scenes:
    jobs: [dev, review]
    until: ${{ needs.review.outputs.pass }}
    max-iterations: 4
    strategy:
      matrix:
        scene: ${{ needs.plan.outputs.scenes }}
      max-parallel: 8
"""


def test_matrix_loop_compiles_to_foreach_of_loop():
    spec = _compile(_LOOP).to_dict()
    nodes = {n.get("id"): n for n in _walk(spec["root"], [])}
    fe = nodes["scenes"]
    assert fe["type"] == "foreach" and fe["items_from"] == "plan.scenes" and fe["as"] == "scene"
    assert fe["max_concurrency"] == 8 and fe["on_error"] == "continue"
    lp = nodes["scenes-item"]
    assert lp["type"] == "loop" and lp["max_iterations"] == 4
    assert lp["until"] == {"ref": "review.pass", "op": "truthy"}
    assert lp["continue_nodes"] == ["dev"]          # 默认：除了 until 那个 job 都接着上一轮会话
    assert "{{scene}}" in nodes["dev"]["input"] and "第 {{loop_iteration}} 轮" in nodes["dev"]["input"]
    assert nodes["review"]["inputs_from"] == ["dev"]
    # 顺序：plan → 循环 → assemble
    order = [s.get("id") for s in spec["root"]["steps"]]
    assert order == ["plan", "scenes", "assemble"]
    assert "{{scenes.items}}" in nodes["assemble"]["input"]


def test_plain_loop_body_ids_referenceable_after():
    text = _LOOP.replace("""    strategy:
      matrix:
        scene: ${{ needs.plan.outputs.scenes }}
      max-parallel: 8
""", "    on-exhausted: continue\n    continue: [dev]\n").replace('"${{ matrix.scene }}"', "pier").replace(
        """  assemble:
    needs: [scenes]""", """  assemble:
    needs: [scenes, review]""").replace("outputs.items }}", "outputs.converged }} ${{ needs.review.outputs.issues }}")
    spec = _compile(text).to_dict()
    nodes = {n.get("id"): n for n in _walk(spec["root"], [])}
    assert nodes["scenes"]["type"] == "loop" and nodes["scenes"]["on_exhausted"] == "continue"
    assert "{{scenes.converged}}" in nodes["assemble"]["input"] and "{{review.issues}}" in nodes["assemble"]["input"]
    assert [s.get("id") for s in spec["root"]["steps"]] == ["plan", "scenes", "assemble"]


def test_loop_errors():
    def probs(text):
        with pytest.raises(WorkflowYamlError) as exc:
            _compile(text)
        return exc.value.problems

    p = probs(_LOOP.replace("until: ${{ needs.review.outputs.pass }}", "until: ${{ needs.review.outputs.ok }}"))
    assert any("没有声明输出 'ok'" in x for x in p)
    p = probs(_LOOP.replace("jobs: [dev, review]", "jobs: [dev, reviewer]"))
    assert any("'reviewer' 不是一个 job" in x for x in p)
    p = probs(_LOOP.replace("max-iterations: 4", "max-iterations: 20"))
    assert any("max-iterations" in x for x in p)
    p = probs(_LOOP.replace("needs: [scenes]", "needs: [review]").replace("outputs.items", "outputs.pass"))
    assert any("needs 循环本身" in x for x in p)
    p = probs(_LOOP.replace('round: "第 ${{ loop.iteration }} 轮"', 'round: "${{ needs.plan.output }}"').replace(
        "prompt: 你把场景组装起来。", "prompt: 你把场景组装起来。\n    with:\n      x: \"${{ loop.iteration }}\"").replace(
        "    with:\n      result:", "      result:"))
    assert any("只能用在循环里" in x for x in p)
