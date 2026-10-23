"""YAML（GHA 风格）→ WorkflowSpec 编译器：结构、编译期校验、文件产出与行切片。"""

from __future__ import annotations

import pytest

from power_loop.workflow.yaml import (
    DEFAULT_PREVIEW_LINES,
    WorkflowYamlError,
    compile_workflow_yaml,
    output_path,
    resolve_file_refs,
)

# ── 一套最小但真实的技能内容（权威在 PG；测试用 dict 假装） ──────────────────────

_ACTION = """
name: 静态审读
charter: charters/static-reviewer.md
tools: [read_file, glob, grep, write_file]
inputs:
  target:       {required: true, description: 产物目录}
  requirements: {required: true, description: 用户需求原话}
outputs:
  count: {type: integer, description: 发现条数}
"""

_JUDGE_ACTION = """
name: 判定
charter: charters/judge.md
tools: [read_file, write_file]
inputs:
  findings: {required: true, description: 上游 findings}
"""

FILES = {
    ("av", "actions/static/action.yaml"): _ACTION,
    ("av", "actions/static/charters/static-reviewer.md"): "你是静态审读方。",
    ("av", "actions/judge/action.yaml"): _JUDGE_ACTION,
    ("av", "actions/judge/charters/judge.md"): "你是判定方。",
    ("av", "charters/plain.md"): "裸 charter，无 IO 契约。",
}


def _load(skill: str, path: str) -> str | None:
    return FILES.get((skill, path))


def _compile(text: str):
    return compile_workflow_yaml(text, load_file=_load)


def _problems(text: str) -> list[str]:
    with pytest.raises(WorkflowYamlError) as exc:
        _compile(text)
    return exc.value.problems


_GOOD = """
name: 对抗验证
inputs:
  requirements: |
    我要一个旅行手帐。
    两个人一起用。
jobs:
  static:
    name: 静态审读
    uses: skill:av/actions/static
    with:
      target: app/
      requirements: ${{ inputs.requirements }}
  runtime:
    name: 运行时取证
    uses: skill:av/charters/plain.md
    continue-on-error: true
  judge:
    name: 判定
    needs: [static, runtime]
    uses: skill:av/actions/judge
    with:
      findings: ${{ needs.static.output }} ${{ needs.runtime.output[-50:] }}
"""


# ── 结构 ──────────────────────────────────────────────────────────────────────


def test_needs_gives_implicit_parallelism_and_layering():
    """并行是隐式的：没有 needs 的 job 自动同层并发——不用手写 parallel 节点。"""
    spec = _compile(_GOOD)
    assert spec.name == "对抗验证"
    root = spec.root
    assert root.type == "sequence"
    layer0, layer1 = root.steps
    assert layer0.type == "parallel"
    assert sorted(b.id for b in layer0.branches) == ["runtime", "static"]
    assert layer1.type == "agent" and layer1.id == "judge"      # 单节点层不套 parallel
    assert list(layer1.inputs_from) == ["static", "runtime"]     # needs = DAG 的边


def test_leaf_prompt_is_a_charter_sentinel_not_authored_text():
    # 裁判提示词正文由平台在 spec_transform 注入并盖哈希戳；YAML 里只有引用。
    spec = _compile(_GOOD)
    static = next(b for b in spec.root.steps[0].branches if b.id == "static")
    assert static.spec.system_prompt == "charter:skill:av/actions/static/charters/static-reviewer.md"
    assert static.spec.metadata["_action_tools"] == ["read_file", "glob", "grep", "write_file"]


def test_each_job_gets_its_own_output_file_and_is_told_to_append():
    spec = _compile(_GOOD)
    static = next(b for b in spec.root.steps[0].branches if b.id == "static")
    assert static.spec.metadata["output_file"] == output_path("static") == "outputs/static.md"
    assert "outputs/static.md" in static.input
    assert "追加" in static.input                                 # 边做边写=中断恢复点


def test_declared_outputs_become_a_structured_schema():
    spec = _compile(_GOOD)
    static = next(b for b in spec.root.steps[0].branches if b.id == "static")
    assert static.output_schema["schema"]["properties"]["count"]["type"] == "integer"


def test_block_scalar_multiline_input_is_substituted_verbatim():
    spec = _compile(_GOOD)
    static = next(b for b in spec.root.steps[0].branches if b.id == "static")
    assert "我要一个旅行手帐。\n两个人一起用。" in static.input   # 不再有 \\n 转义


def test_single_job_workflow_has_no_wrapper():
    spec = _compile(
        "name: 单跑\njobs:\n  only:\n    uses: skill:av/charters/plain.md\n"
    )
    assert spec.root.type == "agent" and spec.root.id == "only"


# ── 编译期校验：花第一个 token 之前就该发现的问题 ──────────────────────────────


def test_gha_reflexes_get_named_corrections():
    """模型一定会写 runs-on / steps —— 报错必须点名纠正，不能只说 unknown key。"""
    bad = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
    runs-on: ubuntu-latest
    steps:
      - run: npm test
"""
    problems = "\n".join(_problems(bad))
    assert "runs-on" in problems and "没有 runner" in problems
    assert "steps" in problems and "一个 job = 一个 agent 叶子" in problems


def test_marketplace_action_ref_is_rejected_with_the_right_shape():
    problems = "\n".join(_problems("name: x\njobs:\n  a:\n    uses: actions/checkout@v4\n"))
    assert "skill:<技能名>/<路径>" in problems


def test_with_keys_are_checked_against_declared_inputs():
    bad = """
name: x
jobs:
  a:
    uses: skill:av/actions/static
    with:
      target: app/
      requirements: r
      typo_key: 1
"""
    problems = "\n".join(_problems(bad))
    assert "没有声明输入 'typo_key'" in problems
    assert "requirements" in problems      # 顺带列出可用输入


def test_missing_required_input_is_caught():
    bad = "name: x\njobs:\n  a:\n    uses: skill:av/actions/static\n    with:\n      target: app/\n"
    assert any("缺必填输入 'requirements'" in p for p in _problems(bad))


def test_reference_without_needs_is_rejected():
    """引用上游产物却不写 needs——运行时那个文件根本还不存在。"""
    bad = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
  b:
    uses: skill:av/charters/plain.md
    with: {x: "${{ needs.a.output }}"}
"""
    assert any("没把它写进 needs" in p for p in _problems(bad))


def test_unknown_job_in_needs_and_in_expression():
    bad = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
    needs: [ghost]
"""
    assert any("'ghost' 不存在" in p for p in _problems(bad))


def test_cycles_are_reported():
    bad = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
    needs: [b]
  b:
    uses: skill:av/charters/plain.md
    needs: [a]
"""
    assert any("环" in p for p in _problems(bad))


def test_undeclared_structured_output_reference_is_caught():
    """跨节点引用的静态校验——conv134 那次 run failed 就死在这条链上。"""
    bad = """
name: x
jobs:
  a:
    uses: skill:av/actions/static
    with: {target: t, requirements: r}
  b:
    uses: skill:av/charters/plain.md
    needs: [a]
    with: {x: "${{ needs.a.outputs.nope }}"}
"""
    problems = "\n".join(_problems(bad))
    assert "没有声明输出 'nope'" in problems and "count" in problems


def test_step_slice_is_rejected_but_line_slices_are_fine():
    bad = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
  b:
    uses: skill:av/charters/plain.md
    needs: [a]
    with: {x: "${{ needs.a.output[::2] }}"}
"""
    assert any("不支持步长" in p for p in _problems(bad))


def test_duplicate_yaml_keys_are_rejected_not_silently_overwritten():
    bad = "name: x\nname: y\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n"
    assert any("重复的键" in p for p in _problems(bad))


def test_all_problems_are_aggregated_in_one_shot():
    bad = """
name: x
jobs:
  a:
    uses: skill:av/actions/static
    runs-on: ubuntu-latest
    with: {bogus: 1}
"""
    # 未知键 + with 不对账 + 两个必填缺失 —— 一次全报，模型改一轮就能全对
    assert len(_problems(bad)) >= 4


def test_missing_action_yaml_points_at_the_two_valid_shapes():
    problems = "\n".join(_problems("name: x\njobs:\n  a:\n    uses: skill:av/actions/ghost\n"))
    assert "action.yaml" in problems and ".md" in problems


# ── 运行时：文件引用 → 路径 + 片段 + 截断留痕 ────────────────────────────────


def _reader(store: dict[str, list[str]]):
    return lambda path: store.get(path)


def test_default_preview_is_head_and_marks_what_was_left_out():
    body = [f"line {i}" for i in range(200)]
    out = resolve_file_refs("看：@@FILEREF:outputs/a.md@@", _reader({"outputs/a.md": body}))
    assert "共 200 行" in out
    assert f"前 {DEFAULT_PREVIEW_LINES} 行" in out
    assert "line 0" in out and f"line {DEFAULT_PREVIEW_LINES - 1}" in out
    assert f"line {DEFAULT_PREVIEW_LINES}" not in out
    # 🔴 截断必须留痕 + 给出取全文的办法（conv184：静默截断吞掉尾部条目）
    assert "未包含" in out and "read_file('outputs/a.md')" in out


def test_negative_slice_takes_the_tail():
    body = [f"line {i}" for i in range(120)]
    out = resolve_file_refs("@@FILEREF:outputs/a.md[-50:]@@", _reader({"outputs/a.md": body}))
    assert "最后 50 行" in out
    assert "line 119" in out and "line 69" not in out
    assert "未包含" in out


def test_short_file_is_included_whole_without_a_truncation_marker():
    out = resolve_file_refs("@@FILEREF:outputs/a.md@@", _reader({"outputs/a.md": ["only line"]}))
    assert "已全文包含" in out and "未包含" not in out


def test_missing_upstream_file_says_so_instead_of_injecting_nothing():
    out = resolve_file_refs("@@FILEREF:outputs/a.md@@", _reader({}))
    assert "还不存在或为空" in out


def test_multiple_refs_resolve_independently():
    store = {"outputs/a.md": ["a1", "a2"], "outputs/b.md": [f"b{i}" for i in range(10)]}
    out = resolve_file_refs(
        "@@FILEREF:outputs/a.md@@ 和 @@FILEREF:outputs/b.md[-2:]@@", _reader(store)
    )
    assert "a1" in out and "b9" in out and "b0" not in out


# ── YAML 自身的坑（spike 实测撞到的两条） ────────────────────────────────────


def test_yaml11_booleans_are_disabled_so_on_stays_a_string():
    """GHA 反射写出的 `on: [push]`：YAML 1.1 会把键解析成布尔 True，
    错误信息就成了看不懂的「顶层不支持 'True'」。收敛到 YAML 1.2 后能点名纠正。"""
    problems = "\n".join(_problems("name: x\non: [push]\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n"))
    assert "'on'" in problems and "事件触发器" in problems
    assert "True" not in problems


def test_norway_problem_does_not_corrupt_user_text():
    # 需求原话里孤立的 no / yes 不能变成 False / True（挪威问题）
    spec = _compile(
        "name: x\ninputs:\n  a: no\n  b: yes\njobs:\n"
        "  j:\n    uses: skill:av/charters/plain.md\n"
        "    with: {x: '${{ inputs.a }}/${{ inputs.b }}'}\n"
    )
    assert "no/yes" in spec.root.input


def test_true_false_still_parse_as_booleans():
    spec = _compile(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    continue-on-error: true\n"
    )
    assert spec.root.spec.metadata["continue_on_error"] is True


def test_unknown_dependency_is_not_misreported_as_a_cycle():
    """指向不存在的 job 已单独报错；不该再被 Kahn 误报成「有环」。"""
    problems = _problems(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    needs: [ghost]\n"
    )
    assert any("'ghost' 不存在" in p for p in problems)
    assert not any("环" in p for p in problems)


# ── retry / fallback / output-file（power-loop 5.2.0 的叶子级语义） ────────────


def test_retry_block_is_translated_to_the_ir_shape():
    spec = _compile(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n"
        "    retry: {max-attempts: 3, on: [failed], backoff-s: 2}\n"
    )
    r = spec.root.retry
    assert r.max_attempts == 3 and r.on == ("failed",) and r.backoff_s == 2.0


def test_retry_rejects_unknown_keys():
    bad = ("name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n"
           "    retry: {max-attempts: 2, nope: 1}\n")
    assert any("retry 不支持 'nope'" in p for p in _problems(bad))


def test_retry_values_are_validated_by_the_ir_schema_not_duplicated_here():
    """取值合法性（上限 5、trigger 白名单）只在 IR 的严格 schema 里定义一次。

    分层的代价：编译器层的问题会先抛，IR 层的取值问题要等形状无误才轮到——
    两类同时写错时模型要改两轮。可接受，换来的是不必维护两套口径。"""
    bad = ("name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n"
           "    retry: {max-attempts: 99, on: [bogus]}\n")
    problems = "\n".join(_problems(bad))
    assert "capped at 5" in problems and "unknown trigger" in problems


def test_continue_on_error_reaches_the_node_not_just_metadata():
    spec = _compile(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    continue-on-error: true\n"
    )
    assert spec.root.continue_on_error is True
    assert spec.root.spec.metadata["continue_on_error"] is True     # 面板仍读 metadata


def test_output_file_override():
    spec = _compile(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    output-file: r/custom.md\n"
    )
    assert spec.root.output_file == "r/custom.md"


_FALLBACK = """
name: x
jobs:
  deep:
    uses: skill:av/charters/plain.md
    retry: {max-attempts: 2}
    fallback: shallow
  shallow:
    uses: skill:av/charters/plain.md
  report:
    needs: [deep]
    uses: skill:av/charters/plain.md
"""


def test_fallback_job_is_attached_to_its_primary_and_kept_out_of_the_main_dag():
    spec = _compile(_FALLBACK)
    # 主图只有 deep → report；shallow 不占一层
    assert spec.root.type == "sequence"
    deep, report = spec.root.steps
    assert deep.id == "deep" and report.id == "report"
    assert deep.fallback is not None and deep.fallback.id == "shallow"


def test_fallback_target_must_exist_and_must_not_nest_or_be_depended_on():
    assert any("fallback 指向不存在的 job 'ghost'" in p for p in _problems(
        "name: x\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    fallback: ghost\n"))
    nested = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
    fallback: b
  b:
    uses: skill:av/charters/plain.md
    fallback: c
  c:
    uses: skill:av/charters/plain.md
"""
    assert any("兜底的兜底" in p for p in _problems(nested))
    depended = """
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
    fallback: b
  b:
    uses: skill:av/charters/plain.md
  c:
    needs: [b]
    uses: skill:av/charters/plain.md
"""
    assert any("不在主图里，不能被依赖" in p for p in _problems(depended))


# ── strategy.matrix（动态/静态铺开）────────────────────────────────────────────


_MATRIX_STATIC = """
name: x
jobs:
  review:
    uses: skill:av/charters/plain.md
    strategy:
      matrix:
        file: [a.js, b.js, c.js]
      max-parallel: 2
    with: {target: "${{ matrix.file }}"}
"""


def test_static_matrix_becomes_a_foreach():
    spec = _compile(_MATRIX_STATIC)
    fe = spec.root
    assert fe.type == "foreach" and fe.id == "review"
    assert fe.as_var == "file" and list(fe.items) == ["a.js", "b.js", "c.js"]
    assert fe.max_concurrency == 2
    # body 另起 id：节点 id 全局唯一，且引擎禁止 inputs_from 指向 body
    assert fe.body.id == "review-item"
    assert "{{file}}" in fe.body.input          # 交给引擎按迭代替换


def test_dynamic_matrix_reads_an_upstream_declared_output():
    spec = _compile("""
name: x
jobs:
  plan:
    uses: skill:av/actions/static
    with: {target: t, requirements: r}
  work:
    needs: [plan]
    uses: skill:av/charters/plain.md
    strategy:
      matrix:
        item: ${{ needs.plan.outputs.count }}
    with: {x: "${{ matrix.item }}"}
""")
    work = spec.root.steps[1]
    assert work.type == "foreach" and work.items_from == "plan.count"


def test_matrix_needs_a_declared_output_not_a_product_file():
    bad = _MATRIX_STATIC.replace("        file: [a.js, b.js, c.js]",
                                 "        file: ${{ needs.plan.output }}")
    problems = chr(10).join(_problems(bad))
    assert "needs.<job>.outputs.<键>" in problems and "不能当列表铺开" in problems


def test_matrix_variable_name_must_match_the_reference():
    bad = _MATRIX_STATIC.replace('${{ matrix.file }}', '${{ matrix.wrong }}')
    assert any("matrix 变量是 'file'" in p for p in _problems(bad))


def test_multi_dimensional_matrix_is_rejected_with_a_way_out():
    bad = _MATRIX_STATIC.replace("        file: [a.js, b.js, c.js]",
                                 "        file: [a.js]\n        os: [linux]")
    assert any("正好有**一个**变量" in p for p in _problems(bad))


def test_unsupported_strategy_keys_are_named():
    bad = _MATRIX_STATIC.replace("      max-parallel: 2", "      fail-fast: true")
    assert any("strategy 不支持 'fail-fast'" in p for p in _problems(bad))


def test_downstream_cannot_reference_a_matrix_jobs_output_file():
    """matrix job 的产出是每次迭代一个文件，没有单一的 .output。"""
    bad = _MATRIX_STATIC + """  merge:
    needs: [review]
    uses: skill:av/charters/plain.md
    with: {all: "${{ needs.review.output }}"}
"""
    problems = "\n".join(_problems(bad))
    assert "是 matrix job" in problems and "outputs/review-item.*.md" in problems


def test_matrix_continue_on_error_maps_to_the_batch_policy():
    spec = _compile(_MATRIX_STATIC.replace("    strategy:", "    continue-on-error: true\n    strategy:"))
    assert spec.root.on_error == "continue"      # 一次迭代失败不毁掉整批
    assert spec.root.body.continue_on_error is True


# ── DAG 编码：分量之间不共用屏障 ──────────────────────────────────────────────


def test_independent_chains_do_not_share_a_barrier():
    """IR 是树结构，依赖只能用「分层+屏障」编码；但互不相干的两条链不该互相等
    （正确性没问题，纯属白等——一条慢链会拖住另一条快链）。"""
    spec = _compile("""
name: x
jobs:
  a1:
    uses: skill:av/charters/plain.md
  a2:
    needs: [a1]
    uses: skill:av/charters/plain.md
  b1:
    uses: skill:av/charters/plain.md
  b2:
    needs: [b1]
    uses: skill:av/charters/plain.md
""")
    assert spec.root.type == "parallel"                  # 两条链真并发
    chains = {tuple(n.id for n in br.steps): br for br in spec.root.branches}
    assert set(chains) == {("a1", "a2"), ("b1", "b2")}   # 各自内部仍是顺序


def test_one_connected_graph_stays_a_single_layered_chain():
    spec = _compile("""
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
  b:
    uses: skill:av/charters/plain.md
  c:
    needs: [a, b]
    uses: skill:av/charters/plain.md
""")
    assert spec.root.type == "sequence"
    layer0, layer1 = spec.root.steps
    assert layer0.type == "parallel" and sorted(n.id for n in layer0.branches) == ["a", "b"]
    assert layer1.id == "c"


def test_diamond_dag_compiles():
    """菱形：a → (b, c) → d。DAG 的基本形状都要能表达。"""
    spec = _compile("""
name: x
jobs:
  a:
    uses: skill:av/charters/plain.md
  b:
    needs: [a]
    uses: skill:av/charters/plain.md
  c:
    needs: [a]
    uses: skill:av/charters/plain.md
  d:
    needs: [b, c]
    uses: skill:av/charters/plain.md
""")
    steps = spec.root.steps
    assert [s.id if s.type == "agent" else s.type for s in steps] == ["a", "parallel", "d"]
    assert sorted(n.id for n in steps[1].branches) == ["b", "c"]
    assert list(steps[2].inputs_from) == ["b", "c"]


def test_compilation_is_deterministic():
    # 同一份 YAML 每次都编出同样的图（分量顺序按最小 job id 稳定）
    y = "name: x\njobs:\n  z:\n    uses: skill:av/charters/plain.md\n  a:\n    uses: skill:av/charters/plain.md\n"
    assert _compile(y).to_dict() == _compile(y).to_dict()


# ── 整图模板引用（template + args） ───────────────────────────────────────────


from power_loop.workflow.yaml import (  # noqa: E402
    compile_workflow_document,
    compile_workflow_template,
)

_TPL = """
name: 特性验证
inputs:
  requirements:        # 空值 = 必填
  app_id:
  depth: standard      # 有值 = 默认值
jobs:
  probe:
    uses: skill:av/charters/plain.md
    with: {r: "${{ inputs.requirements }}", a: "${{ inputs.app_id }}", d: "${{ inputs.depth }}"}
"""

_TPL_FILES = {**FILES, ("av", "graphs/c.yaml"): _TPL}


def _tpl_load(skill: str, path: str) -> str | None:
    return _TPL_FILES.get((skill, path))


def _tpl(args):
    return compile_workflow_template("skill:av/graphs/c.yaml", args, load_file=_tpl_load)


def test_template_reference_substitutes_args_and_keeps_defaults():
    spec = _tpl({"requirements": "用户原话", "app_id": 170})
    inp = spec.root.input
    assert "用户原话" in inp and "170" in inp
    assert "standard" in inp                      # 未传的参数用模板默认值


def test_missing_required_arg_is_caught_instead_of_running_with_a_placeholder():
    """模板里若用占位文本当默认值，忘传就会拿占位文本当真需求跑完一轮。
    空值=必填让它在编译期就炸。"""
    with pytest.raises(WorkflowYamlError) as exc:
        _tpl({"app_id": 1})
    assert any("缺必填参数 'requirements'" in p for p in exc.value.problems)


def test_unknown_arg_lists_what_the_template_declares():
    with pytest.raises(WorkflowYamlError) as exc:
        _tpl({"requirements": "r", "app_id": 1, "typo": 1})
    joined = "\n".join(exc.value.problems)
    assert "没有声明参数 'typo'" in joined and "depth" in joined


def test_unknown_template_and_non_yaml_ref_are_rejected():
    with pytest.raises(WorkflowYamlError) as exc:
        compile_workflow_template("skill:av/graphs/ghost.yaml", {}, load_file=_tpl_load)
    assert any("不存在" in p for p in exc.value.problems)
    with pytest.raises(WorkflowYamlError) as exc:
        compile_workflow_template("skill:av/charters/plain.md", {}, load_file=_tpl_load)
    assert any("必须指向一份 .yaml 模板" in p for p in exc.value.problems)


def test_matrix_can_fan_out_over_a_workflow_input_list():
    """配合 template+args：调用方传几项就跑几路。"""
    spec = _compile("""
name: x
inputs:
  candidates: [A, B, C]
jobs:
  assess:
    uses: skill:av/charters/plain.md
    strategy:
      matrix: {c: "${{ inputs.candidates }}"}
    with: {who: "${{ matrix.c }}"}
""")
    assert spec.root.type == "foreach" and list(spec.root.items) == ["A", "B", "C"]


def test_matrix_over_a_non_list_input_is_rejected():
    bad = ("name: x\ninputs:\n  candidates: 一个字符串\njobs:\n  a:\n"
           "    uses: skill:av/charters/plain.md\n    strategy:\n"
           "      matrix: {c: \"${{ inputs.candidates }}\"}\n    with: {w: \"${{ matrix.c }}\"}\n")
    assert any("必须是**非空列表**" in p for p in _problems(bad))


def test_matrix_over_unknown_input_lists_whats_available():
    bad = ("name: x\ninputs:\n  known: [1]\njobs:\n  a:\n"
           "    uses: skill:av/charters/plain.md\n    strategy:\n"
           "      matrix: {c: \"${{ inputs.ghost }}\"}\n    with: {w: \"${{ matrix.c }}\"}\n")
    assert any("不存在的 inputs.ghost" in p and "known" in p for p in _problems(bad))


# ── matrix job 的产出说明（design/78 §7 实施中发现）──────────────────────────
# 引擎按迭代序号给每一路分配 `outputs/<job>.<N>.md`，但那个路径只进 spec.metadata，
# **从不注入叶子的提示词**；编译期又只知道 job_id，算不出序号。沿用普通 job 那段说明的
# 结果是：N 路并行叶子被同时告知去写同一个 `outputs/<job>.md`——共享工作区上的读改写
# 竞态，先写完的被后写的整段盖掉，而且不报错。所以并行路一律不写文件，报告走文本回。

_MATRIX_PRODUCT = """
name: 并行走查
jobs:
  audit:
    uses: skill:av/actions/static
    strategy:
      matrix:
        item: [a, b]
    with: {target: "${{ matrix.item }}", requirements: r}
"""

_PLAIN_PRODUCT = """
name: 单路
jobs:
  audit:
    uses: skill:av/actions/static
    with: {target: x, requirements: r}
"""


def test_matrix_job_is_told_to_name_its_own_output_file():
    """并行路不能写共用产出文件（静默互相覆盖），但**可以**写文件——
    用自己那一路的唯一标识命名即可（matrix 的每一项天然带一个唯一值）。"""
    inp = _compile(_MATRIX_PRODUCT).root.body.input
    assert "不要写 `outputs/audit.md`" in inp        # 共用名被点名禁止
    assert "唯一标识给文件命名" in inp                # 并给出了可执行的替代
    assert "outputs/audit-<你的标识>.md" in inp


def test_plain_job_still_writes_its_output_file():
    """非 matrix 的 job 行为一个字不变：产出落盘仍是它的恢复点。"""
    inp = _compile(_PLAIN_PRODUCT).root.input
    assert "outputs/audit.md" in inp and "追加" in inp
    assert "不要写文件" not in inp


def test_job_continuation_compiles_to_ir():
    """DSL：continuation（kebab）→ AgentNode.continuation（snake），与 retry 正交并存。"""
    spec = _compile("""
name: t
jobs:
  a:
    uses: skill:av/charters/plain.md
    retry: { max-attempts: 2 }
    continuation: { extra-rounds: 6, max-continuations: 2, gate: always }
""")
    node = spec.root
    while getattr(node, "type", "") != "agent":
        node = (getattr(node, "steps", None) or getattr(node, "branches"))[0]
    assert node.continuation is not None
    assert node.continuation.extra_rounds == 6
    assert node.continuation.max_continuations == 2
    assert node.continuation.gate == "always"
    assert node.retry.max_attempts == 2


def test_job_continuation_rejects_unknown_keys():
    problems = _problems("""
name: t
jobs:
  a:
    uses: skill:av/charters/plain.md
    continuation: { rounds: 6 }
""")
    assert any("continuation 不支持" in p for p in problems)


# ── design/90：就地角色（prompt: / uses: file:）——YAML 不再只能引用技能库 charter ──


def _inline(text: str, args=None, ws=None):
    return compile_workflow_document(
        text, args or {}, load_file=_tpl_load,
        read_workspace=(ws or {}).get if ws is not None else None,
    )


def test_prompt_writes_the_leaf_role_inline():
    """`uses:` 曾是必填且只认 skill: —— 全平台只有 3 本技能带 charter，于是「调研 / 审查 /
    复盘」这类没有现成 charter 的活根本写不出 YAML（design/90 §1.2）。prompt: 是 GHA `run:`
    的对偶：角色就地写，引用留给需要平台钉死判据的验证类图。"""
    spec = _inline("name: w\njobs:\n  a:\n    prompt: |\n      你是研究员。\n")
    assert spec.root.spec.system_prompt.strip() == "你是研究员。"
    meta = spec.root.spec.metadata
    # 就地角色明确标 unpinned：journal 里「有没有 charter_sha256」仍是「平台钉死」的唯一判据
    assert meta["leaf_prompt_source"] == "prompt" and meta["charter_pinned"] is False
    assert len(meta["leaf_prompt_sha256"]) == 64


def test_uses_and_prompt_are_mutually_exclusive_and_one_is_required():
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    prompt: x\n")
    assert any("二选一" in p for p in exc.value.problems)
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    needs: []\n")
    assert any("prompt" in p and "uses" in p for p in exc.value.problems)


def test_job_level_role_switches_only_apply_to_inline_roles():
    spec = _inline("name: w\njobs:\n  a:\n    prompt: x\n    tools: [bash, read_file]\n"
                   "    role-mode: assigned\n    skills: liveapp-verify\n")
    meta = spec.root.spec.metadata
    assert meta["_action_tools"] == ["bash", "read_file"]
    assert meta["_leaf_role_mode"] == "assigned" and meta["_leaf_skills"] == ["liveapp-verify"]
    # skill: charter 的开关住在 charter frontmatter 里——job 上再写一遍就是第二个真相源
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    uses: skill:av/charters/plain.md\n    tools: [bash]\n")
    assert any("单一真相源" in p for p in exc.value.problems)
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    prompt: x\n    role-mode: whatever\n")
    assert any("role-mode" in p for p in exc.value.problems)


def test_uses_file_reads_a_workspace_charter_with_frontmatter():
    ws = {"charters/researcher.md": "---\ntools: web_search\nrole_mode: assigned\n---\n你是研究员。"}
    spec = _inline("name: w\njobs:\n  a:\n    uses: file:charters/researcher.md\n", ws=ws)
    assert spec.root.spec.system_prompt.strip() == "你是研究员。"
    assert spec.root.spec.metadata["_action_tools"] == ["web_search"]
    assert spec.root.spec.metadata["leaf_prompt_source"] == "file:charters/researcher.md"
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    uses: file:charters/ghost.md\n", ws=ws)
    assert any("没有 charter 文件" in p for p in exc.value.problems)
    # 没有工作区（演练场）时给出口，而不是一句 uses 语法错
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    uses: file:charters/researcher.md\n")
    assert any("拿不到工作区" in p and "prompt" in p for p in exc.value.problems)


def test_inline_role_body_takes_inputs_but_refuses_runtime_refs():
    spec = _inline("name: w\ninputs:\n  topic:\njobs:\n  a:\n    prompt: 研究 ${{ inputs.topic }}\n",
                   {"topic": "光刻机"})
    assert spec.root.spec.system_prompt == "研究 光刻机"
    # needs/matrix 是运行时替换，引擎只对 node.input 做——放进 system_prompt 会原样送给模型
    with pytest.raises(WorkflowYamlError) as exc:
        _inline("name: w\njobs:\n  a:\n    prompt: 看 ${{ needs.b.output }}\n  b:\n    prompt: x\n")
    assert any("with:" in p for p in exc.value.problems)
