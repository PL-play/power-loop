"""YAML（GitHub Actions 风格）→ power-loop WorkflowSpec 编译器。

**为什么不自创语法**：模型见过的 GHA workflow 文件以百万计，`jobs/needs/uses/with` 这套惯例
是它最熟的编排词汇；自创 DSL 会把「只有 schema 是陌生的」升级成「语法和 schema 都陌生」。
所以这里不发明语言，只写编译器——**语法借现成的，语义和错误信息全部自己拥有**。

借来的惯例与它解决的问题：

===================  ==========================================================
`needs: [a, b]`      DAG 的边。**并行是隐式的**：没有 needs 的 job 自动同层并发，
                     所以再也不用手写 `{"type":"parallel","branches":[…]}` 包装。
`uses:`              复用的唯一入口。既可指一份 charter（.md），也可指一个 action
                     目录（含 action.yaml 的 IO 契约）——「引用 charter」和「引用
                     整套动作」统一成一个关键字。
`with:`              给 action 传参；键必须对得上 action 声明的 inputs（编译期查）。
`continue-on-error`  节点级错误语义（IR 里 foreach 之外一直没有）。
块标量 `|`           多行提示词/需求原话不再是满屏 `\\n` 转义。
===================  ==========================================================

**数据流默认走文件**（design 讨论 2026-08-14）：每个 job 自动分到一个产出文件
`outputs/<job_id>.md`，边做边追加写；下游用 `${{ needs.x.output }}` 引用时，注入的是
**路径 + 预览**而不是整段全文——token 成本从「强制」变成「按需」，且文件在盘上就是
中断恢复点。切片 `${{ needs.x.output[-50:] }}` 按**行**取（报告是按行组织的，按字符切
会把表格/markdown 拦腰截断），禁 step。
截断一律留痕：conv184 那次报告被静默截断 4000 字、尾部条目直接消失，才有了后来的
verify_report 工具——同样的失败模式不许在新机制里复活。

编译产物是现有的 ``WorkflowSpec``，引擎一行不改。叶子的 system_prompt 编成
``charter:skill:…`` 哨兵，交给宿主的 charter resolver 做注入与盖戳，
所以「裁判提示词由平台注入 + journal 留哈希」这条保证在 YAML 路径上原样成立。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ImportError("Workflow YAML requires PyYAML; install power-loop[workflow].") from exc

from .spec import WorkflowSpec, WorkflowSpecError

#: 运行时文件引用占位符。刻意不用 `{{…}}`——那是引擎的模板变量语法，混用会互相吃掉。
#: 形如 ``@@FILEREF:outputs/static.md@@`` 或 ``@@FILEREF:outputs/runtime.md[-50:]@@``。
FILEREF_RE = re.compile(r"@@FILEREF:([^\[\]@]+)(\[[^\]]*\])?@@")

#: 无切片时注入的预览行数。够 judge 判断「要不要读全文」，又不至于吃掉整个上下文。
DEFAULT_PREVIEW_LINES = 80

_EXPR_RE = re.compile(r"\$\{\{(.+?)\}\}", re.S)
_SLICE_RE = re.compile(r"^\[(-?\d*)(?::(-?\d*))?\]$")
_JOB_ID_RE = re.compile(r"^[A-Za-z_][\w-]*$")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

#: GHA 里几乎每份文件都有、但本平台没有的键——模型一定会写出来，所以要给点名的纠错，
#: 而不是一句笼统的 "unknown key"。
_GHA_FALSE_FRIENDS = {
    "runs-on": "本平台没有 runner：job 跑在本会话的服务端工作区。删掉这行。",
    "steps": "本平台一个 job = 一个 agent 叶子，没有 steps 数组。要跑命令请在叶子里用 bash 工具。",
    "run": "本平台不直接跑 shell：把要做的事写进 action 的 charter，或让叶子用 bash 工具。",
    "on": "本平台的 workflow 由 agent 主动发起，没有事件触发器。删掉 on:。",
    "env": "本平台不注入环境变量；要传值用 with: + action 声明的 inputs。",
    "container": "本平台没有容器选择：叶子跑在平台统一的沙箱里。删掉这行。",
    "services": "本平台不支持 services。",
    "permissions": "本平台的权限由 action 的 tools 白名单决定，不在 YAML 里配。",
    "concurrency": "本平台的并发由平台钳制（max_concurrency），不在 YAML 里配。",
}

#: 编出来的 parallel / foreach 的并发数（design/170 §1，10-17）：写库的硬上限，由平台 ``max_concurrency``（后台可配，默认 8）
#: 往下钳。原来不写，落到库默认 5——后台配 8 也只跑 5 路，工具描述却说「最多 8」。
_FANOUT = 64

_TOP_KEYS = {"name", "inputs", "jobs", "timeout-minutes", "loops", "concurrency-scope"}
_LOOP_KEYS = {"jobs", "until", "max-iterations", "continue", "on-exhausted", "strategy"}
#: 循环自己交的结构化结果（引擎 LoopNode 的聚合）；带 matrix 的循环交 {items: [每条一份]}
_LOOP_OUTPUTS = {"converged": {"type": "boolean", "description": "最后一轮是否满足 until"},
                 "iterations": {"type": "integer", "description": "跑了几轮"},
                 "why": {"type": "string", "description": "until 最后一次的判断说明"}}
_LOOP_MATRIX_OUTPUTS = {"items": {"type": "array", "description": "每个条目一份 {converged, iterations, why, last}"}}
_JOB_KEYS = {"name", "uses", "prompt", "needs", "with", "continue-on-error", "retry", "fallback",
             "output-file", "strategy", "continuation", "tools", "role-mode", "skills", "outputs", "if"}

#: `uses: skill:` 的角色开关住在 charter 的 frontmatter 里（平台钉死那份）——job 上再写一遍
#: 会出现两个真相源。就地写的角色（prompt: / uses: file:）才允许在 job 上写。
_INLINE_ONLY_JOB_KEYS = ("tools", "role-mode", "skills", "outputs")


class WorkflowYamlError(Exception):
    """编译失败。聚合**全部**问题一次抛出——模型改一轮就该能全对，而不是改一条报一条。"""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"workflow YAML 有 {len(self.problems)} 个问题：\n{body}")


@dataclass
class ActionDef:
    """一份角色定义（编译期可校验的那部分）。三种来源，形状统一：

    * ``uses: skill:<技能>/<路径>`` —— 平台钉死的 charter：只递 ``charter_ref`` 哨兵，
      正文由 `app/tools/workflow.py::_resolve_charter` 在建 run 时从技能库注入并盖哈希戳。
      验证/审计类角色必须走这条（journal 里有戳 = 判据不是编排方写的）。
    * ``uses: file:<工作区路径>.md`` —— 自己写在工作区里的 charter：编译期读进来内联，
      盖 sha 但标 unpinned（可复用、可版本管理，但不是平台钉死的判据）。
    * ``prompt: |`` —— 就地写在 YAML 里的角色提示词（GHA ``run:`` 的对偶）。

    没有第三种以外的口子：`uses` 与 `prompt` 二选一，两个都写或都不写都在编译期报错。
    """

    ref: str                                  # uses: 的原值（inline prompt 时是 "prompt"）
    charter_ref: str = ""                     # charter:skill:<skill>/<path> 哨兵（仅 skill: 来源）
    name: str = ""
    tools: list[str] | None = None
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 就地正文（prompt: / uses: file: 两种来源）；为 None 表示走 charter_ref 哨兵。
    inline_body: str | None = None
    #: 正文来源标记，进叶子 metadata 供事后审计（"prompt" / "file:<路径>"）。
    inline_source: str = ""
    role_mode: str | None = None              # assigned | discovery
    skills: list[str] | None = None           # 环境块技能收窄


@dataclass
class _Job:
    job_id: str
    name: str
    uses: str
    prompt: str
    needs: list[str]
    with_: dict[str, Any]
    continue_on_error: bool
    retry: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    fallback: str | None = None
    matrix_var: str | None = None          # strategy.matrix 的变量名
    matrix_items: list[Any] | None = None  # 静态列表
    matrix_from: str | None = None         # 动态：<job>.<key>（上游结构化输出）
    max_parallel: int | None = None
    output_file: str | None = None
    action: ActionDef | None = None
    job_tools: list[str] | None = None       # 仅就地角色：job 级工具白名单（只收窄）
    job_skills: list[str] | None = None      # 仅就地角色：环境块技能收窄
    job_role_mode: str | None = None         # 仅就地角色：assigned | discovery
    job_outputs: dict[str, dict[str, Any]] | None = None   # 仅就地角色：结构化输出（design/170 §2）
    if_expr: str | None = None                # 条件（design/170 第二批）：编成节点的 when
    loop_id: str | None = None                # 属于哪个循环（design/170 第三批）
    loop_matrix_var: str | None = None        # 所在循环带 matrix 时的变量名（体内 ${{ matrix.x }} 用）
    is_loop: bool = False                     # 这是循环本身（编译期的占位 job，不是叶子）


@dataclass
class _Loop:
    loop_id: str
    body: list[str]
    until: str
    max_iterations: int = 3
    continue_nodes: list[str] | None = None
    on_exhausted: str = "fail"
    strategy: dict[str, Any] = field(default_factory=dict)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader + 重复键报错 + YAML 1.2 布尔。

    重复键：YAML 默认静默后者覆盖前者——那是最难查的一类错。
    布尔：YAML 1.1 把 ``on/off/yes/no/y/n`` 都当布尔（著名的「挪威问题」：国家代码 ``NO``
    变成 False）。两个后果都真实存在——① 用户需求原话里一个孤立的 `no` 会变成 False；
    ② GHA 反射写出的 ``on: [push]``，键会变成 `True`，报错就成了「顶层不支持 'True'」这种
    看不懂的话。这里收敛到 YAML 1.2：只有 true/false 是布尔，其余是字符串。
    """


_StrictLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in rules if tag != "tag:yaml.org,2002:bool"]
    for ch, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"重复的键 {key!r}（YAML 默认静默覆盖，本平台拒绝）", key_node.start_mark
            )
        seen.add(key)
    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def parse_charter_frontmatter(raw: str) -> tuple[str, list[str] | None, list[str] | None, str | None]:
    """charter 文件的极简 frontmatter：``tools:``（工具白名单）、``skills:``（环境块收窄）、
    ``role_mode:``（工作区契约形态，design/84）。返回 (正文, tools, skills, role_mode)。

    住在编译器里而不是工具面：`uses: file:` 的 charter 由编译器读，`uses: skill:` 的 charter 由
    `_resolve_charter` 读——两条路必须用**同一个**解析器，否则同一份 frontmatter 在两条路上
    含义会漂（design/90 §5.3）。无 frontmatter → (原文, None, None, None)。
    """
    if not raw.startswith("---\n"):
        return raw, None, None, None
    end = raw.find("\n---", 4)
    if end < 0:
        return raw, None, None, None
    head, body = raw[4:end], raw[end + 4:].lstrip("\n")

    def _list(value: str) -> list[str]:
        items = value.strip().strip("[]")
        return [t.strip().strip("'\"") for t in items.split(",") if t.strip().strip("'\"")]

    def _tool_names(value: str) -> list[str]:
        """工具白名单。作者常写成 `liveapp_run(action="ui")` 来表达「只用这个动作」——
        **括号后面的部分不是工具名的一部分**，这里剥掉。

        不剥的代价实测过（会话 261 首发验证）：`liveapp_app(action="list")` 整串当成工具名，
        匹配不上任何已注册工具 → 静默丢掉 → 叶子调用时报 `Unknown tool: liveapp_app`，
        而任务书明明写着它有。动作级限制由正文表达，白名单只管「哪些工具」。
        """
        out: list[str] = []
        for raw in _list(value):
            name = raw.split("(", 1)[0].strip()
            if name and name not in out:
                out.append(name)
        return out

    tools: list[str] | None = None
    skills: list[str] | None = None
    role_mode: str | None = None
    for line in head.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "tools":
            tools = _tool_names(value)
        elif key.strip() == "skills":
            skills = _list(value)
        elif key.strip() == "role_mode":
            v = value.strip().strip("'\"")
            role_mode = v if v in ("assigned", "discovery") else None
    return body, tools, skills, role_mode


def output_path(job_id: str) -> str:
    """job 的产出文件——每个 job 有且只有一个，写别人的会被拒（并发共享工作区，不隔离就是竞态）。"""
    return f"outputs/{job_id}.md"


# ── 表达式 ${{ … }} ────────────────────────────────────────────────────────────


_OUT_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def _output_prop(v: dict[str, Any]) -> dict[str, Any]:
    typ = str((v or {}).get("type") or "string")
    prop: dict[str, Any] = {"type": typ if typ in _OUT_TYPES else "string"}
    if (v or {}).get("description"):
        prop["description"] = str(v["description"])
    if prop["type"] == "array":
        prop["items"] = (v or {}).get("items") if isinstance((v or {}).get("items"), dict) else {}
    return prop


def _parse_job_outputs(raw: Any, where: str, problems: list[str]) -> dict[str, dict[str, Any]] | None:
    """就地 job 的 ``outputs:``：``键: 类型`` 或 ``键: {type, description}``。下游用 ``${{ needs.<job>.outputs.<键> }}`` 引用。"""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        problems.append(f"{where}: outputs 要是一个对象：键: 类型（string / number / integer / boolean / array / object）"
                        "或 键: {{type, description}}。")
        return None
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if not re.match(r"^[A-Za-z_]\w*$", str(k)):
            problems.append(f"{where}: outputs 的键 '{k}' 不合法（字母、数字、下划线，字母或下划线开头）。")
            continue
        spec = {"type": v} if isinstance(v, str) else (v if isinstance(v, dict) else {})
        if str(spec.get("type") or "string") not in _OUT_TYPES:
            problems.append(f"{where}: outputs.{k} 的 type '{spec.get('type')}' 不认识（{'/'.join(sorted(_OUT_TYPES))}）。")
            continue
        out[str(k)] = spec
    return out or None


_IF_RE = re.compile(
    r"^\$\{\{\s*(?P<neg>!)?\s*needs\.(?P<job>[A-Za-z_][\w-]*)\.outputs\.(?P<key>[A-Za-z_]\w*)\s*"
    r"(?:(?P<op>==|!=)\s*(?P<lit>true|false|-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"))?\s*\}\}$")


def _compile_if(expr: str, job: _Job, jobs: dict[str, _Job], problems: list[str],
                where: str | None = None) -> dict[str, Any] | None:
    """job 的 ``if:`` → 引擎节点的 ``when``。只认四种（够用、好校验）：
    ``${{ needs.j.outputs.k }}``（为真）· ``${{ !needs.j.outputs.k }}``（为假）· ``== 字面量`` · ``!= 字面量``。
    循环的 ``until:`` 也走这里（同一套写法）。"""
    where = where or f"job '{job.job_id}' if"
    m = _IF_RE.match(expr.strip())
    if not m:
        problems.append(f"{where}: 看不懂 `{expr}`。只支持 ${{{{ needs.<job>.outputs.<键> }}}}、${{{{ !needs.<job>.outputs.<键> }}}}、"
                        f"${{{{ needs.<job>.outputs.<键> == 值 }}}}、!= 值（值是 true / false / 数字 / '文字'）。")
        return None
    dep, key = m.group("job"), m.group("key")
    if dep not in jobs:
        problems.append(f"{where}: 引用了不存在的 job '{dep}'。")
        return None
    if dep not in job.needs:
        problems.append(f"{where}: 引用了 '{dep}'，要把它写进 needs（没有先后就判断不了）。")
        return None
    declared = set((jobs[dep].action.outputs if jobs[dep].action else None) or {})
    if declared and key not in declared:
        problems.append(f"{where}: '{dep}' 没有声明输出 '{key}'。它声明的是：{'、'.join(sorted(declared))}")
        return None
    if not declared:
        problems.append(f"{where}: '{dep}' 没有声明 outputs，没有结构化结果可判断。给它加 outputs:。")
        return None
    ref = f"{dep}.{key}"
    if m.group("op"):
        lit = m.group("lit")
        val: Any = (lit == "true") if lit in ("true", "false") else (
            lit[1:-1] if lit[:1] in ("'", '"') else (float(lit) if "." in lit else int(lit)))
        op = "eq" if m.group("op") == "==" else "ne"
        if m.group("neg"):
            op = "ne" if op == "eq" else "eq"
        return {"ref": ref, "op": op, "value": val}
    return {"ref": ref, "op": "falsy" if m.group("neg") else "truthy"}


def _compile_expr(
    raw: str, job: _Job, jobs: dict[str, _Job], wf_inputs: dict[str, Any], problems: list[str]
) -> str:
    """把一个 ``${{ … }}`` 表达式编译成字面量或运行时占位符。"""
    expr = raw.strip()
    where = f"job '{job.job_id}'"

    if expr.startswith("matrix."):
        var = expr[len("matrix."):].strip()
        mvar = job.matrix_var or job.loop_matrix_var
        if not mvar:
            problems.append(f"{where}: 用了 ${{{{ matrix.{var} }}}} 但这个 job 没有 strategy.matrix"
                            f"（也不在带 matrix 的循环里）。")
            return ""
        if var != mvar:
            problems.append(
                f"{where}: matrix 变量是 '{mvar}'，不是 '{var}'。"
            )
            return ""
        return "{{" + mvar + "}}"      # 交给引擎按迭代替换

    if expr == "loop.iteration":
        if not job.loop_id:
            problems.append(f"{where}: ${{{{ loop.iteration }}}} 只能用在循环里的 job 上。")
            return ""
        return "{{loop_iteration}}"     # 引擎每轮注入（从 1 开始）

    if expr.startswith("inputs."):
        key = expr[len("inputs."):].strip()
        if key not in wf_inputs:
            known = "、".join(sorted(wf_inputs)) or "(顶层没声明 inputs)"
            problems.append(f"{where}: 引用了不存在的 inputs.{key}。可用：{known}")
            return ""
        return str(wf_inputs[key])

    if expr.startswith("needs."):
        rest = expr[len("needs."):]
        # needs.<job>.output[切片]  |  needs.<job>.outputs.<key>
        m = re.match(r"^([A-Za-z_][\w-]*)\.(output|outputs)\b(.*)$", rest, re.S)
        if not m:
            problems.append(
                f"{where}: 看不懂的表达式 ${{{{ {expr} }}}}。needs 只支持两种形式："
                f"`needs.<job>.output`（产出文件，可加行切片如 [-50:]）与 "
                f"`needs.<job>.outputs.<键>`（结构化输出）。"
            )
            return ""
        dep, kind, tail = m.group(1), m.group(2), m.group(3).strip()
        if dep not in jobs:
            problems.append(f"{where}: needs 引用了不存在的 job '{dep}'。可用：{'、'.join(jobs)}")
            return ""
        if dep not in job.needs:
            problems.append(
                f"{where}: 引用了 '{dep}' 的产出，但没把它写进 needs——"
                f"没有 needs 就没有先后顺序，运行时那个文件还不存在。请写 needs: [{dep}, …]。"
            )
            return ""

        if kind == "output":
            if jobs[dep].is_loop:
                problems.append(
                    f"{where}: '{dep}' 是循环，没有产出文件。要循环里某个 job 的文件就 needs 那个 job、"
                    f"写 needs.<job>.output；要结果写 needs.{dep}.outputs.converged / iterations / why。"
                )
                return ""
            if jobs[dep].loop_matrix_var:
                problems.append(
                    f"{where}: '{dep}' 在带 matrix 的循环 '{jobs[dep].loop_id}' 里，每个条目各有一份产出，"
                    f"没有单一的 needs.{dep}.output。"
                    + ("同一个循环里 needs 它就够了：它的最终回复会自动进你的上下文（让它在回复里写文件路径）。"
                       if jobs[dep].loop_id == job.loop_id else f"needs 那个循环（'{jobs[dep].loop_id}'）就够了。")
                )
                return ""
            if jobs[dep].matrix_var:
                problems.append(
                    f"{where}: '{dep}' 是 matrix job，它的产出是**每次迭代一个文件**，没有单一的 "
                    f"needs.{dep}.output。把 '{dep}' 写进 needs 就够了——各次迭代的结论会自动聚合进"
                    f"你的上下文；要读全文用 glob 找 `outputs/{dep}-item.*.md`。"
                )
                return ""
            slice_expr = ""
            if tail:
                if not _SLICE_RE.match(tail):
                    problems.append(
                        f"{where}: 切片 `{tail}` 不合法。按**行**取，支持 [:80]、[-50:]、[10:20]；"
                        f"不支持步长（[::2] 对文本没有意义）。"
                    )
                    return ""
                slice_expr = tail
            return f"@@FILEREF:{jobs[dep].output_file or output_path(dep)}{slice_expr}@@"

        # outputs.<key>：必须是上游 action 声明过的键——这正是「跨节点引用无法静态校验」的补丁
        key = tail[1:].strip() if tail.startswith(".") else ""
        if not key:
            problems.append(f"{where}: `needs.{dep}.outputs` 后面要跟具体的键名。")
            return ""
        dep_action = jobs[dep].action
        declared = set((dep_action.outputs if dep_action else {}) or {})
        if declared and key not in declared:
            problems.append(
                f"{where}: '{dep}' 没有声明输出 '{key}'。它声明的是：{'、'.join(sorted(declared))}"
            )
            return ""
        return f"{{{{{dep}.{key}}}}}"   # 交给引擎的结构化输出引用

    problems.append(
        f"{where}: 不支持的表达式 ${{{{ {expr} }}}}。本平台只认 `inputs.<键>`、"
        f"`needs.<job>.output[切片]`、`needs.<job>.outputs.<键>`、`matrix.<变量>`、`loop.iteration`。"
    )
    return ""


def _render(
    value: Any, job: _Job, jobs: dict[str, _Job], wf_inputs: dict[str, Any], problems: list[str]
) -> str:
    text = value if isinstance(value, str) else str(value)
    return _EXPR_RE.sub(
        lambda m: _compile_expr(m.group(1), job, jobs, wf_inputs, problems), text
    )


# ── 解析与校验 ────────────────────────────────────────────────────────────────


def _parse_uses(uses: str, problems: list[str], where: str) -> tuple[str, str] | None:
    if not uses.startswith("skill:"):
        problems.append(
            f"{where}: uses 必须是 `skill:<技能名>/<路径>`（技能库里的 action 目录或 charter .md）"
            f"或 `file:<工作区路径>.md`（你自己写的 charter）。得到 {uses!r}。"
            f"（本平台没有 actions/checkout@v4 这类市场 action；要就地写提示词用 prompt: |。）"
        )
        return None
    body = uses[len("skill:"):]
    skill, _, rel = body.partition("/")
    if not skill.strip() or not rel.strip():
        problems.append(f"{where}: uses 缺技能名或路径：{uses!r}")
        return None
    return skill.strip(), rel.strip()


def _load_action(
    uses: str,
    load_file: Callable[[str, str], str | None],
    problems: list[str],
    where: str,
    *,
    read_workspace: Callable[[str], str | None] | None = None,
) -> ActionDef | None:
    if uses.startswith("file:"):
        rel = uses[len("file:"):].strip()
        if not rel.endswith(".md"):
            problems.append(f"{where}: uses: file: 必须指向一份 .md charter，得到 {uses!r}。")
            return None
        if read_workspace is None:
            problems.append(
                f"{where}: 这次编译拿不到工作区（演练场/无会话），不能用 uses: file:。"
                f"改用 prompt: | 就地写，或 uses: skill:<技能>/<路径>。"
            )
            return None
        raw = read_workspace(rel)
        if raw is None:
            problems.append(
                f"{where}: 工作区里没有 charter 文件 '{rel}'（相对工作区根，不能越界）。"
            )
            return None
        body, tools, skills, role_mode = parse_charter_frontmatter(raw)
        if not body.strip():
            problems.append(f"{where}: charter 文件 '{rel}' 是空的。")
            return None
        return ActionDef(ref=uses, inline_body=body, inline_source=f"file:{rel}",
                         tools=tools, skills=skills, role_mode=role_mode)

    parsed = _parse_uses(uses, problems, where)
    if parsed is None:
        return None
    skill, rel = parsed

    if rel.endswith(".md"):
        # 直接指一份 charter：没有 IO 契约，工具白名单仍由 charter frontmatter 决定。
        if load_file(skill, rel) is None:
            problems.append(f"{where}: charter 文件 '{rel}' 在技能 '{skill}' 中不存在（或技能未发布）。")
            return None
        return ActionDef(ref=uses, charter_ref=f"charter:skill:{skill}/{rel}")

    raw = load_file(skill, f"{rel}/action.yaml")
    if raw is None:
        problems.append(
            f"{where}: 找不到 '{rel}/action.yaml'（技能 '{skill}'）。uses 指向 action 目录时，"
            f"该目录下必须有 action.yaml；若想直接用一份 charter，请把路径写到 .md。"
        )
        return None
    try:
        data = yaml.load(raw, Loader=_StrictLoader) or {}
    except yaml.YAMLError as exc:
        problems.append(f"{where}: '{rel}/action.yaml' 解析失败：{exc}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{where}: '{rel}/action.yaml' 顶层必须是对象。")
        return None

    charter_rel = str(data.get("charter") or "").strip()
    if not charter_rel:
        problems.append(f"{where}: '{rel}/action.yaml' 缺 charter:（指向本 action 的角色提示词文件）。")
        return None
    # action.yaml 里的路径**始终相对 action 目录**（同 GHA 的复合 action：包内路径就是包内路径）。
    # 想指技能根下的共享 charter，用前导 `/`。
    charter_path = (
        charter_rel.lstrip("/") if charter_rel.startswith("/") else f"{rel}/{charter_rel}"
    )
    if load_file(skill, charter_path) is None:
        problems.append(f"{where}: action 声明的 charter '{charter_path}' 在技能 '{skill}' 中不存在。")
        return None

    tools = data.get("tools")
    if tools is not None and (
        not isinstance(tools, list) or not all(isinstance(t, str) for t in tools)
    ):
        problems.append(f"{where}: action.yaml 的 tools 必须是字符串数组。")
        tools = None
    if isinstance(tools, list):
        # 与 charter frontmatter 同一条规矩：`名字(action="x")` 里括号后面的不是工具名。
        tools = list(dict.fromkeys(t.split("(", 1)[0].strip() for t in tools if t.split("(", 1)[0].strip()))

    def _io(block: Any, label: str) -> dict[str, dict[str, Any]]:
        if block is None:
            return {}
        if not isinstance(block, dict):
            problems.append(f"{where}: action.yaml 的 {label} 必须是对象（键=名字）。")
            return {}
        out: dict[str, dict[str, Any]] = {}
        for k, v in block.items():
            out[str(k)] = v if isinstance(v, dict) else {}
        return out

    return ActionDef(
        ref=uses,
        charter_ref=f"charter:skill:{skill}/{charter_path}",
        name=str(data.get("name") or ""),
        tools=list(tools) if tools is not None else None,
        inputs=_io(data.get("inputs"), "inputs"),
        outputs=_io(data.get("outputs"), "outputs"),
    )


def _parse_strategy(
    raw: Any, job_id: str, problems: list[str], wf_inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """``strategy: {matrix: {file: [...]}, max-parallel: 4}`` → foreach 的构件。

    借的是 GHA 的 matrix：**同一个 job 对一组值各跑一次**。值可以是写死的列表，也可以是
    ``${{ needs.<job>.outputs.<键> }}``（上游必须在 action.yaml 里声明过那个 output——
    那正是「动态铺开」与「静态列表」的唯一区别）。
    矩阵变量在本 job 的 with 里用 ``${{ matrix.<变量名> }}`` 取。
    """
    empty: dict[str, Any] = {}
    if raw is None:
        return empty
    if not isinstance(raw, dict):
        problems.append(f"job '{job_id}': strategy 必须是对象，如 strategy: {{matrix: {{file: [a, b]}}}}")
        return empty
    for key in set(raw) - {"matrix", "max-parallel"}:
        problems.append(
            f"job '{job_id}': strategy 不支持 '{key}'。只认 matrix / max-parallel"
            f"（本平台没有 fail-fast / include / exclude）。"
        )
    matrix = raw.get("matrix")
    if not isinstance(matrix, dict) or len(matrix) != 1:
        problems.append(
            f"job '{job_id}': matrix 必须正好有**一个**变量，如 matrix: {{file: [a.js, b.js]}}"
            f"（本平台不做多维笛卡尔积——要多维就自己展开成多个 job）。"
        )
        return empty
    var, values = next(iter(matrix.items()))
    var = str(var)
    if not _IDENT_RE.match(var):
        problems.append(f"job '{job_id}': matrix 变量名 '{var}' 非法（字母/下划线开头，后接字母数字下划线）。")
        return empty
    out: dict[str, Any] = {"matrix_var": var}
    if isinstance(values, str) and _EXPR_RE.search(values):
        expr = _EXPR_RE.search(values).group(1).strip()
        m = re.match(r"^needs\.([A-Za-z_][\w-]*)\.outputs\.([A-Za-z_]\w*)$", expr)
        if m:
            out["matrix_from"] = f"{m.group(1)}.{m.group(2)}"
        elif expr.startswith("inputs."):
            # 顶层 inputs 里的列表：编译期就能定下来（配合 template+args = 调用方传几项跑几路）
            key = expr[len("inputs."):].strip()
            src = (wf_inputs or {}).get(key)
            if key not in (wf_inputs or {}):
                known = "、".join(sorted(wf_inputs or {})) or "(顶层没声明 inputs)"
                problems.append(f"job '{job_id}': matrix 引用了不存在的 inputs.{key}。可用：{known}")
                return empty
            if not isinstance(src, list) or not src:
                problems.append(
                    f"job '{job_id}': matrix 引用的 inputs.{key} 必须是**非空列表**"
                    f"（得到 {type(src).__name__}）。用 template 调用时把它作为数组传进 args。"
                )
                return empty
            out["matrix_items"] = list(src)
        else:
            problems.append(
                f"job '{job_id}': matrix 的取值只支持写死的列表、${{{{ inputs.<键> }}}}"
                f"（顶层 inputs 里的列表）或 ${{{{ needs.<job>.outputs.<键> }}}}，得到 {values!r}。"
                f"（产出文件 needs.x.output 是一段文本，不能当列表铺开。）"
            )
            return empty
    elif isinstance(values, list) and values:
        out["matrix_items"] = list(values)
    else:
        problems.append(
            f"job '{job_id}': matrix 的值要么是非空列表，要么是 ${{{{ needs.<job>.outputs.<键> }}}}。"
        )
        return empty
    mp = raw.get("max-parallel")
    if mp is not None:
        if not isinstance(mp, int) or isinstance(mp, bool) or mp < 1:
            problems.append(f"job '{job_id}': max-parallel 必须是 >= 1 的整数。")
        else:
            out["max_parallel"] = mp
    return out


def _parse_continuation_block(raw: Any, job_id: str, problems: list[str]) -> dict[str, Any] | None:
    """``continuation: {max-continuations: 1, extra-rounds: 8, gate: todo_remaining}``
    → IR snake_case（design/85：hit_round_limit 原会话续跑，与 retry 正交——那是新会话重跑）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problems.append(
            f"job '{job_id}': continuation 必须是对象，如 continuation: {{extra-rounds: 8}}")
        return None
    keymap = {"max-continuations": "max_continuations", "extra-rounds": "extra_rounds",
              "gate": "gate"}
    unknown = set(raw) - set(keymap)
    if unknown:
        problems.append(
            f"job '{job_id}': continuation 不支持 {sorted(unknown)}。"
            f"只认 max-continuations / extra-rounds / gate。")
        return None
    out = {keymap[k]: v for k, v in raw.items()}
    return out or {}


def _parse_retry_block(raw: Any, job_id: str, problems: list[str]) -> dict[str, Any] | None:
    """``retry: {max-attempts: 3, on: [failed, empty], backoff-s: 2}`` → IR 的 snake_case 形状。

    取值合法性（上限 5、trigger 白名单）由 WorkflowSpec 的严格 schema 兜底并报错，这里只做
    形状与键名转换——两处都校验会出现两套口径。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problems.append(
            f"job '{job_id}': retry 必须是对象，如 retry: {{max-attempts: 3, on: [failed, empty]}}"
        )
        return None
    alias = {"max-attempts": "max_attempts", "backoff-s": "backoff_s",
             "backoff-factor": "backoff_factor", "on": "on"}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key)
        if k in alias:
            out[alias[k]] = value
        elif k in ("max_attempts", "backoff_s", "backoff_factor"):
            out[k] = value            # snake 也认，模型两种都可能写
        else:
            problems.append(
                f"job '{job_id}': retry 不支持 '{k}'。只认 max-attempts / on / backoff-s / backoff-factor。"
            )
    return out or None


def _parse_loops(raw: Any, jobs: dict[str, _Job], wf_inputs: dict[str, Any], problems: list[str]) -> dict[str, _Loop]:
    """顶层 ``loops:``（design/170 第三批）：把几个 job 圈成一个循环——跑完一轮看 ``until``，不成立再跑一轮，最多 N 轮。

    ``continue`` 里的 job 从第 2 轮起在**自己上一轮的会话**里接着做（收到上一轮 until 那个 job 交回的意见）；
    不写时默认是除了 until 引用的那个 job 以外的全部（开发接着改，审核每轮新开、保持独立）。
    带 ``strategy.matrix`` 时每个条目各跑一路循环（并行），体内用 ``${{ matrix.<变量> }}`` 取条目。
    循环自己也是一个 job id：别的 job 可以 needs 它，取 ``needs.<循环>.outputs.converged / iterations / why``。

    会把循环登记成占位 job（``is_loop``）放进 ``jobs``，并给体内的 job 标上 ``loop_id``。
    """
    loops: dict[str, _Loop] = {}
    if raw is None:
        return loops
    if not isinstance(raw, dict) or not raw:
        problems.append("loops: 必须是对象（键=循环 id，值={jobs, until, max-iterations?, continue?, on-exhausted?, strategy?}）。")
        return loops
    owner: dict[str, str] = {}
    for lid_raw, spec in raw.items():
        lid = str(lid_raw)
        where = f"loop '{lid}'"
        if not _JOB_ID_RE.match(lid):
            problems.append(f"{where}: 循环 id 只能是字母/数字/下划线/连字符，且以字母或下划线开头。")
            continue
        if lid in jobs:
            problems.append(f"{where}: 和 job '{lid}' 重名。循环自己也是一个 id（别的 job 会 needs 它），换个名字。")
            continue
        if not isinstance(spec, dict):
            problems.append(f"{where}: 必须是对象。")
            continue
        for key in set(spec) - _LOOP_KEYS:
            problems.append(f"{where}: 不支持 '{key}'。循环只认 jobs / until / max-iterations / continue / on-exhausted / strategy。")
        body = spec.get("jobs")
        if isinstance(body, str):
            body = [body]
        if not isinstance(body, list) or not body or not all(isinstance(b, str) for b in body):
            problems.append(f"{where}: jobs 要列出循环里的 job id（至少一个），如 jobs: [dev, review]。")
            continue
        body = list(dict.fromkeys(body))
        ok = True
        for b in body:
            if b not in jobs or jobs[b].is_loop:
                problems.append(f"{where}: jobs 里的 '{b}' 不是一个 job。可用：{'、'.join(sorted(j for j in jobs if not jobs[j].is_loop))}")
                ok = False
            elif b in owner:
                problems.append(f"{where}: job '{b}' 已经在循环 '{owner[b]}' 里了。一个 job 只能属于一个循环。")
                ok = False
            elif jobs[b].matrix_var:
                problems.append(f"{where}: job '{b}' 自己带了 strategy.matrix。循环里的 job 不能各自铺开——"
                                f"要每个条目各跑一路循环，把 matrix 写在循环上（loops.{lid}.strategy）。")
                ok = False
        until = spec.get("until")
        if not isinstance(until, str) or not until.strip():
            problems.append(f"{where}: 缺 until:（循环什么时候算做完），如 until: ${{{{ needs.review.outputs.pass }}}}。")
            ok = False
        mi = spec.get("max-iterations", 3)
        if isinstance(mi, bool) or not isinstance(mi, int) or not 1 <= mi <= 10:
            problems.append(f"{where}: max-iterations 是 1–10 的整数（默认 3）。")
            mi = 3
        oe = str(spec.get("on-exhausted") or "fail")
        if oe not in ("fail", "continue"):
            problems.append(f"{where}: on-exhausted 只能是 fail（默认：到了轮数还没过，循环算失败）或 continue（照常往下走）。")
            oe = "fail"
        cont = spec.get("continue")
        if isinstance(cont, str):
            cont = [cont]
        if cont is not None and (not isinstance(cont, list) or not all(isinstance(c, str) for c in cont)):
            problems.append(f"{where}: continue 是 job id 数组（第 2 轮起接着上一轮会话做的 job）。")
            cont = None
        if cont is not None:
            for c in cont:
                if c not in body:
                    problems.append(f"{where}: continue 里的 '{c}' 不在这个循环的 jobs 里。")
        strategy = _parse_strategy(spec.get("strategy"), lid, problems, wf_inputs) if spec.get("strategy") is not None else {}
        if not ok:
            continue
        for b in body:
            owner[b] = lid
            jobs[b].loop_id = lid
            jobs[b].loop_matrix_var = strategy.get("matrix_var")
        outs = _LOOP_MATRIX_OUTPUTS if strategy.get("matrix_var") else _LOOP_OUTPUTS
        jobs[lid] = _Job(job_id=lid, name=lid, uses="", prompt="", needs=[], with_={}, continue_on_error=False,
                         is_loop=True, action=ActionDef(ref="loop", outputs=dict(outs)))
        loops[lid] = _Loop(loop_id=lid, body=body, until=until.strip(), max_iterations=mi,
                           continue_nodes=list(dict.fromkeys(cont)) if cont is not None else None,
                           on_exhausted=oe, strategy=strategy)
    return loops


def _components(jobs: dict[str, _Job]) -> list[dict[str, _Job]]:
    """把 job 图切成**弱连通分量**（互不相干的几条链）。

    IR 是树结构（sequence / parallel），没有「谁的依赖齐了谁就开跑」的 DAG 调度器，所以
    只能用「分层 + 屏障」来编码依赖。分量之间没有任何依赖关系，**不该共用屏障**——
    不拆的话，一条慢链会把另一条快链一起拖住（正确性没问题，纯属白等）。
    """
    parent = {jid: jid for jid in jobs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for jid, job in jobs.items():
        for dep in job.needs:
            if dep in jobs:
                union(jid, dep)
    groups: dict[str, dict[str, _Job]] = {}
    for jid in jobs:
        groups.setdefault(find(jid), {})[jid] = jobs[jid]
    # 稳定顺序：按各分量里最小的 job id 排，保证同一份 YAML 每次编出同样的图
    return [g for _, g in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def _layers(jobs: dict[str, _Job], problems: list[str]) -> list[list[str]]:
    """按依赖分层（Kahn）。同层 = 可并发；**并行是隐式的，不用写 parallel 节点**。

    只按**存在的** job 排依赖：指向不存在的 job 已经单独报过错，若把它留在待满足集合里，
    Kahn 会把「永远满足不了」误报成「有环」——一条与真实原因无关的错误信息。
    """
    pending = {jid: {d for d in j.needs if d in jobs} for jid, j in jobs.items()}
    out: list[list[str]] = []
    done: set[str] = set()
    while pending:
        ready = sorted(jid for jid, deps in pending.items() if not (deps - done))
        if not ready:
            stuck = "、".join(sorted(pending))
            problems.append(f"needs 里有环（或指向环的依赖），涉及 job：{stuck}。DAG 不能有环。")
            return out
        out.append(ready)
        done.update(ready)
        for jid in ready:
            pending.pop(jid)
    return out


_PRODUCT_NOTE = (
    "\n\n## 你的产出\n"
    "- 边做边把结论**追加**写入 `{path}`（不要等最后一轮才写：中途被打断时，"
    "已写下的部分就是恢复点）。\n"
    "- 首次执行时创建文件（write_file 会自动建目录）；返工或恢复时先读已有报告，保留有效结论，更新受影响部分。\n"
    "- 这是你的交接报告；任务要求的代码、资源照常修改。其它 job 的报告只读不写。\n"
    "- 最终回复只要一句指向该文件即可，正文写在文件里。"
)

#: matrix（foreach）job 的产出说明 —— **不能**沿用上面那段。
#: 引擎按迭代序号给每一路分配 `outputs/<job>.<N>.md`，但那个路径只进 spec.metadata，
#: **从不注入叶子的提示词**；而编译期只知道 job_id，算不出序号。于是沿用上面那段的结果是：
#: N 路并行的叶子被同时告知去写同一个 `outputs/<job>.md` —— 共享工作区上的读改写竞态，
#: 先写完的会被后写的整段盖掉，而且不报错。
#: 所以并行路一律**不写文件**，把结论作为最终文本回复交回编排方（它本来就要拿去改东西）。
_PRODUCT_NOTE_MATRIX = (
    "\n\n## 你的产出\n"
    "- **不要写 `{shared}`。** 你是并行多路中的一路，那是所有路共用的名字，"
    "一起写会互相整段覆盖，而且不会报错。\n"
    "- 要落盘就**用你这一路的唯一标识给文件命名**（你的输入里那个只属于你的值，"
    "如条目名 / 屏名 / 文件名），例如 `outputs/{job}-<你的标识>.md`——"
    "这样 N 路各写各的，永不撞车。命名规则以你的任务说明为准。\n"
    "- 最终文本回复给**一句摘要 + 文件路径**，正文写在文件里；"
    "编排方要把 N 路的结果一起读完再动手，你多写的每一行都摊在它的上下文里。"
)


def compile_workflow_template(
    ref: str,
    args: dict[str, Any] | None,
    *,
    load_file: Callable[[str, str], str | None],
    read_workspace: Callable[[str], str | None] | None = None,
    output_mode: str = "file",
) -> WorkflowSpec:
    """``skill:<技能>/<路径>.yaml`` + args → WorkflowSpec（整图复用，不用抄模板）。

    模板就是一份普通的 workflow YAML，只是它的 ``inputs`` 兼作**参数声明**：

    * 值为空（YAML null）⇒ **必填参数**，调用时必须由 args 提供。
      这条是关键：模板里写占位文本（「（用户需求原话）」）当默认值，忘传就会拿占位文本
      当真需求跑完整轮验证——最贵的一种静默失败。空值让它在编译期就炸。
    * 有值 ⇒ 默认值，args 可以覆盖。
    """
    problems: list[str] = []
    parsed = _parse_uses(ref, problems, "template")
    if problems:
        raise WorkflowYamlError(problems)
    assert parsed is not None
    skill, rel = parsed
    if not rel.endswith((".yaml", ".yml")):
        raise WorkflowYamlError([f"template 必须指向一份 .yaml 模板，得到 {ref!r}"])
    raw = load_file(skill, rel)
    if raw is None:
        raise WorkflowYamlError(
            [f"模板 '{rel}' 在技能 '{skill}' 中不存在（或技能未发布/不对 agent 开放）。"]
        )
    return compile_workflow_document(raw, args, load_file=load_file,
                                     read_workspace=read_workspace, where=ref, output_mode=output_mode)


def compile_workflow_document(
    raw: str,
    args: dict[str, Any] | None,
    *,
    load_file: Callable[[str, str], str | None],
    read_workspace: Callable[[str], str | None] | None = None,
    where: str = "workflow",
    output_mode: str = "file",
) -> WorkflowSpec:
    """一份 workflow YAML 文档（技能库模板 或 工作区文件）+ args → WorkflowSpec。

    两种来源走**同一条**参数合并路：`inputs` 里值为空 = 必填参数（args 必须给），有值 = 默认值。
    为什么必填要空着：模板里写占位文本（「（用户需求原话）」）当默认值，忘传就会拿占位文本
    当真需求跑完整轮——最贵的一种静默失败。空值让它在编译期就炸。
    """
    problems: list[str] = []
    try:
        data = yaml.load(raw, Loader=_StrictLoader) or {}
    except yaml.YAMLError as exc:
        raise WorkflowYamlError([f"'{where}' 解析失败：{exc}"]) from exc
    if not isinstance(data, dict) or not isinstance(data.get("inputs", {}), dict):
        raise WorkflowYamlError([f"'{where}' 顶层必须是对象，且 inputs 是对象。"])

    declared = dict(data.get("inputs") or {})
    supplied = dict(args or {})
    for key in set(supplied) - set(declared):
        problems.append(
            f"'{where}' 没有声明参数 '{key}'。它的 inputs 是："
            + ("、".join(sorted(declared)) or "(无)")
        )
    merged: dict[str, Any] = {}
    for key, default in declared.items():
        if key in supplied and supplied[key] is not None:
            merged[key] = supplied[key]
        elif default is None:
            problems.append(
                f"'{where}' 缺必填参数 '{key}'——inputs 里它没有默认值，必须由 args 提供。"
            )
        else:
            merged[key] = default
    if problems:
        raise WorkflowYamlError(problems)

    data["inputs"] = merged
    return compile_workflow_yaml(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                 load_file=load_file, read_workspace=read_workspace, output_mode=output_mode)


def compile_workflow_yaml(
    text: str,
    *,
    load_file: Callable[[str, str], str | None],
    read_workspace: Callable[[str], str | None] | None = None,
    output_mode: str = "file",
) -> WorkflowSpec:
    """YAML 源码 → WorkflowSpec。失败抛 :class:`WorkflowYamlError`（问题一次报全）。

    ``load_file(skill_name, rel_path)`` 从技能库读文件（权威在 PG，只认已发布 + agent_visible
    的版本；读不到返回 None）——与 charter 注入同一个信任面，绝不读磁盘物化副本。
    """
    if output_mode not in ("file", "text"):
        raise WorkflowYamlError(["output_mode must be 'file' or 'text'."])
    problems: list[str] = []
    try:
        data = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise WorkflowYamlError([f"YAML 解析失败：{exc}"]) from exc
    if not isinstance(data, dict):
        raise WorkflowYamlError(["顶层必须是对象：{name, inputs?, jobs}。"])

    for key in set(data) - _TOP_KEYS:
        hint = _GHA_FALSE_FRIENDS.get(str(key))
        problems.append(
            f"顶层不支持 '{key}'。{hint}" if hint
            else f"顶层不支持 '{key}'。只认 name / inputs / jobs / timeout-minutes。"
        )

    name = str(data.get("name") or "").strip()
    if not name:
        problems.append("缺 name:（这个 workflow 叫什么）。")
        name = "workflow"

    # 整个 run 的时限（design/170 第二批）：到点停掉没跑完的节点，run 判失败。不写用平台默认（6 小时）。
    timeout_s: float | None = None
    tm = data.get("timeout-minutes")
    if tm is not None:
        if isinstance(tm, bool) or not isinstance(tm, (int, float)) or not 1 <= float(tm) <= 10080:
            problems.append("timeout-minutes: 整个 run 的时限（分钟），1–10080。")
        else:
            timeout_s = float(tm) * 60

    wf_inputs = data.get("inputs") or {}
    if not isinstance(wf_inputs, dict):
        problems.append("inputs: 必须是对象（键=名字，值=字面量）。")
        wf_inputs = {}

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, dict) or not raw_jobs:
        problems.append("缺 jobs:（至少要有一个 job）。")
        raise WorkflowYamlError(problems)

    jobs: dict[str, _Job] = {}
    for jid, raw in raw_jobs.items():
        job_id = str(jid)
        where = f"job '{job_id}'"
        if not _JOB_ID_RE.match(job_id):
            problems.append(f"{where}: job id 只能是字母/数字/下划线/连字符，且以字母或下划线开头。")
            continue
        if not isinstance(raw, dict):
            problems.append(f"{where}: 必须是对象。")
            continue
        for key in set(raw) - _JOB_KEYS:
            hint = _GHA_FALSE_FRIENDS.get(str(key))
            problems.append(
                f"{where}: 不支持 '{key}'。{hint}" if hint
                else f"{where}: 不支持 '{key}'。job 只认 name / uses / needs / with / continue-on-error。"
            )
        uses = str(raw.get("uses") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        if uses and prompt:
            problems.append(
                f"{where}: uses 与 prompt 二选一——uses 是引用（技能库 charter / 工作区 .md），"
                f"prompt 是就地写的角色提示词。两个都写了，说不清这个叶子跑的是哪份。"
            )
        elif not uses and not prompt:
            problems.append(
                f"{where}: 每个 job 要有一个角色提示词。二选一：`prompt: |` 就地写，"
                f"或 `uses: skill:<技能>/<路径>` / `uses: file:<工作区路径>.md` 引用。"
            )
        if uses.startswith("skill:"):
            for k in _INLINE_ONLY_JOB_KEYS:
                if k in raw:
                    problems.append(
                        f"{where}: '{k}' 不能写在引用了 skill: charter 的 job 上——"
                        f"平台钉死的角色，它的 tools / role_mode / skills 写在 charter 的 "
                        f"frontmatter 里（单一真相源）。要自定义就改用 prompt: | 或 uses: file:。"
                    )
        role_mode = str(raw.get("role-mode") or "").strip() or None
        if role_mode is not None and role_mode not in ("assigned", "discovery"):
            problems.append(
                f"{where}: role-mode 只能是 assigned（只做任务书指派的那份）或 "
                f"discovery（默认：自己发现该做什么），得到 {role_mode!r}。"
            )
            role_mode = None

        def _str_list(key: str, raw=raw, where=where) -> list[str] | None:
            v = raw.get(key)
            if v is None:
                return None
            if isinstance(v, str):
                v = [t.strip() for t in v.split(",")]
            if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
                problems.append(f"{where}: {key} 必须是字符串数组（或逗号分隔的字符串）。")
                return None
            out = [t.strip() for t in v if t.strip()]
            return out if key == "tools" else out or None

        job_tools = _str_list("tools")
        if job_tools:
            job_tools = list(dict.fromkeys(t.split("(", 1)[0].strip() for t in job_tools if t.split("(", 1)[0].strip()))
        job_skills = _str_list("skills")
        needs_raw = raw.get("needs") or []
        if isinstance(needs_raw, str):
            needs_raw = [needs_raw]
        if not isinstance(needs_raw, list) or not all(isinstance(n, str) for n in needs_raw):
            problems.append(f"{where}: needs 必须是 job id 数组（或单个字符串）。")
            needs_raw = []
        with_raw = raw.get("with") or {}
        if not isinstance(with_raw, dict):
            problems.append(f"{where}: with 必须是对象。")
            with_raw = {}
        jobs[job_id] = _Job(
            job_id=job_id,
            name=str(raw.get("name") or job_id),
            uses=uses,
            prompt=prompt,
            needs=[str(n) for n in needs_raw],
            with_=with_raw,
            continue_on_error=bool(raw.get("continue-on-error") or False),
            retry=_parse_retry_block(raw.get("retry"), job_id, problems),
            continuation=_parse_continuation_block(raw.get("continuation"), job_id, problems),
            **_parse_strategy(raw.get("strategy"), job_id, problems, wf_inputs),
            fallback=(str(raw["fallback"]).strip() or None) if raw.get("fallback") else None,
            output_file=(str(raw["output-file"]).strip() or None) if raw.get("output-file") else None,
            job_tools=job_tools,
            job_skills=job_skills,
            job_role_mode=role_mode,
            job_outputs=_parse_job_outputs(raw.get("outputs"), where, problems),
            if_expr=(str(raw["if"]).strip() or None) if raw.get("if") is not None else None,
        )

    if not jobs:
        raise WorkflowYamlError(problems)

    loops = _parse_loops(data.get("loops"), jobs, wf_inputs, problems)

    # needs 指向存在的 job
    for job in jobs.values():
        for dep in job.needs:
            if dep not in jobs:
                problems.append(
                    f"job '{job.job_id}': needs 里的 '{dep}' 不存在。可用：{'、'.join(sorted(jobs))}"
                )

    # 载入角色定义（引用 or 就地）
    for job in jobs.values():
        if job.is_loop:
            continue
        if job.uses:
            job.action = _load_action(job.uses, load_file, problems, f"job '{job.job_id}'",
                                      read_workspace=read_workspace)
        elif job.prompt:
            job.action = ActionDef(ref="prompt", inline_body=job.prompt, inline_source="prompt")
        if job.action is not None and job.action.inline_body is not None:
            # 就地角色：job 级开关覆盖文件 frontmatter（写在眼前的更具体）
            if job.job_tools is not None:
                job.action.tools = job.job_tools
            if job.job_skills is not None:
                job.action.skills = job.job_skills
            if job.job_outputs:
                job.action.outputs = job.job_outputs
            if job.job_role_mode is not None:
                job.action.role_mode = job.job_role_mode

    # with ↔ action.inputs 对账（编译期，不用花一个 token 就能发现）
    for job in jobs.values():
        act = job.action
        if act is None or not act.inputs:
            continue
        for key in set(job.with_) - set(act.inputs):
            problems.append(
                f"job '{job.job_id}': action 没有声明输入 '{key}'。它声明的是："
                f"{'、'.join(sorted(act.inputs))}"
            )
        for key, meta in act.inputs.items():
            if meta.get("required") and key not in job.with_:
                desc = str(meta.get("description") or "")
                problems.append(
                    f"job '{job.job_id}': 缺必填输入 '{key}'"
                    + (f"（{desc}）" if desc else "")
                )

    # fallback 指向的 job 不是 DAG 的一员：它只在主 job 用尽 attempt 后跑，所以要从分层里摘掉。
    fallback_ids: set[str] = set()
    for job in jobs.values():
        if not job.fallback:
            continue
        target = jobs.get(job.fallback)
        if target is not None and (target.is_loop or target.loop_id):
            problems.append(f"job '{job.job_id}': fallback 不能指向循环或循环里的 job（'{job.fallback}'）。")
            continue
        if target is None:
            problems.append(
                f"job '{job.job_id}': fallback 指向不存在的 job '{job.fallback}'。"
                f"可用：{'、'.join(sorted(jobs))}"
            )
            continue
        if target.fallback:
            problems.append(
                f"job '{job.job_id}': 兜底 job '{job.fallback}' 自己又声明了 fallback——"
                f"兜底的兜底会变成没人能推理的失败链，不允许。"
            )
        if target.needs:
            problems.append(
                f"job '{job.job_id}': 兜底 job '{job.fallback}' 不能有 needs——"
                f"它不在主图里，等不到任何上游。"
            )
        for other in jobs.values():
            if job.fallback in other.needs:
                problems.append(
                    f"job '{other.job_id}': needs 里的 '{job.fallback}' 是 "
                    f"'{job.job_id}' 的兜底 job，不在主图里，不能被依赖。"
                )
                break
        fallback_ids.add(job.fallback)

    main_jobs = {jid: j for jid, j in jobs.items() if jid not in fallback_ids}

    # 循环在主图里当一个 job：体内 job 收起来；体内对体外的 needs 变成循环的 needs，体外对体内的 needs 指向循环。
    # 主图只用来分层（_node_for 拿原 job，inputs_from 照原 needs 写——循环之后体内 id 照样能引用）。
    def _graph_id(jid: str) -> str:
        return jobs[jid].loop_id or jid if jid in jobs else jid

    graph: dict[str, _Job] = {}
    for jid, j in main_jobs.items():
        if j.loop_id:
            continue
        if j.is_loop:
            ext = [d for b in loops[jid].body for d in jobs[b].needs if d in jobs and jobs[d].loop_id != jid]
            src = str(loops[jid].strategy.get("matrix_from") or "").split(".", 1)[0]
            if src:
                if src not in jobs:
                    problems.append(f"loop '{jid}': matrix 引用了不存在的 job '{src}'。")
                else:
                    ext.append(src)     # 条目列表来自它：它得先跑完
            graph[jid] = replace(j, needs=list(dict.fromkeys(_graph_id(d) for d in ext)))
        else:
            graph[jid] = replace(j, needs=list(dict.fromkeys(_graph_id(d) for d in j.needs if _graph_id(d) != jid)))
    for jid, j in main_jobs.items():
        lid = j.loop_id
        for dep in j.needs:
            dj = jobs.get(dep)
            if dj is None:
                continue
            if dj.loop_id and dj.loop_id != lid and dj.loop_matrix_var:
                problems.append(
                    f"job '{jid}': needs 了 '{dep}'，它在带 matrix 的循环 '{dj.loop_id}' 里（每个条目各一份）。"
                    f"needs 循环本身（'{dj.loop_id}'）。")
            if lid and dep == lid:
                problems.append(f"job '{jid}': 在循环 '{lid}' 里，不能 needs 循环自己。")
    _layers(graph, problems)      # 环检测（分量内分层在下面按分量再算一次）
    for _lid, lp in loops.items():
        _layers({b: replace(jobs[b], needs=[d for d in jobs[b].needs if d in lp.body]) for b in lp.body}, problems)

    # 就地角色的正文里也认 ${{ inputs.* }}（同一份 charter 想按参数改口径时不必复制一遍）。
    # 🔴 只认 inputs：needs.*.output 与 matrix.* 是**运行时**替换，而引擎只对 node.input 做替换、
    # 不碰 system_prompt——放进 prompt 里会原样留着一串 `${{ … }}` 送给模型（静默错）。
    for job in jobs.values():
        act = job.action
        if act is None or act.inline_body is None:
            continue
        where = f"job '{job.job_id}'"
        for m in _EXPR_RE.finditer(act.inline_body):
            expr = m.group(1).strip()
            if not expr.startswith("inputs."):
                problems.append(
                    f"{where}: 角色正文（prompt / file charter）里只能用 ${{{{ inputs.<键> }}}}，"
                    f"不能用 ${{{{ {expr} }}}}——needs / matrix 是运行时数据，写进 with: 里传给它。"
                )
        act.inline_body = _EXPR_RE.sub(
            lambda m, job=job: _compile_expr(m.group(1), job, jobs, wf_inputs, problems)
            if m.group(1).strip().startswith("inputs.") else m.group(0),
            act.inline_body,
        )

    # output-file 里认 ${{ inputs.* }}：同一张图同时开几份 run（「一个包一个 workflow」，两套就开两个）时，
    # 靠它给产出文件带上编号——几份 run 共用一个工作区，都写 outputs/<job>.md 会互相整段覆盖，而且不报错。
    # 下游 needs.x.output、「你的产出」提示、节点元数据都用替换后的这一个路径。
    for job in jobs.values():
        if not job.output_file:
            continue
        where = f"job '{job.job_id}' output-file"
        for m in _EXPR_RE.finditer(job.output_file):
            if not m.group(1).strip().startswith("inputs."):
                problems.append(f"{where}: 只能用 ${{{{ inputs.<键> }}}}，得到 ${{{{ {m.group(1).strip()} }}}}")
        job.output_file = _EXPR_RE.sub(
            lambda m, job=job: _compile_expr(m.group(1), job, jobs, wf_inputs, problems)
            if m.group(1).strip().startswith("inputs.") else "",
            job.output_file,
        ).strip() or None

    # 渲染每个 job 的 input（表达式在这一步编译；错误继续累积）
    rendered: dict[str, str] = {}
    for job in jobs.values():
        if job.is_loop:
            continue
        parts: list[str] = []
        for key, value in job.with_.items():
            parts.append(f"### {key}\n{_render(value, job, jobs, wf_inputs, problems)}")
        if output_mode == "text":
            if job.output_file or any(FILEREF_RE.search(part) for part in parts):
                problems.append(f"job '{job.job_id}': text output mode cannot use output-file or file output references; use structured outputs or file mode.")
            rendered[job.job_id] = "\n\n".join(parts) or "Complete your role's task and return the requested final answer."
            continue
        note = (
            _PRODUCT_NOTE_MATRIX.format(shared=output_path(job.job_id), job=job.job_id)
            if job.matrix_var or job.loop_matrix_var
            else _PRODUCT_NOTE.format(path=job.output_file or output_path(job.job_id))
        )
        outs = (job.action.outputs if job.action else None) or {}
        if outs:
            # 声明了结构化输出：最终回复是 JSON（原来一律写「最终回复只要一句指向文件」，和 output_schema 冲突，解析失败下游就拿不到值）
            keys = "；".join(f"`{k}`（{_output_prop(v).get('type')}）：{v.get('description') or ''}" for k, v in outs.items())
            note = note.replace("- 最终回复只要一句指向该文件即可，正文写在文件里。",
                                "- **最终回复是一个 JSON 对象**（不要别的文字），键：" + keys + "。详细内容写在文件里。")
        parts.append(note)
        rendered[job.job_id] = "\n\n".join(parts)

    if problems:
        raise WorkflowYamlError(problems)

    # ── 组装 WorkflowSpec：分层 → sequence[ parallel[…] ]，单节点层不套 parallel ──
    def _node_for(job: _Job) -> dict[str, Any]:
        """job → 节点。matrix job 包一层 foreach：**同一个 job 对一组值各跑一次**。"""
        node = _agent_node(job)
        if not job.matrix_var:
            return node
        # foreach 的 id 用 job id（下游 needs 引用的是这个聚合），body 另起 id——
        # 节点 id 全局唯一，且引擎禁止 inputs_from 指向 body（body 每次迭代都是同一个 id）。
        node["id"] = f"{job.job_id}-item"
        fe: dict[str, Any] = {
            "type": "foreach",
            "id": job.job_id,
            "as": job.matrix_var,
            "body": node,
            "parallel": True,
            # 一次迭代失败不该毁掉整批——与 job 级 continue-on-error 同一个开关。
            "on_error": "continue" if job.continue_on_error else "halt",
        }
        if job.matrix_items is not None:
            fe["items"] = job.matrix_items
        else:
            fe["items_from"] = job.matrix_from
        fe["max_concurrency"] = job.max_parallel or _FANOUT
        return fe

    def _agent_node(job: _Job) -> dict[str, Any]:
        act = job.action
        assert act is not None
        meta: dict[str, Any] = {
            **({"output_file": job.output_file or output_path(job.job_id)} if output_mode == "file" else {}),
            "continue_on_error": job.continue_on_error,
            **({"_action_tools": act.tools} if act.tools is not None else {}),
            **({"_leaf_skills": act.skills} if act.skills else {}),
            **({"_leaf_role_mode": act.role_mode} if act.role_mode else {}),
        }
        if act.inline_body is not None:
            # 就地角色（prompt: / uses: file:）：正文此刻就定了，不走 charter 注入。
            # 盖 sha 但明确标 unpinned——journal 里「有没有 charter_sha256」仍是「平台钉死」的
            # 唯一判据，凭证账本不会把自写角色误认成平台判据（design/90 §5.3）。
            import hashlib as _hl

            meta["leaf_prompt_source"] = act.inline_source
            meta["leaf_prompt_sha256"] = _hl.sha256(act.inline_body.encode("utf-8")).hexdigest()
            meta["charter_pinned"] = False
        node: dict[str, Any] = {
            "type": "agent",
            "id": job.job_id,
            "spec": {
                "name": job.name or act.name or job.job_id,
                **({"tools": act.tools} if act.tools is not None else {}),
                # charter 哨兵：正文由平台在 spec_transform 注入并盖哈希戳（app/tools/workflow.py）；
                # 就地角色则是正文本身。
                "system_prompt": act.inline_body if act.inline_body is not None else act.charter_ref,
                "metadata": meta,
            },
            "input": rendered[job.job_id],
        }
        if job.needs:
            node["inputs_from"] = list(job.needs)
        if job.retry:
            node["retry"] = job.retry
        if job.continuation:
            node["continuation"] = job.continuation
        if job.continue_on_error:
            node["continue_on_error"] = True
        if job.output_file:
            node["output_file"] = job.output_file
        if job.fallback and job.fallback in jobs:
            node["fallback"] = _agent_node(jobs[job.fallback])   # 兜底不支持 matrix
        if job.if_expr:
            when = _compile_if(job.if_expr, job, jobs, problems)
            if when:
                node["when"] = when
        if act.outputs:
            node["output_schema"] = {
                "name": job.job_id,
                "schema": {
                    "type": "object",
                    "properties": {k: _output_prop(v) for k, v in act.outputs.items()},
                    "required": list(act.outputs),
                },
            }
        return node

    def _loop_node(lp: _Loop) -> dict[str, Any]:
        """循环 → LoopNode（带 matrix 时外面包一层 foreach：每个条目各跑一路循环）。"""
        sub = {b: replace(jobs[b], needs=[d for d in jobs[b].needs if d in lp.body]) for b in lp.body}
        body = _chain(sub, f"{lp.loop_id}-body")
        holder = replace(jobs[lp.loop_id], needs=list(lp.body))
        until = _compile_if(lp.until, holder, jobs, problems, where=f"loop '{lp.loop_id}' until")
        ref_job = str((until or {}).get("ref") or "").split(".", 1)[0]
        cont = lp.continue_nodes if lp.continue_nodes is not None else [b for b in lp.body if b != ref_job]
        node: dict[str, Any] = {
            "type": "loop", "id": lp.loop_id, "body": body,
            "until": until or {"ref": f"{lp.body[0]}", "op": "truthy"},
            "max_iterations": lp.max_iterations, "continue_nodes": cont, "on_exhausted": lp.on_exhausted,
        }
        if not lp.strategy.get("matrix_var"):
            return node
        node["id"] = f"{lp.loop_id}-item"
        fe: dict[str, Any] = {"type": "foreach", "id": lp.loop_id, "as": lp.strategy["matrix_var"], "body": node,
                              "parallel": True, "on_error": "continue",
                              "max_concurrency": lp.strategy.get("max_parallel") or _FANOUT}
        if lp.strategy.get("matrix_items") is not None:
            fe["items"] = lp.strategy["matrix_items"]
        else:
            fe["items_from"] = lp.strategy["matrix_from"]
        return fe

    def _chain(group: dict[str, _Job], tag: str) -> dict[str, Any]:
        """一个分量 → 分层的 sequence（层内并发）。"""
        steps: list[dict[str, Any]] = []
        for layer in _layers(group, problems):
            nodes = [_loop_node(loops[jid]) if jid in loops else _node_for(jobs[jid]) for jid in layer]
            steps.append(
                nodes[0] if len(nodes) == 1
                else {"type": "parallel", "id": f"{tag}-l{len(steps)}", "branches": nodes, "max_concurrency": _FANOUT}
            )
        if len(steps) == 1:
            return steps[0]
        return {"type": "sequence", "id": tag, "steps": steps}

    groups = _components(graph)
    if len(groups) == 1:
        root = _chain(groups[0], "root")
    else:
        # 互不相干的几条链之间没有屏障：一条慢链不拖住另一条。
        root = {
            "type": "parallel",
            "id": "root",
            "branches": [_chain(g, f"chain{i}") for i, g in enumerate(groups)],
            "max_concurrency": _FANOUT,
        }

    scope = data.get("concurrency-scope", "branch")
    if scope not in ("branch", "leaf"):
        problems.append("concurrency-scope 只能是 branch 或 leaf")
    if scope == "leaf":
        def leaf_scope(node):
            if node.get("type") == "parallel":
                node["concurrency_scope"] = "leaf"
            for key in ("branches", "steps"):
                for child in node.get(key, []):
                    leaf_scope(child)
            if isinstance(node.get("body"), dict):
                leaf_scope(node["body"])
        leaf_scope(root)
        # Even a single component has a global leaf ceiling after platform clamping.
        if root.get("type") != "parallel":
            root = {"type":"parallel", "branches":[root], "max_concurrency":_FANOUT, "concurrency_scope":"leaf"}

    if problems:   # 组装节点时（if: 条件）还可能报问题
        raise WorkflowYamlError(problems)

    try:
        # 解析后的顶层 inputs 捎进 spec.metadata：凭证账本的 app 绑定从这里取
        # （模板/内联 YAML 把 app_id 写成 input 默认值、不传 args 也是合法用法——
        # 只认 args 会漏掉这一路，2026-08-19 B 档真跑实锤）。只带标量，防大对象进 journal。
        scalars = {
            k: v for k, v in wf_inputs.items()
            if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 500
        }
        meta = {"workflow_inputs": scalars} if scalars else {}
        return WorkflowSpec.from_json({"name": name, "root": root, "metadata": meta,
                                       **({"timeout_s": timeout_s} if timeout_s else {})})
    except WorkflowSpecError as exc:
        # IR 的严格 schema 也会报问题（retry 取值、节点结构…）。它是另一种异常类型——
        # 不转换的话调用方拿到的是原始异常而不是我们的格式化清单，模型看到的就是一段 traceback。
        raise WorkflowYamlError([str(p) for p in exc.problems]) from exc


# ── 运行时：把 @@FILEREF:…@@ 换成「路径 + 预览/切片 + 截断留痕」 ──────────────────


def resolve_file_refs(text: str, read_lines: Callable[[str], list[str] | None]) -> str:
    """叶子启动前调用（executor 的 run_agent）：上游产物此时才存在。

    注入的是**路径 + 内容片段**，不是整段全文——下游要全文自己 read_file。
    截断必须留痕，不许静默丢尾巴（conv184 教训）。
    """

    def _sub(m: re.Match[str]) -> str:
        path = m.group(1)
        raw_slice = m.group(2) or ""
        lines = read_lines(path)
        if lines is None:
            return f"（上游产物 `{path}` 还不存在或为空——上游可能没写产出文件。）"
        total = len(lines)
        sliced, how = _apply_slice(lines, raw_slice)
        body = "\n".join(sliced)
        head = f"（上游产物 `{path}`，共 {total} 行，{how}"
        if len(sliced) < total:
            head += f"；**其余 {total - len(sliced)} 行未包含**，需要全文请 read_file('{path}')）"
        else:
            head += "，已全文包含）"
        return f"{head}\n```\n{body}\n```"

    return FILEREF_RE.sub(_sub, text)


def _apply_slice(lines: list[str], raw: str) -> tuple[list[str], str]:
    if not raw:
        if len(lines) <= DEFAULT_PREVIEW_LINES:
            return lines, "全部"
        return lines[:DEFAULT_PREVIEW_LINES], f"此处给出前 {DEFAULT_PREVIEW_LINES} 行"
    m = _SLICE_RE.match(raw)
    if not m:                      # 编译期已拦，这里兜底
        return lines, "全部"
    start_s, stop_s = m.group(1), m.group(2)
    if stop_s is None:             # [n] —— 单行索引
        idx = int(start_s)
        try:
            return [lines[idx]], f"此处给出第 {idx} 行"
        except IndexError:
            return [], f"第 {idx} 行不存在"
    start = int(start_s) if start_s else None
    stop = int(stop_s) if stop_s else None
    out = lines[start:stop]
    if start is not None and start < 0:
        how = f"此处给出最后 {abs(start)} 行"
    elif stop is not None and stop >= 0 and not start:
        how = f"此处给出前 {stop} 行"
    else:
        how = f"此处给出第 {start or 0}–{stop if stop is not None else '末'} 行"
    return out, how
