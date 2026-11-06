"""Graph interpretation over independently owned durable node jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import ChainMap
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Protocol

from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.structured import StructuredOutputError, parse_structured

from .result import AgentResult, SharedBudget, WorkflowResult
from .spec import (
    MAX_BACKOFF_S,
    MAX_FOREACH_ITEMS,
    AgentNode,
    BranchNode,
    ForeachNode,
    LoopNode,
    ParallelNode,
    SequenceNode,
    WorkflowNode,
    WorkflowSpec,
)

#: 循环信息在 env 里的键（模板变量正则匹配不到它，不会被当成 {{变量}}）
_LOOP_KEY = "\x00loop"

#: 当前 foreach 迭代自己的结果表（6.46.1）。body 的节点在每次迭代里 id 都一样，并行迭代共用全局结果表时，
#: 体内的 inputs_from / when / 循环的 until 会读到别的迭代的结果。每个迭代在自己的 task 里开一层，嵌套的迭代再往里叠一层。
_ITER_SCOPE: ContextVar[ChainMap[str, AgentResult] | None] = ContextVar("pl_wf_iter_scope", default=None)
_DEFER_STEP: ContextVar[str | None] = ContextVar("pl_wf_defer_task_step", default=None)


class _ScopedResults(dict):
    """节点结果表：写的时候全局和当前迭代都写；读的时候先看当前迭代（再往外层迭代），最后看全局。
    ``dict(self)`` / ``.items()`` 拿到的是全局那份（run 结果、journal 用）。"""

    def __setitem__(self, key: str, value: AgentResult) -> None:
        super().__setitem__(key, value)
        scope = _ITER_SCOPE.get()
        if scope is not None:
            scope[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        scope = _ITER_SCOPE.get()
        if scope is not None and key in scope:
            return scope[key]
        return super().get(key, default)

    def __getitem__(self, key: str) -> AgentResult:
        scope = _ITER_SCOPE.get()
        if scope is not None and key in scope:
            return scope[key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        scope = _ITER_SCOPE.get()
        return (scope is not None and key in scope) or super().__contains__(key)


__all__ = ["Executor", "NodeRequest", "WorkflowEngine", "WorkflowRunError", "in_workflow"]

logger = logging.getLogger(__name__)

#: Hard ceiling on the total number of real leaf executions in one run (H3 — BUG_REVIEW_3.4). A
#: backstop independent of the optional budget: caps nested fanout (foreach-in-foreach) and any
#: programmatically-built spec that bypasses the static validation caps. Fail-closed when exceeded.
MAX_TOTAL_LEAVES = 10_000
ITERATION_ENV_KEY = "\x00iteration"


#: ``{{var}}``（foreach 变量等）和 ``{{node.key}}``（上游节点结构化输出里的一个键，6.45：之前不认带点的名字，
#: YAML 的 ``${{ needs.x.outputs.k }}`` 编成 ``{{x.k}}`` 后原样送给了模型）。节点 id 可以带短横线。
def _agent_ids(node: WorkflowNode) -> list[str]:
    out: list[str] = []
    if isinstance(node, AgentNode):
        return [node.id]
    for child in getattr(node, "steps", None) or getattr(node, "branches", None) or ():
        out += _agent_ids(child)
    if isinstance(node, BranchNode):
        for c in node.cases.values():
            out += _agent_ids(c)
        if node.default is not None:
            out += _agent_ids(node.default)
    if isinstance(node, (ForeachNode, LoopNode)):
        out += _agent_ids(node.body)
    return out


def _loop_prompt(lc: dict[str, Any], user_input: str) -> str:
    """循环第 2 轮起给 continue_nodes 的话：上一轮为什么没过、意见是什么、在已有成果上改。"""
    return (
        f"第 {lc['iteration']}/{lc['max']} 轮：上一轮没通过（{lc.get('why') or ''}）。\n"
        f"上一轮的意见：\n{lc.get('feedback') or ''}\n\n"
        "按你的节点职责处理反馈：开发节点局部修复，审核节点独立复验、不能代改。上一轮会话和文件都还在，不从头重做；按原来的要求交卷。\n\n"
        "先用 todo 列出本轮审核意见对应的修复或复验任务；上一轮已完成待办只代表历史成果。"
        "完成待办后还需提交本轮最终结果；若审核缺少具体问题，先读取审核报告核实。\n\n"
        f"（这一轮的任务输入照旧附在下面）\n{user_input}"
    )


#: 重启后在原会话接着跑时给叶子的话（6.46）
_RESUME_PROMPT = (
    "进程重启打断了你，你的会话、文件和半成品都还在。先看一眼自己做到哪了（todo、产出文件），"
    "接着把原来的任务做完，然后照原来的要求给出最终回复。"
)

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][\w-]*(?:\.[a-zA-Z_]\w*)?)\s*\}\}")

# Set while a workflow run is in flight. Propagates into child sub-agent tasks
# (asyncio copies the context), so a leaf that calls the create_workflow tool can
# detect it is already inside a workflow and refuse — a simple recursion guard
# for this in-process tier.
_IN_WORKFLOW: ContextVar[bool] = ContextVar("power_loop_in_workflow", default=False)


def in_workflow() -> bool:
    """True if the current execution is inside a running workflow."""
    return _IN_WORKFLOW.get()


class WorkflowRunError(RuntimeError):
    """Raised when a node references missing data, or a ``halt`` branch fails."""


def _continuation_prompt(remaining: list[str], i: int, n: int) -> str:
    todo_block = ("\n剩余待办：\n" + "\n".join(f"- {t}" for t in remaining)) if remaining else ""
    return (
        f"[continuation {i}/{n}] 你在轮数用尽前写下了「做了什么、还剩什么」的总结（见上）。"
        f"现在给你追加了预算：**接着做剩余的部分，不要重做已完成的**。{todo_block}\n"
        "做完后按原任务要求给出最终结果；仍完不成就明确说剩什么。"
    )


@dataclass(frozen=True)
class NodeRequest:
    """Serializable leaf command; identity survives driver replacement."""

    run_id: str
    node_path: str
    iteration_path: list
    attempt: int
    spec: dict
    body: str
    continuation: int = 0
    session_id: str | None = None


class Executor(Protocol):
    async def run_agent(self, request: NodeRequest, *, allow_start: bool = True) -> dict[str, Any]: ...
    async def todo_remaining(self, session_id: str) -> list[str]: ...


FILEREF_RE = re.compile(r"@@FILEREF:([^\[\]@]+)(\[[^\]]*\])?@@")


class WorkflowFileIO(Protocol):
    def output_path(self, node_id: str, *, iteration: int | None = None) -> str: ...
    def render_ref(self, path: str, slice_expr: str) -> str: ...


class WorkflowEngine:
    """Interpret the frozen graph; the executor owns durable node admission."""

    def __init__(
        self, executor: Executor, *, run_id: str, driver_sid: str, allowed_tools=None, stop_event=None, file_io=None
    ):
        self._executor, self._run_id, self._driver_sid = executor, run_id, driver_sid
        self._allowed_tools, self._file_io = allowed_tools, file_io
        self._cancel = stop_event or CancellationToken()
        self._results = _ScopedResults()
        self._errors, self._iter_broken, self._tolerated = [], [], set()
        self._last = None
        self._budget = None
        self._budget_hit = self._cancelled = self._leaf_cap_hit = False
        self._leaves_run = 0
        self._usage_acc = {}

    async def run(self, spec: WorkflowSpec) -> WorkflowResult:
        self._spec = spec
        if spec.budget:
            self._budget = SharedBudget(spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct)
            self._budget.commit(self._usage_acc)
        token = _IN_WORKFLOW.set(True)
        status = "completed"
        try:
            await self._exec(spec.root, {"input": spec.input}, self._driver_sid)
        except WorkflowRunError as exc:
            self._errors.append(str(exc))
            status = "failed"
        finally:
            _IN_WORKFLOW.reset(token)
        if self._cancel.is_cancelled() or self._cancelled:
            status = "cancelled"
        elif self._budget_hit:
            status = "budget_exceeded"
        elif self._iter_broken or any(
            r.status not in ("completed", "skipped") and nid not in self._tolerated for nid, r in self._results.items()
        ):
            status = "failed"
        return self._build_result(spec, status)

    def _build_result(self, spec: WorkflowSpec, status: str) -> WorkflowResult:
        return WorkflowResult(
            name=spec.name,
            status=status,
            results=dict(self._results),
            final=self._last,
            usage=self._total_usage(),
            errors=list(self._errors),
            tolerated=sorted(self._tolerated),
        )

    # ── node dispatch ──────────────────────────────────────────────────────

    @staticmethod
    def _children(node: WorkflowNode) -> list[WorkflowNode]:
        if isinstance(node, SequenceNode):
            return list(node.steps)
        if isinstance(node, ParallelNode):
            return list(node.branches)
        if isinstance(node, (LoopNode, ForeachNode)):
            return [node.body]
        if isinstance(node, BranchNode):
            return list(node.cases.values()) + ([node.default] if node.default else [])
        return []

    async def _exec(self, node: WorkflowNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        return await self._exec_dispatch(node, env, driver_sid)

    async def _exec_dispatch(self, node: WorkflowNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        if isinstance(node, AgentNode):
            return await self._exec_agent(node, env, driver_sid)
        if isinstance(node, SequenceNode):
            return await self._exec_sequence(node, env, driver_sid)
        if isinstance(node, ParallelNode):
            return await self._exec_parallel(node, env, driver_sid)
        if isinstance(node, ForeachNode):
            return await self._exec_foreach(node, env, driver_sid)
        if isinstance(node, BranchNode):
            return await self._exec_branch(node, env, driver_sid)
        if isinstance(node, LoopNode):
            return await self._exec_loop(node, env, driver_sid)
        raise WorkflowRunError(f"unknown node type: {type(node).__name__}")

    async def _exec_loop(self, node: LoopNode, env: dict[str, Any], driver_sid: str) -> AgentResult:
        """反复跑循环体直到 ``until`` 成立（6.46，design/170 第三批）。循环信息随 env 往下传（几个循环可以并行跑）。"""
        body_ids = _agent_ids(node.body)
        feedback_node = str(node.until.get("ref") or "").split(".", 1)[0]
        prev: dict[str, str] = {}
        converged, why, iteration = False, "", 0
        for iteration in range(1, node.max_iterations + 1):
            if self._cancel.is_cancelled():
                break
            lc = {
                "loop_id": node.id,
                "iteration": iteration,
                "max": node.max_iterations,
                "continue": set(node.continue_nodes),
                "prev": dict(prev),
                "why": why,
                "feedback": self._loop_feedback(feedback_node) if iteration > 1 else "",
            }
            from .journal import ROUTE_KEY

            body_env = {
                **env,
                _LOOP_KEY: lc,
                "loop_iteration": iteration,
                ROUTE_KEY: [*env.get(ROUTE_KEY, []), {"loop": node.id, "iteration": iteration}],
            }
            body_result = await self._exec(node.body, body_env, driver_sid)
            if (
                body_result is not None
                and body_result.status not in ("completed", "skipped")
                and body_result.node_id not in self._tolerated
            ):
                why = f"上游 {body_result.node_id} 未完成：{body_result.status}；审核与下一轮已阻断"
                break
            for nid in node.continue_nodes:
                r = self._results.get(nid)
                if r is not None and r.session_id:
                    prev[nid] = r.session_id
            converged, why = self._eval_when(node.until)
            if converged:
                break
            logger.info("workflow loop %s: iteration %d/%d not done (%s)", node.id, iteration, node.max_iterations, why)
        last = {
            nid: {"status": r.status, "text": (r.text or "")[:4000], "payload": r.payload, "session_id": r.session_id}
            for nid in body_ids
            if (r := self._results.get(nid)) is not None
        }
        payload = {"converged": converged, "iterations": iteration, "why": why, "last": last}
        broken = any(
            r.status not in ("completed", "skipped") and nid not in self._tolerated
            for nid in body_ids
            if (r := self._results.get(nid)) is not None
        )
        ok = converged or (node.on_exhausted == "continue" and not broken and not self._cancel.is_cancelled())
        err = (
            None
            if converged
            else (f"跑了 {iteration} 轮仍不满足 until：{why}" if not self._cancel.is_cancelled() else "循环被取消")
        )
        res = AgentResult(
            node_id=node.id,
            status="completed" if ok else ("cancelled" if self._cancel.is_cancelled() else "failed"),
            text=f"循环 {node.id}：{'满足' if converged else '没满足'}（{iteration} 轮）：{why}",
            payload=payload,
            error=err,
        )
        self._results[node.id] = res
        await self._emit_step(res)
        return res

    def _loop_feedback(self, node_id: str) -> str:
        """上一轮 until 引用的那个节点交回的意见（结构化结果优先）。"""
        r = self._results.get(node_id)
        if r is None:
            return "（上一轮没有意见）"
        if r.payload is not None:
            return json.dumps(r.payload, ensure_ascii=False)[:4000]
        return (r.text or r.error or "")[:4000]

    async def _exec_agent(self, node, env, driver_sid):
        from dataclasses import asdict

        from .admission import admit
        from .journal import ROUTE_KEY

        if self._cancel.is_cancelled():
            self._cancelled = True
            return AgentResult(node.id, "cancelled", "")
        if node.when and not self._eval_when(node.when)[0]:
            result = AgentResult(node.id, "skipped", "")
            self._results[node.id] = result
            return result
        if self._leaves_run >= MAX_TOTAL_LEAVES:
            raise WorkflowRunError("workflow leaf ceiling exceeded")
        self._leaves_run += 1
        body = _render(node.input, env, self._resolve_ref_soft)
        extras = []
        for name in node.inputs_from or []:
            result = self._results.get(name)
            extras.append(
                result.text
                if result and result.status == "completed"
                else f"[Upstream {name} did not complete: {result.error or result.status if result else 'missing'}]"
            )
        if extras:
            body += "\n\n--- context ---\n" + "\n\n".join(extras)
        output_file = node.output_file
        if self._file_io:
            output_file = output_file or self._file_io.output_path(node.id, iteration=env.get(ITERATION_ENV_KEY))
            body = FILEREF_RE.sub(lambda match: self._file_io.render_ref(match.group(1), match.group(2) or ""), body)
        elif FILEREF_RE.search(body):
            raise WorkflowRunError("workflow file references require the configured resource profile")
        tools = node.spec.tools
        if self._allowed_tools is not None:
            tools = sorted(self._allowed_tools) if tools is None else [t for t in tools if t in self._allowed_tools]
        spec = replace(
            node.spec,
            system_prompt=_render(node.spec.system_prompt, env, self._resolve_ref_soft),
            tools=tools,
            lifecycle="linked",
            output_schema=node.output_schema,
            metadata={**node.spec.metadata, **({"output_file": output_file} if output_file else {})},
        )
        route = env.get(ROUTE_KEY, [])
        lc = env.get(_LOOP_KEY)
        previous = lc["prev"].get(node.id) if lc and node.id in lc["continue"] else None
        if previous:
            body = _loop_prompt(lc, body)
        attempts = node.retry.max_attempts if node.retry else 1
        raw = {}
        node_usage = {}

        def record_usage(raw):
            value = raw.get("usage") or {}
            current = raw.get("accounting_usage", value)
            self._add_usage(current)
            if self._budget:
                self._budget.commit(current)
            if raw["status"] == "budget_exceeded":
                self._budget_hit = True
            for name, count in value.items():
                node_usage[name] = node_usage.get(name, 0) + count

        async with admit(self._cancel):
            for attempt in range(1, attempts + 1):
                request = NodeRequest(self._run_id, node.id, route, attempt, asdict(spec), body, session_id=previous)
                raw = await self._executor.run_agent(request, allow_start=not self._budget or self._budget.can_spawn())
                record_usage(raw)
                # Unknown actions are never automatically retried.
                if (
                    raw.get("status") in {"unknown", "interrupted", "cancelled", "budget_exceeded"}
                    or not node.retry
                    or not self._should_retry(raw, set(node.retry.on))
                    or attempt == attempts
                ):
                    break
                if node.retry.backoff_s:
                    await asyncio.sleep(
                        min(MAX_BACKOFF_S, node.retry.backoff_s * node.retry.backoff_factor ** (attempt - 1))
                    )
            cont = node.continuation
            for index in range(1, cont.max_continuations + 1 if cont else 1):
                if raw.get("status") not in {"hit_round_limit", "context_checkpoint"}:
                    break
                if self._cancel.is_cancelled():
                    break
                # A gate admits new work; it cannot invalidate an input that
                # was already accepted (and may have completed its todo).
                remaining = []
                if not raw.get("has_next_continuation"):
                    remaining = await self._executor.todo_remaining(raw["session_id"])
                    if cont.gate == "todo_remaining" and not remaining:
                        break
                more = await self._executor.run_agent(
                    replace(
                        request,
                        continuation=index,
                        session_id=raw["session_id"],
                        spec=asdict(replace(spec, max_rounds=cont.extra_rounds)),
                        body=_continuation_prompt(remaining, index, cont.max_continuations),
                    ),
                    allow_start=not self._budget or self._budget.can_spawn(),
                )
                record_usage(more)
                usage = dict(raw.get("usage") or {})
                for key, value in (more.get("usage") or {}).items():
                    usage[key] = usage.get(key, 0) + value
                raw = {**more, "usage": usage}
        payload, error = raw.get("structured_output"), raw.get("error")
        if node.output_schema and raw["status"] == "completed" and payload is None:
            try:
                payload = parse_structured(raw.get("final_text", ""), schema=node.output_schema.get("schema"))
            except StructuredOutputError as exc:
                error, raw = str(exc), {**raw, "status": "failed"}
        result = AgentResult(
            node.id,
            raw["status"],
            raw.get("final_text", ""),
            payload=payload,
            usage=node_usage,
            session_id=raw.get("session_id"),
            error=error,
        )
        self._results[node.id] = self._last = result
        if (
            result.status not in {"completed", "cancelled", "interrupted", "unknown", "budget_exceeded"}
            and node.fallback
        ):
            fallback = await self._exec(node.fallback, env, driver_sid)
            if fallback and fallback.ok:
                result = replace(fallback, node_id=node.id)
                self._results[node.id] = self._last = result
        if node.continue_on_error and not result.ok:
            self._tolerated.add(node.id)
        return result

    async def _emit_step(self, result):
        # Leaf results are persisted by the executor. Aggregate results are
        # deterministic projections, rebuilt from those exact node references.
        pass

    def _add_usage(self, usage):
        for key, value in (usage or {}).items():
            self._usage_acc[key] = self._usage_acc.get(key, 0) + value

    def _eval_when(self, when: dict[str, Any]) -> tuple[bool, str]:
        """``when`` 成立吗？引用取不到（上游失败 / 跳过 / 没交结构化结果）按不成立。返回 (成立, 说明)。"""
        ref = str(when.get("ref") or "")
        op = str(when.get("op") or "truthy")
        try:
            value = self._resolve_ref(ref)
        except WorkflowRunError as exc:
            return False, f"取不到 {ref}（{exc}）"

        def norm(x: Any) -> Any:
            if isinstance(x, str) and x.strip().lower() in ("true", "false"):
                return x.strip().lower() == "true"
            return x

        v = norm(value)
        target = when.get("value")
        if op == "truthy":
            ok = bool(v)
        elif op == "falsy":
            ok = not bool(v)
        elif op == "eq":
            ok = v == norm(target) or str(value) == str(target)
        elif op == "ne":
            ok = not (v == norm(target) or str(value) == str(target))
        else:  # in
            ok = any(v == norm(x) or str(value) == str(x) for x in (target or []))
        shown = json.dumps(value, ensure_ascii=False)[:80] if not isinstance(value, str) else value[:80]
        return (
            ok,
            f"{ref} = {shown}（要求 {op}{'' if op in ('truthy', 'falsy') else ' ' + json.dumps(target, ensure_ascii=False)[:60]}）",
        )

    @staticmethod
    def _should_retry(raw: dict[str, Any], triggers: set[str]) -> bool:
        """这次 attempt 该不该重试。``failed``=状态不是 completed **也不是 hit_round_limit**
        （耗尽单列——它有自己的触发名与 ContinuationPolicy 原会话续跑，新会话重跑只会同预算
        再耗尽一次）;``empty``=完成但没正文;``hit_round_limit``=显式要旧行为时用。"""
        if raw.get("completion_unresolved"):
            return False  # Never re-run business work to repair result/ACK persistence.
        status = str(raw.get("status") or "")
        if status in ("hit_round_limit", "context_checkpoint"):
            return "hit_round_limit" in triggers
        if status == "cancelled":
            return False  # 取消不是「没跑好」
        if status != "completed":
            return "failed" in triggers
        return "empty" in triggers and not str(raw.get("final_text") or "").strip()

    async def _exec_sequence(self, node: SequenceNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        last: AgentResult | None = None
        for step in node.steps:
            last = await self._exec(step, env, driver_sid)
            if self._budget_hit or self._cancelled:
                break
            if last is not None and last.status not in ("completed", "skipped") and last.node_id not in self._tolerated:
                break
        return last

    async def _gather_branches(self, coros: list[Any], *, on_error: str) -> list[Any]:
        """Run branch coroutines concurrently with halt-aware cancellation (H1.2/C2).

        ``on_error="continue"``: collect every result/exception (return_exceptions).
        ``on_error="halt"``: on the FIRST failure, **cancel the still-running
        siblings** (so they stop burning real LLM calls and can't clobber the journal
        with a late step), drain them, then re-raise — unlike ``asyncio.gather``,
        which re-raises but leaves the siblings running detached.

        Halt is scoped to THIS node's own siblings (C6): we cancel only the tasks of
        *this* gather and re-raise. We deliberately do NOT flip the run-wide
        ``self._cancel`` token or set ``self._cancelled`` here — doing so would tear
        down unrelated in-flight branches of an *enclosing* ``on_error="continue"``
        node and force the whole run to ``"cancelled"``. The raised exception
        propagates naturally: a containing ``continue`` collects it as a branch error,
        a containing ``halt``/``sequence`` re-raises, and the top level becomes
        ``"failed"`` (or surfaces the raw exception) — never a spurious ``"cancelled"``.
        """

        async def branch(coro):
            result = await coro
            if (
                on_error == "halt"
                and result is not None
                and result.status not in {"completed", "skipped"}
                and result.node_id not in self._tolerated
            ):
                raise WorkflowRunError(f"branch {result.node_id}: {result.status}")
            return result

        tasks = [asyncio.create_task(branch(c)) for c in coros]
        if not tasks:
            return []
        if on_error == "continue":
            return await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        first_exc: BaseException | None = next(
            (t.exception() for t in tasks if t.done() and not t.cancelled() and t.exception() is not None),
            None,
        )
        if first_exc is not None:
            # Cancel only THIS gather's own in-flight siblings; scope it here so a
            # nested halt cannot poison unrelated subtrees or the run-wide token (C6).
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise first_exc
        return [t.result() for t in tasks]

    async def _exec_parallel(self, node: ParallelNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        sem = asyncio.Semaphore(node.max_concurrency)
        request_sem = asyncio.Semaphore(node.max_concurrency)

        async def one(branch: WorkflowNode) -> AgentResult | None:
            if node.concurrency_scope == "leaf":
                from power_loop.workflow.admission import GATES, REQUEST_GATES

                token = GATES.set((*GATES.get(), sem))
                requests = REQUEST_GATES.set((*REQUEST_GATES.get(), request_sem))
                try:
                    return await self._exec(branch, dict(env), driver_sid)
                finally:
                    REQUEST_GATES.reset(requests)
                    GATES.reset(token)
            async with sem:
                return await self._exec(branch, dict(env), driver_sid)

        gathered = await self._gather_branches([one(b) for b in node.branches], on_error=node.on_error)
        for r in gathered:
            if isinstance(r, BaseException):
                self._errors.append(f"parallel branch error: {r}")
            elif node.on_error == "continue" and r is not None and r.status not in {"completed", "skipped"}:
                self._tolerated.add(r.node_id)
                self._errors.append(f"parallel branch {r.node_id}: {r.status}")
        broken = next(
            (
                r
                for r in gathered
                if isinstance(r, AgentResult)
                and r.status not in ("completed", "skipped")
                and r.node_id not in self._tolerated
            ),
            None,
        )
        return broken or next((r for r in reversed(gathered) if isinstance(r, AgentResult)), None)

    async def _exec_foreach(self, node: ForeachNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        items = self._resolve_items(node)
        sem = asyncio.Semaphore(node.max_concurrency)

        async def one(item: Any, index: int) -> AgentResult | None:
            # ITERATION_ENV_KEY：body 的所有迭代共享同一个 node_id，宿主的 FileIO 端口靠它
            # 给每次迭代分不同的产出文件——不区分的话 N 个并发迭代会追加进同一个文件，互相搅乱。
            child_env = {**env, node.as_var: item, ITERATION_ENV_KEY: index}
            from .journal import ROUTE_KEY

            child_env[ROUTE_KEY] = [*env.get(ROUTE_KEY, []), {"foreach": node.id, "index": index}]
            outer = _ITER_SCOPE.get()
            scope: ChainMap[str, AgentResult] = ChainMap() if outer is None else outer.new_child()
            tok = _ITER_SCOPE.set(scope)
            try:
                if node.parallel:
                    async with sem:
                        return await self._exec(node.body, child_env, driver_sid)
                return await self._exec(node.body, child_env, driver_sid)
            finally:
                _ITER_SCOPE.reset(tok)
                # 这一路里没成功的节点（6.46.1）：全局结果表里体内 id 只留最后写的那一路，run 终态要看每一路
                for nid, r in scope.maps[0].items():
                    if r.status not in ("completed", "skipped") and nid not in self._tolerated:
                        failure = f"{node.id or 'foreach'}[{index}].{nid}"
                        if node.on_error == "continue":
                            self._tolerated.add(nid)
                            self._errors.append(failure)
                        else:
                            self._iter_broken.append(failure)

        if node.parallel:
            gathered = await self._gather_branches([one(it, i) for i, it in enumerate(items)], on_error=node.on_error)
        else:
            gathered = []
            for i, it in enumerate(items):
                try:
                    gathered.append(await one(it, i))
                except Exception as exc:  # noqa: BLE001
                    if node.on_error == "halt":
                        raise
                    gathered.append(exc)

        leaves = [r for r in gathered if isinstance(r, AgentResult)]
        for r in gathered:
            if isinstance(r, BaseException):
                self._errors.append(f"foreach item error: {r}")
        if node.id:
            agg = AgentResult(
                node_id=node.id,
                status="completed",
                text="\n\n".join(r.text for r in leaves),
                payload={"items": [r.payload if r.payload is not None else r.text for r in leaves]},
                # Sum leaf usage so total/resume accounting includes the fan-out
                # (leaves already counted live; the aggregate carries it for the
                # journal so a replayed foreach contributes its tokens).
                usage=_sum_usage(leaves),
            )
            self._results[node.id] = agg
            # Journal the aggregate so resume can skip the whole foreach.
            await self._emit_step(agg)
            return agg
        return leaves[-1] if leaves else None

    async def _exec_branch(self, node: BranchNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        value = self._resolve_ref(node.on)
        chosen = node.cases.get(str(value), node.default)
        if chosen is None:
            raise WorkflowRunError(f"branch on '{node.on}' got value {value!r} with no matching case and no default")
        return await self._exec(chosen, env, driver_sid)

    # ── data references ────────────────────────────────────────────────────

    def _resolve_items(self, node: ForeachNode) -> list[Any]:
        if node.items is not None:
            return list(node.items)
        assert node.items_from is not None
        value = self._resolve_ref(node.items_from)
        if not isinstance(value, list):
            raise WorkflowRunError(
                f"foreach items_from '{node.items_from}' did not resolve to a list (got {type(value).__name__})"
            )
        # H3: cap a DYNAMIC items_from list before _exec_foreach eagerly creates one task per item
        # (the static `items` literal is already capped at parse time). Fail-closed — a runaway
        # upstream node can't fan out into millions of leaves/LLM calls.
        if len(value) > MAX_FOREACH_ITEMS:
            raise WorkflowRunError(
                f"foreach items_from '{node.items_from}' resolved to {len(value)} items (max {MAX_FOREACH_ITEMS})"
            )
        return value

    def _resolve_ref(self, ref: str) -> Any:
        node_id, _, key = ref.partition(".")
        res = self._results.get(node_id)
        if res is None:
            raise WorkflowRunError(f"reference '{ref}': node '{node_id}' has not produced a result")
        if not key:
            return res.text
        if res.payload is None:
            raise WorkflowRunError(
                f"reference '{ref}': node '{node_id}' has no structured payload "
                f"(does it declare an output_schema and complete?)"
            )
        if key not in res.payload:
            raise WorkflowRunError(
                f"reference '{ref}': key '{key}' not in node '{node_id}' payload (keys: {sorted(res.payload)})"
            )
        return res.payload[key]

    def _resolve_ref_soft(self, ref: str) -> str:
        """渲染节点输入里的 ``{{node.key}}``：取不到时给一句明确说明，不让整个 run 崩（下游自己判断怎么办）。"""
        try:
            value = self._resolve_ref(ref)
        except WorkflowRunError as exc:
            return f"[取不到 {ref}：{exc}]"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _total_usage(self) -> dict[str, int]:
        # Accumulated once per real leaf / replayed result — not summed from
        # _results (where foreach bodies share one id and would mis-count).
        return dict(self._usage_acc)


def _sum_usage(results: list[AgentResult]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in results:
        for k in total:
            total[k] += int((r.usage or {}).get(k, 0) or 0)
    return total


def _render(template: str, env: dict[str, Any], resolve_ref: Callable[[str], str] | None = None) -> str:
    """Substitute ``{{var}}`` from ``env`` and ``{{node.key}}`` via ``resolve_ref``; leave unknown placeholders untouched."""

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in env:
            return str(env[name])
        if "." in name and resolve_ref is not None:
            return resolve_ref(name)
        return m.group(0)

    return _VAR_RE.sub(repl, template)
