"""Deterministic in-process engine that interprets a :class:`WorkflowSpec`.

The engine walks the spec's control-flow tree and runs each ``agent`` leaf as a
sub-agent via :func:`power_loop.runtime.spec.run_agent_spec`. It is deterministic
— the only LLM calls are the leaves; ``sequence`` / ``parallel`` / ``foreach`` /
``branch`` are ordinary code.

Execution model (this tier): **in-process, single writer.** All sub-agents are
children of one *driver session* on the parent loop's shared ``SessionStore``,
run with ``asyncio`` concurrency (one ``asyncio.Lock`` per session in the loop
makes that safe). This respects the store's one-process-per-file write
constraint.

Executor seam
-------------
Leaf execution goes through the :class:`Executor` protocol so a future
out-of-process backend (``python -m power_loop.runner --spec …``) can be slotted
in without touching the interpreter. The default :class:`InProcessExecutor` calls
``run_agent_spec`` directly. A subprocess executor must additionally solve the
single-writer constraint (a single owning writer funnel) — deliberately deferred.

Sharp edge handled here: ``run_agent_spec`` resolves the *parent* session from a
``contextvar`` (``get_session_id``). The engine sets that contextvar to the
driver session id **inside each leaf coroutine**, so concurrent fan-out tasks
(which each get their own copied context) never clobber one another.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import ChainMap
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Protocol

from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.spec import _ON_CHILD_SESSION, AgentSpec, run_agent_spec
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

__all__ = ["Executor", "InProcessExecutor", "WorkflowEngine", "WorkflowRunError", "in_workflow"]

logger = logging.getLogger(__name__)

#: Hard ceiling on the total number of real leaf executions in one run (H3 — BUG_REVIEW_3.4). A
#: backstop independent of the optional budget: caps nested fanout (foreach-in-foreach) and any
#: programmatically-built spec that bypasses the static validation caps. Fail-closed when exceeded.
MAX_TOTAL_LEAVES = 10_000

#: ``{{var}}``（foreach 变量等）和 ``{{node.key}}``（上游节点结构化输出里的一个键，6.45：之前不认带点的名字，
#: YAML 的 ``${{ needs.x.outputs.k }}`` 编成 ``{{x.k}}`` 后原样送给了模型）。节点 id 可以带短横线。
def _agent_ids(node: WorkflowNode) -> list[str]:
    out: list[str] = []
    if isinstance(node, AgentNode):
        return [node.id]
    for child in (getattr(node, "steps", None) or getattr(node, "branches", None) or ()):
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
    return (f"第 {lc['iteration']}/{lc['max']} 轮：上一轮没通过（{lc.get('why') or ''}）。\n"
            f"上一轮的意见：\n{lc.get('feedback') or ''}\n\n"
            "按你的节点职责处理反馈：开发节点局部修复，审核节点独立复验、不能代改。上一轮会话和文件都还在，不从头重做；按原来的要求交卷。\n\n"
            "先用 todo 列出本轮审核意见对应的修复或复验任务；上一轮已完成待办只代表历史成果。"
            "完成待办后还需提交本轮最终结果；若审核缺少具体问题，先读取审核报告核实。\n\n"
            f"（这一轮的任务输入照旧附在下面）\n{user_input}")


#: 重启后在原会话接着跑时给叶子的话（6.46）
_RESUME_PROMPT = ("进程重启打断了你，你的会话、文件和半成品都还在。先看一眼自己做到哪了（todo、产出文件），"
                  "接着把原来的任务做完，然后照原来的要求给出最终回复。")

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


class Executor(Protocol):
    """How a single ``agent`` leaf is executed.

    Implementations return the ``run_agent_spec`` result dict augmented with a
    ``usage`` key (``{"prompt_tokens", "completion_tokens", "total_tokens"}``).
    """

    async def run_agent(
        self,
        spec: AgentSpec,
        user_input: str,
        *,
        parent_loop: Any,
        driver_sid: str,
        stop_event: CancellationLike = None,
    ) -> dict[str, Any]: ...


def _continuation_prompt(remaining: list[str], i: int, n: int) -> str:
    todo_block = ("\n剩余待办：\n" + "\n".join(f"- {t}" for t in remaining)) if remaining else ""
    return (
        f"[continuation {i}/{n}] 你在轮数用尽前写下了「做了什么、还剩什么」的总结（见上）。"
        f"现在给你追加了预算：**接着做剩余的部分，不要重做已完成的**。{todo_block}\n"
        "做完后按原任务要求给出最终结果；仍完不成就明确说剩什么。"
    )


class InProcessExecutor:
    """Default executor: run the sub-agent in this process via ``run_agent_spec``."""

    async def run_agent(
        self,
        spec: AgentSpec,
        user_input: str,
        *,
        parent_loop: Any,
        driver_sid: str,
        stop_event: CancellationLike = None,
    ) -> dict[str, Any]:
        # Make this leaf a child of the driver session. Set inside the coroutine
        # so concurrent fan-out tasks (each with their own context copy) don't
        # clobber each other's parent id.
        token = set_session_id(driver_sid)
        try:
            raw = await run_agent_spec(
                spec, user_input, parent_loop=parent_loop, stop_event=stop_event
            )
        finally:
            reset_session_id(token)
        # run_agent_spec now surfaces usage; fall back to persisted session stats.
        if not raw.get("usage"):
            raw["usage"] = await _usage_for(parent_loop, raw.get("session_id"))
        return raw

    async def continue_agent(
        self,
        session_id: str,
        user_input: str,
        *,
        parent_loop: Any,
        extra_rounds: int,
        stop_event: CancellationLike = None,
    ) -> dict[str, Any]:
        """耗尽续跑的默认实现：对**同一叶子会话** ``follow_up`` 一段续命输入。

        走 parent_loop 的 child-run guards（host 注册过的 per-wake 钩子挂起，与
        run_agent_spec 同一套保障），预算 ``extra_rounds`` 由调用方（引擎的
        ContinuationPolicy）钳制。"""
        import contextlib as _ctxlib

        from power_loop.runtime.spec import _USAGE_SINK, _add_usage, workflow_continuation_loop

        leaf_loop = await workflow_continuation_loop(parent_loop, session_id)
        sink: dict[str, int] = {}   # 续跑这几轮里派的子 agent 的用量也算进来（design/170 §5）
        sink_tok = _USAGE_SINK.set(sink)
        try:
            with _ctxlib.ExitStack() as guards:
                for _name, guard in getattr(parent_loop, "_child_run_guards", ()) or ():
                    guards.enter_context(guard())
                res = await leaf_loop.follow_up(
                    user_input, session_id=session_id,
                    stop_event=stop_event, max_rounds=extra_rounds,
                )
        finally:
            _USAGE_SINK.reset(sink_tok)
        status = str(getattr(res, "status", "failed") or "failed")
        usage = await _usage_delta(parent_loop, session_id, getattr(res, "usage", None))
        _add_usage(usage, sink)
        return {
            "status": status,
            "final_text": str(getattr(res, "final_text", "") or ""),
            "rounds": int(getattr(res, "rounds", 0) or 0),
            "usage": usage,
            "session_id": session_id,
        }


async def _usage_delta(parent_loop: Any, session_id: str, res_usage: Any) -> dict[str, int]:
    """续跑轮的增量用量：优先取本次 send 结果自带的 usage；缺失再退回全会话统计（可能偏大）。"""
    if isinstance(res_usage, dict) and res_usage:
        return {k: int(v or 0) for k, v in res_usage.items() if isinstance(v, (int, float))}
    return await _usage_for(parent_loop, session_id)


async def _usage_for(parent_loop: Any, session_id: str | None) -> dict[str, int]:
    """Best-effort token usage for a finished child session.

    ``run_agent_spec`` drops ``result.usage`` today, so we read it back from the
    persisted session stats (works because workflow leaves use ``linked``
    lifecycle, keeping the session). Returns ``{}`` if unavailable.
    """
    if not session_id:
        return {}
    try:
        stats = await parent_loop.get_session_stats(session_id)
    except Exception:
        return {}
    if stats is None:
        return {}
    return {
        "prompt_tokens": int(getattr(stats, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(stats, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(stats, "total_tokens", 0) or 0),
    }


#: 叶子输入里的文件引用占位符：``@@FILEREF:outputs/static.md[-50:]@@``。
#: 引擎只认**语法**（在哪、指哪个文件、切片表达式原样是什么）；读文件与渲染文案由 host 的
#: :class:`WorkflowFileIO` 实现——那部分带语言和产品口径，不该焊进库里。
FILEREF_RE = re.compile(r"@@FILEREF:([^\[\]@]+)(\[[^\]]*\])?@@")

#: foreach 把当前迭代序号放进 body 的 env（保留键），供 FileIO 端口分配 per-iteration 产出文件。
ITERATION_ENV_KEY = "__iteration__"


class WorkflowFileIO(Protocol):
    """叶子产出文件与文件引用的宿主端口（design/64 §2）。引擎自己不碰文件系统。"""

    def output_path(self, node_id: str, *, iteration: int | None = None) -> str:
        """该节点产出文件的路径（相对 run 工作区）。foreach 的每次迭代给不同的 ``iteration``——
        body 节点共享一个 node_id，不区分就会互相覆盖。"""

    def render_ref(self, path: str, slice_expr: str) -> str:
        """把一处文件引用渲染成要注入叶子输入的文本（路径 + 片段 + **截断留痕**）。"""

    def before_attempt(self, node_id: str, path: str, attempt: int) -> None:
        """第 2 次及以后的 attempt 开始前调用：host 可以把上一次的产出归档，
        免得重跑的内容追加到旧证据后面分不清哪次。"""


class WorkflowEngine:
    """Interpret a :class:`WorkflowSpec` to completion (in-process)."""

    def __init__(
        self,
        parent_loop: Any,
        *,
        executor: Executor | None = None,
        budget: SharedBudget | None = None,
        on_step: Callable[[AgentResult], Awaitable[None]] | None = None,
        on_node_start=None,
        stop_event: CancellationLike = None,
        replay: dict[str, AgentResult] | None = None,
        run_id: str | None = None,
        close_driver: bool = True,
        allowed_tools: frozenset[str] | None = None,
        file_io: WorkflowFileIO | None = None,
        on_node_session: Callable[[str, str], Awaitable[None]] | None = None,
        resume_sessions: dict[str, str] | None = None,
    ) -> None:
        self._loop = parent_loop
        self._executor = executor or InProcessExecutor()
        self._budget = budget
        # Capability clamp: every leaf's tools are intersected with this set at
        # the SPEC level (before the executor runs it), so it holds for any
        # executor — including out-of-process leaves, where the parent's ambient
        # contextvar filter can't reach. None = unrestricted.
        self._allowed_tools = allowed_tools
        # M-workflow-engine-2: close the driver session (+ its linked leaf subtree) when run()
        # finishes so each run doesn't leak a driver row + one linked leaf row per node. Nothing
        # reads those sessions after the result is built (outputs/usage are already in `results`).
        # Set False to retain leaf sessions for post-hoc inspection.
        self._close_driver = close_driver
        # Optional per-step observer fired when each agent node settles
        # (completed / failed / budget_exceeded). Used by the detached runner to
        # journal live progress. Must not raise; errors are swallowed.
        self._on_step = on_step
        # Optional observer fired when a REAL leaf execution begins (not for replays
        # or guard short-circuits). The detached runner journals it as the run's live
        # heartbeat — without it a single-leaf run reads as "running, steps: []" for
        # its whole duration and looks hung. Must not raise; errors are swallowed.
        self._on_node_start = on_node_start
        # Cooperative cancellation: forwarded into every in-flight leaf so they
        # stop at their next checkpoint, and checked before spawning new leaves
        # so a cancelled run stops fanning out.
        # 6.46：引擎自己也持有一个令牌（run 超时用它停整个 run），和外面传进来的「谁取消都算」。
        self._own_cancel = CancellationToken()
        self._cancel: CancellationToken = CancellationToken.any_of(
            CancellationToken.from_any(stop_event), self._own_cancel)
        self._timed_out = False
        # 单节点取消（6.46）：每个节点一个子令牌——取消它只停这个节点，run 和别的节点照跑。
        self._node_tokens: dict[str, list[CancellationToken]] = {}
        self._node_cancelled: set[str] = set()
        self._admission_tokens: dict[str, list[CancellationToken]] = {}
        # 叶子会话建好时的观察者（runner 把「节点 → 会话」记进 journal）；resume 时有会话的节点在原会话接着跑。
        self._on_node_session = on_node_session
        self._resume_sessions: dict[str, str] = dict(resume_sessions or {})
        # Resume support: results of already-completed nodes, keyed by node id,
        # rehydrated from a run's journal. A node whose id is here is replayed
        # (its cached result is returned) instead of re-executed. Foreach-body
        # ids must NOT appear here (they share one id across iterations) — only
        # individually-addressable nodes and foreach *aggregate* ids belong.
        self._replay: dict[str, AgentResult] = dict(replay or {})
        # Stable id for this run; threaded into each leaf's metadata as an
        # idempotency key so side-effecting tools can dedupe across a resume.
        self._run_id = run_id
        self._results: dict[str, AgentResult] = _ScopedResults()
        self._iter_broken: list[str] = []
        self._errors: list[str] = []
        self._last: AgentResult | None = None
        self._budget_hit = False
        self._cancelled = False
        # H3: hard backstop on total real leaf executions per run, independent of the optional
        # budget. Guards nested fanout (foreach-in-foreach) and specs built programmatically
        # (bypassing the parse-time caps). Counted in _exec_agent; fail-closed past the ceiling.
        self._leaves_run = 0
        self._leaf_cap_hit = False
        # 叶子产出文件与文件引用的宿主端口（design/64 §2）。引擎不碰文件系统：它只决定
        # 「哪个节点有产出文件」「输入里的引用占位符在哪」，读写与文案渲染都由 host 实现。
        # None = 不启用（老行为）。
        self._file_io = file_io
        # 明知会失败仍放行的节点（continue_on_error）——run 终态判定要跳过它们。
        self._tolerated: set[str] = set()
        # Token usage accumulated exactly once per real leaf execution (and once
        # per replayed result on resume). Robust against the foreach-body id
        # collision, which would double/under-count if summed from _results.
        self._usage_acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # >0 while executing inside a foreach body: suppresses per-iteration step
        # journaling (those leaves share one id and aren't individually
        # replayable). The foreach's aggregate is journaled instead.
        self._emit_suppressed = 0
        self._frames: dict[int, dict[str, Any]] = {}
        self._repairs: dict[int, asyncio.Task[Any]] = {}
        self._resume_instructions: dict[str, tuple[str, int]] = {}
        self._replay_no_charge: set[str] = set()
        self._steering: dict[str, list[asyncio.Task[Any]]] = {}

    async def run(self, spec: WorkflowSpec) -> WorkflowResult:
        self._spec = spec
        if self._budget is None and spec.budget is not None:
            self._budget = SharedBudget(
                spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct
            )
        # On resume, pre-charge the (fresh) budget with tokens already spent by
        # replayed steps so the shared ceiling accounts for the whole run.
        if self._budget is not None and self._replay:
            for r in self._replay.values():
                self._budget.commit(r.usage)
        driver_sid = await self._loop.new_session(metadata={"kind": "wf_driver", "workflow": spec.name})
        try:
            env: dict[str, Any] = {"input": spec.input}
            status = "completed"
            guard = _IN_WORKFLOW.set(True)
            watchdog: asyncio.Task[None] | None = None
            if spec.timeout_s:
                async def _watchdog(limit: float = float(spec.timeout_s)) -> None:
                    await asyncio.sleep(limit)
                    self._timed_out = True
                    self._own_cancel.cancel(f"run timeout ({int(limit)}s)")
                watchdog = asyncio.create_task(_watchdog(), name="workflow-timeout")
            try:
                await self._exec(spec.root, env, driver_sid)
            except WorkflowRunError as exc:
                status = "failed"
                self._errors.append(str(exc))
            finally:
                _IN_WORKFLOW.reset(guard)
                if watchdog is not None:
                    watchdog.cancel()
            if self._timed_out:
                status = "failed"
                self._errors.append(f"run 超时（{int(spec.timeout_s or 0)} 秒）：没跑完的节点已停，没开始的不再启动")
            elif self._cancelled or self._cancel.is_cancelled():
                status = "cancelled"
            elif self._budget_hit:
                status = "budget_exceeded"
            elif status == "completed" and not self._leaf_cap_hit:
                # 以前只要没抛异常就报 completed——哪怕 judge 节点整个失败了。
                # 现在:任一叶子 failed 即 run failed,除非它显式 continue_on_error(或被兜底救回,
                # 那种情况结果已被顶替成 completed)。这是 continue_on_error 得以有意义的前提。
                # (叶子上限触顶时跳过:那时下面那条「ceiling」才是准确的根因诊断。)
                # 6.7.0：hit_round_limit 也算 broken——被截断的叶子曾让 run 谎报 completed
                # （只认 "failed" 的老判定漏掉它；continuation 救回来的已经是 completed 了）。
                broken = [
                    nid for nid, r in self._results.items()
                    if r.status not in ("completed", "skipped") and nid not in self._tolerated
                ]
                # foreach 各路里没成功的（全局表里只留了最后写的那一路，见 _iter_broken）
                broken += [b for b in self._iter_broken if b.rsplit(".", 1)[-1] not in self._tolerated]
                if broken:
                    status = "failed"
                    self._errors.append(
                        "failed node(s): " + ", ".join(sorted(set(broken)))
                        + "（如果这些失败是可接受的，给对应节点加 continue_on_error: true）"
                    )
            if self._leaf_cap_hit and status == "completed":
                # The leaf ceiling fail-closes individual leaves (status=failed); surface it at the
                # run level so a truncated fanout isn't reported as a clean "completed". (H3)
                status = "failed"
                self._errors.append(
                    f"workflow leaf ceiling ({MAX_TOTAL_LEAVES}) exceeded; fanout truncated")
            return self._build_result(spec, status)
        finally:
            # Close ONLY the synthetic driver session; the LINKED leaf sessions survive
            # (re-parented to NULL by cascade=False) as the run's audit trail — their
            # transcripts are the only tool-by-tool record of what each role actually did.
            # (Formerly M-workflow-engine-2 cascaded the whole subtree away, which erased
            # every leaf transcript the moment a run finished.) Best-effort — a cleanup
            # failure must not fail an otherwise-successful run.
            if self._close_driver:
                try:
                    await self._loop.close_session(driver_sid, cascade=False)
                except Exception:  # noqa: BLE001
                    logger.debug("workflow driver session cleanup failed", exc_info=True)

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

    async def steer_node(self, node_id: str, sid: str, message: str):
        """Accept steering only while this engine owns execution, never launch an idle send."""
        import uuid
        pending = self._steering.get(node_id)
        if pending is None:
            raise WorkflowRunError("node is settling; retry nudge after its journal step settles")
        task = asyncio.create_task(self._loop.follow_up(
            message, session_id=sid, mode="steer", enqueue_only=True,
            item_id=f"workflow-steer-{uuid.uuid4().hex}"))
        pending.append(task)  # registered before any await; the leaf joins delivery before settling
        return await asyncio.shield(task)

    async def continue_node(self, node_id: str, message: str, rounds: int) -> AgentResult | None:
        """Recover a settled branch while siblings run, using the original leaf session.

        Active dependent scopes are rejected before executing anything. Foreach IDs are
        ambiguous and require iteration-aware orchestration instead of guessing a session.
        """
        def path_to(node):
            if isinstance(node, AgentNode) and node.id == node_id:
                return [node]
            for child in self._children(node):
                path = path_to(child)
                if path:
                    return [node, *path]
            return None

        path = path_to(self._spec.root)
        if not path or any(isinstance(n, ForeachNode) for n in path):
            raise WorkflowRunError("node is missing or requires an explicit foreach iteration")
        leaf = path[-1]
        frame = self._frames.get(id(leaf))
        if not frame or frame["active"]:
            raise WorkflowRunError("node is active or has not settled; deliver a steering message to its sid")
        scope = leaf
        for parent in reversed(path[:-1]):
            pf = self._frames.get(id(parent))
            if not pf or pf["active"]:
                if not isinstance(parent, ParallelNode):
                    raise WorkflowRunError("dependent scope is still active; wait for its current step to settle")
                break
            scope = parent
        key = id(scope)
        if key in self._repairs:
            raise WorkflowRunError("this branch already has a continuation in progress")
        prior = self._results.get(node_id)
        if prior is None or not prior.session_id:
            raise WorkflowRunError("node has no resumable session")
        sf = self._frames[key]

        async def repair():
            from .resume import continuation_replay
            repaired_spec = replace(self._spec, root=scope)
            for key in _agent_ids(scope):
                self._replay.pop(key, None)
            invalidated: set[str] = set()
            replay = continuation_replay(repaired_spec, node_id, dict(self._results), invalidated=invalidated)
            self._replay.update(replay)
            self._replay_no_charge.update(replay)
            for invalid_id in invalidated:
                self._replay.pop(invalid_id, None)
                old = self._results.pop(invalid_id, None)
                await self._emit_step(AgentResult(
                    node_id=invalid_id, status="pending", text="",
                    session_id=old.session_id if old else None,
                    error="invalidated by node continuation"))
            self._resume_sessions[node_id] = prior.session_id
            self._resume_instructions[node_id] = (message, rounds)
            guard = _IN_WORKFLOW.set(True)
            try:
                return await self._exec(scope, sf["env"], sf["driver"])
            finally:
                _IN_WORKFLOW.reset(guard)

        task = asyncio.create_task(repair(), name=f"workflow-continue-{node_id}")
        self._repairs[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            # Keep the task registered until completed, even if the caller disconnects.
            if task.done():
                self._repairs.pop(key, None)

    async def _exec(self, node: WorkflowNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        if ITERATION_ENV_KEY in env:
            return await self._exec_dispatch(node, env, driver_sid)
        frame = {"active": True, "env": dict(env), "driver": driver_sid, "result": None}
        self._frames[id(node)] = frame
        try:
            frame["result"] = await self._exec_dispatch(node, env, driver_sid)
            return frame["result"]
        finally:
            frame["active"] = False

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
        replayed = self._replay.pop(node.id, None)
        if replayed is not None:
            # 整体重放：把最后一轮各节点的结果还原，循环之后对它们的引用照样能用
            for nid, r in ((replayed.payload or {}).get("last") or {}).items():
                self._results[nid] = AgentResult(node_id=nid, status=str(r.get("status") or "completed"),
                                                 text=str(r.get("text") or ""), payload=r.get("payload"),
                                                 session_id=r.get("session_id"))
            self._results[node.id] = replayed
            if node.id not in self._replay_no_charge:
                self._add_usage(replayed.usage)
            self._replay_no_charge.discard(node.id)
            return replayed
        body_ids = _agent_ids(node.body)
        feedback_node = str(node.until.get("ref") or "").split(".", 1)[0]
        prev: dict[str, str] = {}
        converged, why, iteration = False, "", 0
        for iteration in range(1, node.max_iterations + 1):
            if self._cancel.is_cancelled():
                break
            lc = {"loop_id": node.id, "iteration": iteration, "max": node.max_iterations,
                  "continue": set(node.continue_nodes), "prev": dict(prev), "why": why,
                  "feedback": self._loop_feedback(feedback_node) if iteration > 1 else ""}
            body_result = await self._exec(node.body, {**env, _LOOP_KEY: lc, "loop_iteration": iteration}, driver_sid)
            if body_result is not None and body_result.status not in ("completed", "skipped") and body_result.node_id not in self._tolerated:
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
        last = {nid: {"status": r.status, "text": (r.text or "")[:4000], "payload": r.payload, "session_id": r.session_id}
                for nid in body_ids if (r := self._results.get(nid)) is not None}
        payload = {"converged": converged, "iterations": iteration, "why": why, "last": last}
        broken = any(r.status not in ("completed", "skipped") and nid not in self._tolerated
                     for nid in body_ids if (r := self._results.get(nid)) is not None)
        ok = converged or (node.on_exhausted == "continue" and not broken and not self._cancel.is_cancelled())
        err = None if converged else (f"跑了 {iteration} 轮仍不满足 until：{why}" if not self._cancel.is_cancelled()
                                      else "循环被取消")
        res = AgentResult(node_id=node.id, status="completed" if ok else ("cancelled" if self._cancel.is_cancelled() else "failed"),
                          text=f"循环 {node.id}：{'满足' if converged else '没满足'}（{iteration} 轮）：{why}",
                          payload=payload, error=err)
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

    async def _exec_agent(self, node: AgentNode, env: dict[str, Any], driver_sid: str) -> AgentResult:
        from time import monotonic

        from power_loop.workflow.admission import AdmissionCancelled, admit
        started = monotonic()
        token = self._cancel.child()
        waiting = self._admission_tokens.setdefault(node.id, [])
        waiting.append(token)
        try:
            async with admit(token) as lease:
                waiting.remove(token)
                result = await self._exec_agent_body(node, env, driver_sid)
        except AdmissionCancelled:
            result = AgentResult(node_id=node.id, status="cancelled", text="", error="cancelled while queued")
            self._results[node.id] = result
            if not self._cancel.is_cancelled():
                self._tolerated.add(node.id)
            await self._emit_step(result)
            return result
        finally:
            if token in waiting:
                waiting.remove(token)
        if result.session_id and hasattr(self._loop, "ensure_store"):
            try:
                store = await self._loop.ensure_store()
                old = await store.get_runtime_state(result.session_id, "workflow_admission", {})
                await store.set_runtime_state(result.session_id, "workflow_admission", {
                    "attempts": int(old.get("attempts", 0)) + 1,
                    "queue_s": float(old.get("queue_s", 0)) + lease.queue_s,
                    "idle_s": float(old.get("idle_s", 0)) + lease.idle_s,
                    "metrics": {key: old.get("metrics", {}).get(key, 0) + lease.metrics.get(key, 0)
                                for key in old.get("metrics", {}).keys() | lease.metrics.keys()},
                    "elapsed_s": float(old.get("elapsed_s", 0)) + monotonic() - started,
                })
            except Exception:
                logger.warning("workflow admission metric persistence failed", exc_info=True)
        return result

    async def _exec_agent_body(self, node: AgentNode, env: dict[str, Any], driver_sid: str) -> AgentResult:
        # Resume: a completed node is replayed from the journal, not re-run. Pop
        # so a body id that somehow leaked in can only ever replay once.
        replayed = self._replay.pop(node.id, None)
        if replayed is not None:
            self._results[node.id] = replayed
            self._last = replayed
            if node.id not in self._replay_no_charge:
                self._add_usage(replayed.usage)
            self._replay_no_charge.discard(node.id)
            return replayed

        if self._cancel.is_cancelled():
            self._cancelled = True
            res = AgentResult(node_id=node.id, status="cancelled", text="",
                              error="workflow cancelled before this step started")
            self._results[node.id] = res
            await self._emit_step(res)
            return res

        if node.when is not None:
            ok, why = self._eval_when(node.when)
            if not ok:
                res = AgentResult(node_id=node.id, status="skipped", text="", error=f"条件不成立，跳过：{why}")
                self._results[node.id] = res
                await self._emit_step(res)
                return res

        if self._budget is not None and not self._budget.can_spawn():
            self._budget_hit = True
            res = AgentResult(node_id=node.id, status="budget_exceeded", text="",
                              error="shared token budget exhausted")
            self._results[node.id] = res
            await self._emit_step(res)
            return res

        if self._leaves_run >= MAX_TOTAL_LEAVES:
            self._leaf_cap_hit = True
            res = AgentResult(node_id=node.id, status="failed", text="",
                              error=f"workflow leaf ceiling ({MAX_TOTAL_LEAVES}) exceeded")
            self._results[node.id] = res
            await self._emit_step(res)
            return res
        self._leaves_run += 1

        user_input = _render(node.input, env, self._resolve_ref_soft)
        if node.inputs_from:
            extras = []
            for ref in node.inputs_from:
                up = self._results.get(ref)
                if up is None:
                    continue
                if up.status == "skipped":
                    extras.append(f"[上游节点 '{ref}' 条件不成立，跳过了：{up.error or ''}]")
                elif up.status != "completed":
                    # 🔴 上游失败时**绝不能**给空字符串:下游模型会把「没说话」读成「没意见」,
                    # 进而在缺证据的情况下下结论。给一个显式的失败说明。
                    extras.append(
                        f"[上游节点 '{ref}' 未产出结果：{up.error or up.status}。"
                        f"本次没有它的证据——不要把「没有发现」当成「没有问题」。]"
                    )
                else:
                    extras.append(up.text)
            if extras:
                user_input = user_input + "\n\n--- context ---\n" + "\n\n".join(extras)

        # 文件产出与文件引用（design/64 §2）：产出路径进 metadata 供 host/叶子使用；
        # 输入里的 @@FILEREF:…@@ 就地换成「路径 + 片段 + 截断留痕」。
        output_file: str | None = None
        if self._file_io is not None:
            try:
                output_file = node.output_file or self._file_io.output_path(
                    node.id, iteration=env.get(ITERATION_ENV_KEY)
                )
            except Exception:  # noqa: BLE001 — 产出文件是增益,不该让整个叶子起不来
                logger.debug("file_io.output_path failed for %s", node.id, exc_info=True)
            user_input = FILEREF_RE.sub(
                lambda m: self._render_fileref(m.group(1), m.group(2) or ""), user_input
            )

        # keep trace + readable usage (linked); enforce structured output at the
        # provider via the node's output_schema so real models return JSON, not prose.
        # Thread a stable idempotency key (run_id:node_id) into the leaf's
        # metadata so side-effecting tools can dedupe if the step re-runs on resume.
        leaf_tools = node.spec.tools
        if self._allowed_tools is not None:
            # inherit (None) → the clamp set becomes the explicit whitelist;
            # explicit list → intersect. Unknown names are dropped by the
            # registry subset downstream, so an over-broad clamp is harmless.
            leaf_tools = (
                sorted(self._allowed_tools)
                if leaf_tools is None
                else [t for t in leaf_tools if t in self._allowed_tools]
            )
        spec = replace(
            node.spec,
            tools=leaf_tools,
            lifecycle="linked",
            output_schema=node.output_schema,
            metadata={
                **node.spec.metadata,
                **self._step_idempotency(node.id),
                **({"output_file": output_file} if output_file else {}),
                # foreach 迭代号（6.8.1）：body 各迭代共享一个 node_id，观测端（活动面板）
                # 没有它就把 N 路并发画成同一个节点反复亮灭——看起来像串行循环。
                **({"workflow_iteration": env[ITERATION_ENV_KEY]}
                   if ITERATION_ENV_KEY in env else {}),
            },
        )
        if self._on_node_start is not None:
            try:
                await self._on_node_start(node.id)
            except Exception:  # noqa: BLE001 — observability must not fail the run
                pass

        self._steering[node.id] = []
        node_tok = self._cancel.child()
        self._node_tokens.setdefault(node.id, []).append(node_tok)
        # 重启后接着跑（6.46）：这个节点上次有叶子会话、没做完 → 先在原会话补轮；失败保留会话，不隐式重放。
        resumed_raw: dict[str, Any] | None = None
        rs = self._resume_sessions.pop(node.id, None) if not self._emit_suppressed else None
        if rs:
            try:
                prompt, rounds = self._resume_instructions.pop(
                    node.id, (_RESUME_PROMPT, max(int(node.spec.max_rounds or 30), 10)))
                resumed_raw = await self._continue_leaf(
                    rs, prompt, extra_rounds=rounds, stop=node_tok)
                resumed_raw.setdefault("session_id", rs)
                logger.info("workflow leaf %s resumed in its original session %s", node.id, rs)
            except Exception as exc:  # noqa: BLE001 — 保留原会话，防止重复副作用
                logger.warning("workflow leaf %s: resume in session %s failed (%s); preserving original session", node.id, rs, exc)
                resumed_raw = {"status": "failed", "session_id": rs,
                               "error": f"continuation failed without replay: {exc}"}

        lc = env.get(_LOOP_KEY)
        if (resumed_raw is None and lc and lc["iteration"] > 1 and node.id in lc["continue"]
                and lc["prev"].get(node.id)):
            prev_sid = lc["prev"][node.id]
            try:
                await self._prepare_rework(prev_sid, node.id, lc)
                resumed_raw = await self._continue_leaf(
                    prev_sid, _loop_prompt(lc, user_input), extra_rounds=max(int(node.spec.max_rounds or 30), 10), stop=node_tok)
                resumed_raw.setdefault("session_id", prev_sid)
            except Exception as exc:  # noqa: BLE001 — 续跑失败不能变成首次执行
                logger.warning("workflow loop %s: continuing %s in %s failed (%s); preserving original session",
                               lc["loop_id"], node.id, prev_sid, exc)
                resumed_raw = {"status": "failed", "session_id": prev_sid,
                               "error": f"continuation failed without replay: {exc}"}

        # ── 重试循环（design/64 §1）──────────────────────────────────────────────
        # 每次 attempt 都起**新的叶子会话**（失败会话的上下文可能已被半截的工具调用污染）。
        # idempotency_key 跨 attempt 不变（变了工具就没法去重），attempt 号单独进 metadata。
        policy = node.retry
        max_attempts = policy.max_attempts if policy is not None else 1
        triggers = set(policy.on) if policy is not None else set()
        raw: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                # 重跑会把产出**追加**到已有文件里,证据混在一起分不清哪次——交给 host 处理
                # (DeepTalk:改名成 .attempt<N>.md 留证据但不混淆)。
                if self._file_io is not None and output_file:
                    try:
                        self._file_io.before_attempt(node.id, output_file, attempt)
                    except Exception:  # noqa: BLE001 — 归档失败不该毁掉这次重试
                        logger.debug("file_io.before_attempt failed", exc_info=True)
                if policy is not None and policy.backoff_s > 0:
                    # 第 N 次重试等 backoff_s * factor**(N-1)；factor=1 就是固定间隔。
                    delay = policy.backoff_s * (policy.backoff_factor ** (attempt - 2))
                    await asyncio.sleep(min(delay, MAX_BACKOFF_S))
            if attempt == 1 and resumed_raw is not None:
                raw = resumed_raw
            else:
                cb_tok = (_ON_CHILD_SESSION.set(self._session_cb(node.id))
                          if self._on_node_session is not None and not self._emit_suppressed else None)
                try:
                    raw = await self._executor.run_agent(
                        replace(spec, metadata={**spec.metadata, "attempt": attempt}),
                        user_input, parent_loop=self._loop, driver_sid=driver_sid,
                        stop_event=node_tok,
                    )
                finally:
                    if cb_tok is not None:
                        _ON_CHILD_SESSION.reset(cb_tok)
            if resumed_raw is not None or attempt >= max_attempts or not self._should_retry(raw, triggers):
                break
            # 取消/预算耗尽时不再重试:那不是「这次没跑好」,是整个 run 该停了。
            if node_tok.is_cancelled() or (
                self._budget is not None and not self._budget.can_spawn()
            ):
                break
            logger.info(
                "workflow leaf %s attempt %d/%d failed (%s); retrying",
                node.id, attempt, max_attempts, raw.get("status"),
            )
        # ── 耗尽续跑（6.7.0，ContinuationPolicy）：hit_round_limit + 门条件成立 → 原会话补轮。
        # 与 retry 正交：retry 起新会话防污染，这里恰恰要**同一会话**接着成果做——引擎在
        # 耗尽时已逼叶子写下「做了什么+还剩什么」，续命提示再喂它自己的剩余 todo。
        cont = node.continuation
        if cont is not None:
            for c_i in range(1, cont.max_continuations + 1):
                if str(raw.get("status")) not in ("hit_round_limit", "context_checkpoint") or not raw.get("session_id"):
                    break
                if node_tok.is_cancelled() or (
                    self._budget is not None and not self._budget.can_spawn()
                ):
                    break
                remaining = await self._todo_remaining(str(raw["session_id"]))
                if cont.gate == "todo_remaining" and not remaining:
                    break  # 没有账本（或全勾完）：不知道从哪接，续跑只会瞎跑
                prompt = _continuation_prompt(remaining, c_i, cont.max_continuations)
                logger.info("workflow leaf %s hit_round_limit; continuation %d/%d (+%d rounds)",
                            node.id, c_i, cont.max_continuations, cont.extra_rounds)
                try:
                    more = await self._continue_leaf(
                        str(raw["session_id"]), prompt, extra_rounds=cont.extra_rounds, stop=node_tok)
                except Exception as exc:  # noqa: BLE001 — 续跑失败不毁掉已有成果
                    logger.warning("workflow leaf %s continuation failed: %s", node.id, exc)
                    break
                merged_usage = dict(raw.get("usage") or {})
                for k, v in (more.get("usage") or {}).items():
                    merged_usage[k] = int(merged_usage.get(k, 0) or 0) + int(v or 0)
                raw = {
                    **raw,
                    "status": more.get("status"),
                    "final_text": more.get("final_text") or raw.get("final_text"),
                    "rounds": int(raw.get("rounds") or 0) + int(more.get("rounds") or 0),
                    "usage": merged_usage,
                }
                self._add_usage(more.get("usage") or {})
                if self._budget is not None:
                    self._budget.commit(more.get("usage") or {})

        if str(raw.get("status")) == "cancelled":
            if node_tok.is_cancelled() and not self._cancel.is_cancelled():
                # 只取消了这一个节点（cancel_node）：run 照跑，这个节点按「放行的失败」记
                self._node_cancelled.add(node.id)
                raw = {**raw, "error": raw.get("error") or f"节点被单独取消：{node_tok.reason}"}
            else:
                self._cancelled = True
        # Close intake, then join every accepted delivery. Any input arriving after the
        # final round drain is flushed before schema validation, journaling or review.
        accepted_steering = self._steering.pop(node.id, [])
        if accepted_steering:
            await asyncio.gather(*accepted_steering)
            sid = raw.get("session_id")
            pending = await self._loop.inbox_pending(sid) if sid else {}
            if pending.get("pending", 0):
                more = await self._continue_leaf(
                    str(sid), "继续处理收件箱中的补充要求，完成后按原任务格式交付。",
                    extra_rounds=max(int(node.spec.max_rounds or 30), 10), stop=node_tok)
                combined_usage = dict(raw.get("usage") or {})
                for key, value in (more.get("usage") or {}).items():
                    combined_usage[key] = combined_usage.get(key, 0) + value
                raw = {**more, "session_id": sid, "usage": combined_usage}

        payload: dict[str, Any] | None = None
        err: str | None = None
        if node.output_schema is not None and raw.get("status") == "completed":
            try:
                payload = parse_structured(
                    raw.get("final_text") or "", schema=node.output_schema.get("schema")
                )
            except StructuredOutputError as exc:
                err = f"output_schema parse failed for '{node.id}': {exc}"
                self._errors.append(err)

        res = AgentResult(
            node_id=node.id,
            status="failed" if err else str(raw.get("status")),
            text=raw.get("final_text") or "",
            payload=payload,
            usage=dict(raw.get("usage") or {}),
            session_id=raw.get("session_id"),
            # output_schema parse error takes precedence; otherwise surface any
            # error the executor reported (e.g. a subprocess timeout / crash).
            error=err or raw.get("error"),
            db_path=raw.get("db_path"),
        )
        self._results[node.id] = res
        self._last = res
        self._add_usage(res.usage)
        if self._budget is not None:
            self._budget.commit(res.usage)

        # ── 兜底节点：所有 attempt 用尽仍失败时跑，成功则**顶替**本节点的结果供下游引用 ──
        if res.status != "completed" and node.fallback is not None and not self._cancelled:
            logger.info("workflow leaf %s failed; running fallback", node.id)
            fb = await self._exec(node.fallback, env, driver_sid)
            if fb is not None and fb.status == "completed":
                # 下游引用的是主节点 id,所以结果要挂在主节点上;兜底自己的记录也留着(它有自己的 id)。
                res = replace(
                    fb, node_id=node.id,
                    error=f"primary failed ({res.error or res.status}); recovered by fallback "
                          f"'{node.fallback.id}'",
                )
                self._results[node.id] = res
                self._last = res
                await self._emit_step(res)
                return res

        # continue_on_error:失败但明确放行——记下来,run 终态判定跳过它。被单独取消的节点同样放行。
        if (res.status == "failed" and node.continue_on_error) or node.id in self._node_cancelled:
            self._tolerated.add(node.id)
        await self._emit_step(res)
        return res

    def _session_cb(self, node_id: str) -> Callable[[str], Awaitable[None]]:
        async def _cb(sid: str) -> None:
            if self._on_node_session is not None:
                try:
                    await self._on_node_session(node_id, sid)
                except Exception:  # noqa: BLE001 — 观测失败不毁叶子
                    logger.debug("on_node_session failed for %s", node_id, exc_info=True)
        return _cb

    def cancel_node(self, node_id: str, reason: str = "cancelled by host") -> int:
        """只取消一个节点（6.46）：它在跑就在下一个检查点停下，run 和别的节点照跑；被取消的节点按「放行的失败」记。
        返回取消了几个在跑的叶子（foreach 的迭代共用一个 id，会一起停）。"""
        n = 0
        for tok in [*(self._node_tokens.get(node_id) or []), *(self._admission_tokens.get(node_id) or [])]:
            if not tok.is_cancelled():
                tok.cancel(reason)
                n += 1
        return n

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
        return ok, f"{ref} = {shown}（要求 {op}{'' if op in ('truthy', 'falsy') else ' ' + json.dumps(target, ensure_ascii=False)[:60]}）"

    def _render_fileref(self, path: str, slice_expr: str) -> str:
        if self._file_io is None:
            return ""
        try:
            return self._file_io.render_ref(path, slice_expr)
        except Exception:  # noqa: BLE001 — 引用渲染失败要说出来,不能悄悄注入空串
            logger.debug("file_io.render_ref failed for %s", path, exc_info=True)
            return f"[无法读取上游产物 {path}]"

    async def _todo_remaining(self, session_id: str) -> list[str]:
        """叶子自己的未完成 todo 文本（runtime_state key='todo'）。读不到=空（门自然关）。"""
        try:
            store = await self._loop.ensure_store()
            state = await store.get_runtime_state(session_id, "todo")
            items = state.get("items") if isinstance(state, dict) else None
            if not isinstance(items, list):
                return []
            return [str(i.get("text") or "") for i in items
                    if isinstance(i, dict) and i.get("status") != "completed"][:12]
        except Exception:  # noqa: BLE001
            return []

    async def _prepare_rework(self, session_id: str, node_id: str, lc: dict[str, Any]) -> None:
        """Archive the previous todo only for a new review iteration, never for a budget continuation."""
        from power_loop.runtime.todos import render_todos, todo_counts, validate_todos

        if not hasattr(self._loop, "ensure_store"):
            return
        store = await self._loop.ensure_store()
        key = f"workflow_rework:{lc['loop_id']}:{node_id}:{lc['iteration']}"
        if await store.get_runtime_state(session_id, key, default=None):
            return
        old = await store.get_runtime_state(session_id, "todo", default={})
        await store.set_runtime_state(session_id, key + ":previous_todo", old)
        items = validate_todos([{"text": f"第 {lc['iteration']} 轮：核实审核意见，列出修复与复验任务", "status": "in_progress"}])
        await store.set_runtime_state(session_id, "todo", {
            "items": items, "rendered": render_todos(items), "counts": todo_counts(items),
        })
        state = {"iteration": lc["iteration"], "node_id": node_id, "loop_id": lc["loop_id"],
                 "reason": lc.get("why"), "previous_todo_ref": key + ":previous_todo"}
        await store.set_runtime_state(session_id, "workflow_rework", state)
        await store.set_runtime_state(session_id, key, state)

    async def _continue_leaf(self, session_id: str, prompt: str, *, extra_rounds: int,
                             stop: CancellationToken | None = None) -> dict[str, Any]:
        """原会话补轮。走 executor 的 ``continue_agent``（host 可换实现，如加自己的 guards）；
        没有该方法的旧 executor 退回 InProcessExecutor 的默认路径。"""
        fn = getattr(self._executor, "continue_agent", None)
        if fn is None:
            fn = InProcessExecutor().continue_agent
        from power_loop.contracts.event_payloads import SubagentCompletedPayload, SubagentTaskStartPayload
        from power_loop.contracts.events import AgentEventType
        from power_loop.core.agent_context import get_session_id
        from power_loop.runtime.spec import _publish_subagent_event

        parent_sid = get_session_id()
        _publish_subagent_event(self._loop, parent_sid, AgentEventType.SUBAGENT_TASK_START,
                               SubagentTaskStartPayload(task=prompt, preset="core", sub_session_id=session_id, depth=1))
        result = None
        try:
            result = await fn(session_id, prompt, parent_loop=self._loop,
                              extra_rounds=extra_rounds, stop_event=stop or self._cancel)
            return result
        finally:
            _publish_subagent_event(self._loop, parent_sid, AgentEventType.SUBAGENT_COMPLETED,
                                   SubagentCompletedPayload(sub_session_id=session_id,
                                       status=(result or {}).get("status", "failed"),
                                       rounds=int((result or {}).get("rounds") or 0),
                                       final_text=str((result or {}).get("final_text") or "")))

    @staticmethod
    def _should_retry(raw: dict[str, Any], triggers: set[str]) -> bool:
        """这次 attempt 该不该重试。``failed``=状态不是 completed **也不是 hit_round_limit**
        （耗尽单列——它有自己的触发名与 ContinuationPolicy 原会话续跑，新会话重跑只会同预算
        再耗尽一次）;``empty``=完成但没正文;``hit_round_limit``=显式要旧行为时用。"""
        status = str(raw.get("status") or "")
        if status in ("hit_round_limit", "context_checkpoint"):
            return "hit_round_limit" in triggers
        if status == "cancelled":
            return False          # 取消不是「没跑好」
        if status != "completed":
            return "failed" in triggers
        return "empty" in triggers and not str(raw.get("final_text") or "").strip()

    def _add_usage(self, usage: dict[str, int] | None) -> None:
        for k in self._usage_acc:
            self._usage_acc[k] += int((usage or {}).get(k, 0) or 0)

    def _step_idempotency(self, node_id: str) -> dict[str, Any]:
        """Stable per-step keys injected into a leaf's metadata. Empty for an
        ad-hoc (non-journaled) run with no run_id."""
        if not self._run_id:
            return {}
        return {
            "workflow_run_id": self._run_id,
            "workflow_node_id": node_id,
            "idempotency_key": f"{self._run_id}:{node_id}",
        }

    async def _emit_step(self, res: AgentResult) -> None:
        # Inside a foreach body, per-iteration leaves are not journaled (they
        # share one id and aren't individually replayable); the aggregate is.
        if self._on_step is None or self._emit_suppressed > 0:
            return
        try:
            await self._on_step(res)
        except Exception:  # noqa: BLE001 — observer must never break the run
            pass

    async def _exec_sequence(self, node: SequenceNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        last: AgentResult | None = None
        for step in node.steps:
            last = await self._exec(step, env, driver_sid)
            if self._budget_hit or self._cancelled:
                break
            if last is not None and last.status not in ("completed", "skipped") and last.node_id not in self._tolerated:
                break
        return last

    async def _gather_branches(
        self, coros: list[Any], *, on_error: str
    ) -> list[Any]:
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
        tasks = [asyncio.ensure_future(c) for c in coros]
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
            (t.exception() for t in tasks
             if t.done() and not t.cancelled() and t.exception() is not None),
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

        gathered = await self._gather_branches(
            [one(b) for b in node.branches], on_error=node.on_error
        )
        # A closed branch may have been repaired while siblings were executing.
        # Join accepted repairs before releasing dependent sequence/assembly steps.
        for branch in node.branches:
            task = self._repairs.get(id(branch))
            if task is not None:
                await asyncio.shield(task)
        gathered = [self._frames.get(id(b), {}).get("result", r)
                    if not isinstance(r, BaseException) and ITERATION_ENV_KEY not in env else r
                    for b, r in zip(node.branches, gathered, strict=True)]
        for r in gathered:
            if isinstance(r, BaseException):
                self._errors.append(f"parallel branch error: {r}")
        broken = next((r for r in gathered if isinstance(r, AgentResult)
                       and r.status not in ("completed", "skipped") and r.node_id not in self._tolerated), None)
        return broken or next((r for r in reversed(gathered) if isinstance(r, AgentResult)), None)

    async def _exec_foreach(self, node: ForeachNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        # Resume: a foreach is replayed atomically via its journaled aggregate.
        # (Its body leaves share one id across iterations, so they can't be
        # replayed individually — a half-done foreach re-runs in full.)
        if node.id:
            replayed = self._replay.pop(node.id, None)
            if replayed is not None:
                self._results[node.id] = replayed
                self._add_usage(replayed.usage)  # aggregate carries summed leaf usage
                return replayed

        items = self._resolve_items(node)
        sem = asyncio.Semaphore(node.max_concurrency)

        async def one(item: Any, index: int) -> AgentResult | None:
            # ITERATION_ENV_KEY：body 的所有迭代共享同一个 node_id，宿主的 FileIO 端口靠它
            # 给每次迭代分不同的产出文件——不区分的话 N 个并发迭代会追加进同一个文件，互相搅乱。
            child_env = {**env, node.as_var: item, ITERATION_ENV_KEY: index}
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
                        self._iter_broken.append(f"{node.id or 'foreach'}[{index}].{nid}")

        # Per-iteration leaves are not journaled individually (see _emit_step);
        # the aggregate below is the resume unit. Reset even if a body raises.
        self._emit_suppressed += 1
        try:
            if node.parallel:
                gathered = await self._gather_branches(
                    [one(it, i) for i, it in enumerate(items)], on_error=node.on_error
                )
            else:
                gathered = []
                for i, it in enumerate(items):
                    try:
                        gathered.append(await one(it, i))
                    except Exception as exc:  # noqa: BLE001
                        if node.on_error == "halt":
                            raise
                        gathered.append(exc)
        finally:
            self._emit_suppressed -= 1

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
            raise WorkflowRunError(
                f"branch on '{node.on}' got value {value!r} with no matching case and no default"
            )
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
                f"foreach items_from '{node.items_from}' resolved to {len(value)} items "
                f"(max {MAX_FOREACH_ITEMS})"
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
                f"reference '{ref}': key '{key}' not in node '{node_id}' payload "
                f"(keys: {sorted(res.payload)})"
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
