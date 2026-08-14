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
import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Protocol

from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.spec import AgentSpec, run_agent_spec
from power_loop.runtime.structured import StructuredOutputError, parse_structured

from .result import AgentResult, SharedBudget, WorkflowResult
from .spec import (
    MAX_BACKOFF_S,
    MAX_FOREACH_ITEMS,
    AgentNode,
    BranchNode,
    ForeachNode,
    ParallelNode,
    SequenceNode,
    WorkflowNode,
    WorkflowSpec,
)

__all__ = ["Executor", "InProcessExecutor", "WorkflowEngine", "WorkflowRunError", "in_workflow"]

logger = logging.getLogger(__name__)

#: Hard ceiling on the total number of real leaf executions in one run (H3 — BUG_REVIEW_3.4). A
#: backstop independent of the optional budget: caps nested fanout (foreach-in-foreach) and any
#: programmatically-built spec that bypasses the static validation caps. Fail-closed when exceeded.
MAX_TOTAL_LEAVES = 10_000

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_]\w*)\s*\}\}")

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
        self._cancel: CancellationToken = CancellationToken.from_any(stop_event)
        # Resume support: results of already-completed nodes, keyed by node id,
        # rehydrated from a run's journal. A node whose id is here is replayed
        # (its cached result is returned) instead of re-executed. Foreach-body
        # ids must NOT appear here (they share one id across iterations) — only
        # individually-addressable nodes and foreach *aggregate* ids belong.
        self._replay: dict[str, AgentResult] = dict(replay or {})
        # Stable id for this run; threaded into each leaf's metadata as an
        # idempotency key so side-effecting tools can dedupe across a resume.
        self._run_id = run_id
        self._results: dict[str, AgentResult] = {}
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

    async def run(self, spec: WorkflowSpec) -> WorkflowResult:
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
            try:
                await self._exec(spec.root, env, driver_sid)
            except WorkflowRunError as exc:
                status = "failed"
                self._errors.append(str(exc))
            finally:
                _IN_WORKFLOW.reset(guard)
            if self._cancelled or self._cancel.is_cancelled():
                status = "cancelled"
            elif self._budget_hit:
                status = "budget_exceeded"
            elif status == "completed" and not self._leaf_cap_hit:
                # 以前只要没抛异常就报 completed——哪怕 judge 节点整个失败了。
                # 现在:任一叶子 failed 即 run failed,除非它显式 continue_on_error(或被兜底救回,
                # 那种情况结果已被顶替成 completed)。这是 continue_on_error 得以有意义的前提。
                # (叶子上限触顶时跳过:那时下面那条「ceiling」才是准确的根因诊断。)
                broken = [
                    nid for nid, r in self._results.items()
                    if r.status == "failed" and nid not in self._tolerated
                ]
                if broken:
                    status = "failed"
                    self._errors.append(
                        "failed node(s): " + ", ".join(sorted(broken))
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
        )

    # ── node dispatch ──────────────────────────────────────────────────────

    async def _exec(self, node: WorkflowNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
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
        raise WorkflowRunError(f"unknown node type: {type(node).__name__}")

    async def _exec_agent(self, node: AgentNode, env: dict[str, Any], driver_sid: str) -> AgentResult:
        # Resume: a completed node is replayed from the journal, not re-run. Pop
        # so a body id that somehow leaked in can only ever replay once.
        replayed = self._replay.pop(node.id, None)
        if replayed is not None:
            self._results[node.id] = replayed
            self._last = replayed
            self._add_usage(replayed.usage)
            return replayed

        if self._cancel.is_cancelled():
            self._cancelled = True
            res = AgentResult(node_id=node.id, status="cancelled", text="",
                              error="workflow cancelled before this step started")
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

        user_input = _render(node.input, env)
        if node.inputs_from:
            extras = []
            for ref in node.inputs_from:
                up = self._results.get(ref)
                if up is None:
                    continue
                if up.status != "completed":
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
            },
        )
        if self._on_node_start is not None:
            try:
                await self._on_node_start(node.id)
            except Exception:  # noqa: BLE001 — observability must not fail the run
                pass

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
            raw = await self._executor.run_agent(
                replace(spec, metadata={**spec.metadata, "attempt": attempt}),
                user_input, parent_loop=self._loop, driver_sid=driver_sid,
                stop_event=self._cancel,
            )
            if attempt >= max_attempts or not self._should_retry(raw, triggers):
                break
            # 取消/预算耗尽时不再重试:那不是「这次没跑好」,是整个 run 该停了。
            if self._cancel.is_cancelled() or (
                self._budget is not None and not self._budget.can_spawn()
            ):
                break
            logger.info(
                "workflow leaf %s attempt %d/%d failed (%s); retrying",
                node.id, attempt, max_attempts, raw.get("status"),
            )
        if str(raw.get("status")) == "cancelled":
            self._cancelled = True
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
            status=str(raw.get("status")),
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
        await self._emit_step(res)

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

        # continue_on_error:失败但明确放行——记下来,run 终态判定跳过它。
        if res.status == "failed" and node.continue_on_error:
            self._tolerated.add(node.id)
        return res

    def _render_fileref(self, path: str, slice_expr: str) -> str:
        if self._file_io is None:
            return ""
        try:
            return self._file_io.render_ref(path, slice_expr)
        except Exception:  # noqa: BLE001 — 引用渲染失败要说出来,不能悄悄注入空串
            logger.debug("file_io.render_ref failed for %s", path, exc_info=True)
            return f"[无法读取上游产物 {path}]"

    @staticmethod
    def _should_retry(raw: dict[str, Any], triggers: set[str]) -> bool:
        """这次 attempt 该不该重试。``failed``=状态不是 completed;``empty``=完成但没正文。"""
        status = str(raw.get("status") or "")
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

        async def one(branch: WorkflowNode) -> AgentResult | None:
            async with sem:
                return await self._exec(branch, dict(env), driver_sid)

        gathered = await self._gather_branches(
            [one(b) for b in node.branches], on_error=node.on_error
        )
        for r in gathered:
            if isinstance(r, BaseException):
                self._errors.append(f"parallel branch error: {r}")
        return next((r for r in reversed(gathered) if isinstance(r, AgentResult)), None)

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
            if node.parallel:
                async with sem:
                    return await self._exec(node.body, child_env, driver_sid)
            return await self._exec(node.body, child_env, driver_sid)

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


def _render(template: str, env: dict[str, Any]) -> str:
    """Substitute ``{{var}}`` from ``env``; leave unknown placeholders untouched."""

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        return str(env[name]) if name in env else m.group(0)

    return _VAR_RE.sub(repl, template)
