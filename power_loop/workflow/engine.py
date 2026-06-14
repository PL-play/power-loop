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
import re
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Protocol

from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.spec import AgentSpec, run_agent_spec
from power_loop.runtime.structured import StructuredOutputError, parse_structured

from .result import AgentResult, SharedBudget, WorkflowResult
from .spec import (
    AgentNode,
    BranchNode,
    ForeachNode,
    ParallelNode,
    SequenceNode,
    WorkflowNode,
    WorkflowSpec,
)

__all__ = ["Executor", "InProcessExecutor", "WorkflowEngine", "WorkflowRunError", "in_workflow"]

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
        self, spec: AgentSpec, user_input: str, *, parent_loop: Any, driver_sid: str
    ) -> dict[str, Any]: ...


class InProcessExecutor:
    """Default executor: run the sub-agent in this process via ``run_agent_spec``."""

    async def run_agent(
        self, spec: AgentSpec, user_input: str, *, parent_loop: Any, driver_sid: str
    ) -> dict[str, Any]:
        # Make this leaf a child of the driver session. Set inside the coroutine
        # so concurrent fan-out tasks (each with their own context copy) don't
        # clobber each other's parent id.
        token = set_session_id(driver_sid)
        try:
            raw = await run_agent_spec(spec, user_input, parent_loop=parent_loop)
        finally:
            reset_session_id(token)
        raw["usage"] = _usage_for(parent_loop, raw.get("session_id"))
        return raw


def _usage_for(parent_loop: Any, session_id: str | None) -> dict[str, int]:
    """Best-effort token usage for a finished child session.

    ``run_agent_spec`` drops ``result.usage`` today, so we read it back from the
    persisted session stats (works because workflow leaves use ``linked``
    lifecycle, keeping the session). Returns ``{}`` if unavailable.
    """
    if not session_id:
        return {}
    try:
        stats = parent_loop.get_session_stats(session_id)
    except Exception:
        return {}
    if stats is None:
        return {}
    return {
        "prompt_tokens": int(getattr(stats, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(stats, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(stats, "total_tokens", 0) or 0),
    }


class WorkflowEngine:
    """Interpret a :class:`WorkflowSpec` to completion (in-process)."""

    def __init__(
        self,
        parent_loop: Any,
        *,
        executor: Executor | None = None,
        budget: SharedBudget | None = None,
        on_step: Callable[[AgentResult], None] | None = None,
    ) -> None:
        self._loop = parent_loop
        self._executor = executor or InProcessExecutor()
        self._budget = budget
        # Optional per-step observer fired when each agent node settles
        # (completed / failed / budget_exceeded). Used by the detached runner to
        # journal live progress. Must not raise; errors are swallowed.
        self._on_step = on_step
        self._results: dict[str, AgentResult] = {}
        self._errors: list[str] = []
        self._last: AgentResult | None = None
        self._budget_hit = False

    async def run(self, spec: WorkflowSpec) -> WorkflowResult:
        if self._budget is None and spec.budget is not None:
            self._budget = SharedBudget(
                spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct
            )
        driver_sid = self._loop.new_session(metadata={"kind": "wf_driver", "workflow": spec.name})
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
        if self._budget_hit:
            status = "budget_exceeded"
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
        if self._budget is not None and not self._budget.can_spawn():
            self._budget_hit = True
            res = AgentResult(node_id=node.id, status="budget_exceeded", text="",
                              error="shared token budget exhausted")
            self._results[node.id] = res
            self._emit_step(res)
            return res

        user_input = _render(node.input, env)
        if node.inputs_from:
            extras = [self._results[r].text for r in node.inputs_from if r in self._results]
            if extras:
                user_input = user_input + "\n\n--- context ---\n" + "\n\n".join(extras)

        spec = replace(node.spec, lifecycle="linked")  # keep trace + readable usage
        raw = await self._executor.run_agent(
            spec, user_input, parent_loop=self._loop, driver_sid=driver_sid
        )
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
            error=err,
        )
        self._results[node.id] = res
        self._last = res
        if self._budget is not None:
            self._budget.commit(res.usage)
        self._emit_step(res)
        return res

    def _emit_step(self, res: AgentResult) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(res)
        except Exception:  # noqa: BLE001 — observer must never break the run
            pass

    async def _exec_sequence(self, node: SequenceNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        last: AgentResult | None = None
        for step in node.steps:
            last = await self._exec(step, env, driver_sid)
            if self._budget_hit:
                break
        return last

    async def _exec_parallel(self, node: ParallelNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        sem = asyncio.Semaphore(node.max_concurrency)

        async def one(branch: WorkflowNode) -> AgentResult | None:
            async with sem:
                return await self._exec(branch, dict(env), driver_sid)

        gathered = await asyncio.gather(
            *(one(b) for b in node.branches),
            return_exceptions=(node.on_error == "continue"),
        )
        for r in gathered:
            if isinstance(r, BaseException):
                self._errors.append(f"parallel branch error: {r}")
        return next((r for r in reversed(gathered) if isinstance(r, AgentResult)), None)

    async def _exec_foreach(self, node: ForeachNode, env: dict[str, Any], driver_sid: str) -> AgentResult | None:
        items = self._resolve_items(node)
        sem = asyncio.Semaphore(node.max_concurrency)

        async def one(item: Any) -> AgentResult | None:
            child_env = {**env, node.as_var: item}
            if node.parallel:
                async with sem:
                    return await self._exec(node.body, child_env, driver_sid)
            return await self._exec(node.body, child_env, driver_sid)

        if node.parallel:
            gathered = await asyncio.gather(
                *(one(it) for it in items),
                return_exceptions=(node.on_error == "continue"),
            )
        else:
            gathered = []
            for it in items:
                try:
                    gathered.append(await one(it))
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
            )
            self._results[node.id] = agg
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
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in self._results.values():
            for k in total:
                total[k] += int(r.usage.get(k, 0) or 0)
        return total


def _render(template: str, env: dict[str, Any]) -> str:
    """Substitute ``{{var}}`` from ``env``; leave unknown placeholders untouched."""

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        return str(env[name]) if name in env else m.group(0)

    return _VAR_RE.sub(repl, template)
