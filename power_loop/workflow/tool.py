"""LLM-facing workflow tools.

``create_workflow`` — author + run a :class:`WorkflowSpec` JSON (validated on
submission). Synchronous by default; ``detached: true`` runs it in the background
and returns a run id, so the agent can ``pass_turn`` and be woken on completion
(requires the host to run a ``TimerRunner`` and to have called
:func:`power_loop.workflow.register_wake_guard`).

``workflow_status`` — list runs or fetch one run's status/detail (D4).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id

from .api import create_workflow
from .engine import in_workflow
from .introspect import get_workflow, list_workflows
from .result import WorkflowRunHandle
from .spec import WorkflowSpec, WorkflowSpecError

__all__ = [
    "CREATE_WORKFLOW_DEFINITION",
    "WORKFLOW_STATUS_DEFINITION",
    "register_workflow_tools",
]

CREATE_WORKFLOW_DEFINITION = ToolDefinition(
    name="create_workflow",
    description=(
        "Author and run a deterministic multi-agent workflow from a WorkflowSpec "
        "JSON. Use this when a task needs FIXED control flow over several "
        "sub-agents — run steps in order (sequence), concurrently (parallel), "
        "map one step over a list (foreach), or pick a step by a prior result "
        "(branch) — rather than deciding each delegation yourself. Each leaf is "
        "an 'agent' node carrying an AgentSpec. Reference an earlier agent's "
        "structured output via 'node_id.key' (that node must declare an "
        "output_schema). Strict schema: unknown keys are rejected and all "
        "problems are reported at once. Set 'detached' to run in the background "
        "and be woken when it finishes (you may pass_turn after). Returns a run "
        "summary, or a run id when detached."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "A WorkflowSpec: {name, input?, budget?, root}. 'root' is a node. "
                    "Node types: "
                    "{type:'agent', id, spec:<AgentSpec>, input?, inputs_from?, output_schema?}; "
                    "{type:'sequence', steps:[node...]}; "
                    "{type:'parallel', branches:[node...], max_concurrency?, on_error?}; "
                    "{type:'foreach', as, items_from|items, body:<node>, parallel?, max_concurrency?, on_error?}; "
                    "{type:'branch', on:'node_id.key', cases:{value:node}, default?}. "
                    "Use {{var}} in an agent's 'input'/system_prompt for the foreach var or {{input}}."
                ),
            },
            "input": {
                "type": "string",
                "description": "Optional initial input, overrides spec.input (available as {{input}}).",
            },
            "detached": {
                "type": "boolean",
                "description": "Run in the background and wake you on completion (default false = run now).",
            },
        },
        "required": ["spec"],
    },
    required_params=("spec",),
)

WORKFLOW_STATUS_DEFINITION = ToolDefinition(
    name="workflow_status",
    description=(
        "Inspect workflows you started. With no args, lists all runs (run_id, "
        "name, status). With 'run_id', returns that run's status and steps; add "
        "'detail' for per-step token usage."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "A specific run id to inspect (optional)."},
            "detail": {"type": "boolean", "description": "Include per-step session stats."},
        },
    },
)


def _make_create_workflow_handler(
    *,
    executor_factory: Callable[[Any, str | None], Any] | None = None,
    budget_factory: Callable[[Any, str | None], Any] | None = None,
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None,
) -> Callable[..., Any]:
    """Build the ``create_workflow`` tool handler, with optional HOST injection
    points (PROVISIONAL, 3.14) — see :func:`register_workflow_tools`."""

    async def _handle(**kwargs: Any) -> str:
        loop = get_current_loop()
        if loop is None:
            return (
                "Error: create_workflow must be invoked from inside an active "
                "StatefulAgentLoop run."
            )
        if in_workflow():
            return (
                "Error: nested workflow creation is not allowed in this tier. "
                "Express the additional steps as nodes inside the current WorkflowSpec."
            )
        spec_payload = kwargs.get("spec")
        if spec_payload is None:
            return "Error: create_workflow requires 'spec'."
        try:
            spec = WorkflowSpec.from_json(spec_payload)
        except WorkflowSpecError as exc:
            return f"Error: {exc}"  # aggregated problems — the model can repair and retry
        if kwargs.get("input"):
            spec = replace(spec, input=str(kwargs["input"]))

        parent_sid = get_session_id()
        if spec_transform is not None:
            # Host policy rewrite (clamps, forced budgets/models …). A
            # WorkflowSpecError raised here reads exactly like a validation
            # failure — aggregated problems back to the model, which can repair
            # and retry. Any other exception is a host bug and propagates (the
            # pipeline surfaces it as this tool call's error).
            try:
                spec = spec_transform(spec)
            except WorkflowSpecError as exc:
                return f"Error: {exc}"
            if not isinstance(spec, WorkflowSpec):
                raise TypeError(
                    "spec_transform must return a WorkflowSpec "
                    f"(got {type(spec).__name__})"
                )
        wf = create_workflow(
            spec,
            parent_loop=loop,
            executor=executor_factory(loop, parent_sid) if executor_factory else None,
            budget=budget_factory(loop, parent_sid) if budget_factory else None,
            parent_session_id=parent_sid,
        )

        if kwargs.get("detached"):
            if not parent_sid:
                return (
                    "Error: detached workflows require an active session; "
                    "run without 'detached'."
                )
            handle = await wf.start(detached=True)
            # start(detached=True) always returns a WorkflowRunHandle (the union's other
            # arm is the non-detached path); narrow so .run_id is well-typed.
            assert isinstance(handle, WorkflowRunHandle)
            return (
                f"Started detached workflow '{spec.name}' as run {handle.run_id}. "
                f"You will be woken with the result when it finishes — you may pass_turn now. "
                f"Check progress with workflow_status(run_id='{handle.run_id}')."
            )
        result = await wf.run()
        return result.summary()

    return _handle


#: Default handler (no host injection) — the pre-3.14 behavior, kept as a
#: module-level symbol for direct use in tests/embedding.
_handle_create_workflow = _make_create_workflow_handler()


async def _handle_workflow_status(**kwargs: Any) -> str:
    loop = get_current_loop()
    if loop is None:
        return "Error: workflow_status must be invoked from inside an active StatefulAgentLoop run."
    parent_sid = get_session_id()
    if not parent_sid:
        return "Error: no active session."
    run_id = kwargs.get("run_id")
    if run_id:
        j = await get_workflow(loop, parent_sid, str(run_id), detail=bool(kwargs.get("detail")))
        if j is None:
            return f"No workflow run '{run_id}' found."
        return json.dumps(j, ensure_ascii=False, indent=2)
    runs = await list_workflows(loop, parent_sid)
    if not runs:
        return "No workflow runs."
    return "\n".join(f"{r['run_id']}  {r['status']:<14} {r['workflow']} ({r['steps']} steps)" for r in runs)


def register_workflow_tools(
    registry: Any,
    *,
    overwrite: bool = False,
    executor_factory: Callable[[Any, str | None], Any] | None = None,
    budget_factory: Callable[[Any, str | None], Any] | None = None,
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None,
) -> None:
    """Register the ``create_workflow`` + ``workflow_status`` tools on ``registry``.

    For detached runs to wake the agent, the host must also run a ``TimerRunner``
    and call :func:`power_loop.workflow.register_wake_guard(loop)` once.

    Do NOT grant these tools to sub-agents spawned inside a workflow — nested
    runs are refused in this tier.

    HOST injection points (all optional; omitting all three = pre-3.14 behavior;
    PROVISIONAL, 3.14). Each is evaluated PER TOOL INVOCATION with
    ``(loop, parent_session_id)`` from the invoking run's context:

    * ``executor_factory`` → the run's :class:`~power_loop.workflow.engine.Executor`
      (e.g. a capability-clamping wrapper, or a ``SubprocessExecutor``).
      ``None``/returning ``None`` → the default ``InProcessExecutor``.
    * ``budget_factory`` → a per-run ``SharedBudget`` (e.g. from host config).
      Returning ``None`` → no host budget (the spec's own ``budget`` still applies).
    * ``spec_transform`` → rewrite the validated :class:`WorkflowSpec` before it
      runs (policy clamps, forced limits). May raise
      :class:`~power_loop.workflow.spec.WorkflowSpecError` to reject — the
      aggregated problems go back to the model like a validation failure.
    """
    handler = (
        _make_create_workflow_handler(
            executor_factory=executor_factory,
            budget_factory=budget_factory,
            spec_transform=spec_transform,
        )
        if (executor_factory or budget_factory or spec_transform)
        else _handle_create_workflow
    )
    registry.register(CREATE_WORKFLOW_DEFINITION, handler, overwrite=overwrite)
    registry.register(WORKFLOW_STATUS_DEFINITION, _handle_workflow_status, overwrite=overwrite)
