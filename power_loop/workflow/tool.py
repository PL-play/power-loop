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


async def _handle_create_workflow(**kwargs: Any) -> str:
    loop = get_current_loop()
    if loop is None:
        return "Error: create_workflow must be invoked from inside an active StatefulAgentLoop run."
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
    wf = create_workflow(spec, parent_loop=loop, parent_session_id=parent_sid)

    if kwargs.get("detached"):
        if not parent_sid:
            return "Error: detached workflows require an active session; run without 'detached'."
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


async def _handle_workflow_status(**kwargs: Any) -> str:
    loop = get_current_loop()
    if loop is None:
        return "Error: workflow_status must be invoked from inside an active StatefulAgentLoop run."
    parent_sid = get_session_id()
    if not parent_sid:
        return "Error: no active session."
    run_id = kwargs.get("run_id")
    if run_id:
        j = get_workflow(loop, parent_sid, str(run_id), detail=bool(kwargs.get("detail")))
        if j is None:
            return f"No workflow run '{run_id}' found."
        return json.dumps(j, ensure_ascii=False, indent=2)
    runs = list_workflows(loop, parent_sid)
    if not runs:
        return "No workflow runs."
    return "\n".join(f"{r['run_id']}  {r['status']:<14} {r['workflow']} ({r['steps']} steps)" for r in runs)


def register_workflow_tools(registry: Any, *, overwrite: bool = False) -> None:
    """Register the ``create_workflow`` + ``workflow_status`` tools on ``registry``.

    For detached runs to wake the agent, the host must also run a ``TimerRunner``
    and call :func:`power_loop.workflow.register_wake_guard(loop)` once.

    Do NOT grant these tools to sub-agents spawned inside a workflow — nested
    runs are refused in this tier.
    """
    registry.register(CREATE_WORKFLOW_DEFINITION, _handle_create_workflow, overwrite=overwrite)
    registry.register(WORKFLOW_STATUS_DEFINITION, _handle_workflow_status, overwrite=overwrite)
