"""create_workflow — an LLM-facing meta-tool.

Lets the parent agent author a :class:`WorkflowSpec` JSON at runtime (same idiom
as emitting an ``AgentSpec`` for ``run_agent``) and execute it. The spec is
validated on submission; invalid specs come back as a single aggregated error
string the model can read and repair.

The tool returns the run's :meth:`WorkflowResult.summary`. It refuses to run
*inside* an already-running workflow (a recursion guard for this tier).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop

from .api import create_workflow
from .engine import in_workflow
from .spec import WorkflowSpec, WorkflowSpecError

__all__ = ["CREATE_WORKFLOW_DEFINITION", "register_workflow_tools"]

CREATE_WORKFLOW_DEFINITION = ToolDefinition(
    name="create_workflow",
    description=(
        "Author and run a deterministic multi-agent workflow from a WorkflowSpec "
        "JSON. Use this when a task needs FIXED control flow over several "
        "sub-agents — run steps in order (sequence), concurrently (parallel), "
        "map one step over a list (foreach), or pick a step by a prior result "
        "(branch) — rather than deciding each delegation yourself. Each leaf is "
        "an 'agent' node carrying an AgentSpec. Reference an earlier agent's "
        "structured output via 'node_id.key' (the earlier node must declare an "
        "output_schema). Strict schema: unknown keys are rejected and all "
        "problems are reported at once. Returns a summary of the run."
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
        },
        "required": ["spec"],
    },
    required_params=("spec",),
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

    wf = create_workflow(spec, parent_loop=loop)
    result = await wf.run()
    return result.summary()


def register_workflow_tools(registry: Any, *, overwrite: bool = False) -> None:
    """Register the ``create_workflow`` meta-tool on ``registry``.

    Note: do NOT grant this tool to sub-agents you spawn inside a workflow — the
    in-process tier guards against nested runs, but the cleanest boundary is to
    keep ``create_workflow`` off leaf agents' tool whitelists.
    """
    registry.register(CREATE_WORKFLOW_DEFINITION, _handle_create_workflow, overwrite=overwrite)
