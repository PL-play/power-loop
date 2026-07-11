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
from power_loop.runtime.spec import AgentSpec

from .api import create_workflow
from .engine import in_workflow
from .introspect import get_workflow, list_workflows
from .result import WorkflowRunHandle
from .spec import WorkflowSpec, WorkflowSpecError

__all__ = [
    "CREATE_WORKFLOW_DEFINITION",
    "VALIDATE_WORKFLOW_DEFINITION",
    "WORKFLOW_STATUS_DEFINITION",
    "register_workflow_tools",
]

# Per-field one-liners for the AgentSpec block of the create_workflow description.
# COVERAGE IS ENFORCED: a guard test asserts this dict's keys equal
# dataclasses.fields(AgentSpec) exactly, so adding/renaming an AgentSpec field
# without updating the model-facing description fails CI (anti-drift).
_AGENT_SPEC_FIELD_DOCS: dict[str, str] = {
    "name": "REQUIRED — short label for the sub-agent",
    "system_prompt": "REQUIRED — the sub-agent's instructions",
    "tools": "optional list of tool names; omit to inherit ALL of your current tools",
    "max_rounds": "optional round budget (default 8)",
    "max_tokens": "optional per-reply token cap (default 4000)",
    "temperature": "optional sampling temperature (default 0)",
    "model": "optional model override (omit to use the service default)",
    "output_schema": (
        "optional {\"name\", \"schema\"(JSON Schema)} — forces the leaf to emit "
        "validated JSON that downstream items_from / branch.on / .key references can read"
    ),
    "lifecycle": "managed by the engine — do not set",
    "metadata": "managed by the engine — do not set",
}


def _agent_spec_fields_block() -> str:
    """The AgentSpec field list, DERIVED from the live dataclass so it can
    never silently omit a field (see _AGENT_SPEC_FIELD_DOCS)."""
    from dataclasses import fields as _dc_fields

    lines = []
    for f in _dc_fields(AgentSpec):
        doc = _AGENT_SPEC_FIELD_DOCS.get(f.name, "(undocumented)")
        lines.append(f'  "{f.name}": {doc}')
    return "\n".join(lines)


def _build_create_workflow_description() -> str:
    return (
        "Author and run a deterministic multi-agent workflow from a WorkflowSpec "
        "JSON. Use it when you can write the control flow up front — a pipeline, "
        "a fan-out/fan-in, a map over a list, a branch on a result. For ad-hoc, "
        "one-at-a-time delegation decided as you go, use spawn_agent/run_agent "
        "instead.\n"
        "\n"
        "SPEC: {\"name\": str, \"input\"?: str, \"root\": <node>}. Node types:\n"
        "- {\"type\":\"agent\", \"id\": globally-unique str, \"spec\": <AgentSpec>, "
        "\"input\"?: template, \"inputs_from\"?: [earlier node ids], "
        "\"output_schema\"?: {\"name\", \"schema\"}}\n"
        "- {\"type\":\"sequence\", \"steps\": [node, ...]} — run in order\n"
        "- {\"type\":\"parallel\", \"branches\": [node, ...], \"max_concurrency\"?: int} "
        "— run concurrently, barrier at the end\n"
        "- {\"type\":\"foreach\", \"as\": var, \"items\": [...] OR \"items_from\": "
        "\"node_id.key\", \"body\": node, \"parallel\"?: bool, \"max_concurrency\"?: int} "
        "— map body over a list\n"
        "- {\"type\":\"branch\", \"on\": \"node_id.key\", \"cases\": {value: node}, "
        "\"default\"?: node} — pick one child by a prior result\n"
        "\n"
        "AGENTSPEC (a leaf's \"spec\") — these fields and NO others (unknown keys "
        "are rejected):\n"
        f"{_agent_spec_fields_block()}\n"
        "\n"
        "DATA FLOW: in an agent node's \"input\" template, {{input}} = spec.input "
        "and {{var}} = the enclosing foreach's \"as\" variable. \"inputs_from\" "
        "appends the named nodes' output TEXT to this node's input. items_from / "
        "branch.on read a prior node's STRUCTURED output via \"node_id.key\" — "
        "that node MUST declare an output_schema. Every node id is globally "
        "unique. Reference only nodes that complete EARLIER on the same path: "
        "never a parallel sibling, a later node, or a node inside a foreach body "
        "(reference the foreach's own aggregate id instead).\n"
        "\n"
        "DETACHED: set \"detached\": true to run in the background — returns a "
        "run id immediately and you may pass_turn; when the run finishes you "
        "will be woken, then call workflow_status(run_id=..., detail=true) to "
        "read the results. Without it (default) the call blocks and returns a "
        "per-step summary.\n"
        "\n"
        "EXAMPLE: {\"name\":\"research\",\"input\":\"topic X\",\"root\":{\"type\":"
        "\"sequence\",\"steps\":[{\"type\":\"agent\",\"id\":\"plan\",\"spec\":"
        "{\"name\":\"planner\",\"system_prompt\":\"Reply ONLY JSON: "
        "{\\\"subtopics\\\":[\\\"...\\\"]}\"},\"output_schema\":{\"name\":\"Plan\","
        "\"schema\":{\"type\":\"object\",\"required\":[\"subtopics\"],\"properties\":"
        "{\"subtopics\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}}}},"
        "{\"type\":\"foreach\",\"id\":\"res\",\"items_from\":\"plan.subtopics\","
        "\"as\":\"t\",\"body\":{\"type\":\"agent\",\"id\":\"r\",\"spec\":{\"name\":"
        "\"researcher\",\"system_prompt\":\"Write 2-3 factual sentences.\"},"
        "\"input\":\"Subtopic: {{t}}\"}},{\"type\":\"agent\",\"id\":\"sum\",\"spec\":"
        "{\"name\":\"writer\",\"system_prompt\":\"Synthesize into one paragraph.\"},"
        "\"inputs_from\":[\"res\"]}]}}\n"
        "\n"
        "Validation reports ALL problems at once — fix everything in one revision "
        "rather than retrying one error at a time. For a large or detached "
        "workflow, dry-run the spec with validate_workflow first; a spec it "
        "accepts is guaranteed to be accepted here."
    )


CREATE_WORKFLOW_DEFINITION = ToolDefinition(
    name="create_workflow",
    description=_build_create_workflow_description(),
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

VALIDATE_WORKFLOW_DEFINITION = ToolDefinition(
    name="validate_workflow",
    description=(
        "Dry-run check a WorkflowSpec JSON WITHOUT running it: full validation "
        "(plus the platform's policy checks, when the host installed any), "
        "reporting ALL problems at once so you can fix everything in one "
        "revision. On success returns a structure summary (nodes, leaves, "
        "referenced outputs). Use it to cheaply verify a large or detached "
        "workflow before create_workflow commits real sub-agent runs; for "
        "small specs you may call create_workflow directly — it validates too."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": "The WorkflowSpec to check — same shape create_workflow accepts.",
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
        "name, status). With 'run_id', returns that run's status and per-step "
        "results; add 'detail' for per-step token usage. After being woken by a "
        "detached run's completion, call this first to read its results before "
        "continuing."
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


def _spec_summary(spec: WorkflowSpec) -> str:
    """A compact structure recap for a VALID spec (validate_workflow's reply)."""
    from .spec import AgentNode, BranchNode, ForeachNode, ParallelNode, SequenceNode

    counts: dict[str, int] = {}
    leaf_ids: list[str] = []
    schema_ids: list[str] = []

    def walk(node: Any) -> None:
        counts[node.type] = counts.get(node.type, 0) + 1
        if isinstance(node, AgentNode):
            leaf_ids.append(node.id)
            if node.output_schema is not None:
                schema_ids.append(node.id)
            return
        if isinstance(node, SequenceNode):
            for s in node.steps:
                walk(s)
        elif isinstance(node, ParallelNode):
            for b in node.branches:
                walk(b)
        elif isinstance(node, ForeachNode):
            walk(node.body)
        elif isinstance(node, BranchNode):
            for c in node.cases.values():
                walk(c)
            if node.default is not None:
                walk(node.default)

    walk(spec.root)
    shape = ", ".join(f"{t}×{n}" for t, n in sorted(counts.items()))
    out = (
        f"workflow '{spec.name}': {sum(counts.values())} node(s) ({shape}); "
        f"{len(leaf_ids)} leaf agent(s): {', '.join(leaf_ids)}"
    )
    if schema_ids:
        out += f"; structured output declared on: {', '.join(schema_ids)}"
    if spec.budget is not None:
        out += f"; spec budget {spec.budget.max_tokens} tokens"
    return out


def _make_validate_workflow_handler(
    *,
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None,
) -> Callable[..., Any]:
    """Build the ``validate_workflow`` handler.

    Runs the SAME checks ``create_workflow`` would — strict parse/validation
    AND the host's ``spec_transform`` policy pass — but never executes anything,
    so a "valid" verdict here is a guarantee create_workflow will accept the
    spec (it may still fail at runtime, e.g. a leaf hitting its round limit).
    """

    async def _handle(**kwargs: Any) -> str:
        payload = kwargs.get("spec")
        if payload is None:
            return "Error: validate_workflow requires 'spec'."
        try:
            spec = WorkflowSpec.from_json(payload)
        except WorkflowSpecError as exc:
            return f"INVALID — {exc}"
        if spec_transform is not None:
            try:
                spec = spec_transform(spec)
            except WorkflowSpecError as exc:
                return f"INVALID (platform policy) — {exc}"
            if not isinstance(spec, WorkflowSpec):
                raise TypeError(
                    "spec_transform must return a WorkflowSpec "
                    f"(got {type(spec).__name__})"
                )
        return f"VALID. {_spec_summary(spec)}. Ready for create_workflow."

    return _handle


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
    description_suffix: str | None = None,
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
    * ``description_suffix`` → appended to ``create_workflow``'s model-facing
      description. The default description documents power-loop's own facts
      (grammar, AgentSpec fields, data flow); a host whose ``spec_transform``
      enforces policy limits, or whose wake delivery has its own protocol,
      states those HOST facts here so the model writes within them first try.
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
    create_def = CREATE_WORKFLOW_DEFINITION
    if description_suffix:
        create_def = replace(
            create_def, description=f"{create_def.description}\n\n{description_suffix.strip()}"
        )
    registry.register(create_def, handler, overwrite=overwrite)
    registry.register(WORKFLOW_STATUS_DEFINITION, _handle_workflow_status, overwrite=overwrite)
    # Dry-run checker: same validation + host policy pass as create_workflow,
    # zero execution. Registered as a bundle with the other two.
    registry.register(
        VALIDATE_WORKFLOW_DEFINITION,
        _make_validate_workflow_handler(spec_transform=spec_transform),
        overwrite=overwrite,
    )
