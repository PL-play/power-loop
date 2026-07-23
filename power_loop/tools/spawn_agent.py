"""spawn_agent + run_agent — meta-tools the LLM uses to delegate work.

Two flavours of subagent invocation sit on the same plumbing
(:func:`power_loop.runtime.spec.run_agent_spec`):

* ``spawn_agent(task, preset=…)`` — *imperative*. Simple kwargs, the library
  builds an :class:`AgentSpec` with sensible defaults for ``system_prompt`` and
  the default tool preset. Designed for the common "go do this" case.
* ``run_agent(spec_json, input)`` — *declarative*. The LLM provides a full
  :class:`AgentSpec` JSON (custom system prompt, explicit tool whitelist,
  max_rounds, etc). Designed for dynamic-workflow patterns where the parent
  agent reasons about what a child should look like.

Both tools require an active :class:`StatefulAgentLoop` context (set by
:meth:`StatefulAgentLoop._run_loop`). Calling them outside one returns a
clear error string.
"""

from __future__ import annotations

from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.spec import AgentSpec, AgentSpecError, run_agent_spec

DEFAULT_MAX_ROUNDS = 20

SPAWN_AGENT_DEFINITION = ToolDefinition(
    name="spawn_agent",
    description=(
        "Spawn a sub-agent to handle a delegated task in an isolated session. "
        "The sub-agent inherits the parent's tool registry (filterable via "
        "the 'tools' arg). Returns the sub-agent's final text."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description / instructions for the sub-agent.",
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "Optional system prompt override. Defaults to a generic "
                    "task-completion prompt."
                ),
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional whitelist of tool names from the parent registry. "
                    "Omit to grant the sub-agent the full parent toolset."
                ),
            },
            "max_rounds": {
                "type": "integer",
                "description": f"Maximum rounds (default {DEFAULT_MAX_ROUNDS}, min 1).",
            },
        },
        "required": ["task"],
    },
    required_params=("task",),
)

RUN_AGENT_DEFINITION = ToolDefinition(
    name="run_agent",
    description=(
        "Materialize a full AgentSpec JSON as a one-shot sub-agent. Use when "
        "you want explicit control over name / system_prompt / tools / "
        "max_rounds / max_tokens / temperature / lifecycle. Strict schema: "
        "unknown fields are rejected."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": "AgentSpec object (or JSON string).",
            },
            "input": {
                "type": "string",
                "description": "The initial user message sent to the sub-agent.",
            },
        },
        "required": ["spec", "input"],
    },
    required_params=("spec", "input"),
)


# ── handlers ──────────────────────────────────────────────────────────────


_DEFAULT_SUB_SYSTEM_PROMPT = (
    "You are a delegated sub-agent. Complete the task the parent gave you, "
    "be concise, and return your final answer in the last assistant message."
)


async def _handle_spawn_agent(**kwargs: Any) -> str:
    loop = get_current_loop()
    if loop is None:
        return (
            "Error: spawn_agent must be invoked from inside an active "
            "StatefulAgentLoop run."
        )
    task = str(kwargs.get("task") or "").strip()
    if not task:
        return "Error: spawn_agent requires 'task'."

    spec = AgentSpec(
        name=str(kwargs.get("name") or "delegate"),
        system_prompt=str(kwargs.get("system_prompt") or _DEFAULT_SUB_SYSTEM_PROMPT),
        tools=kwargs.get("tools"),
        max_rounds=int(kwargs.get("max_rounds") or DEFAULT_MAX_ROUNDS),
    )
    result = await run_agent_spec(spec, task, parent_loop=loop)
    return _format_subagent_result(result)


async def _handle_run_agent(**kwargs: Any) -> str:
    loop = get_current_loop()
    if loop is None:
        return (
            "Error: run_agent must be invoked from inside an active "
            "StatefulAgentLoop run."
        )
    spec_payload = kwargs.get("spec")
    user_input = str(kwargs.get("input") or "")
    if spec_payload is None:
        return "Error: run_agent requires 'spec'."
    if not user_input:
        return "Error: run_agent requires 'input'."
    try:
        spec = AgentSpec.from_json(spec_payload) if not isinstance(spec_payload, AgentSpec) else spec_payload
    except AgentSpecError as exc:
        return f"Error: invalid AgentSpec — {exc}"

    result = await run_agent_spec(spec, user_input, parent_loop=loop)
    return _format_subagent_result(result)


def _format_subagent_result(result: dict[str, Any]) -> str:
    text = result.get("final_text") or "(no output)"
    status = result.get("status")
    if status and status != "completed":
        return f"[sub-agent status={status}]\n{text}"
    return text


# ── registration helpers ──────────────────────────────────────────────────


def register_spawn_agent(registry, *, include_run_agent: bool = True, overwrite: bool = False) -> None:
    """Register the spawn_agent (and optionally run_agent) tools on ``registry``.

    Usage::

        from power_loop import create_default_tool_registry, register_spawn_agent
        registry = create_default_tool_registry(workspace_dir="/path/to/project")
        register_spawn_agent(registry)
    """
    registry.register(SPAWN_AGENT_DEFINITION, _handle_spawn_agent, overwrite=overwrite)
    if include_run_agent:
        registry.register(RUN_AGENT_DEFINITION, _handle_run_agent, overwrite=overwrite)


__all__ = [
    "SPAWN_AGENT_DEFINITION",
    "RUN_AGENT_DEFINITION",
    "register_spawn_agent",
]
