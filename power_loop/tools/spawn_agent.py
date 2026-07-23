"""spawn_agent — the meta-tool the LLM uses to delegate work.

A single imperative flavour of subagent invocation on top of
:func:`power_loop.runtime.spec.run_agent_spec`: simple kwargs
(``task`` plus optional ``name`` / ``system_prompt`` / ``tools`` /
``max_rounds``), the library builds an :class:`AgentSpec` with sensible
defaults. The former declarative ``run_agent`` (full AgentSpec JSON) was
merged into this tool in 4.0.0 — ``system_prompt`` was its only capability
that mattered in practice; hosts that need a fully declarative spec call
:func:`run_agent_spec` directly.

The tool requires an active :class:`StatefulAgentLoop` context (set by
:meth:`StatefulAgentLoop._run_loop`). Calling it outside one returns a
clear error string.
"""

from __future__ import annotations

from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.spec import AgentSpec, run_agent_spec

DEFAULT_MAX_ROUNDS = 20

SPAWN_AGENT_DEFINITION = ToolDefinition(
    name="spawn_agent",
    description=(
        "Spawn a sub-agent to handle a delegated task in an isolated session "
        "and return its final text. The sub-agent inherits the parent's tool "
        "registry (filterable via the 'tools' arg); give it a custom persona "
        "via 'system_prompt' when the default task-completion prompt isn't "
        "enough."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description / instructions for the sub-agent.",
            },
            "name": {
                "type": "string",
                "description": "Optional short label for the sub-agent (cosmetic only).",
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


# ── handler ───────────────────────────────────────────────────────────────


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


def _format_subagent_result(result: dict[str, Any]) -> str:
    text = result.get("final_text") or "(no output)"
    status = result.get("status")
    if status and status != "completed":
        return f"[sub-agent status={status}]\n{text}"
    return text


# ── registration helpers ──────────────────────────────────────────────────


def register_spawn_agent(registry, *, overwrite: bool = False) -> None:
    """Register the spawn_agent tool on ``registry``.

    Usage::

        from power_loop import create_default_tool_registry, register_spawn_agent
        registry = create_default_tool_registry(workspace_dir="/path/to/project")
        register_spawn_agent(registry)
    """
    registry.register(SPAWN_AGENT_DEFINITION, _handle_spawn_agent, overwrite=overwrite)


__all__ = [
    "SPAWN_AGENT_DEFINITION",
    "register_spawn_agent",
]
