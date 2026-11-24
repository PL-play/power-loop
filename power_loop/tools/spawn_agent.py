"""Asynchronous create/nudge meta-tool with durable child completion results.

Host code may synchronously await run_agent_spec. Model-visible calls always return
acceptance receipts and deliver results to the parent's durable task_done inbox.
"""

from __future__ import annotations

import json
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import (
    AgentSpec,
    AgentSpecError,
    normalize_output_schema,
)

DEFAULT_MAX_ROUNDS = 20

SPAWN_AGENT_DEFINITION = ToolDefinition(pricing=Free("composite"),
    name="spawn_agent",
    description=(
        "Delegate work in an isolated child session, always asynchronously. create (default) "
        "starts a child session; send adds a message to the same child (urgent joins its next boundary). "
        "Returns target_ref/input_ref/session_id/run_id and acceptance, not completion. Results arrive "
        "as ordinary agent_result inputs in your session inbox with final_text, status and usage. "
        "You may pass_turn while waiting; inspect the original session for additional context."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "send"], "default": "create",
                       "description": "create needs task; send needs sub_session and message and retains the original execution contract."},
            "sub_session": {"type": "string", "description": "send only: direct child's session_id from a previous receipt; deleted children cannot continue."},
            "message": {"type": "string", "description": "send only: correction, additional input or follow-up question."},
            "urgent": {"type": "boolean", "description": "send only: deliver before the child's not-yet-started tool calls."},
            "task": {
                "type": "string",
                "description": "create requires task: clear instructions, input locations and expected result.",
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
            "output_schema": {
                "type": "object",
                "description": (
                    "要子 agent 交回一个 JSON 对象时给出它的 JSON Schema（根须是 object）；"
                    "结果会被解析后返回"
                ),
            },
        },
        "required": [],
    },
    required_params=(),
    # design/124 §8.4: a stop gives the sub-agent subtask_s to wind down instead of killing it.
    interrupt="background",
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
    action = kwargs.get("action", "create")
    if action == "send":
        from power_loop.runtime.child_tasks import send_input
        sid, message = str(kwargs.get("sub_session") or "").strip(), kwargs.get("message")
        if not sid or not isinstance(message, str) or not message.strip():
            return "Error: send requires sub_session and message."
        if any(k in kwargs for k in ("task", "name", "system_prompt", "tools", "max_rounds", "output_schema")):
            return "Error: send retains the original contract; provide only sub_session, message and urgent."
        try:
            return json.dumps(await send_input(loop, sid, message, urgent=kwargs.get("urgent") is True),
                              ensure_ascii=False)
        except ValueError as exc:
            return "Error: " + str(exc)
    if action != "create":
        return "Error: action must be create or send."
    task = str(kwargs.get("task") or "").strip()
    if not task:
        return "Error: spawn_agent requires 'task'."

    output_schema = None
    raw_schema = kwargs.get("output_schema")
    if raw_schema not in (None, "", {}):
        try:
            # agent 自己写的 schema 很少符合 strict 规范（每层禁多余字段、所有键必填），
            # 默认不开 strict，免得原生 json_schema 的服务端直接 400。
            output_schema = normalize_output_schema(raw_schema, default_strict=False)
        except AgentSpecError as exc:
            return f"Error: output_schema 无效，子 agent 没有启动：{exc}"

    try:
        spec = AgentSpec(
            name=str(kwargs.get("name") or "delegate"),
            system_prompt=str(kwargs.get("system_prompt") or _DEFAULT_SUB_SYSTEM_PROMPT),
            tools=kwargs.get("tools"),
            max_rounds=int(kwargs.get("max_rounds") or DEFAULT_MAX_ROUNDS),
            output_schema=output_schema,
            lifecycle="linked",
        )
    except AgentSpecError as exc:
        return f"Error: spawn_agent 参数有误，子 agent 没有启动：{exc}"
    from power_loop.runtime.delegation import start
    try:
        result = await start(loop, spec, task)
        return json.dumps(result, ensure_ascii=False)
    except ValueError as exc:
        return "Error: " + str(exc)


# ── registration helpers ──────────────────────────────────────────────────


def register_spawn_agent(registry, *, overwrite: bool = False) -> None:
    """Register the spawn_agent tool on ``registry``.

    Usage::

        from power_loop import create_default_tool_registry, register_spawn_agent
        registry = create_default_tool_registry(workspace_dir="/path/to/project")
        register_spawn_agent(registry)
    """
    from power_loop.runtime.delegation import result_view
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    registry.register(SPAWN_AGENT_DEFINITION, FunctionRuntimeTool(
        _handle_spawn_agent, ToolRuntimeContract("sdk.subagent", background_allowed=False, allowed_roles=("main",)),
        policy=lambda arguments: ToolDispatchPolicy(native_admitted=True), formatter=result_view,
    ), overwrite=overwrite)


__all__ = [
    "SPAWN_AGENT_DEFINITION",
    "register_spawn_agent",
]
