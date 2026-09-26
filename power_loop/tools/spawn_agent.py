"""spawn_agent — the meta-tool the LLM uses to delegate work.

A single imperative flavour of subagent invocation on top of
:func:`power_loop.runtime.spec.run_agent_spec`: simple kwargs
(``task`` plus optional ``name`` / ``system_prompt`` / ``tools`` /
``max_rounds`` / ``output_schema``), the library builds an :class:`AgentSpec`
with sensible defaults. With ``output_schema`` the child is asked for one JSON
object and the tool returns it parsed (design/126 §2). The former declarative
``run_agent`` (full AgentSpec JSON) was merged into this tool in 4.0.0 —
``system_prompt`` was its only capability that mattered in practice; hosts
that need a fully declarative spec call :func:`run_agent_spec` directly.

The tool requires an active :class:`StatefulAgentLoop` context (set by
:meth:`StatefulAgentLoop._run_loop`). Calling it outside one returns a
clear error string.
"""

from __future__ import annotations

import json
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.spec import (
    AgentSpec,
    AgentSpecError,
    normalize_output_schema,
    run_agent_spec,
)
from power_loop.runtime.structured import StructuredOutputError, parse_structured

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
            "output_schema": {
                "type": "object",
                "description": (
                    "要子 agent 交回一个 JSON 对象时给出它的 JSON Schema（根须是 object）；"
                    "结果会被解析后返回"
                ),
            },
        },
        "required": ["task"],
    },
    required_params=("task",),
    # design/124 §7.3: a sub-agent is long, useful work — a steer moves it to the background
    # (its result is delivered when it finishes) instead of killing it or blocking the turn.
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
        )
    except AgentSpecError as exc:
        return f"Error: spawn_agent 参数有误，子 agent 没有启动：{exc}"
    result = await run_agent_spec(spec, task, parent_loop=loop)
    if output_schema is not None and result.get("status") == "completed":
        return _format_structured_result(result.get("final_text") or "", output_schema["schema"])
    return _format_subagent_result(result)


def _format_subagent_result(result: dict[str, Any]) -> str:
    text = result.get("final_text") or "(no output)"
    status = result.get("status")
    if status and status != "completed":
        return f"[sub-agent status={status}]\n{text}"
    return text


#: 解析失败时带回的原文上限：够主 agent 看清子 agent 写了什么，又不把一整篇散文塞回上下文。
_RAW_TEXT_CAP = 4000

_PARSE_FAILURE_REASONS = {
    "no_json": "回复里没有 JSON 对象",
    "invalid_json": "JSON 格式不对",
    "not_object": "不是 JSON 对象",
}


def _format_structured_result(text: str, schema: dict[str, Any]) -> str:
    """子 agent 的最终回复 → 解析后的紧凑 JSON；解析不了就带上原因和原文。

    这里的子会话用完即删，没法像 DeepTalk 那样在同一个会话里补一轮修复，失败就交给主 agent 处理。
    """
    try:
        value = parse_structured(text, schema=schema)
    except StructuredOutputError as exc:
        reason = exc.reason
        if reason.startswith("missing_required:"):
            reason = f"缺少必填字段 {reason.split(':', 1)[1]}"
        else:
            reason = _PARSE_FAILURE_REASONS.get(reason, reason)
        raw = text or "(no output)"
        if len(raw) > _RAW_TEXT_CAP:
            raw = raw[:_RAW_TEXT_CAP] + f"…（原文共 {len(text)} 字，后面已截掉）"
        return f"结构化结果解析失败（{reason}），原文：{raw}"
    return "结构化结果：" + json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
