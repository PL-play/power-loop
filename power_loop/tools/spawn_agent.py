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
from power_loop.runtime.structured import StructuredOutputError, parse_structured

DEFAULT_MAX_ROUNDS = 20

SPAWN_AGENT_DEFINITION = ToolDefinition(pricing=Free("composite"),
    name="spawn_agent",
    description=(
        "Delegate work in an isolated child session, always asynchronously. create (default) "
        "starts a task; nudge continues the same child or steers it if running. "
        "Returns session_id/run_id and acceptance, not completion. Results arrive in your "
        "task_done inbox with final_text, status, usage and messages/projection ranges. "
        "Read them with recall_send(sub_session=session_id, source='messages'|'projection', "
        "ids=the_range). You may pass_turn while waiting."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "nudge"], "default": "create",
                       "description": "create needs task; nudge needs sub_session and message and retains the original execution contract."},
            "sub_session": {"type": "string", "description": "nudge only: direct child's session_id from a previous receipt; deleted children cannot continue."},
            "message": {"type": "string", "description": "nudge only: correction, additional input or follow-up question."},
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
    action = kwargs.get("action", "create")
    if action == "nudge":
        from power_loop.runtime.delegation import nudge
        sid, message = str(kwargs.get("sub_session") or "").strip(), kwargs.get("message")
        if not sid or not isinstance(message, str) or not message.strip():
            return "Error: nudge requires sub_session and message."
        if any(k in kwargs for k in ("task", "name", "system_prompt", "tools", "max_rounds", "output_schema")):
            return "Error: nudge retains the original contract; provide only sub_session and message."
        try:
            return json.dumps(await nudge(loop, sid, message, finalize=lambda sid, raw: _finalize(loop, sid, raw)), ensure_ascii=False)
        except ValueError as exc:
            return "Error: " + str(exc)
    if action != "create":
        return "Error: action must be create or nudge."
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
        result = await start(loop, spec, task, finalize=lambda sid, raw: _finalize(loop, sid, raw))
        return json.dumps(result, ensure_ascii=False)
    except ValueError as exc:
        return "Error: " + str(exc)


async def _finalize(loop, sid, raw):
    from power_loop.runtime.spec import workflow_continuation_loop
    store = await loop.ensure_store()
    contract = await store.get_runtime_state(sid, "workflow_execution_contract")
    schema = contract["spec"].get("output_schema")
    if schema is None or raw.get("status") != "completed":
        return raw
    result = dict(raw)
    for attempt in range(2):
        try:
            result["payload"] = parse_structured(result.get("final_text") or "", schema=schema["schema"])
            if str(result.get("error", "")).startswith("structured_output:"):
                result.pop("error")
            return result
        except StructuredOutputError as exc:
            result["error"] = "structured_output: " + exc.reason
        if attempt == 0 and str(result.get("final_text") or "").strip():
            from power_loop.runtime.delegation import readmit_session
            child = await workflow_continuation_loop(loop, sid)
            import contextlib
            with contextlib.ExitStack() as guards:
                for _, guard in getattr(loop, "_child_run_guards", ()):
                    guards.enter_context(guard())
                async with readmit_session(loop, sid):
                    repaired = await child.send("Rewrite the previous answer as the requested JSON object. Do not gather new facts.",
                                                session_id=sid, tools=[], max_rounds=1)
            result["final_text"] = repaired.final_text
            result["rounds"] += repaired.rounds
            usage = dict(result.get("usage") or {})
            for name, value in (repaired.usage or {}).items():
                usage[name] = usage.get(name, 0) + value
            result["usage"] = usage
            result["new_input_after_pass"] = bool(getattr(repaired, "new_input_after_pass", False))
            if repaired.status != "completed":
                result.pop("payload", None)
                result["status"] = repaired.status
                result["error"] = "structured_output_repair_ended: " + str(repaired.status)
                return result
            if result["new_input_after_pass"]:
                result.pop("payload", None)
                return result
        else:
            break
    result["status"] = "failed"
    return result


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
    from power_loop.runtime.delegation import result_view
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    registry.register(SPAWN_AGENT_DEFINITION, FunctionRuntimeTool(
        _handle_spawn_agent, ToolRuntimeContract("sdk.subagent", background_allowed=False, allowed_roles=("main", "workflow_node")),
        policy=lambda arguments: ToolDispatchPolicy(native_admitted=True), formatter=result_view,
    ), overwrite=overwrite)


__all__ = [
    "SPAWN_AGENT_DEFINITION",
    "register_spawn_agent",
]
