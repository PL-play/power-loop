from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from dotenv import load_dotenv

from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

from power_loop.agent.loop import AgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.events import AgentEventType
from power_loop.contracts.hooks import HookContext, HookPoint
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.tools.registry import ToolRegistry


def _load_llm_env() -> None:
    power_loop_dir = Path(__file__).resolve().parents[1]
    # Prefer `.env`, but allow filling `.env.example` directly.
    load_dotenv(power_loop_dir / ".env", override=False)
    load_dotenv(power_loop_dir / ".env.example", override=False)


def _require_env() -> Dict[str, str]:
    _load_llm_env()
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    if not (model and base_url and api_key):
        pytest.skip(
            "Real smoke test skipped: please fill OPENAI_COMPAT_MODEL/BASE_URL/API_KEY in power-loop/.env "
            "or power-loop/.env.example (lines 1-8)."
        )
    run_flag = (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower()
    if run_flag not in {"1", "true", "yes", "y"}:
        pytest.skip("Real smoke test skipped: set POWER_LOOP_RUN_REAL_SMOKE=1 to actually call the LLM.")
    return {"model": model, "base_url": base_url, "api_key": api_key}


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"...(truncated {len(s) - n} chars)"


def _print_event(event: Any) -> None:
    payload = getattr(event, "payload", {}) or {}
    # Avoid huge spam for streaming deltas.
    if "text" in payload:
        payload_str = f"text={_truncate(str(payload.get('text', '')), 120)}"
    elif "reason" in payload:
        payload_str = f"reason={payload.get('reason')}"
    elif payload:
        payload_str = _truncate(str(payload), 300)
    else:
        payload_str = "{}"

    print(
        f"[event] {event.type} "
        f"(session_id={event.session_id}, round={event.round_index}, stream_id={event.stream_id}) "
        f"payload={payload_str}"
    )


def _print_hook(hook_point: HookPoint, ctx: HookContext) -> None:
    values = getattr(ctx, "values", {}) or {}
    # Keep prints short; just show keys and a small excerpt.
    excerpt = ""
    if isinstance(values, dict):
        if "messages" in values and isinstance(values["messages"], list):
            excerpt = f"messages_last={values['messages'][-1] if values['messages'] else None}"
            excerpt = _truncate(str(excerpt), 200)
        elif "tool_output" in values:
            excerpt = _truncate(str(values.get("tool_output")), 120)
        elif "response" in values:
            excerpt = _truncate(str(values.get("response")), 120)
        else:
            excerpt = _truncate(str(values), 200)
    print(f"[hook] {hook_point} values={excerpt}")


def _build_tools() -> ToolRegistry:
    # Real smoke test safety:
    # - Prefer the model to not call tools.
    # - Still register a harmless `echo_tool` so that accidental tool calls won't break the loop.
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_tool",
            description="Harmless echo tool for real smoke testing.",
            required_params=("text",),
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        ),
        lambda text: f"echo:{text}",
    )
    return registry


def _build_event_bus() -> AgentEventBus:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    event_types = [
        AgentEventType.SESSION_STARTED,
        AgentEventType.SESSION_ENDED,
        AgentEventType.ROUND_STARTED,
        AgentEventType.ROUND_COMPLETED,
        AgentEventType.ROUND_TOOLS_PRESENT,
        AgentEventType.STREAM_STARTED,
        AgentEventType.STREAM_DELTA,
        AgentEventType.STREAM_THINK_DELTA,
        AgentEventType.STREAM_COMPLETED,
        AgentEventType.TOOL_CALL_STARTED,
        AgentEventType.TOOL_CALL_COMPLETED,
        AgentEventType.TOOL_CALL_FAILED,
        AgentEventType.STATUS_CHANGED,
        AgentEventType.USAGE_UPDATED,
        AgentEventType.USER_NOTIFICATION,
        AgentEventType.AGENT_ERROR,
        AgentEventType.SYSTEM_LOG,
        AgentEventType.SUBAGENT_TASK_START,
        AgentEventType.SUBAGENT_TEXT,
        AgentEventType.SUBAGENT_LIMIT,
    ]

    for et in event_types:
        bus.subscribe(et, _print_event)

    return bus


def _build_hooks() -> AgentHooks:
    hooks = AgentHooks()

    # Register all supported hook points with print-only handlers.
    def _make_handler(hp: HookPoint):
        def _handler(ctx: HookContext) -> HookContext:
            print(f"[hook] {hp}")
            _print_hook(hp, ctx)
            return ctx

        return _handler

    hooks.register(HookPoint.SESSION_START, _make_handler(HookPoint.SESSION_START))
    hooks.register(HookPoint.SESSION_END, _make_handler(HookPoint.SESSION_END))
    hooks.register(HookPoint.ROUND_START, _make_handler(HookPoint.ROUND_START))
    hooks.register(HookPoint.ROUND_END, _make_handler(HookPoint.ROUND_END))
    hooks.register(HookPoint.LLM_BEFORE, _make_handler(HookPoint.LLM_BEFORE))
    hooks.register(HookPoint.LLM_AFTER, _make_handler(HookPoint.LLM_AFTER))
    hooks.register(HookPoint.TOOLS_BATCH_BEFORE, _make_handler(HookPoint.TOOLS_BATCH_BEFORE))
    hooks.register(HookPoint.TOOLS_BATCH_AFTER, _make_handler(HookPoint.TOOLS_BATCH_AFTER))
    hooks.register(HookPoint.TOOL_BEFORE, _make_handler(HookPoint.TOOL_BEFORE))
    hooks.register(HookPoint.TOOL_AFTER, _make_handler(HookPoint.TOOL_AFTER))

    return hooks


def test_real_agent_loop_smoke_print_only() -> None:
    # This is an integration smoke test intended for manual runs.
    # It will be auto-skipped if the real LLM env variables are not configured.
    creds = _require_env()

    llm_cfg = OpenAICompatibleChatConfig(
        base_url=creds["base_url"],
        api_key=creds["api_key"],
        model=creds["model"]
    )
    llm = OpenAICompatibleChatLLMService(llm_cfg)

    event_bus = _build_event_bus()
    hooks = _build_hooks()
    tool_registry = _build_tools()

    system_prompt = (
        "你在执行 power-loop 的真实 smoke 测试。"
        "请严格遵循：不要调用任何工具，直接用一句话回答用户问题。"
        "一句话必须简短明确。"
    )
    agent_cfg = AgentLoopConfig(max_rounds=2, max_tokens=8000, temperature=0.0, system_prompt=system_prompt)
    loop = AgentLoop(llm=llm, config=agent_cfg, tool_registry=tool_registry, event_bus=event_bus, hooks=hooks)

    async def _run() -> Any:
        try:
            print("[smoke] start real agent loop")
            result = await loop.run(
                messages=[{"role": "user", "content": "power-loop real smoke test：你能否正常回答问题？"}],
                session_id="real-smoke",
            )
            print("[smoke] done:", result.status, "final_text=", _truncate(result.final_text, 400))
            return result
        finally:
            await llm.close()

    result = asyncio.run(_run())
    assert result.status == "completed"
    assert str(result.final_text).strip()

