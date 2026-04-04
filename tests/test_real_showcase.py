"""
power-loop framework showcase — real LLM integration tests.

From simple to complex, each level demonstrates a new capability:

  Level 1: Minimal — just call LLM, get response
  Level 2: Events — subscribe to streaming events for real-time output
  Level 3: Tools — register custom tools, let LLM decide to call them
  Level 4: Hooks — security policy (TOOL_BEFORE SKIP), message audit (MESSAGE_APPEND),
                    response enhancement (LLM_AFTER)
  Level 5: Combo — token budget guard + rate limiter + round tracker
  Level 6: Spawn agent — sub-agent with event bubbling, depth control

Usage:
    POWER_LOOP_RUN_REAL_SMOKE=1 PYTHONPATH=. pytest tests/test_real_showcase.py -s -v

    Or run directly:
    POWER_LOOP_RUN_REAL_SMOKE=1 PYTHONPATH=. python tests/test_real_showcase.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# ── LLM client ──
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

# ── power-loop public API ──
from power_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    HookContext,
    HookDirective,
    HookPoint,
    HookResult,
    SystemPromptBuilder,
    SystemPromptContext,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    register_spawn_agent,
)


# =====================================================================
# Helpers
# =====================================================================

def _creds_or_skip() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    if not (model and base_url and api_key):
        pytest.skip("Fill OPENAI_COMPAT_MODEL/BASE_URL/API_KEY in .env")
    if (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower() not in ("1", "true", "yes"):
        pytest.skip("Set POWER_LOOP_RUN_REAL_SMOKE=1 to run")
    return {"model": model, "base_url": base_url, "api_key": api_key}


def _load_env_or_exit() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    if not (model and base_url and api_key):
        print("SKIP: fill OPENAI_COMPAT_MODEL/BASE_URL/API_KEY in .env")
        sys.exit(0)
    if (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower() not in ("1", "true", "yes"):
        print("SKIP: set POWER_LOOP_RUN_REAL_SMOKE=1 to run")
        sys.exit(0)
    return {"model": model, "base_url": base_url, "api_key": api_key}


def _make_llm(creds: Dict[str, str]) -> OpenAICompatibleChatLLMService:
    return OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(
            base_url=creds["base_url"],
            api_key=creds["api_key"],
            model=creds["model"],
        )
    )


def _sep(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# =====================================================================
# Level 1: Minimal — just LLM, no tools, no hooks
# =====================================================================

async def level_1_minimal(llm: OpenAICompatibleChatLLMService):
    """
    The simplest possible usage of power-loop:
    - Create an AgentLoop with default config
    - Send a single user message
    - Get the response

    This demonstrates: AgentLoop, AgentLoopConfig, basic run()
    """
    _sep("Level 1: Minimal — just call LLM")

    config = AgentLoopConfig(
        system_prompt="You are a helpful assistant. Answer concisely in 1-2 sentences.",
        max_rounds=1,
        max_tokens=256,
        temperature=0.0,
    )

    loop = AgentLoop(llm=llm, config=config)

    result = await loop.run(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        session_id="level-1",
    )

    print(f"  Status : {result.status}")
    print(f"  Rounds : {result.rounds}")
    print(f"  Answer : {result.final_text}")

    assert result.status == "completed"
    assert result.rounds == 1
    assert "Paris" in result.final_text or "paris" in result.final_text.lower()
    print("  [PASS]")


# =====================================================================
# Level 2: Events — streaming + lifecycle observation
# =====================================================================

async def level_2_events(llm: OpenAICompatibleChatLLMService):
    """
    Subscribe to various events to observe the agent loop in real-time.

    Events are purely observational — they cannot affect execution.
    Think of them as "printf debugging" or telemetry hooks.

    This demonstrates: AgentEventBus, AgentEventType, event subscribers
    """
    _sep("Level 2: Events — streaming + lifecycle observation")

    bus = AgentEventBus(suppress_subscriber_errors=True)
    event_log: List[str] = []

    # 1. Streaming: print LLM output token-by-token
    def on_delta(e: AgentEvent):
        text = e.payload.get("text", "")
        if text:
            print(text, end="", flush=True)
    bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)

    # 2. Lifecycle: log session/round start/end
    bus.subscribe(AgentEventType.SESSION_STARTED,
                  lambda e: event_log.append(f"SESSION_STARTED({e.session_id})"))
    bus.subscribe(AgentEventType.SESSION_ENDED,
                  lambda e: event_log.append(f"SESSION_ENDED({e.payload.get('reason')})"))
    bus.subscribe(AgentEventType.ROUND_STARTED,
                  lambda e: event_log.append(f"ROUND_STARTED({e.round_index})"))
    bus.subscribe(AgentEventType.ROUND_COMPLETED,
                  lambda e: event_log.append(f"ROUND_COMPLETED({e.round_index})"))

    # 3. Usage: capture token usage
    usage_records: List[dict] = []
    def on_usage(e: AgentEvent):
        usage = e.payload.get("usage", {})
        usage_records.append(usage)
        prompt = usage.get("prompt_tokens") or usage.get("input")
        comp = usage.get("completion_tokens") or usage.get("output")
        print(f"\n  [usage] prompt={prompt} completion={comp}", flush=True)
    bus.subscribe(AgentEventType.USAGE_UPDATED, on_usage)

    config = AgentLoopConfig(
        system_prompt="You are a helpful assistant. Answer in 2-3 sentences.",
        max_rounds=2,
        max_tokens=512,
    )

    loop = AgentLoop(llm=llm, config=config, event_bus=bus)

    result = await loop.run(
        messages=[{"role": "user", "content": "Explain what a Python decorator is."}],
        session_id="level-2",
    )

    print(f"\n\n  Event log: {event_log}")
    print(f"  Usage records: {len(usage_records)}")

    assert "SESSION_STARTED(level-2)" in event_log
    assert any("SESSION_ENDED" in e for e in event_log)
    assert any("ROUND_STARTED" in e for e in event_log)
    assert len(usage_records) >= 1
    print("  [PASS]")


# =====================================================================
# Level 3: Tools — custom tools + LLM tool calling
# =====================================================================

async def level_3_tools(llm: OpenAICompatibleChatLLMService):
    """
    Register custom tools and let the LLM decide when to call them.

    The agent loop:
    1. Sends the user message + tool definitions to LLM
    2. LLM returns a tool_call → pipeline executes it via ToolRegistry
    3. Tool result is appended to history
    4. LLM is called again to produce a final answer

    This demonstrates: ToolDefinition, ToolRegistry, multi-round tool calling
    """
    _sep("Level 3: Tools — custom tools + LLM tool calling")

    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STREAM_DELTA,
                  lambda e: print(e.payload.get("text", ""), end="", flush=True))
    bus.subscribe(AgentEventType.TOOL_CALL_STARTED,
                  lambda e: print(f"\n  >>> [tool] {e.payload.get('name')}({e.payload.get('tool_input')})", flush=True))
    bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED,
                  lambda e: print(f"  <<< [tool] {e.payload.get('name')} -> {str(e.payload.get('output', ''))[:100]}", flush=True))

    # Define custom tools
    registry = ToolRegistry()

    # Tool 1: get_weather — returns fake weather data
    registry.register(
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city. Returns temperature and conditions.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            required_params=("city",),
        ),
        lambda city: json.dumps({
            "city": city,
            "temperature": 22,
            "unit": "celsius",
            "conditions": "sunny with light clouds",
            "humidity": 45,
        }),
    )

    # Tool 2: calculate — evaluates a math expression
    registry.register(
        ToolDefinition(
            name="calculate",
            description="Evaluate a mathematical expression and return the result.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '2 + 3 * 4'"},
                },
                "required": ["expression"],
            },
            required_params=("expression",),
        ),
        lambda expression: str(eval(expression)),  # noqa: S307 — demo only
    )

    config = AgentLoopConfig(
        system_prompt=(
            "You are a helpful assistant with access to tools.\n"
            "Use the get_weather tool to check weather, and calculate for math.\n"
            "Always use tools when the user asks about weather or math. Answer concisely."
        ),
        max_rounds=5,
        max_tokens=1024,
    )

    loop = AgentLoop(llm=llm, config=config, tool_registry=registry, event_bus=bus)

    result = await loop.run(
        messages=[{"role": "user", "content": "What's the weather in Tokyo? Also, what is 17 * 23 + 5?"}],
        session_id="level-3",
    )

    print(f"\n\n  Status: {result.status}, Rounds: {result.rounds}")
    print(f"  Final: {result.final_text[:200]}")

    assert result.status == "completed"
    assert result.rounds >= 2  # at least 1 tool round + 1 final
    # Check tool messages exist in history
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1, f"Expected tool messages, got {len(tool_msgs)}"
    print("  [PASS]")


# =====================================================================
# Level 4: Hooks — security policy, audit log, response enhancement
# =====================================================================

async def level_4_hooks(llm: OpenAICompatibleChatLLMService):
    """
    Hooks are the control-flow mechanism of power-loop. Unlike events (read-only),
    hooks can modify data and return directives that change execution.

    We demonstrate 3 hooks:

    1. TOOL_BEFORE + SKIP: Security policy that blocks dangerous tool calls
    2. MESSAGE_APPEND: Audit log that records every message to a list
    3. LLM_AFTER: Response enhancement that adds metadata

    This demonstrates: HookPoint, HookDirective, HookResult, AgentHooks
    """
    _sep("Level 4: Hooks — security, audit, enhancement")

    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STREAM_DELTA,
                  lambda e: print(e.payload.get("text", ""), end="", flush=True))
    bus.subscribe(AgentEventType.TOOL_CALL_STARTED,
                  lambda e: print(f"\n  >>> [tool] {e.payload.get('name')}", flush=True))
    bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED,
                  lambda e: print(f"  <<< [tool] {e.payload.get('name')} -> {str(e.payload.get('output', ''))[:80]}", flush=True))

    hooks = AgentHooks()
    audit_log: List[Dict[str, Any]] = []
    blocked_tools: List[str] = []

    # ── Hook 1: Security policy (TOOL_BEFORE → SKIP) ──
    # Block any tool call that contains "rm", "delete", or "drop"
    def security_policy(ctx: HookContext) -> HookResult:
        tool_name = ctx.values.get("tool_name", "")
        tool_args = ctx.values.get("tool_args", {})
        args_str = json.dumps(tool_args, ensure_ascii=False)

        dangerous_keywords = ["rm ", "rm -", "delete", "drop ", "rmdir", "unlink"]
        for kw in dangerous_keywords:
            if kw in args_str.lower():
                blocked_tools.append(f"{tool_name}({args_str[:60]})")
                ctx.values["output"] = f"BLOCKED by security policy: '{kw}' detected in arguments."
                print(f"\n  [SECURITY] BLOCKED: {tool_name} with '{kw}'", flush=True)
                return HookResult(context=ctx, directive=HookDirective.SKIP)

        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.TOOL_BEFORE, security_policy, order=0)

    # ── Hook 2: Audit log (MESSAGE_APPEND) ──
    # Record every message with timestamp
    def audit_hook(ctx: HookContext) -> HookContext:
        msg = ctx.values.get("message")
        if msg:
            audit_log.append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "round": ctx.values.get("round_index"),
                "role": msg.get("role", "?"),
                "content_preview": str(msg.get("content", ""))[:80],
                "has_tool_calls": bool(msg.get("tool_calls")),
            })
        return ctx

    hooks.register(HookPoint.MESSAGE_APPEND, audit_hook)

    # ── Hook 3: Session lifecycle logging ──
    def on_session_start(ctx: HookContext) -> HookContext:
        print(f"  [HOOK] Session starting, messages={len(ctx.values.get('messages', []))}", flush=True)
        return ctx

    def on_session_end(ctx: HookContext) -> HookContext:
        reason = ctx.values.get("reason", "?")
        print(f"  [HOOK] Session ending, reason={reason}", flush=True)
        return ctx

    hooks.register(HookPoint.SESSION_START, on_session_start)
    hooks.register(HookPoint.SESSION_END, on_session_end)

    # ── Tools: include bash + a safe tool ──
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="bash",
            description="Run a shell command and return its output.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                },
                "required": ["command"],
            },
            required_params=("command",),
        ),
        lambda command: os.popen(command).read().strip()[:2000],
    )

    registry.register(
        ToolDefinition(
            name="get_time",
            description="Get the current date and time.",
            input_schema={"type": "object", "properties": {}},
            required_params=(),
        ),
        lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    config = AgentLoopConfig(
        system_prompt=(
            "You are a helpful assistant with bash and get_time tools.\n"
            "The user may ask you to perform operations. Use tools as needed.\n"
            "Answer concisely."
        ),
        max_rounds=5,
        max_tokens=1024,
    )

    loop = AgentLoop(llm=llm, config=config, tool_registry=registry, event_bus=bus, hooks=hooks)

    # The task asks to both use a safe command AND a dangerous one
    result = await loop.run(
        messages=[{"role": "user", "content": (
            "Please do two things:\n"
            "1. Use get_time to tell me the current time\n"
            "2. Use bash to run: echo hello"
        )}],
        session_id="level-4",
    )

    print(f"\n\n  Status: {result.status}, Rounds: {result.rounds}")
    print(f"\n  Audit log ({len(audit_log)} entries):")
    for entry in audit_log:
        print(f"    [{entry['time']}] round={entry['round']} role={entry['role']} "
              f"tools={entry['has_tool_calls']} content={entry['content_preview'][:50]}")

    if blocked_tools:
        print(f"\n  Blocked tool calls: {blocked_tools}")

    assert result.status == "completed"
    assert len(audit_log) >= 2, f"Expected audit log entries, got {len(audit_log)}"
    print("  [PASS]")


# =====================================================================
# Level 5: Combo — budget guard + round tracker + tool rate limiter
# =====================================================================

async def level_5_combo(llm: OpenAICompatibleChatLLMService):
    """
    Combine multiple hooks and events to build a sophisticated control system:

    1. ROUND_START hook: token budget guard — if total tokens exceed threshold, BREAK
    2. ROUND_START hook: round progress tracker — prints "[Round N/M]" status
    3. TOOL_BEFORE hook: rate limiter — if >3 tool calls in 10s, inject delay warning
    4. LLM_AFTER hook: response length monitor — warn if response is too long
    5. Events: collect full telemetry for post-run analysis

    This demonstrates: multiple hooks on same HookPoint (ordered), combining hooks + events
    """
    _sep("Level 5: Combo — budget guard + round tracker + rate limiter")

    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STREAM_DELTA,
                  lambda e: print(e.payload.get("text", ""), end="", flush=True))

    hooks = AgentHooks()
    telemetry: Dict[str, Any] = {
        "rounds": [],
        "tool_calls": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "budget_break": False,
    }

    TOKEN_BUDGET = 50000  # generous budget for real LLM calls
    tool_call_timestamps: List[float] = []

    # ── Hook 1: Round progress tracker (order=0, runs first) ──
    def round_tracker(ctx: HookContext) -> HookContext:
        round_idx = ctx.values.get("round_index", 0)
        print(f"\n  [Round {round_idx + 1}] starting...", flush=True)
        telemetry["rounds"].append({"index": round_idx, "start_time": time.time()})
        return ctx

    hooks.register(HookPoint.ROUND_START, round_tracker, order=0)

    # ── Hook 2: Token budget guard (order=1, runs after tracker) ──
    def budget_guard(ctx: HookContext) -> HookResult:
        if telemetry["total_prompt_tokens"] > TOKEN_BUDGET:
            print(f"\n  [BUDGET] Token budget exceeded: {telemetry['total_prompt_tokens']} > {TOKEN_BUDGET}", flush=True)
            ctx.values["reason"] = "token_budget_exceeded"
            telemetry["budget_break"] = True
            return HookResult(context=ctx, directive=HookDirective.BREAK)
        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.ROUND_START, budget_guard, order=1)

    # ── Hook 3: Tool rate limiter (TOOL_BEFORE) ──
    def rate_limiter(ctx: HookContext) -> HookResult:
        now = time.time()
        tool_call_timestamps.append(now)

        # Count calls in last 10 seconds
        recent = [t for t in tool_call_timestamps if now - t < 10]
        tool_name = ctx.values.get("tool_name", "?")
        telemetry["tool_calls"].append({"name": tool_name, "time": now})

        if len(recent) > 10:
            print(f"\n  [RATE LIMIT] Too many tool calls ({len(recent)} in 10s), injecting warning", flush=True)
            ctx.values["output"] = (
                "Rate limit: too many tool calls in quick succession. "
                "Please slow down and batch your operations."
            )
            return HookResult(context=ctx, directive=HookDirective.SKIP)

        print(f"\n  [rate] tool={tool_name}, recent_calls={len(recent)}/10", flush=True)
        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.TOOL_BEFORE, rate_limiter, order=0)

    # ── Hook 4: Response length monitor (LLM_AFTER) ──
    from llm_client.interface import LLMResponse
    def response_monitor(ctx: HookContext) -> HookContext:
        output = ctx.values.get("output")
        if isinstance(output, LLMResponse):
            text = output.raw_text or ""
            if len(text) > 2000:
                print(f"\n  [MONITOR] Long response: {len(text)} chars", flush=True)
        return ctx

    hooks.register(HookPoint.LLM_AFTER, response_monitor)

    # ── Event: collect token usage for budget tracking ──
    def on_usage(e: AgentEvent):
        usage = e.payload.get("usage", {})
        prompt = usage.get("prompt_tokens") or usage.get("input") or 0
        comp = usage.get("completion_tokens") or usage.get("output") or 0
        telemetry["total_prompt_tokens"] += (prompt or 0)
        telemetry["total_completion_tokens"] += (comp or 0)
        print(f"\n  [usage] cumulative: prompt={telemetry['total_prompt_tokens']}, "
              f"completion={telemetry['total_completion_tokens']}", flush=True)

    bus.subscribe(AgentEventType.USAGE_UPDATED, on_usage)

    # ── Tools ──
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="search_knowledge",
            description="Search the knowledge base for information about a topic.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            required_params=("query",),
        ),
        lambda query: json.dumps({
            "results": [
                {"title": f"Article about {query}", "snippet": f"This is a comprehensive overview of {query}..."},
                {"title": f"Advanced {query} guide", "snippet": f"Deep dive into {query} techniques and best practices..."},
            ]
        }),
    )

    registry.register(
        ToolDefinition(
            name="save_note",
            description="Save a note or summary for later reference.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
            required_params=("title", "content"),
        ),
        lambda title, content: f"Note saved: '{title}' ({len(content)} chars)",
    )

    config = AgentLoopConfig(
        system_prompt=(
            "You are a research assistant. Use search_knowledge to find information, "
            "and save_note to save important findings.\n"
            "Answer the user's question based on your research. Be concise."
        ),
        max_rounds=5,
        max_tokens=1024,
    )

    loop = AgentLoop(llm=llm, config=config, tool_registry=registry, event_bus=bus, hooks=hooks)

    result = await loop.run(
        messages=[{"role": "user", "content": (
            "Research 'Python async programming' and save a brief note summarizing the key concepts."
        )}],
        session_id="level-5",
    )

    print(f"\n\n  Status: {result.status}, Rounds: {result.rounds}")
    print(f"  Telemetry:")
    print(f"    Rounds executed: {len(telemetry['rounds'])}")
    print(f"    Tool calls: {len(telemetry['tool_calls'])}")
    print(f"    Total prompt tokens: {telemetry['total_prompt_tokens']}")
    print(f"    Total completion tokens: {telemetry['total_completion_tokens']}")
    print(f"    Budget break triggered: {telemetry['budget_break']}")

    for tc in telemetry["tool_calls"]:
        print(f"    - {tc['name']}")

    assert result.status == "completed"
    assert len(telemetry["rounds"]) >= 1
    print("  [PASS]")


# =====================================================================
# Level 6: Spawn agent — sub-agent with event bubbling
# =====================================================================

async def level_6_spawn_agent(llm: OpenAICompatibleChatLLMService, model: str):
    """
    The crown jewel: sub-agent spawning.

    The main agent has a `spawn_agent` tool. When it decides to delegate,
    a new AgentLoop is created with:
    - Its own session (isolated ContextVar)
    - Its own event bus (with bubbling to parent)
    - Its own tool set (configurable via preset)
    - Depth control (prevents unbounded recursion)

    The sub-agent runs to completion and returns its final_text as the tool result.

    This demonstrates: register_spawn_agent, event bubbling, session isolation,
                       ContextVar, depth control, tool presets
    """
    _sep("Level 6: Spawn agent — sub-agent with event bubbling")

    bus = AgentEventBus(suppress_subscriber_errors=True)

    # ── Main agent events ──
    bus.subscribe(AgentEventType.STREAM_DELTA,
                  lambda e: (print(e.payload.get("text", ""), end="", flush=True)
                             if not e.payload.get("source", "").startswith("subagent:") else None))

    bus.subscribe(AgentEventType.TOOL_CALL_STARTED,
                  lambda e: print(f"\n  >>> [main] {e.payload.get('name')}", flush=True))
    bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED,
                  lambda e: print(f"  <<< [main] {e.payload.get('name')} done", flush=True))

    # ── Sub-agent lifecycle events ──
    def on_subagent_start(e: AgentEvent):
        p = e.payload
        print(f"\n  {'='*40}", flush=True)
        print(f"  SUBAGENT SPAWNED (depth={p.get('depth')}, preset={p.get('preset')})", flush=True)
        print(f"  Task: {p.get('task', '')[:80]}", flush=True)
        print(f"  {'='*40}", flush=True)

    def on_subagent_text(e: AgentEvent):
        p = e.payload
        status = p.get("status", "?")
        rounds = p.get("rounds", "?")
        text_preview = (p.get("final_text") or "")[:100]
        print(f"\n  {'='*40}", flush=True)
        print(f"  SUBAGENT COMPLETED (status={status}, rounds={rounds})", flush=True)
        print(f"  Result: {text_preview}...", flush=True)
        print(f"  {'='*40}", flush=True)

    def on_subagent_limit(e: AgentEvent):
        print(f"\n  [WARNING] Sub-agent hit round limit: {e.payload}", flush=True)

    bus.subscribe(AgentEventType.SUBAGENT_TASK_START, on_subagent_start)
    bus.subscribe(AgentEventType.SUBAGENT_TEXT, on_subagent_text)
    bus.subscribe(AgentEventType.SUBAGENT_LIMIT, on_subagent_limit)

    # ── Sub-agent streaming (bubbled events have source="subagent:...") ──
    def on_sub_stream(e: AgentEvent):
        source = e.payload.get("source", "")
        if source.startswith("subagent:"):
            text = e.payload.get("text", "")
            if text:
                print(f"    [sub] {text}", end="", flush=True)
    bus.subscribe(AgentEventType.STREAM_DELTA, on_sub_stream)

    # ── Hooks: log tool activity ──
    hooks = AgentHooks()

    tool_activity: List[Dict[str, Any]] = []

    def tool_logger(ctx: HookContext) -> HookContext:
        tool_activity.append({
            "tool": ctx.values.get("tool_name"),
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        return ctx

    hooks.register(HookPoint.TOOL_BEFORE, tool_logger)

    # ── Registry with spawn_agent ──
    registry = create_default_tool_registry(preset="core")
    register_spawn_agent(registry, llm, max_depth=2, bubble_events=True)

    # Build system prompt
    ctx = SystemPromptContext(
        model=model,
        workspace_dir=str(Path.cwd()),
        tool_names=[d.name for d in registry.definitions()],
    )
    prompt = SystemPromptBuilder().use("identity", "tool_guide", "paths").build(ctx)
    prompt += (
        "\n\n# Delegation\n"
        "You have a spawn_agent tool that creates independent sub-agents.\n"
        "Use spawn_agent with preset='explore' for read-only investigation tasks.\n"
        "The sub-agent has its own tools and runs in isolation.\n"
        "Prefer delegation for investigation tasks to keep your context clean."
    )

    config = AgentLoopConfig(
        system_prompt=prompt,
        max_rounds=6,
        max_tokens=2048,
        temperature=0.0,
    )

    loop = AgentLoop(llm=llm, config=config, tool_registry=registry, event_bus=bus, hooks=hooks)

    result = await loop.run(
        messages=[{"role": "user", "content": (
            "I want to understand the structure of the power_loop package. "
            "Please use spawn_agent (preset='explore') to investigate what files are in "
            "the power_loop/ directory and what the __init__.py exports. "
            "Then give me a brief summary."
        )}],
        session_id="level-6-main",
    )

    print(f"\n\n  Status: {result.status}, Rounds: {result.rounds}")
    print(f"  Tool activity log:")
    for entry in tool_activity:
        print(f"    [{entry['time']}] {entry['tool']}")

    # Check that spawn_agent was actually called
    spawn_calls = [t for t in tool_activity if t["tool"] == "spawn_agent"]
    print(f"\n  spawn_agent calls: {len(spawn_calls)}")

    assert result.status == "completed"
    print("  [PASS]")


# =====================================================================
# Bonus: Hook composition — demonstrate @phase-style SHORT_CIRCUIT
# =====================================================================

async def bonus_short_circuit_cache(llm: OpenAICompatibleChatLLMService):
    """
    Demonstrate LLM_BEFORE SHORT_CIRCUIT for caching.

    If we've seen the same question before, return the cached response
    without calling the LLM at all.

    This demonstrates: HookDirective.SHORT_CIRCUIT, LLMResponse construction
    """
    _sep("Bonus: SHORT_CIRCUIT cache — skip LLM on cache hit")

    from llm_client.interface import LLMResponse, LLMTokenUsage

    hooks = AgentHooks()
    cache: Dict[str, LLMResponse] = {}
    cache_hits = 0

    # ── LLM_AFTER: store response in cache ──
    def cache_store(ctx: HookContext) -> HookContext:
        messages = ctx.values.get("messages", [])
        output = ctx.values.get("output")
        if messages and isinstance(output, LLMResponse):
            # Use last user message as cache key
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user:
                key = str(last_user.get("content", ""))[:200]
                cache[key] = output
                print(f"\n  [cache] Stored response for: {key[:50]}...", flush=True)
        return ctx

    hooks.register(HookPoint.LLM_AFTER, cache_store)

    # ── LLM_BEFORE: check cache and SHORT_CIRCUIT if hit ──
    def cache_check(ctx: HookContext) -> HookResult:
        nonlocal cache_hits
        messages = ctx.values.get("messages", [])
        if messages:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user:
                key = str(last_user.get("content", ""))[:200]
                if key in cache:
                    cache_hits += 1
                    print(f"\n  [cache] HIT! Returning cached response (hit #{cache_hits})", flush=True)
                    ctx.values["output"] = cache[key]
                    return HookResult(context=ctx, directive=HookDirective.SHORT_CIRCUIT)
        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.LLM_BEFORE, cache_check)

    bus = AgentEventBus(suppress_subscriber_errors=True)
    bus.subscribe(AgentEventType.STREAM_DELTA,
                  lambda e: print(e.payload.get("text", ""), end="", flush=True))

    config = AgentLoopConfig(
        system_prompt="You are a helpful assistant. Answer concisely.",
        max_rounds=2,
        max_tokens=256,
    )

    # First call — cache miss, calls LLM
    print("  --- First call (cache miss) ---")
    loop1 = AgentLoop(llm=llm, config=config, event_bus=bus, hooks=hooks)
    result1 = await loop1.run(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        session_id="cache-1",
    )
    print(f"\n  Result 1: {result1.final_text}")

    # Second call — same question, cache hit, no LLM call
    print("\n  --- Second call (cache hit) ---")
    loop2 = AgentLoop(llm=llm, config=config, event_bus=bus, hooks=hooks)
    result2 = await loop2.run(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        session_id="cache-2",
    )
    print(f"\n  Result 2: {result2.final_text}")

    assert cache_hits >= 1, f"Expected at least 1 cache hit, got {cache_hits}"
    # Both should have the same answer
    assert result1.final_text == result2.final_text
    print(f"\n  Cache hits: {cache_hits}")
    print("  [PASS]")


# =====================================================================
# pytest entry points
# =====================================================================

def test_level_1_minimal(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_1_minimal(llm))


def test_level_2_events(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_2_events(llm))


def test_level_3_tools(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_3_tools(llm))


def test_level_4_hooks(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_4_hooks(llm))


def test_level_5_combo(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_5_combo(llm))


def test_level_6_spawn_agent(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(level_6_spawn_agent(llm, creds["model"]))


def test_bonus_short_circuit_cache(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    with capsys.disabled():
        asyncio.run(bonus_short_circuit_cache(llm))


# =====================================================================
# main — run all levels sequentially
# =====================================================================

async def main():
    creds = _load_env_or_exit()
    llm = _make_llm(creds)

    try:
        await level_1_minimal(llm)
        await level_2_events(llm)
        await level_3_tools(llm)
        await level_4_hooks(llm)
        await level_5_combo(llm)
        await level_6_spawn_agent(llm, creds["model"])
        await bonus_short_circuit_cache(llm)

        _sep("ALL LEVELS PASSED")
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
