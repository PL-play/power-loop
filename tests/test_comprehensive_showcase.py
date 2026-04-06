"""
Comprehensive showcase test for power-loop framework.

Demonstrates:
  1. Simple single-turn conversation (LLM only)
  2. Multi-round agent with streaming
  3. Custom tools with validation
  4. Hook integration (MESSAGE_APPEND, TOOL_CALL_START, etc.)
  5. Event subscriptions for real-time monitoring
  6. Spawn-agent pattern with delegation
  7. Metrics and token tracking
  8. Complex workflows combining all capabilities

Usage:
    POWER_LOOP_RUN_REAL_SMOKE=1 PYTHONPATH=. pytest tests/test_comprehensive_showcase.py -v

流式输出与 ``test_real_streaming_subagent.py`` 一致：
    真实 LLM 用例使用 **同步** ``def test_*``，在 ``with capsys.disabled():`` 内调用
    ``_configure_streaming_stdio()`` 并以 ``asyncio.run()`` 跑协程；避免 ``pytest-asyncio``
    与 stdout 捕获组合时仍表现为“结束时一次性输出”。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
from dotenv import load_dotenv

# ── LLM client ──────────────────────────────────────────────────────────
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService

# ── power-loop public API ───────────────────────────────────────────────
from power_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentEventBus,
    AgentEventType,
    AgentEvent,
    AgentHooks,
    HookContext,
    HookPoint,
    HookDirective,
    MessageAppendCtx,
    ToolAfterCtx,
    ToolBeforeCtx,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    register_spawn_agent,
)


# =====================================================================
# Configuration & Helpers
# =====================================================================


def _configure_streaming_stdio() -> None:
    """与 ``test_real_streaming_subagent`` 相同：降低 stdout 块缓冲，配合 ``capsys.disabled()`` 实时流式打印。"""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is None:
            continue
        try:
            reconf(line_buffering=True)
        except Exception:
            pass


def _creds_or_skip() -> Dict[str, str]:
    """Load LLM credentials from .env or skip test."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.example", override=False)
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    
    if not (model and base_url and api_key):
        pytest.skip(
            "Fill OPENAI_COMPAT_MODEL / OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY in power-loop/.env"
        )
    
    if (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower() not in ("1", "true", "yes", "y"):
        pytest.skip("Set POWER_LOOP_RUN_REAL_SMOKE=1 to run real-LLM tests.")
    
    return {"model": model, "base_url": base_url, "api_key": api_key}


def _make_llm(creds: Dict[str, str]) -> OpenAICompatibleChatLLMService:
    """Create LLM service from credentials."""
    return OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(
            base_url=creds["base_url"],
            api_key=creds["api_key"],
            model=creds["model"],
        )
    )


# =====================================================================
# Level 1: Simple Single-Turn Conversation
# =====================================================================


async def _async_level1_simple_conversation(creds: Dict[str, str]) -> None:
    """Async body for level 1; called via ``asyncio.run`` inside ``capsys.disabled()``."""
    llm = _make_llm(creds)
    try:
        print("\n" + "=" * 70)
        print("LEVEL 1: Simple Single-Turn Conversation")
        print("=" * 70)

        bus = AgentEventBus()
        output_text: list[str] = []

        def on_stream(event: AgentEvent) -> None:
            text = event.payload.get("text", "")
            if text:
                output_text.append(text)
                print(text, end="", flush=True)

        bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)

        config = AgentLoopConfig(
            system_prompt="You are a helpful assistant. Answer questions concisely.",
            max_rounds=1,
            max_tokens=512,
            temperature=0.7,
        )

        loop = AgentLoop(
            llm=llm,
            config=config,
            tool_registry=None,
            event_bus=bus,
        )

        result = await loop.run(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            session_id="simple-l1",
        )

        print(f"\n\n✓ Rounds: {result.rounds}, Status: {result.status}, Messages: {result.messages}")
        assert result.status == "completed"
        assert len(output_text) > 0
        print(f"✓ Received {len(output_text)} text chunks")
    finally:
        await llm.close()


def test_level1_simple_conversation(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Scenario: Simple single-turn conversation with LLM only.
    Shows: Basic AgentLoop usage without tools.
    """
    creds = _creds_or_skip()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_async_level1_simple_conversation(creds))


# =====================================================================
# Level 2: Multi-Round Agent with File Tools
# =====================================================================


async def _async_level2_multiturn_with_tools(creds: Dict[str, str]) -> None:
    llm = _make_llm(creds)
    try:
        print("\n" + "=" * 70)
        print("LEVEL 2: Multi-Round Agent with Tools")
        print("=" * 70)

        bus = AgentEventBus(suppress_subscriber_errors=True)
        metrics: Dict[str, int] = {
            "stream_chunks": 0,
            "tool_calls": 0,
            "tool_completions": 0,
        }

        def on_stream(event: AgentEvent) -> None:
            metrics["stream_chunks"] += 1
            text = event.payload.get("text", "")
            if text:
                print(text, end="", flush=True)

        def on_tool_call_started(event: AgentEvent) -> None:
            metrics["tool_calls"] += 1
            name = event.payload.get("name", "?")
            print(f"\n>>> [Tool] {name}", flush=True)

        def on_tool_call_completed(event: AgentEvent) -> None:
            metrics["tool_completions"] += 1
            name = event.payload.get("name", "?")
            output = str(event.payload.get("output", ""))[:100]
            print(f"<<< [Result] {output}...\n", flush=True)

        bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
        bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_tool_call_started)
        bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED, on_tool_call_completed)

        registry = create_default_tool_registry(preset="core")

        config = AgentLoopConfig(
            system_prompt=(
                "You are a code analyzer. Your task:\n"
                "1. Find all Python files in the current directory using glob\n"
                "2. Read __init__.py to understand exports\n"
                "3. Summarize the project structure"
            ),
            max_rounds=15,
            max_tokens=2048,
            temperature=0.0,
        )

        loop = AgentLoop(
            llm=llm,
            config=config,
            tool_registry=registry,
            event_bus=bus,
        )

        result = await loop.run(
            messages=[{
                "role": "user",
                "content": "Analyze the power_loop project structure using tools"
            }],
            session_id="multiturn-l2",
        )

        print(f"\n\n✓ Final Status: {result.status}")
        print(f"✓ Rounds: {result.rounds}")
        print(f"✓ Stream chunks: {metrics['stream_chunks']}")
        print(f"✓ Tool calls: {metrics['tool_calls']}")
        assert result.status == "completed"
        assert metrics["tool_calls"] >= 1
    finally:
        await llm.close()


def test_level2_multiturn_with_tools(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Scenario: Multi-round agent with built-in tools (glob, read_file).
    Shows: Tool invocation, streaming, event subscriptions.
    """
    creds = _creds_or_skip()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_async_level2_multiturn_with_tools(creds))


# =====================================================================
# Level 3: Custom Tools with Validation & Hooks
# =====================================================================


async def _async_level3_custom_tools_with_hooks(creds: Dict[str, str]) -> None:
    llm = _make_llm(creds)
    try:
        print("\n" + "=" * 70)
        print("LEVEL 3: Custom Tools + Hooks + Metrics")
        print("=" * 70)

        async def calculate_sum(*numbers: float) -> float:
            """Add multiple numbers together."""
            return sum(numbers)

        async def format_json(data: dict) -> str:
            """Pretty-print JSON data."""
            return json.dumps(data, indent=2, ensure_ascii=False)

        custom_registry = ToolRegistry()

        custom_registry.register(
            ToolDefinition(
                name="calculate_sum",
                description="Calculate the sum of multiple numbers",
                input_schema={
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of numbers to sum"
                        }
                    },
                    "required": ["numbers"],
                },
                required_params=("numbers",),
            ),
            calculate_sum,
        )

        custom_registry.register(
            ToolDefinition(
                name="format_json",
                description="Pretty-print JSON data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Data to format as JSON"
                        }
                    },
                    "required": ["data"],
                },
                required_params=("data",),
            ),
            format_json,
        )

        bus = AgentEventBus(suppress_subscriber_errors=True)
        hooks = AgentHooks()

        state: Dict[str, Any] = {
            "messages_count": 0,
            "tool_inputs_log": [],
            "total_tokens": 0,
            "hook_calls": 0,
        }

        def on_message_append(ctx: MessageAppendCtx) -> None:
            state["messages_count"] += 1
            msg = ctx.message
            if msg:
                role = msg.get("role", "?")
                content_len = len(str(msg.get("content", "")))
                print(f"  [Hook:MESSAGE_APPEND] #{state['messages_count']} role={role} len={content_len}")

        def on_tool_start(ctx: ToolBeforeCtx) -> None:
            state["hook_calls"] += 1
            print(f"  [Hook:TOOL_CALL_START] {ctx.tool_name} with input: {ctx.tool_args}")
            state["tool_inputs_log"].append({
                "tool": ctx.tool_name,
                "input": ctx.tool_args,
            })

        def on_tool_complete(ctx: ToolAfterCtx) -> None:
            state["hook_calls"] += 1

        hooks.register(HookPoint.MESSAGE_APPEND, on_message_append)
        hooks.register(HookPoint.TOOL_BEFORE, on_tool_start)
        hooks.register(HookPoint.TOOL_AFTER, on_tool_complete)

        def on_stream(event: AgentEvent) -> None:
            text = event.payload.get("text", "")
            if text:
                print(text, end="", flush=True)

        def on_usage(event: AgentEvent) -> None:
            usage = event.payload.get("usage", {})
            if usage:
                state["total_tokens"] += usage.get("total_tokens", 0)
                print(f"\n  [Usage] {usage}")

        bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
        bus.subscribe(AgentEventType.USAGE_UPDATED, on_usage)

        config = AgentLoopConfig(
            system_prompt=(
                "You are a math assistant. Use tools to:\n"
                "1. Calculate sum of numbers: 5, 10, 15, 20\n"
                "2. Format the result as JSON with fields 'sum' and 'count'"
            ),
            max_rounds=5,
            max_tokens=2048,
            temperature=0.0,
        )

        loop = AgentLoop(
            llm=llm,
            config=config,
            tool_registry=custom_registry,
            event_bus=bus,
            hooks=hooks,
        )

        result = await loop.run(
            messages=[{
                "role": "user",
                "content": "Calculate the sum of 5, 10, 15, 20 and format as JSON"
            }],
            session_id="custom-tools-l3",
        )

        print(f"\n\n✓ Rounds: {result.rounds}")
        print(f"✓ Messages processed: {state['messages_count']}")
        print(f"✓ Hook calls: {state['hook_calls']}")
        print(f"✓ Tool invocations: {len(state['tool_inputs_log'])}")
        print(f"✓ Total tokens used: {state['total_tokens']}")

        assert result.status == "completed"
        assert len(state["tool_inputs_log"]) >= 1
    finally:
        await llm.close()


def test_level3_custom_tools_with_hooks(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Scenario: Custom tools with input validation + hook integration.
    Shows: Tool registration, hook points, event filtering, metrics.
    """
    creds = _creds_or_skip()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_async_level3_custom_tools_with_hooks(creds))


# =====================================================================
# Level 4: Multi-Agent with Spawn-Agent
# =====================================================================


async def _async_level4_spawn_agent_delegation(creds: Dict[str, str]) -> None:
    llm = _make_llm(creds)
    try:
        print("\n" + "=" * 70)
        print("LEVEL 4: Multi-Agent with Spawn-Agent Delegation")
        print("=" * 70)

        main_bus = AgentEventBus(suppress_subscriber_errors=True)
        main_registry = create_default_tool_registry(preset="core")

        register_spawn_agent(main_registry, llm)

        def on_main_stream(event: AgentEvent) -> None:
            text = event.payload.get("text", "")
            if text:
                print(text, end="", flush=True)

        def on_main_tool(event: AgentEvent) -> None:
            name = event.payload.get("name", "?")
            if name == "spawn_agent":
                print(f"\n>>> [Spawn Subagent]", flush=True)

        main_bus.subscribe(AgentEventType.STREAM_DELTA, on_main_stream)
        main_bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_main_tool)

        main_config = AgentLoopConfig(
            system_prompt=(
                "You are a project coordinator. "
                "Check project structure briefly, then summarize."
            ),
            max_rounds=5,
            max_tokens=1024,
            temperature=0.0,
        )

        main_loop = AgentLoop(
            llm=llm,
            config=main_config,
            tool_registry=main_registry,
            event_bus=main_bus,
        )

        result = await main_loop.run(
            messages=[{
                "role": "user",
                "content": "Briefly check if README.md exists in the project."
            }],
            session_id="coordinator-l4",
        )

        print(f"\n\n✓ Rounds: {result.rounds}")
        print(f"✓ Status: {result.status}")
        assert result.status == "completed"
    finally:
        await llm.close()


def test_level4_spawn_agent_delegation(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Scenario: Main agent with spawn_agent tool for delegation.
    Shows: Agent composition, session isolation, nested contexts.
    """
    creds = _creds_or_skip()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_async_level4_spawn_agent_delegation(creds))


# =====================================================================
# Level 5: Advanced - Complex Workflow with State Management
# =====================================================================


async def _async_level5_advanced_workflow(creds: Dict[str, str]) -> None:
    llm = _make_llm(creds)
    try:
        print("\n" + "=" * 70)
        print("LEVEL 5: Advanced Complex Workflow")
        print("=" * 70)

        workflow_state: Dict[str, Any] = {
            "start_time": datetime.now(),
            "steps": [],
            "tool_calls": {},
            "decisions": [],
            "final_metrics": {},
        }

        async def log_decision(decision: str, reason: str) -> str:
            entry = {"decision": decision, "reason": reason, "time": datetime.now().isoformat()}
            workflow_state["decisions"].append(entry)
            return f"Logged: {decision}"

        async def get_metrics() -> dict:
            elapsed = (datetime.now() - workflow_state["start_time"]).total_seconds()
            return {
                "elapsed_seconds": elapsed,
                "steps_completed": len(workflow_state["steps"]),
                "decisions_made": len(workflow_state["decisions"]),
            }

        registry = create_default_tool_registry(preset="core")

        registry.register(
            ToolDefinition(
                name="log_decision",
                description="Log a decision point with reasoning",
                input_schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["decision", "reason"],
                },
                required_params=("decision", "reason"),
            ),
            log_decision,
        )

        registry.register(
            ToolDefinition(
                name="get_metrics",
                description="Get current workflow metrics",
                input_schema={"type": "object", "properties": {}},
                required_params=(),
            ),
            get_metrics,
        )

        bus = AgentEventBus(suppress_subscriber_errors=True)
        hooks = AgentHooks()

        def on_message_append(ctx: MessageAppendCtx) -> None:
            msg = ctx.message
            if msg and msg.get("role") == "assistant":
                workflow_state["steps"].append({
                    "type": "assistant_message",
                    "timestamp": datetime.now().isoformat(),
                })

        def on_tool_call_start(ctx: ToolBeforeCtx) -> None:
            if ctx.tool_name not in workflow_state["tool_calls"]:
                workflow_state["tool_calls"][ctx.tool_name] = 0
            workflow_state["tool_calls"][ctx.tool_name] += 1
            print(f"  [Decision Point] Using tool: {ctx.tool_name}")

        hooks.register(HookPoint.MESSAGE_APPEND, on_message_append)
        hooks.register(HookPoint.TOOL_BEFORE, on_tool_call_start)

        def on_stream(event: AgentEvent) -> None:
            text = event.payload.get("text", "")
            if text:
                print(text, end="", flush=True)

        def on_round_completed(event: AgentEvent) -> None:
            rid = event.round_index
            payload = event.payload or {}
            has_tools = payload.get("has_tools", False)
            print(f"\n  [Round {rid} Complete] tools_used={has_tools}")

        def on_session_ended(event: AgentEvent) -> None:
            elapsed = (datetime.now() - workflow_state["start_time"]).total_seconds()
            print(f"\n  [Session Ended] Total elapsed: {elapsed:.2f}s")
            workflow_state["final_metrics"]["elapsed_seconds"] = elapsed

        bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
        bus.subscribe(AgentEventType.ROUND_COMPLETED, on_round_completed)
        bus.subscribe(AgentEventType.SESSION_ENDED, on_session_ended)

        config = AgentLoopConfig(
            system_prompt=(
                "You are a workflow orchestrator. Execute this multi-step analysis:\n"
                "\n"
                "Step 1: Use get_metrics to check initial state\n"
                "Step 2: Use log_decision to record your analysis approach\n"
                "Step 3: Use glob to find Python files\n"
                "Step 4: Use log_decision for findings\n"
                "Step 5: Use get_metrics to show progress\n"
                "Step 6: Summarize the workflow execution\n"
                "\n"
                "Use tools strategically to track progress."
            ),
            max_rounds=6,
            max_tokens=2048,
            temperature=0.0,
        )

        loop = AgentLoop(
            llm=llm,
            config=config,
            tool_registry=registry,
            event_bus=bus,
            hooks=hooks,
        )

        result = await loop.run(
            messages=[{
                "role": "user",
                "content": "Execute the workflow analysis with tool tracking"
            }],
            session_id="advanced-l5",
        )

        print(f"\n\n" + "=" * 50)
        print("WORKFLOW SUMMARY")
        print("=" * 50)
        print(f"✓ Final Status: {result.status}")
        print(f"✓ Rounds Used: {result.rounds}")
        print(f"✓ Steps Completed: {len(workflow_state['steps'])}")
        print(f"✓ Decisions Recorded: {len(workflow_state['decisions'])}")
        print(f"✓ Tool Invocations: {workflow_state['tool_calls']}")
        print(f"✓ Elapsed Time: {workflow_state['final_metrics'].get('elapsed_seconds', 0):.2f}s")

        if workflow_state["decisions"]:
            print("\nDecisions Made:")
            for i, d in enumerate(workflow_state["decisions"], 1):
                print(f"  {i}. {d['decision']}: {d['reason']}")

        assert result.status == "completed"
        assert len(workflow_state["steps"]) >= 1
    finally:
        await llm.close()


def test_level5_advanced_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Scenario: Complex workflow combining multiple tools, hooks, events, token reporting.
    Shows: Framework flexibility and composability.
    """
    creds = _creds_or_skip()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_async_level5_advanced_workflow(creds))


def _creds_or_exit() -> Dict[str, str] | None:
    """For ``python tests/test_comprehensive_showcase.py``; returns None if not configured."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.example", override=False)
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    if not (model and base_url and api_key):
        return None
    if (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower() not in ("1", "true", "yes", "y"):
        return None
    return {"model": model, "base_url": base_url, "api_key": api_key}


# =====================================================================
# Test Execution
# =====================================================================

if __name__ == "__main__":
    creds = _creds_or_exit()
    if not creds:
        print("SKIP: set credentials in power-loop/.env and POWER_LOOP_RUN_REAL_SMOKE=1", file=sys.stderr)
        sys.exit(0)
    _configure_streaming_stdio()
    asyncio.run(_async_level1_simple_conversation(creds))
    print("\n" + "=" * 70)
    print("✓ ALL SHOWCASES COMPLETED SUCCESSFULLY")
    print("=" * 70)
