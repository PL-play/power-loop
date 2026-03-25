"""
Real-world integration test demonstrating:
  1. contextvars-based session isolation (two sessions share nothing)
  2. Streaming output with real-time printing
  3. Subagent pattern built from power-loop primitives

Usage:
    # Set env or fill .env first, then:
    POWER_LOOP_RUN_REAL_SMOKE=1 PYTHONPATH=. python tests/test_real_streaming_subagent.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import json
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
    SystemPromptBuilder,
    SystemPromptContext,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
)
from power_loop.core.agent_context import (
    get_event_bus,
    get_hooks,
    get_ctx,
    get_session_id,
)


# =====================================================================
# helpers
# =====================================================================

def _load_env() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.example", override=False)
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

def _creds_or_skip() -> Dict[str, str]:
    """Same rules as ``_load_env()`` but for pytest (skip instead of sys.exit)."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.example", override=False)
    model = (os.getenv("OPENAI_COMPAT_MODEL") or "").strip()
    base_url = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_COMPAT_API_KEY") or "").strip()
    if not (model and base_url and api_key):
        pytest.skip(
            "Fill OPENAI_COMPAT_MODEL / OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY in power-loop/.env "
            "(or .env.example)."
        )
    if (os.getenv("POWER_LOOP_RUN_REAL_SMOKE") or "").strip().lower() not in ("1", "true", "yes", "y"):
        pytest.skip("Set POWER_LOOP_RUN_REAL_SMOKE=1 to run real-LLM parts of this file.")
    return {"model": model, "base_url": base_url, "api_key": api_key}

def _configure_streaming_stdio() -> None:
    """尽量降低 stdout/stderr 块缓冲，使 ``print(..., end='', flush=True)`` 能尽快出现在终端。
    说明：pytest 默认会 *捕获* stdout，捕获模式下无论 flush 与否都要等用例结束才一次性展示。
    对流式用例请配合 ``capsys.disabled()``（见 Part2/3 测试）或命令行 ``pytest -s``。
    """
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is None:
            continue
        try:
            reconf(line_buffering=True)
        except Exception:
            # e.g. io.UnsupportedOperation on non-TTY / wrapped streams
            pass

def _make_llm(creds: Dict[str, str]) -> OpenAICompatibleChatLLMService:
    return OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(
            base_url=creds["base_url"],
            api_key=creds["api_key"],
            model=creds["model"],
        )
    )


# =====================================================================
# Part 1: contextvars walkthrough
# =====================================================================

async def demo_contextvars_isolation():
    """
    演示 contextvars 隔离。

    核心机制说明
    -----------
    Python 的 contextvars.ContextVar 为每个 asyncio Task / 线程提供独立的存储空间。
    power-loop 在 core/agent_context.py 中定义了 4 个 ContextVar：

        _current_event_bus   —— 当前 session 的事件总线
        _current_hooks       —— 当前 session 的 hook 管理器
        _current_ctx         —— 当前 session 的 ContextManager（token 计数、todo 等）
        _current_session_id  —— 当前 session 的 ID

    AgentRunner.session_async() 是一个 async context manager，进入时：
        tok_bus  = set_event_bus(self.event_bus)    # 把 bus 写入当前协程的 ContextVar
        tok_hooks = set_hooks(self.hooks)
        tok_ctx  = set_ctx(ContextManager(role="main"))
        tok_sid  = set_session_id(session_id)

    退出时用 token reset 恢复原值：
        reset_event_bus(tok_bus)
        reset_hooks(tok_hooks)
        ...

    这意味着：
      - 同一个进程里可以并发运行多个 session
      - 每个 session 里调用 get_event_bus() / get_ctx() 拿到的是自己的实例
      - session 结束后自动恢复，不会泄漏到外部
    """
    print("\n" + "=" * 70)
    print("PART 1: contextvars isolation demo")
    print("=" * 70)

    # 创建两个独立的 event bus，各自收集事件
    events_a: list[str] = []
    events_b: list[str] = []

    bus_a = AgentEventBus()
    bus_a.subscribe(AgentEventType.SYSTEM_LOG, lambda e: events_a.append(e.payload.get("msg", "")))

    bus_b = AgentEventBus()
    bus_b.subscribe(AgentEventType.SYSTEM_LOG, lambda e: events_b.append(e.payload.get("msg", "")))

    from power_loop.core.runner import AgentRunner

    runner_a = AgentRunner(event_bus=bus_a)
    runner_b = AgentRunner(event_bus=bus_b)

    async def session_work(runner: AgentRunner, sid: str, label: str):
        async with runner.session_async(session_id=sid):
            # 在 session 内部，get_session_id() / get_event_bus() 返回各自的实例
            assert get_session_id() == sid
            bus = get_event_bus()
            ctx = get_ctx()

            # 发一个事件 —— 只有自己的 bus 能收到
            bus.publish(AgentEvent(
                type=AgentEventType.SYSTEM_LOG,
                payload={"msg": f"hello from {label}"},
                session_id=sid,
            ))

            # ctx 也是隔离的
            ctx.api_calls += 1
            print(f"  [{label}] session_id={get_session_id()}, ctx.api_calls={ctx.api_calls}")

            # 模拟一点延迟，让两个 session 交错执行
            await asyncio.sleep(0.01)

            bus.publish(AgentEvent(
                type=AgentEventType.SYSTEM_LOG,
                payload={"msg": f"bye from {label}"},
                session_id=sid,
            ))

    # 并发运行两个 session
    await asyncio.gather(
        session_work(runner_a, "session-AAA", "A"),
        session_work(runner_b, "session-BBB", "B"),
    )

    # session 外部，get_session_id() 恢复为 None
    assert get_session_id() is None

    print(f"  Bus A received: {events_a}")
    print(f"  Bus B received: {events_b}")
    assert events_a == ["hello from A", "bye from A"], f"Bus A leaked: {events_a}"
    assert events_b == ["hello from B", "bye from B"], f"Bus B leaked: {events_b}"
    print("  ✓ Two sessions ran concurrently, zero cross-talk.")


# =====================================================================
# Part 2: streaming agent loop with real LLM
# =====================================================================

async def demo_streaming_agent(llm: OpenAICompatibleChatLLMService, model: str):
    """
    流式打印 agent 的输出。

    streaming 实现原理
    -----------------
    1. llm.complete() 内部实际走 streaming（OpenAI SDK 的 stream=True）
    2. 每收到一个 chunk，调用 on_chunk_delta_text(text) 回调
    3. agent_loop_async() 里把这个回调挂到 STREAM_DELTA 事件上
    4. 我们订阅 STREAM_DELTA 事件就能实时拿到文本片段

    所以这里只需要给 event_bus 注册一个 subscriber 即可。
    """
    print("\n" + "=" * 70)
    print("PART 2: streaming agent loop (real LLM, multi-round with tools)")
    print("=" * 70)

    bus = AgentEventBus(suppress_subscriber_errors=True)

    # 流式打印 —— 订阅 STREAM_DELTA
    def on_stream_delta(event: AgentEvent):
        text = event.payload.get("text", "")
        if text:
            print(text, end="", flush=True)

    bus.subscribe(AgentEventType.STREAM_DELTA, on_stream_delta)

    # 流式打印 —— 订阅 STREAM_THINK_DELTA（推理/思考内容）
    _current_think_round: int | None = None
    _think_started = False
    def on_stream_think(event: AgentEvent):
        text = event.payload.get("text", "")
        if not text:
            return
        rid = event.round_index
        nonlocal _current_think_round, _think_started
        # 新的一轮 think：前面加一个标记，避免和普通输出混在一起。
        if rid != _current_think_round:
            _current_think_round = rid
            _think_started = True
            print(f"\n[THINK r{rid}] ", end="", flush=True)
        elif not _think_started:
            _think_started = True
            print("\n[THINK] ", end="", flush=True)
        print(text, end="", flush=True)
    bus.subscribe(AgentEventType.STREAM_THINK_DELTA, on_stream_think)

    # 工具调用日志
    def on_tool_start(event: AgentEvent):
        name = event.payload.get("name", "?")
        args = event.payload.get("tool_input", {})
        # 截断长参数
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 120:
            args_str = args_str[:120] + "..."
        print(f"\n>>> [tool call] {name}({args_str})", flush=True)

    def on_tool_done(event: AgentEvent):
        name = event.payload.get("name", "?")
        output = str(event.payload.get("output", ""))
        if len(output) > 200:
            output = output[:200] + "..."
        print(f"<<< [tool result] {name} -> {output}\n", flush=True)

    bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_tool_start)
    bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED, on_tool_done)

    # Session lifecycle markers
    def on_session_started(e: AgentEvent) -> None:
        print(
            f"\n\n{'=' * 30} SESSION_STARTED session_id={e.session_id} {'=' * 30}\n",
            flush=True,
        )
    def on_session_ended(e: AgentEvent) -> None:
        reason = (e.payload or {}).get("reason", "")
        print(
            f"\n\n{'=' * 30} SESSION_ENDED session_id={e.session_id} reason={reason} {'=' * 30}\n",
            flush=True,
        )
    bus.subscribe(AgentEventType.SESSION_STARTED, on_session_started)
    bus.subscribe(AgentEventType.SESSION_ENDED, on_session_ended)
    # Round lifecycle markers (clear separators between rounds)
    def on_round_started(e: AgentEvent) -> None:
        idx = e.round_index
        print(f"\n\n{'-' * 26} ROUND_STARTED round_index={idx} {'-' * 26}\n", flush=True)
    def on_round_completed(e: AgentEvent) -> None:
        idx = e.round_index
        payload = e.payload or {}
        has_tools = payload.get("has_tools")
        used_todo = payload.get("used_todo")
        print(
            f"\n\n{'-' * 26} ROUND_COMPLETED round_index={idx} has_tools={has_tools} used_todo={used_todo} {'-' * 26}\n",
            flush=True,
        )
    bus.subscribe(AgentEventType.ROUND_STARTED, on_round_started)
    bus.subscribe(AgentEventType.ROUND_COMPLETED, on_round_completed)

    def _on_status(e: AgentEvent) -> None:
        kind = e.payload.get("kind", "?")
        print(f"\n--- STATUS_CHANGED kind={kind} payload={e.payload} ---", flush=True)
    bus.subscribe(AgentEventType.STATUS_CHANGED, _on_status)

     # Token 用量：USAGE_UPDATED 里同时有「单次 completion」与「本会话累计」
    _prev_session_prompt_total: int | None = None
    def on_usage_updated(e: AgentEvent) -> None:
        nonlocal _prev_session_prompt_total
        usage = e.payload.get("usage") or {}

        print(
            f"\n--- USAGE_UPDATED round_index={e.round_index!r} ---\n"
            f"  {usage}\n",
            flush=True,
        )
    bus.subscribe(AgentEventType.USAGE_UPDATED, on_usage_updated)

    # 使用 core 工具预设（bash, read_file, write_file, edit_file, glob, grep, ...）
    registry = create_default_tool_registry(preset="core")

    # 构建带 runtime context 的系统提示
    ctx = SystemPromptContext(
        model=model,
        workspace_dir=str(Path.cwd()),
        tool_names=[d.name for d in registry.definitions()],
    )
    prompt = SystemPromptBuilder().use("identity", "tool_guide", "paths").build(ctx)

    config = AgentLoopConfig(
        system_prompt=prompt,
        max_rounds=10,
        max_tokens=4096,
        temperature=0.0,
    )
    loop = AgentLoop(llm=llm, config=config, tool_registry=registry, event_bus=bus)

    task = (
        "请帮我查看当前目录下有哪些 Python 文件（用 glob），然后用 read_file 读取 power_loop/__init__.py 的前 20 行，"
        "最后总结 power-loop 这个库导出了哪些核心类。"
    )

    result = await loop.run(
        messages=[{"role": "user", "content": task}],
        session_id="streaming-demo",
    )

    print(f"\n\n--- Result: status={result.status}, rounds={result.rounds} ---")
    return result


# =====================================================================
# Part 3: subagent pattern using power-loop primitives
# =====================================================================

async def demo_subagent_pattern(llm: OpenAICompatibleChatLLMService, model: str):
    """
    用 power-loop 原语实现 subagent 模式。

    设计思路
    --------
    zero-code 里 subagent 是一个内置工具：agent 调用 sub_agent(prompt, mode)，
    然后框架内部递归调用 _run_subagent_async()。

    power-loop 作为 lib，不内置 subagent。但用户可以：

    1. 定义一个自定义工具 "sub_agent"
    2. 工具的 handler 内部创建一个新的 AgentLoop 并 await 它
    3. 新的 AgentLoop 使用不同的 tool_registry（比如 explore 预设）和不同的系统提示
    4. 每个 AgentLoop.run() 都经过 AgentRunner.session_async()，
       所以内层 session 的 event_bus / hooks / ctx 全部隔离

    这就是 contextvars 的价值 —— 不需要手动传递 session 状态，
    也不需要担心 subagent 的 token 计数污染 parent。
    """
    print("\n" + "=" * 70)
    print("PART 3: subagent pattern (nested AgentLoop)")
    print("=" * 70)

    # ── 主 agent 的 event bus ──
    main_bus = AgentEventBus(suppress_subscriber_errors=True)
    main_bus.subscribe(AgentEventType.STREAM_DELTA, lambda e: print(e.payload.get("text", ""), end="", flush=True))
    main_bus.subscribe(AgentEventType.TOOL_CALL_STARTED, lambda e: print(f"\n>>> [main tool] {e.payload.get('name', '?')}", flush=True))
    main_bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED, lambda e: print(f"<<< [main tool done] {e.payload.get('name', '?')}\n", flush=True))

    # ── subagent 的 event bus（独立的！）──
    sub_bus = AgentEventBus(suppress_subscriber_errors=True)
    sub_bus.subscribe(AgentEventType.STREAM_DELTA, lambda e: print(f"  🔍 {e.payload.get('text', '')}", end="", flush=True))
    sub_bus.subscribe(AgentEventType.TOOL_CALL_STARTED, lambda e: print(f"\n  >>> [sub tool] {e.payload.get('name', '?')}", flush=True))
    sub_bus.subscribe(AgentEventType.TOOL_CALL_COMPLETED, lambda e: print(f"  <<< [sub tool done] {e.payload.get('name', '?')}\n", flush=True))

    # ── subagent handler ──
    # 这个函数就是 sub_agent 工具的实现
    async def run_subagent(prompt: str, mode: str = "explore") -> str:
        """Spawn a child AgentLoop with its own session."""
        print(f"\n  ┌── Subagent spawned (mode={mode}) ──")
        print(f"  │ Task: {prompt[:80]}...")

        # 根据 mode 选择不同的工具和提示
        if mode == "explore":
            sub_registry = create_default_tool_registry(preset="explore")
            sub_prompt_text = "You are a read-only exploration subagent. Search and read files only."
        else:
            sub_registry = create_default_tool_registry(preset="core")
            sub_prompt_text = "You are a coding subagent. Complete the task autonomously."

        sub_ctx = SystemPromptContext(
            model=model,
            workspace_dir=str(Path.cwd()),
            tool_names=[d.name for d in sub_registry.definitions()],
        )
        sub_system_prompt = (
            SystemPromptBuilder(use_defaults=False)
            .add("identity", sub_prompt_text)
            .add("tool_guide", lambda c: SystemPromptBuilder().build(c).split("# Tool Usage")[-1].split("# Paths")[0] if "# Tool Usage" in SystemPromptBuilder().build(c) else "")
            .add("paths", lambda c: f"Workspace: {c.workspace_dir}")
            .build(sub_ctx)
        )

        sub_config = AgentLoopConfig(
            system_prompt=sub_system_prompt,
            max_rounds=5,
            max_tokens=4096,
            temperature=0.0,
        )

        # 关键：sub_bus 是独立的 event bus！
        sub_loop = AgentLoop(
            llm=llm,
            config=sub_config,
            tool_registry=sub_registry,
            event_bus=sub_bus,
        )

        sub_result = await sub_loop.run(
            messages=[{"role": "user", "content": prompt}],
            session_id="subagent-explore",
        )

        print(f"\n  └── Subagent done (rounds={sub_result.rounds}, status={sub_result.status}) ──\n")
        return sub_result.final_text or "(no output)"

    # ── 注册 sub_agent 工具到主 registry ──
    main_registry = create_default_tool_registry(preset="core")
    main_registry.register(
        ToolDefinition(
            name="sub_agent",
            description=(
                "Delegate a task to a child agent with fresh context. "
                "Use mode='explore' for read-only investigation, mode='execute' for tasks that modify files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description for the subagent"},
                    "mode": {"type": "string", "enum": ["explore", "execute"], "description": "Agent mode"},
                },
                "required": ["prompt"],
            },
            required_params=("prompt",),
        ),
        lambda prompt, mode="explore": asyncio.get_event_loop().run_until_complete(run_subagent(prompt, mode))
        if not asyncio.get_event_loop().is_running()
        else run_subagent(prompt, mode),  # async handler — registry.invoke_async will await it
    )

    # ── 主 agent 系统提示 ──
    main_ctx = SystemPromptContext(
        model=model,
        workspace_dir=str(Path.cwd()),
        tool_names=[d.name for d in main_registry.definitions()],
    )
    main_prompt = (
        SystemPromptBuilder()
        .add("delegation", (
            "# Delegation\n"
            "Use sub_agent(prompt, mode) to delegate tasks:\n"
            "- mode='explore': read-only investigation (glob, grep, read_file)\n"
            "- mode='execute': can modify files\n"
            "Prefer delegation for investigation tasks."
        ), after="workflow")
        .build(main_ctx)
    )

    main_config = AgentLoopConfig(
        system_prompt=main_prompt,
        max_rounds=6,
        max_tokens=4096,
        temperature=0.0,
    )

    main_loop = AgentLoop(
        llm=llm,
        config=main_config,
        tool_registry=main_registry,
        event_bus=main_bus,
    )

    task = (
        "我想了解 power_loop/tools/ 目录下都有哪些文件，以及 registry.py 的核心 API。"
        "请使用 sub_agent 来调查，然后给我一个总结。"
    )

    result = await main_loop.run(
        messages=[{"role": "user", "content": task}],
        session_id="main-with-subagent",
    )

    print(f"\n--- Main agent result: status={result.status}, rounds={result.rounds} ---")
    return result


# =====================================================================
# main
# =====================================================================

async def main():
    creds = _load_env()
    llm = _make_llm(creds)

    try:
        # Part 1: 纯 contextvars 隔离演示（不需要 LLM）
        await demo_contextvars_isolation()

        # Part 2: 流式 agent loop
        await demo_streaming_agent(llm, creds["model"])

        # Part 3: subagent 模式
        await demo_subagent_pattern(llm, creds["model"])

        print("\n" + "=" * 70)
        print("ALL DEMOS PASSED")
        print("=" * 70)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())


# =====================================================================
# pytest entry points (PyCharm / CI 用 ``pytest tests/test_real_streaming_subagent.py`` 才能收集到用例)
# =====================================================================
def test_part1_contextvars_isolation() -> None:
    """无需真实 LLM；验证 contextvars session 隔离。"""
    asyncio.run(demo_contextvars_isolation())

def test_part2_streaming_agent_real_llm(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    async def _run() -> None:
        try:
            await demo_streaming_agent(llm, creds["model"])
        finally:
            await llm.close()
    # pytest 默认捕获 stdout：流式 chunk 会攒到用例结束才显示；禁用捕获才能实时看到 token 流
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_run())

def test_part3_subagent_pattern_real_llm(capsys: pytest.CaptureFixture[str]) -> None:
    creds = _creds_or_skip()
    llm = _make_llm(creds)
    async def _run() -> None:
        try:
            await demo_subagent_pattern(llm, creds["model"])
        finally:
            await llm.close()
    with capsys.disabled():
        _configure_streaming_stdio()
        asyncio.run(_run())