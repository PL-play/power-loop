"""23 · Per-call overrides / 每次调用覆盖: restrict tools & system prompt per send

What you learn / 你将学到
--------------------------
- 一个 loop 构造一次（含全部工具 + 基础 system prompt），但**每次 send 可覆盖**
  / build the loop once (all tools + base prompt), then override **per send**
- ``send(..., tools=[...])`` 只允许一个工具子集——模型**根本看不到**其它工具
  / restrict this run to a tool subset — the model never even *sees* the rest
- ``send(..., system_prompt=...)`` 仅对这次运行覆盖系统提示
  / override the system prompt for this run only
- 适合多租户：一个缓存 loop 服务多个调用方，各自不同的工具白名单/提示
  / great for multi-tenant: one cached loop, per-caller allowlists & prompts

``tools`` 接受工具名序列（从 loop 注册表里取子集）或一个 ``ToolRegistry``。
/ ``tools`` accepts a sequence of tool names (subset of the loop registry) or a ``ToolRegistry``.

Run / 运行
----------
    python examples/23_per_send_overrides.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    RuntimeEnv,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    runtime_env_context,
)


def _weather(**kwargs) -> str:
    return f"{kwargs.get('city', '?')}: sunny, 24°C"


def _stock(**kwargs) -> str:
    return f"{kwargs.get('ticker', '?')}: 123.45"


def _build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        ToolDefinition(
            name="get_weather",
            description="Get current weather for a city.",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"]},
            required_params=("city",),
        ),
        _weather,
    )
    r.register(
        ToolDefinition(
            name="get_stock",
            description="Get the current price for a stock ticker.",
            input_schema={"type": "object", "properties": {"ticker": {"type": "string"}},
                          "required": ["ticker"]},
            required_params=("ticker",),
        ),
        _stock,
    )
    return r


async def main() -> None:
    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=":memory:",
        tool_registry=_build_registry(),  # registered with BOTH tools
        config=AgentLoopConfig(system_prompt="You are a helpful assistant.", max_rounds=4),
    )

    # Run A: only expose get_weather — the model cannot call get_stock at all.
    sid_a = loop.new_session()
    res_a = await loop.send(
        "What's the weather in Tokyo, and the price of AAPL?",
        session_id=sid_a,
        tools=["get_weather"],
    )
    print("A (weather only):", res_a.final_text)

    # Run B: only expose get_stock + a per-run system prompt override.
    sid_b = loop.new_session()
    res_b = await loop.send(
        "What's the weather in Tokyo, and the price of AAPL?",
        session_id=sid_b,
        tools=["get_stock"],
        system_prompt="You are a terse finance bot. Answer in one short line.",
    )
    print("B (stock only, terse):", res_b.final_text)

    # One unbound default registry can safely resolve a different workspace
    # for each invocation. The host owns the RuntimeEnv boundary.
    unbound = create_default_tool_registry(include=["read_file"], bind=False)
    with tempfile.TemporaryDirectory() as root:
        tenant_a = Path(root) / "tenant-a"
        tenant_b = Path(root) / "tenant-b"
        tenant_a.mkdir()
        tenant_b.mkdir()
        (tenant_a / "profile.txt").write_text("tenant A", encoding="utf-8")
        (tenant_b / "profile.txt").write_text("tenant B", encoding="utf-8")

        with runtime_env_context(RuntimeEnv(workspace_dir=tenant_a)):
            print("unbound A:", await unbound.invoke_async("read_file", {"path": "profile.txt"}))
        with runtime_env_context(RuntimeEnv(workspace_dir=tenant_b)):
            print("unbound B:", await unbound.invoke_async("read_file", {"path": "profile.txt"}))


if __name__ == "__main__":
    asyncio.run(main())
