# 教程：工具调用

[English](../../en/tutorials/tool-calling.md) | [教程](../index.md)

**目标**：构建一个带天气、搜索和安全门的 Agent——60 行。

**你会学到**：`ToolRegistry`、`ToolDefinition`、JSON Schema、sync 和 async handler、安全门。

## 1. 定义工具

```python
from power_loop import ToolRegistry, ToolDefinition

def get_weather(city: str) -> str:
    return f"{city}天气：晴，22°C"

async def search_web(query: str) -> str:
    return f"搜索结果 '{query}': (mock)"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="获取城市天气。参数：city (string)。",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    get_weather,
)
registry.register(
    ToolDefinition(
        name="search_web",
        description="搜索网络。参数：query (string)。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    search_web,
)
```

## 2. 加安全门

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def safety(ctx: ToolBeforeCtx) -> None:
    if "rm" in str(ctx.tool_args).lower():
        ctx.output = "[已拦截]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, safety)
```

## 3. 运行

```python
loop = StatefulAgentLoop(
    llm=llm, tool_registry=registry, hooks=hooks,
    config=AgentLoopConfig(
        system_prompt="使用工具回答问题。简洁。",
        max_rounds=4,
    ),
)

result = await loop.send("东京天气怎么样？")
print(result.final_text)  # → "东京天气：晴，22°C。"
```

## 完整代码

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, ToolRegistry, ToolDefinition,
    AgentHooks, HookPoint, HookDirective,
    create_llm_service_from_env,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx

def get_weather(city: str) -> str:
    return f"{city}天气：晴，22°C"

async def search_web(query: str) -> str:
    return f"搜索结果 '{query}': (mock)"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="获取城市天气。参数：city。",
        input_schema={"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
    ),
    get_weather,
)
registry.register(
    ToolDefinition(
        name="search_web",
        description="搜索网络。参数：query。",
        input_schema={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    ),
    search_web,
)

hooks = AgentHooks()
def safety(ctx: ToolBeforeCtx) -> None:
    if "rm" in str(ctx.tool_args).lower():
        ctx.output = "[已拦截]"; ctx.directive = HookDirective.SKIP
hooks.register(HookPoint.TOOL_BEFORE, safety)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm, tool_registry=registry, hooks=hooks,
        config=AgentLoopConfig(system_prompt="使用工具。简洁。", max_rounds=4),
    )
    try:
        r = await loop.send("东京天气怎么样？")
        print(f"Bot: {r.final_text}")
    finally:
        loop.close()

asyncio.run(main())
```

## 下一步

- [人在回路](human-in-the-loop.md) — 执行前请求用户确认
- [多 Agent 系统](multi-agent.md) — 委托给子代理