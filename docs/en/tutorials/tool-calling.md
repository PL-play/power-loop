# Tutorial: Tool Calling

[中文](../../zh/tutorials/tool-calling.md) | [Tutorials](../index.md)

**Goal**: Build an agent with weather, calculator, and bash tools — 60 lines.

**You'll learn**: `ToolRegistry`, `ToolDefinition`, JSON Schema, sync & async handlers, multi-round tool use.

## 1. Define Tools

```python
from power_loop import ToolRegistry, ToolDefinition

# Sync tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

# Async tool
async def search_web(query: str) -> str:
    # Real implementation would call an API
    return f"Search results for '{query}': (mock result)"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city. Param: city (string).",
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
        description="Search the web. Param: query (string).",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    search_web,
)
```

## 2. Create the Agent

```python
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, create_llm_service_from_env,
)

llm = create_llm_service_from_env()
loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "You have get_weather and search_web tools. "
            "Use them to answer user questions."
        ),
        max_rounds=4,  # room for tool call + reply
    ),
)
```

## 3. Run

```python
sid = await loop.new_session()

result = await loop.send("What's the weather in Tokyo?", session_id=sid)
print(result.final_text)
# Output: "The weather in Tokyo is sunny, 22°C."

result = await loop.send("Search for Python async patterns", session_id=sid)
print(result.final_text)
# Output: "Search results for 'Python async patterns': (mock result)"
```

## 4. Add a Security Gate

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def block_if_dangerous(ctx: ToolBeforeCtx) -> None:
    """Block any tool call containing 'rm' in its arguments."""
    if "rm" in str(ctx.tool_args).lower():
        ctx.output = "[blocked: potentially destructive]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_if_dangerous)

loop = StatefulAgentLoop(llm=llm, hooks=hooks, tool_registry=registry, ...)
```

## 5. Validate Before Invoke

```python
from power_loop import ToolNotFound, ToolValidationError

try:
    result = await registry.invoke_async("get_weather", {"city": "Tokyo"})
    print(result)  # → "Weather in Tokyo: sunny, 22°C"
except ToolNotFound as exc:
    print(f"Tool not found: {exc.tool_name}")
except ToolValidationError as exc:
    print(f"Invalid args: {exc.message}")
```

## Complete Code

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, ToolRegistry, ToolDefinition,
    AgentHooks, HookPoint, HookDirective,
    create_llm_service_from_env,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

async def search_web(query: str) -> str:
    return f"Search results for '{query}': (mock)"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="Get current weather. Param: city (string).",
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
        description="Search the web. Param: query (string).",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    search_web,
)

hooks = AgentHooks()
def safety(ctx: ToolBeforeCtx) -> None:
    if "rm" in str(ctx.tool_args).lower():
        ctx.output = "[blocked]"
        ctx.directive = HookDirective.SKIP
hooks.register(HookPoint.TOOL_BEFORE, safety)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm, tool_registry=registry, hooks=hooks,
        config=AgentLoopConfig(
            system_prompt="Use tools to answer. Be concise.",
            max_rounds=4,
        ),
    )
    try:
        sid = await loop.new_session()
        result = await loop.send("What's the weather in Tokyo?", session_id=sid)
        print(f"Bot: {result.final_text}")
    finally:
        loop.close()

asyncio.run(main())
```

## Next

- [Human-in-the-Loop](human-in-the-loop.md) — ask users before running tools
- [Multi-Agent System](multi-agent.md) — delegate to sub-agents
