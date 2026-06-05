# Tools

[中文](../../zh/user-guide/tools.md) | [User Guide](../index.md)

Tools give your agent abilities — weather lookup, file operations, API calls, bash commands. power-loop handles registration, JSON Schema validation, and invocation.

## Quick Start

```python
from power_loop import ToolRegistry, ToolDefinition

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    get_weather,
)

loop = StatefulAgentLoop(llm=llm, tool_registry=registry, config=config)
```

## ToolDefinition

```python
from power_loop import ToolDefinition

ToolDefinition(
    name="get_weather",          # unique identifier
    description="Get weather",   # sent to the LLM
    input_schema={               # JSON Schema for arguments
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
    required_params=("city",),   # enforced client-side before handler runs
)
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique tool identifier. Used in registration and LLM tool calls. |
| `description` | `str` | Natural-language description — the LLM uses this to decide when to call the tool. |
| `input_schema` | `dict` | JSON Schema (OpenAI-compatible). Defines the `properties` and `required` fields. |
| `required_params` | `tuple[str, ...]` | Additional client-side validation. `ToolRegistry` checks these before calling the handler. |

## Handler Signatures

### Sync handler

```python
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"
```

### Async handler

```python
async def search_web(query: str) -> str:
    result = await http_client.get(f"/search?q={query}")
    return result.text
```

`ToolRegistry` detects `async def` at register time (`inspect.iscoroutinefunction`). The pipeline always calls `invoke_async()`, which handles both sync and async handlers transparently.

### Callable objects

```python
class WeatherTool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def __call__(self, city: str) -> str:
        # fetch from real API
        return await fetch_weather(city, self.api_key)

registry.register(weather_def, WeatherTool(api_key="..."))
```

`__call__` is inspected for async at register time.

## Validation

`ToolRegistry` validates arguments at two levels:

1. **JSON Schema** — `validate_tool_args(name, args)` checks that required properties exist.
2. **Required params** — `tool.definition.required_params` provides an additional programmatic check.

If validation fails, `invoke_async()` raises `ToolValidationError` (a `PowerLoopError` subclass). The pipeline catches it and returns the error message to the LLM so it can self-correct.

## Error Handling

```python
from power_loop import ToolNotFound, ToolValidationError

try:
    result = await registry.invoke_async("unknown_tool", {})
except ToolNotFound as exc:
    print(f"Tool not found: {exc.tool_name}")
except ToolValidationError as exc:
    print(f"Validation failed for {exc.tool_name}: {exc.message}")
```

## Sync vs Async Invocation

| Method | Use case |
|---|---|
| `invoke(name, args)` | Sync only. Raises `AsyncToolInSyncContext` if the handler is `async def`. |
| `invoke_async(name, args)` | **Universal entry point.** Works with both sync and async handlers. |

```python
# Sync handler, sync call
result = registry.invoke("get_weather", {"city": "Tokyo"})

# Async handler, must use invoke_async
result = await registry.invoke_async("search_web", {"query": "Python"})
```

## Meta-Tools: spawn_agent and run_agent

```python
from power_loop import register_spawn_agent

register_spawn_agent(registry)
# Now the LLM can call:
#   spawn_agent(task="Research X", preset="explore")
#   run_agent(spec='{"name":"researcher", "system_prompt":"...", ...}')
```

See [Sub-agents](subagents.md) for details.

## Next

- [Sub-agents](subagents.md) — `spawn_agent` and `AgentSpec`
- [Hooks](hooks.md) — intercept tool execution with `TOOL_BEFORE` / `TOOL_AFTER`