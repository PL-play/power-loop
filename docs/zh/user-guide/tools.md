# 工具

[English](../../en/user-guide/tools.md) | [用户手册](../index.md)

工具让 Agent 拥有能力——查天气、文件操作、API 调用、bash 命令。power-loop 处理注册、JSON Schema 校验和调用。

## 快速开始

```python
from power_loop import ToolRegistry, ToolDefinition

def get_weather(city: str) -> str:
    return f"{city}天气：晴，22°C"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="获取城市当前天气",
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
    name="get_weather",          # 唯一标识
    description="获取天气",       # 发给 LLM 的描述
    input_schema={                # 参数的 JSON Schema
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"],
    },
    required_params=("city",),   # handler 运行前客户端校验
)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 唯一工具标识。用于注册和 LLM 工具调用。 |
| `description` | `str` | 自然语言描述——LLM 用它决定何时调用工具。 |
| `input_schema` | `dict` | JSON Schema（OpenAI 兼容）。定义 `properties` 和 `required` 字段。 |
| `required_params` | `tuple[str, ...]` | 额外客户端校验。`ToolRegistry` 在调用 handler 前检查这些参数。 |

## Handler 签名

### 同步 handler

```python
def get_weather(city: str) -> str:
    return f"{city}天气：晴"
```

### 异步 handler

```python
async def search_web(query: str) -> str:
    result = await http_client.get(f"/search?q={query}")
    return result.text
```

`ToolRegistry` 在注册时通过 `inspect.iscoroutinefunction` 检测 `async def`。Pipeline 始终调用 `invoke_async()`，透明处理同步和异步 handler。

### Callable 对象

```python
class WeatherTool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def __call__(self, city: str) -> str:
        return await fetch_weather(city, self.api_key)

registry.register(weather_def, WeatherTool(api_key="..."))
```

`__call__` 在注册时被异步检测。

## 校验

`ToolRegistry` 在两个层面校验参数：

1. **JSON Schema** — `validate_tool_args(name, args)` 检查必填属性存在。
2. **Required params** — `tool.definition.required_params` 提供额外的编程式检查。

校验失败时，`invoke_async()` 抛出 `ToolValidationError`（`PowerLoopError` 子类）。Pipeline 捕获后将错误返回给 LLM 使其可以自我纠正。

## 错误处理

```python
from power_loop import ToolNotFound, ToolValidationError

try:
    result = await registry.invoke_async("unknown_tool", {})
except ToolNotFound as exc:
    print(f"工具未找到: {exc.tool_name}")
except ToolValidationError as exc:
    print(f"校验失败 {exc.tool_name}: {exc.message}")
```

## Sync vs Async 调用

| 方法 | 使用场景 |
|---|---|
| `invoke(name, args)` | 仅同步。handler 是 `async def` 时抛出 `AsyncToolInSyncContext`。 |
| `invoke_async(name, args)` | **通用入口。** 同步和异步 handler 都适用。 |

```python
# 同步 handler，同步调用
result = registry.invoke("get_weather", {"city": "Tokyo"})

# 异步 handler，必须用 invoke_async
result = await registry.invoke_async("search_web", {"query": "Python"})
```

## 元工具：spawn_agent 和 run_agent

```python
from power_loop import register_spawn_agent

register_spawn_agent(registry)
# 现在 LLM 可以调用：
#   spawn_agent(task="研究 X", preset="explore")
#   run_agent(spec='{"name":"researcher", "system_prompt":"...", ...}')
```

详见 [子代理](subagents.md)。

## 下一步

- [子代理](subagents.md) — `spawn_agent` 和 `AgentSpec`
- [Hooks](hooks.md) — 用 `TOOL_BEFORE` / `TOOL_AFTER` 拦截工具执行