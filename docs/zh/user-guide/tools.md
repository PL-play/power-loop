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

## 默认工具

`create_default_tool_registry()` 提供一组面向编码 Agent 的默认工具：用专用文件/搜索工具处理精确工作区操作，用 shell 执行确实需要 CLI 的命令。

```python
from power_loop import create_default_tool_registry

registry = create_default_tool_registry(
    preset="core",
    workspace_dir="/path/to/project",
)
```

文件、搜索、shell 和后台命令工具需要显式工作区。传 `workspace_dir=...` 或设置
`POWER_LOOP_WORKSPACE`；power-loop 不会回退到进程当前工作目录。`load_skill` 使用
`AgentLoopConfig.skills_dir`、`skills_dir=...` 或 `POWER_LOOP_SKILLS_DIR`。自定义工具不受影响，
由实现方自己处理路径和配置。

### 每次调用的工具白名单

只注册一次工具全集，每个 run 只向模型暴露当次允许的工具：

```python
result = await loop.send(
    "Inspect the project",
    session_id=sid,
    tools=["read_file", "glob", "grep"],
)
```

名称序列通过 `ToolRegistry.subset()` 解析；未知名称会被忽略，LLM 只会收到
选中工具的 definition。`ToolRegistry.names()` 返回已注册名称；也可以直接传入另一个
`ToolRegistry`。

### 未绑定 registry

同一 registry 需要跨 workspace 复用时，可将环境解析延迟到 handler 调用时：

```python
from power_loop import RuntimeEnv, create_default_tool_registry, runtime_env_context

registry = create_default_tool_registry(preset="core", bind=False)

with runtime_env_context(RuntimeEnv(workspace_dir="/srv/tenant-a")):
    result = await registry.invoke_async("read_file", {"path": "README.md"})
```

`DEFAULT_TOOL_HANDLERS` 也是公开 API，便于 host 用内置 handler 组合自定义 definition。

### Shell 执行边界

默认 `LocalShellBackend` 直接在 host 上启动 `/bin/bash` 并继承 host 环境；它只负责编排，
不提供隔离。不可信命令必须由 host 注入 `ShellBackend`，在 container、gVisor 或其它 sandbox
内启动。`session_key(workspace_dir)` 标识持久 shell 缓存对应的执行目标；不同目标必须返回
不同 key。

预设：

| 预设 | 工具 |
|---|---|
| `core` | `bash`, `read_file`, `write_file`, `edit_file`, `apply_patch`, `glob`, `grep`, `load_skill`, `request_user_input` |
| `explore` | `bash`, `read_file`, `glob`, `grep`, `load_skill`, `request_user_input` |
| `full` | `core` 加上 `todo`、`note_add`/`note_update`/`note_delete`、`schedule_wakeup`/`list_wakeups`/`cancel_wakeup`、`current_time`、`recall_compacted`、`background_run`、`check_background` |

推荐系统提示词：

```text
修改已有文件前先使用 read_file。定位文件优先用 glob，搜索内容优先用 grep，不要用 shell find/grep 代替。单个精确替换用 edit_file，多行或多 hunk 修改用 apply_patch。bash 用于测试、构建、git 检查，以及专用工具无法表达的命令。不要用 bash 绕过文件安全检查。
```

工具行为：

| 工具 | 适用场景 | 安全与精确性说明 |
|---|---|---|
| `read_file` | 按行号读取文本文件，或列目录。 | 拒绝疑似二进制文件。大文件用 `offset` / `limit` 分页。读取会记录文件戳，供写入/编辑/patch 防护使用。 |
| `write_file` | 创建完整新文件，或有意整体覆盖文件。 | 覆盖已有文件前必须先读过，且文件自上次读取后没有变化。会自动创建父目录。 |
| `edit_file` | 替换一个精确片段，或用 `replace_all=true` 替换所有精确出现。 | 空片段、未找到、仅 fuzzy 后找到、多处匹配都会被拒绝并给出修正提示。保留 BOM 和主要换行风格。 |
| `apply_patch` | 对单个文件应用 unified diff 风格 hunk。 | 需要先读文件。过期或歧义 hunk 会被拒绝，不会猜测位置。 |
| `glob` | 用 glob 模式查找路径。 | 裸文件名会递归搜索。默认跳过常见大目录。隐藏路径需要 `include_hidden=true` 或显式隐藏模式。 |
| `grep` | 用正则或字面量搜索文本内容。 | 优先使用 ripgrep，缺失时回退 Python 实现。限制结果数，跳过疑似二进制文件和常见大目录。 |
| `bash` | 运行测试、构建、包管理器和 git 命令。 | 在工作区根目录的持久 bash 会话中运行。超时会重启 shell，避免残留命令。特权/设备级命令（`sudo`、`dd`、`mkfs` 等）以及对 根/家目录/系统目录 的递归 `rm -rf` 会被拦截；`/tmp` 与相对路径允许。 |
| `background_run` / `check_background` | 运行并查看非交互式长命令。 | 使用私有后台任务表，并复用 `bash` 的基础危险命令检查。 |
| `todo` | 维护 Agent 可见任务列表。 | 同一时间只允许一个条目为 `in_progress`。 |
| `load_skill` | 加载指定 skill 的详细说明。 | 未知 skill 会返回错误和可用 skill 名称。 |
| `request_user_input` | 暂停等待调用方/用户输入。 | 返回 `status="waiting_for_input"` 和 `pending_interactions`；用 `submit_input()` 恢复。 |
| `recall_compacted` | 把被[压缩](compaction.md)折叠出活跃窗口的旧消息捞回来。 | 只读、**仅当前会话**。可按 `query`(子串)和/或 `from_seq`/`to_seq` 过滤；按 `limit` 取最近若干条。在 `full` preset 里；也可 `include=["recall_compacted"]` 单挑。 |

可运行示例见 [`examples/20_default_tools.py`](../../../examples/20_default_tools.py)，它不依赖真实 LLM，会逐个演示默认工具。

## 运行时绑定工具

有些默认工具不只是普通函数，它们会参与 agent loop：

- `todo` 会把当前任务列表持久化到 session SQLite 数据库。每轮 LLM 调用前，power-loop 会把这个权威状态投影成临时 `<current_todos>` user 消息。这个投影不会写入 `messages`，所以不会被压缩重复或污染。
- `background_run` 会把任务状态记录到 SQLite。任务从未读变为更新或完成后，下一轮 LLM 会收到临时 `<background_updates>` 消息。`check_background` 读取同一张持久化任务表。
- `load_skill` 在配置了 `AgentLoopConfig.skills_dir` 时会使用该目录。设置 `skills_dir` 后，解析后的系统提示词会包含技能目录和可用 skill 描述。
- `request_user_input` 是控制流工具，不会在 Python 进程里 await 等人。它会把待确认/待输入项持久化，然后返回 `StatefulResult(status="waiting_for_input")`。业务方把 `pending_interactions` 展示给用户或 API 调用方，收集结果后调用 `await loop.submit_input(session_id, interaction_id, value)`，loop 会补上对应 tool result 并继续执行。

这些行为基于公开原语实现。`SessionStore` 暴露 JSON runtime state 和 background task API，`get_tool_runtime_context()` 让工具 handler 获取当前 session/store，`AgentLoopConfig.runtime_projectors` 控制持久化状态如何变成临时 LLM 消息。默认 projector 是 `TodoRuntimeProjector` 和 `BackgroundRuntimeProjector`；你可以传入自己的 `RuntimeProjector` 支持自定义工具，也可以用 `runtime_projectors=()` 关闭默认投影。

```python
from power_loop import RuntimeProjector, get_tool_runtime_context

def remember_custom_state(value: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    ctx.store.set_runtime_state(ctx.session_id, "my_tool", {"value": value})
    return "saved"

class MyToolProjector(RuntimeProjector):
    def project(self, *, store, session_id, round_index, context):
        state = store.get_runtime_state(session_id, "my_tool", default={}) or {}
        if not state:
            return []
        return [{"role": "user", "name": "my_tool_state", "content": str(state)}]
```

因此，只要共享同一个 `SessionStore`，这些运行时状态可以跨新的 `StatefulAgentLoop` 实例恢复。对话历史仍然是协议日志；运行时状态保存在旁路表中，只在需要时投影进 prompt。

你还可以把同一组原语和 hooks/events 组合成更复杂的流程控制：

- `TOOL_BEFORE` hook 可以改写工具参数、要求审批，或跳过执行。
- `TOOL_AFTER` hook 可以通过 `get_tool_runtime_context()` 持久化派生状态。
- event subscriber 可以观察 `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED`，驱动 UI、日志或外部调度器。
- 工具 handler 需要 session 感知行为时，可以查询 `ctx.loop.get_messages(ctx.session_id)` 或 `ctx.store.get_session(ctx.session_id)`。

默认工具也使用这些扩展点。用户自定义工具不需要任何私有通道，就能构建类似流程。

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
