# 高级运行时工具

[English](../../en/user-guide/advanced-runtime-tools.md) | [用户手册](index.md)

有些工具不只是请求/响应函数。它们需要持久状态、当前 loop 上下文、策略门、UI 事件，或者把状态投影到下一轮 LLM。power-loop 把这些能力作为公开原语提供，因此你可以构建自己的高级工具，而不依赖私有内部实现。

## 心智模型

分成四层：

| 层 | 原语 | 用途 |
|---|---|---|
| 工具 handler | `ToolRegistry`, `ToolDefinition`, `get_tool_runtime_context()` | 执行业务逻辑，并访问当前 session/store。 |
| 持久状态 | `SessionStore.get_runtime_state()` / `set_runtime_state()` | 把工具状态保存在对话日志旁边。 |
| Prompt 投影 | `RuntimeProjector`, `AgentLoopConfig.runtime_projectors` | 把持久状态转换成临时 LLM 消息。 |
| 控制与观测 | `AgentHooks`, `AgentEventBus` | 拦截决策并观察生命周期事件。 |

`messages` 表仍然是对话协议日志。运行时状态保存在旁路表中，只在需要时投影给 LLM，因此不会被压缩重复或污染。

## 工具运行时上下文

在工具 handler 内：

```python
from power_loop import get_tool_runtime_context

def save_marker(marker: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    ctx.store.set_runtime_state(ctx.session_id, "marker", {"value": marker})
    return "saved"
```

`ctx` 包含：

| 字段 | 用途 |
|---|---|
| `session_id` | 当前 session id。 |
| `store` | 当前 `SessionStore`；可用于 messages、runtime state、background tasks、session rows。 |
| `loop` | 当前 `StatefulAgentLoop`；可用于 `get_messages()` 等较高层 API。 |
| `config` | 当前 `AgentLoopConfig`。 |

## Runtime Projector

Projector 把持久运行时状态转换成临时 LLM 可见消息：

```python
from power_loop import RuntimeProjector

class MarkerProjector(RuntimeProjector):
    def project(self, *, store, session_id, round_index, context):
        state = store.get_runtime_state(session_id, "marker", default={}) or {}
        if not state:
            return []
        return [{"role": "user", "name": "marker_state", "content": str(state)}]
```

注册：

```python
config = AgentLoopConfig(runtime_projectors=(MarkerProjector(),))
```

传 `runtime_projectors=()` 可以关闭所有默认投影。

## Hooks

Hooks 是策略与控制平面：

```python
def before_tool(ctx):
    if ctx.tool_name == "deploy" and ctx.tool_args["target"] == "production":
        ctx.tool_args["target"] = "staging"

hooks.register(HookPoint.TOOL_BEFORE, before_tool)
```

常见模式：

- 改写不安全的工具参数。
- 跳过工具调用，并返回审批消息。
- 在 `TOOL_AFTER` 持久化派生状态。
- 在 `LLM_BEFORE` 短路 LLM 调用。

## Events

Events 是观测平面：

```python
def on_event(event):
    if event.type is AgentEventType.TOOL_CALL_COMPLETED:
        print(event.data.name, event.data.output)

bus.subscribe(None, on_event)
```

Events 适合 UI 更新、审计日志、指标和外部调度器。Events 不替代持久状态；它们是实时通知。

## 示例

可运行示例在 [`examples/advanced_runtime/`](../../../examples/advanced_runtime/)：

| 示例 | 展示内容 |
|---|---|
| `01_runtime_projector.py` | 把事故状态投影到下一轮 LLM。 |
| `02_tool_runtime_context.py` | 工具查询当前 session metadata/messages 并写入 runtime state。 |
| `03_hooks_control_flow.py` | Hooks 改写部署参数并记录审计状态。 |
| `04_events_observability.py` | Event bus 收集自定义工具生命周期审计。 |

## 设计规则

如果工具状态需要跨压缩、进程重启或未来轮次保留，把它存入 `SessionStore` runtime state。如果 LLM 需要看到这个状态，用 `RuntimeProjector` 投影。如果执行需要策略或监控，用 hooks 和 events。
