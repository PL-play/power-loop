# 快速入门 — 走完每个功能

[English](../../en/user-guide/quickstart.md) | [用户手册](../index.md)

本页带你从 `new_session()` + `send(...)` 一直走到子代理、上下文压缩和跨进程恢复。每节在前一节基础上推进。边看边跑代码。

> **前提**：安装 power-loop 并配置 `POWER_LOOP_*` 环境变量。见 [快速上手](../getting-started.md)。

## 1. 最小用法 — 一条消息

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig,
    create_llm_service_from_env,
)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(
            system_prompt="简洁回复。",
            max_rounds=1,
        ),
    )
    sid = loop.new_session()
    result = await loop.send("你好！", session_id=sid)
    print(result.final_text)
    # → "你好！有什么可以帮你的？"

asyncio.run(main())
```

**要点**：
- `new_session()` 创建会话；`send(..., session_id=sid)` 追加用户轮次并返回 `StatefulResult`。
- `max_rounds=1` 表示"一次 LLM 调用，无工具"——最简单的循环。

## 2. 多轮对话 — 让对话持续

```python
sid = loop.new_session()
r1 = await loop.send("我叫阿岚。", session_id=sid)
print(sid)  # 例如 "sess_abc123..."

r2 = await loop.send("我叫什么？", session_id=sid)
print(r2.final_text)  # → "你叫阿岚。"
```

**要点**：
- 传入同一个 `session_id` 继续同一会话。
- 库自动从 SQLite 加载完整历史——你永远不需要手动管理 `messages`。
- `loop.get_messages(sid)` 可以查看活跃历史。

## 3. 工具调用 — 给 Agent 能力

```python
from power_loop import ToolRegistry, ToolDefinition

def weather(city: str) -> str:
    return f"{city}天气：晴，22°C"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="get_weather",
        description="获取城市当前天气。",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    weather,
)

loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="你有一个 get_weather 工具。用它。",
        max_rounds=4,  # 留空间给工具调用 + 回复
    ),
)

sid = loop.new_session()
result = await loop.send("东京天气怎么样？", session_id=sid)
# LLM 调用 get_weather(city="东京") → result.final_text 提到 "晴，22°C"
```

**要点**：
- `max_rounds=4` 给 LLM 空间去调工具然后回复。
- 工具自动出现在 OpenAI 兼容的 `tools` 字段。
- Async handler 同样支持——`invoke_async()` 在注册时自动检测 `async def`。

## 4. 子代理 — 委托给专门 Agent

```python
from power_loop import register_spawn_agent

# 注册两个 meta-tool：spawn_agent（命令式）和 run_agent（声明式）
register_spawn_agent(registry)

loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "你可以用 spawn_agent 工具把研究任务委托给子代理。"
            "简单任务直接回答。"
        ),
        max_rounds=6,
    ),
)

sid = loop.new_session()
result = await loop.send(
    "研究一下：东京的人口是多少？比伦敦多吗？",
    session_id=sid,
)
# LLM spawn 子代理 → 子代理跑自己的循环 → 父代理拿到结果
```

**要点**：
- `spawn_agent` 是 LLM 调用的工具；库在内部跑一个子 `StatefulAgentLoop`。
- `AgentSpec`（声明式）给你显式控制：工具白名单、model、max_rounds。
- 子代理有独立的 SQLite 行，通过 `parent_session_id` 链接到父代理。

## 5. Hooks — 拦截循环

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def block_dangerous(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash" and "rm -rf" in str(ctx.tool_args):
        ctx.output = "[已拦截]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_dangerous)

loop = StatefulAgentLoop(llm=llm, hooks=hooks, ...)
```

**要点**：
- 18 个 hook 点覆盖每个阶段：`session.start`、`round.start`、`llm.before`、`tool.before`、`compact.before`、……
- Hook 可以改消息、跳过工具、短路 LLM 调用、或直接结束循环。
- Async hook 可用——在跑 `bash` 之前 `await` 用户确认 UI。

## 6. Events — 观测而不干扰

```python
from power_loop import AgentEventBus, AgentEventType

bus = AgentEventBus()

def on_delta(event):
    print(event.data.text, end="", flush=True)  # 打字机效果

bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)

loop = StatefulAgentLoop(llm=llm, event_bus=bus, ...)
```

**要点**：
- Event 只读——不能改变控制流（控制流用 hook）。
- `bus.subscribe(None, fn)` 订阅所有事件（审计日志、调试）。
- 24 种事件类型，带类型化 payload。

## 7. 持久化 — 跨进程恢复

```python
# 进程 1
loop = StatefulAgentLoop(llm=llm, db_path="./chat.db", ...)
sid = loop.new_session()
r1 = await loop.send("记住：我叫阿岚。", session_id=sid)
loop.close()

# 进程 2（几小时后，不同的 Python 进程）
loop2 = StatefulAgentLoop(llm=llm, db_path="./chat.db", ...)
r2 = await loop2.send("我叫什么？", session_id=sid)
print(r2.final_text)  # → "你叫阿岚。"
```

**要点**：
- `db_path` 指向真实文件（默认 `./power_loop_sessions.db`）；`":memory:"` 用于测试。
- 会话活在 SQLite 里——新进程打开同一文件，传入相同 `session_id`，LLM 看到完整历史。
- 跨子进程、容器、重启都可用。

## 8. 下一步

| 功能 | 看这里 |
|---|---|
| 压缩 | [用户手册：压缩](compaction.md) — 自动摘要长会话 |
| 记忆 | [用户手册：记忆](memory.md) — 通过 `MemoryProvider` 跨会话召回 |
| 重试与取消 | [用户手册：重试与取消](retry-cancel.md) — 优雅处理 LLM 失败 |
| 结构化输出 | [用户手册：结构化输出](structured-output.md) — schema 校验的 JSON |
| 完整示例列表 | [Examples](../../../examples/) — 21 个可运行示例，每个教一个概念 |
