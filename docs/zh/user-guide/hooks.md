# Hooks

[English](../../en/user-guide/hooks.md) | [用户手册](../index.md)

Hook 是**控制流**通道。与 [Events](events.md)（只读观测）不同，hook 可以修改消息、跳过工具、短路 LLM 调用或结束循环。

## 快速示例

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def block_dangerous(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash" and "rm -rf" in str(ctx.tool_args):
        ctx.output = "[已拦截：危险命令]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_dangerous)

loop = StatefulAgentLoop(llm=llm, hooks=hooks, config=config)
```

## Hook 点概览

| Hook 点 | 何时 | 常见用途 |
|---|---|---|
| `session.start` | 第一轮之前 | 注入预热消息 |
| `session.end` | 循环结束后 | 审计，关闭连接 |
| `round.start` | 每轮开始前 | 预算检查，`BREAK` 停止 |
| `round.end` | 每轮结束后 | 每轮指标上报 |
| `llm.before` | LLM 调用前 | 修改请求，缓存命中 (`SHORT_CIRCUIT`) |
| `llm.after` | LLM 返回后 | 内容审核，`BREAK` 停止 |
| `round.decide` | 工具执行前 | 跳过所有工具 (`SKIP`) |
| `tools.batch.before` | 工具批次前 | 批次级门控 |
| `tools.batch.after` | 工具批次后 | 只读观测 |
| `tool.before` | 每个工具前 | **安全门**，用户确认 |
| `tool.after` | 工具成功后 | 裁剪输出，修改结果 |
| `tool.error` | 工具异常时 | 吞错 (`SKIP`) 或重试 (`SHORT_CIRCUIT`) |
| `compact.before` | 压缩前 | 跳过本轮压缩 |
| `compact.after` | 压缩后 | 只读观测 |
| `message.append` | 消息存储前 | PII 脱敏，metadata 注入 |
| `memory.recalled` | 记忆召回后 | 过滤/脱敏记忆；`SKIP` 丢弃整批 |

## Hook vs Event

| | Hook | Event |
|---|---|---|
| 能改控制流？ | 是 (`SKIP`、`BREAK`、`SHORT_CIRCUIT`) | 否 |
| 能改数据？ | 是（消息、工具、LLM 请求） | 否 |
| 错误处理 | Handler 错误向上传播 | 订阅者错误默认隔离 |
| 用于 | 安全、路由、缓存、审核 | 观测、审计、流式 |

## 下一步

- [Events](events.md) — 观测而不干扰
- [完整 Hook 参考](../../hooks.md) — 每个 Ctx 字段和 directive 组合