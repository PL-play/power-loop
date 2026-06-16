# 教程：多 Agent 系统

[English](../../en/tutorials/multi-agent.md) | [教程](../index.md)

**目标**：构建一个将研究任务委托给子代理的父 Agent——80 行。

**你会学到**：`spawn_agent`、`AgentSpec`、`run_agent_spec`、子代理生命周期、工具白名单。

## 命令式：spawn_agent

LLM 决定何时委托：

```python
from power_loop import register_spawn_agent

register_spawn_agent(registry)

loop = StatefulAgentLoop(
    llm=llm, tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="你可以用 spawn_agent 委托研究任务。",
        max_rounds=8,
    ),
)

sid = await loop.new_session()
result = await loop.send("找到项目中的认证逻辑代码。", session_id=sid)
# LLM: spawn_agent(task="搜索认证代码", preset="explore")
# → 子代理用 explore 工具集搜索 → 父代理拿到结果
```

## 声明式：AgentSpec

你控制子代理配置：

```python
from power_loop import AgentSpec, run_agent_spec

spec = AgentSpec(
    name="security-auditor",
    system_prompt="你是安全审计员。只报告确认的问题。",
    tools=["grep", "read", "glob"],  # 白名单
    max_rounds=5,
    lifecycle="ephemeral",
)

result = await run_agent_spec(spec, "审计认证模块的 SQL 注入漏洞。", parent_loop=loop)
print(result["final_text"])
```

## 两种方式对比

| | spawn_agent | AgentSpec |
|---|---|---|
| 谁决定 | LLM | 你 |
| 工具白名单 | 通过 preset | 显式 `tools` 列表 |
| 使用场景 | 动态委托 | 受控、可审计的子任务 |

## 完整代码

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, ToolRegistry, ToolDefinition,
    AgentSpec, run_agent_spec, register_spawn_agent,
    create_llm_service_from_env,
)

registry = ToolRegistry()
registry.register(
    ToolDefinition(name="read_file", description="读取文件。参数: path。",
        input_schema={"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
    lambda path: f"(内容: {path})",
)
register_spawn_agent(registry)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm, tool_registry=registry,
        config=AgentLoopConfig(system_prompt="用 spawn_agent 委托。简洁。", max_rounds=8),
    )
    try:
        sid = await loop.new_session()
        r1 = await loop.send("研究：read_file 返回什么？", session_id=sid)
        print(f"命令式: {r1.final_text[:200]}")

        spec = AgentSpec(name="r", system_prompt="简洁回答。", tools=["read_file"], max_rounds=3)
        r2 = await run_agent_spec(spec, "读 config.py", parent_loop=loop)
        print(f"声明式: {r2['final_text'][:200]}")
    finally:
        loop.close()

asyncio.run(main())
```

## 下一步

- [子代理用户手册](../user-guide/subagents.md) — 完整参考
- [Hooks 用户手册](../user-guide/hooks.md) — 拦截每个阶段
