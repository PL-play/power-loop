# Tutorial: Multi-Agent System

[中文](../../zh/tutorials/multi-agent.md) | [Tutorials](../index.md)

**Goal**: Build a parent agent that delegates research to sub-agents with tool whitelists — 80 lines.

**You'll learn**: `spawn_agent`, `AgentSpec`, `run_agent_spec`, child lifecycle, tool whitelisting.

## 1. Register Sub-Agent Tools

```python
from power_loop import ToolRegistry, ToolDefinition, register_spawn_agent

registry = ToolRegistry()

# Parent tools
def read_file(path: str) -> str:
    return f"(contents of {path})"

registry.register(
    ToolDefinition(
        name="read_file",
        description="Read a file. Param: path (string).",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    read_file,
)

# Register both meta-tools: spawn_agent and run_agent
register_spawn_agent(registry)
```

## 2. Imperative: spawn_agent

The LLM decides when to delegate:

```python
loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "You are a project manager. You can delegate research tasks "
            "to sub-agents using spawn_agent. Use preset='explore' for "
            "code/file searches. For simple tasks, answer directly."
        ),
        max_rounds=8,
    ),
)

result = await loop.send(
    "Find where authentication logic is implemented in this project. "
    "Then tell me if it uses JWT or session tokens."
)
# LLM: spawn_agent(task="search for auth code", preset="explore")
# → child runs with explore tools (grep, read, glob)
# → parent gets result, synthesizes answer
```

## 3. Declarative: AgentSpec

You control the child's configuration:

```python
from power_loop import AgentSpec, run_agent_spec

# Define a specialized researcher
spec = AgentSpec(
    name="security-auditor",
    system_prompt=(
        "You are a security auditor. Find vulnerabilities in the code "
        "snippets provided. Report only confirmed issues, not speculation."
    ),
    tools=["grep", "read", "glob"],  # whitelist: only these tools
    max_rounds=5,
    max_tokens=2000,
    temperature=0.0,
    lifecycle="ephemeral",
)

# Run directly (no LLM deciding — you control it)
result = await run_agent_spec(
    spec,
    "Audit the authentication module for SQL injection.",
    parent_loop=loop,
)
print(result["final_text"])
```

## 4. Two Approaches Compared

| | spawn_agent | AgentSpec |
|---|---|---|
| Who decides | LLM | You |
| Tool whitelist | Via preset | Explicit `tools` list |
| Max rounds | Default | You set |
| Model | Parent's | Override per spec |
| Use case | Dynamic delegation | Controlled, auditable sub-tasks |

## 5. Lifecycle

```python
# EPHEMERAL (default) — deleted on success, kept on failure for debug
AgentSpec(name="x", system_prompt="...", lifecycle="ephemeral")

# LINKED — cascade-deleted when parent closes
AgentSpec(name="x", system_prompt="...", lifecycle="linked")

# DETACHED — independent, survives parent close
AgentSpec(name="x", system_prompt="...", lifecycle="detached")
```

## 6. Depth Limit

`MAX_SPAWN_DEPTH = 3` — a sub-agent can spawn its own sub-agent, but the chain cannot exceed 3 levels. Trying to spawn a 4th level raises an error.

## Complete Code

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, ToolRegistry, ToolDefinition,
    AgentSpec, run_agent_spec, register_spawn_agent,
    create_llm_service_from_env,
)

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="read_file",
        description="Read a file. Param: path (string).",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    lambda path: f"(contents of {path})",
)
register_spawn_agent(registry)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm, tool_registry=registry,
        config=AgentLoopConfig(
            system_prompt="Delegate research with spawn_agent. Be concise.",
            max_rounds=8,
        ),
    )
    try:
        # Imperative
        r1 = await loop.send("Research: what does read_file return?")
        print(f"Imperative: {r1.final_text[:200]}")

        # Declarative
        spec = AgentSpec(
            name="researcher",
            system_prompt="Answer concisely. Use tools if available.",
            tools=["read_file"], max_rounds=3,
        )
        r2 = await run_agent_spec(spec, "Read file 'config.py'", parent_loop=loop)
        print(f"Declarative: {r2['final_text'][:200]}")
    finally:
        loop.close()

asyncio.run(main())
```

## Next

- [Sub-agents User Guide](../user-guide/subagents.md) — full reference
- [Hooks User Guide](../user-guide/hooks.md) — intercept every phase