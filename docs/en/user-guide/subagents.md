# Sub-Agents

[中文](../../zh/user-guide/subagents.md) | [User Guide](../index.md)

Sub-agents let the parent agent delegate tasks to specialized child agents. Each child gets its own `StatefulAgentLoop` with a custom system prompt and tool whitelist.

## Two Paths

| Path | Trigger | Control |
|---|---|---|
| **Imperative** (`spawn_agent`) | LLM calls the `spawn_agent` tool | LLM decides what to delegate |
| **Declarative** (`run_agent` / `AgentSpec`) | LLM submits a JSON spec | You control system prompt, tools, model, max_rounds |

## Imperative: spawn_agent

```python
from power_loop import register_spawn_agent

register_spawn_agent(registry)

loop = StatefulAgentLoop(
    llm=llm,
    tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "You can delegate research tasks using spawn_agent. "
            "Use preset='explore' for file/code searches."
        ),
        max_rounds=6,
    ),
)

result = await loop.send("Find where authentication logic is defined in this project.")
# LLM: spawn_agent(task="search for auth code", preset="explore")
# → child runs its own loop → parent gets the result
```

## Declarative: AgentSpec

```python
from power_loop import AgentSpec

spec = AgentSpec(
    name="researcher",
    system_prompt="You are a code researcher. Use grep, read, and glob to find answers.",
    tools=["grep", "read", "glob"],   # whitelist — only these tools
    max_rounds=5,
    max_tokens=2000,
    temperature=0.0,
    lifecycle="ephemeral",            # deleted on success, kept on failure for debug
)

# Direct call (no LLM involved)
from power_loop import run_agent_spec
result = await run_agent_spec(spec, "Find all SQL injection vulnerabilities", parent_loop=loop)
```

### AgentSpec Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique name. Non-empty string. |
| `system_prompt` | `str` | required | The child's system prompt. |
| `tools` | `list[str] \| None` | `None` | Whitelist of tool names from the parent registry. `None` = all tools. |
| `max_rounds` | `int` | `5` | Max LLM + tool rounds. Range: [1, 50]. |
| `max_tokens` | `int` | `2000` | Per-request token cap. |
| `temperature` | `float` | `0.0` | LLM temperature. |
| `model` | `str \| None` | `None` | Override model. `None` = use parent's. |
| `lifecycle` | `str` | `"ephemeral"` | `"ephemeral"` / `"linked"` / `"detached"` |
| `metadata` | `dict` | `{}` | Free-form metadata for audit. |

### Validation

`AgentSpec` has **strict validation**. Unknown fields, invalid lifecycle values, or out-of-range `max_rounds` raise `AgentSpecError` (a `SpecValidationError` → `PowerLoopError`):

```python
try:
    spec = AgentSpec.from_dict({"name": "", "system_prompt": ""})
except AgentSpecError as exc:
    print(exc)  # "AgentSpec.name must be a non-empty string"
```

## Lifecycle

| Lifecycle | Behavior |
|---|---|
| `EPHEMERAL` (default) | Deleted on success. Preserved on failure for debugging. |
| `LINKED` | Cascade-deleted when the parent session is closed. |
| `DETACHED` | Independent of the parent. Survives parent close. |

## Depth Limit

`MAX_SPAWN_DEPTH = 3` — a child can spawn its own sub-agent, but the chain cannot exceed 3 levels deep. Enforced at `SessionStore.create_session()`.

## Session Tree

```mermaid
flowchart TD
    P[Parent Session<br/>sess_abc] --> C1[Child: researcher<br/>sess_def]
    P --> C2[Child: reviewer<br/>sess_ghi]
    C1 --> GC1[Grandchild: searcher<br/>sess_jkl]
```

All children share the same `SessionStore` as the parent. `close_session(parent_sid, cascade=True)` recursively deletes the entire tree.

## Next

- [Hooks](hooks.md) — intercept tool execution
- [Compaction](compaction.md) — auto-summarize long sessions