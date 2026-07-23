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

## Default Tools

`create_default_tool_registry()` gives agents a coding-oriented tool set modeled after the practical split used by current agent runtimes: dedicated file/search tools for precise workspace operations, and shell execution for commands that genuinely need a CLI.

```python
from power_loop import create_default_tool_registry

registry = create_default_tool_registry(
    preset="core",
    workspace_dir="/path/to/project",
)
```

Filesystem, search, shell, and background tools require an explicit workspace.
Pass `workspace_dir=...` or set `POWER_LOOP_WORKSPACE`; power-loop does not
fall back to the process current working directory. `load_skill` uses
`AgentLoopConfig.skills_dir`, `skills_dir=...`, or `POWER_LOOP_SKILLS_DIR`.
Custom tools are unaffected and own their own path/config handling.

### Per-call allowlists

Register the superset once, then expose only the tools allowed for one run:

```python
result = await loop.send(
    "Inspect the project",
    session_id=sid,
    tools=["read_file", "glob", "grep"],
)
```

The sequence is resolved through `ToolRegistry.subset()`, so unknown names are
ignored and the LLM only receives definitions for the selected tools.
`ToolRegistry.names()` returns the registered names. You may also pass a
separate `ToolRegistry` directly.

### Unbound registries

For a host that reuses one registry across workspaces, defer environment
resolution until handler invocation:

```python
from power_loop import RuntimeEnv, create_default_tool_registry, runtime_env_context

registry = create_default_tool_registry(preset="core", bind=False)

with runtime_env_context(RuntimeEnv(workspace_dir="/srv/tenant-a")):
    result = await registry.invoke_async("read_file", {"path": "README.md"})
```

`DEFAULT_TOOL_HANDLERS` is public for hosts that need to compose the built-in
handlers with their own definitions.

### Shell execution boundary

The default `LocalShellBackend` starts `/bin/bash` on the host and inherits the
host environment. It is orchestration, not isolation. Untrusted commands need
a host-provided `ShellBackend` that launches inside a container, gVisor, or
another sandbox. `session_key(workspace_dir)` identifies the execution target
used to cache persistent shell sessions; different targets must return
different keys.

Presets:

| Preset | Tools |
|---|---|
| `core` | `bash`, `read_file`, `write_file`, `edit_file`, `apply_patch`, `glob`, `grep`, `load_skill`, `request_user_input` |
| `explore` | `bash`, `read_file`, `glob`, `grep`, `load_skill`, `request_user_input` |
| `full` | `core` plus `todo`, `note`, `schedule_wakeup`/`list_wakeups`/`cancel_wakeup`, `current_time`, `recall_compacted`, `background_run`, `check_background` |

Recommended system prompt guidance:

```text
Use read_file before modifying existing files. Prefer glob for locating files and grep for searching content instead of shell find/grep. Prefer edit_file for a single exact replacement and apply_patch for multi-line or multi-hunk edits. Use bash for tests, builds, git inspection, and commands that dedicated tools cannot express. Never use bash to bypass file safety checks.
```

Tool behavior:

| Tool | Use it for | Safety and precision notes |
|---|---|---|
| `read_file` | Read text files with line numbers or list directories. | Refuses binary-looking files. Large files are paged by `offset` / `limit`. Reading records a file stamp used by write/edit/patch guards. |
| `write_file` | Create a complete new file or intentionally overwrite a whole file. | Existing files must have been read first and must not have changed since that read. Parent directories are created automatically. |
| `edit_file` | Replace one exact snippet, or all exact occurrences with `replace_all=true`. | Empty, missing, fuzzy-only, or ambiguous snippets are rejected with corrective errors. Preserves BOM and dominant line endings. |
| `apply_patch` | Apply unified-diff style hunks to one file. | Requires a prior read. Stale or ambiguous hunks are rejected instead of guessed. |
| `glob` | Find paths by glob pattern. | Bare names search recursively. Common bulky directories are skipped. Hidden paths require `include_hidden=true` or an explicit hidden pattern. |
| `grep` | Search text content by regex or literal string. | Uses ripgrep when available, with Python fallback. Results are capped, binary-looking files and bulky directories are skipped. |
| `bash` | Run tests, builds, package managers, and git commands. | Runs in a persistent workspace-rooted bash session. Timeouts restart the shell to avoid leftover commands. Privileged/device-level commands (`sudo`, `dd`, `mkfs`, …) and recursive `rm -rf` of root/home/system directories are blocked; `/tmp` and relative paths are allowed. |
| `background_run` / `check_background` | Run and inspect non-interactive long commands. | Uses a private background task table and the same basic dangerous-command checks as `bash`. |
| `todo` | Maintain an agent-visible task list. | Only one item can be `in_progress`. |
| `load_skill` | Load a named skill's detailed instructions. | Unknown skills return an error listing available skill names. |
| `request_user_input` | Pause for caller/user input. | Returns `status="waiting_for_input"` with `pending_interactions`; resume with `submit_input()`. |
| `recall_compacted` | Pull back messages that [compaction](compaction.md) folded out of the active window. | Read-only, **current session only**. Filter by `query` (substring) and/or `from_seq`/`to_seq`; capped to the most recent `limit`. In the `full` preset; cherry-pick via `include=["recall_compacted"]`. |

See [`examples/20_default_tools.py`](../../../examples/20_default_tools.py) for a runnable script that exercises every default tool without requiring a real LLM.

## Runtime-Bound Tools

Some default tools are not just functions. They participate in the agent loop:

- `todo` persists its current item list in the session SQLite database. Before each LLM round, power-loop projects that authoritative state into a transient `<current_todos>` user message. The projection is not saved into `messages`, so compaction cannot duplicate or corrupt it.
- `background_run` records task status in SQLite. When a task changes from unseen to updated or completed, the next LLM round receives a transient `<background_updates>` message. `check_background` reads the same persisted task table.
- `load_skill` uses `AgentLoopConfig.skills_dir` when configured. When `skills_dir` is set, the resolved system prompt includes the skills directory and available skill descriptions.
- `request_user_input` is a control-flow tool. It does not wait inside the Python process. Instead, it persists a pending interaction and returns `StatefulResult(status="waiting_for_input")`. The caller shows `pending_interactions` to a user or API client, then calls `await loop.submit_input(session_id, interaction_id, value)` to append the matching tool result and continue.
- `recall_compacted` reads the current session's `compacted_out` rows via `get_tool_runtime_context()`. Compaction summarizes old turns into a `compact_note` and marks the originals `compacted_out` — they leave the active window but are not deleted. This tool surfaces them on demand (read-only, session-scoped) when the note lacks a specific detail. See [Compaction](compaction.md).

This behavior is built on public primitives. `SessionStore` exposes JSON runtime state and background task APIs, `get_tool_runtime_context()` gives tool handlers the current session/store, and `AgentLoopConfig.runtime_projectors` controls how persisted state becomes transient LLM messages. The default projectors are `TodoRuntimeProjector` and `BackgroundRuntimeProjector`; pass your own `RuntimeProjector` objects to support custom tools or disable the defaults with `runtime_projectors=()`.

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

This means session state survives a new `StatefulAgentLoop` instance that shares the same `SessionStore`. Conversation history remains the protocol log; runtime state lives beside it and is projected into the prompt only when needed.

You can combine the same primitives with hooks and events for richer flow control:

- A `TOOL_BEFORE` hook can rewrite tool arguments, require approval, or skip execution.
- A `TOOL_AFTER` hook can persist derived state with `get_tool_runtime_context()`.
- Event subscribers can observe `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED` and drive UI, logs, or external schedulers.
- Tool handlers can query `ctx.loop.get_messages(ctx.session_id)` or `ctx.store.get_session(ctx.session_id)` when they need session-aware behavior.

The default tools use these same extension points. There is no private channel required for user-defined tools to build similar workflows.

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
