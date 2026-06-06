# power-loop

[Documentation](docs/en/index.md) | [中文文档](docs/zh/index.md) | [Examples](examples/README.md) | [Changelog](CHANGELOG.md)

Embeddable, stateful agent execution for Python.

power-loop gives application code one small interface, `StatefulAgentLoop`, and handles the repetitive agent runtime work around it: multi-turn LLM loops, tool calls, hooks, events, context compaction, sub-agents, retry/cancel, structured output, memory, and SQLite-backed session persistence.

It is a library, not a service or a full application framework. You keep ownership of product logic, HTTP APIs, auth, queues, RAG, UI, and deployment.

## Install

```bash
pip install power-loop
```

For local development:

```bash
git clone https://github.com/PL-play/power-loop.git
cd power-loop
pip install -e ".[dev]"
```

Python 3.10+ is required.

## Quick Example

```python
import asyncio

from power_loop import AgentLoopConfig, StatefulAgentLoop, create_llm_service_from_env


async def main() -> None:
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        db_path="./power_loop_sessions.db",
        config=AgentLoopConfig(
            system_prompt="You are a concise assistant.",
            max_rounds=4,
        ),
    )

    sid = loop.new_session(metadata={"user_id": "demo"})
    first = await loop.send("My favorite color is teal.", session_id=sid)
    second = await loop.send("What is my favorite color?", session_id=sid)

    print(second.final_text)


asyncio.run(main())
```

Configure any OpenAI-compatible endpoint with environment variables:

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

See [Getting Started](docs/en/getting-started.md) for the complete first run.

## What It Provides

| Capability | Where to read more |
|---|---|
| Stateful sessions and cross-process resume | [Sessions](docs/en/user-guide/sessions.md) |
| Tool calling with JSON Schema validation | [Tools](docs/en/user-guide/tools.md) |
| Lifecycle hooks for control flow | [Hooks](docs/en/user-guide/hooks.md) |
| Typed events for streaming, audit, and metrics | [Events](docs/en/user-guide/events.md) |
| Context compaction | [Compaction](docs/en/user-guide/compaction.md) |
| Sub-agents with `AgentSpec` | [Sub-agents](docs/en/user-guide/subagents.md) |
| Retry, timeout, and cancellation | [Retry & Cancel](docs/en/user-guide/retry-cancel.md) |
| Structured JSON output | [Structured Output](docs/en/user-guide/structured-output.md) |
| Pluggable cross-session memory | [Memory](docs/en/user-guide/memory.md) |
| Provider configuration | [Providers](docs/en/user-guide/providers.md) |

## Public API

Stable imports are re-exported from `power_loop`:

```python
from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    StatefulResult,
    ToolDefinition,
    ToolRegistry,
)
```

The stability tiers are:

| Tier | Meaning |
|---|---|
| Stable | Backward compatible across minor releases. Listed in `power_loop.STABLE_API`. |
| Provisional | Available from the top-level package during 0.x, but may change. |
| Internal | Submodule imports such as `power_loop.core.*`; no compatibility promise. |

See the [API reference](docs/en/api/index.md) for the current surface.

## Examples

The `examples/` directory is ordered from minimal usage to full chatbot composition:

```bash
python examples/00_hello_world.py
python examples/02_tool_calling.py
python examples/19_full_chatbot.py
```

The full list is in [examples/README.md](examples/README.md).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q --no-real
```

Real LLM examples/tests use `POWER_LOOP_*` or the legacy `OPENAI_COMPAT_*` variables.

## Project Links

- [Documentation index](docs/README.md)
- [Architecture](docs/en/architecture.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
