# Installation

[中文](../../zh/user-guide/installation.md) | [User Guide](../index.md)

## Requirements

- **Python 3.10+** — uses `str | None` syntax and `asyncio` improvements.
- **SQLite 3.35+** — bundled with Python on all major platforms.
- An **OpenAI-compatible LLM endpoint** — any provider that exposes a `/chat/completions` API.

## Install

```bash
# From PyPI
pip install power-loop

# From GitHub (development)
pip install git+https://github.com/deep-talk0/power-loop.git

# Editable install (multi-repo setup like DeepTalk)
git clone https://github.com/deep-talk0/power-loop.git
cd power-loop
pip install -e .

# With dev dependencies (linting, testing)
pip install -e ".[dev]"
```

## Verify

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig, __version__
print(__version__)  # → "0.2.0"
```

## Next

[Configuration](configuration.md) — set up your LLM credentials and tune the loop.
