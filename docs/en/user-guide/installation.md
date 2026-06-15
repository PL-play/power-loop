# Installation

[中文](../../zh/user-guide/installation.md) | [User Guide](../index.md)

## Requirements

- **Python 3.10+** — uses `str | None` syntax and `asyncio` improvements.
- **SQLite 3.35+** — bundled with Python on all major platforms.
- An **OpenAI-compatible LLM endpoint** (or an Anthropic Messages API endpoint) — plus the
  matching transport extra (`[openai]` or `[anthropic]`); the core ships SDK-free.

## Install

The core has **zero runtime dependencies** (pure stdlib); the OpenAI/Anthropic SDK is pulled
in only by the extra you install. Constructing a provider without its extra raises a clear `ImportError`.

```bash
# Pick a transport extra:
pip install 'power-loop[openai]'      # OpenAI-compatible /chat/completions (DashScope, DeepSeek, Ollama/vLLM, …)
pip install 'power-loop[anthropic]'   # Anthropic Messages API endpoints
pip install 'power-loop[skills]'      # YAML skill frontmatter (load_skill)
pip install 'power-loop[pdf]'         # PDF input in the multimodal helper
pip install 'power-loop[all]'         # both transports + skills + pdf

# From GitHub (development)
pip install "power-loop[openai] @ git+https://github.com/PL-play/power-loop.git"

# Editable install (multi-repo setup like DeepTalk)
git clone https://github.com/PL-play/power-loop.git
cd power-loop
pip install -e '.[openai]'            # add a transport extra so a provider can be built

# With dev dependencies (linting, testing — includes both transports)
pip install -e ".[dev]"
```

## Verify

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig, __version__
print(__version__)  # → "0.14.0"
```

## Next

[Configuration](configuration.md) — set up your LLM credentials and tune the loop.
