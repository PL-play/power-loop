# Contributing to power-loop

Thanks for considering contributing! This document explains how to set up your environment, run tests, and submit changes.

## Development Setup

```bash
git clone https://github.com/PL-play/power-loop.git
cd power-loop
pip install -e ".[dev]"
```

## Environment

Copy `.env.example` to `.env` and fill in your LLM credentials:

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-…
POWER_LOOP_MODEL=gpt-4o-mini
```

## Running tests

```bash
# All tests (including real LLM)
pytest

# Unit tests only (no LLM needed)
pytest -m "not real_llm"
```

## Code Style

- **ruff** for linting and formatting: `ruff check . && ruff format .`
- **mypy** for type checking: `mypy power_loop/`
- Line length: 120 characters
- Python 3.10+ syntax

## Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Make your changes and add tests.
4. Run `ruff check .` and `pytest`.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Submit a PR against `main`.

## Commit Convention

Write commits in the imperative mood. Reference milestone tags (`M1.1`, `M2.5`, etc.) when relevant.

## Where to Start

- [ROADMAP.md](ROADMAP.md) — current milestones and priorities
- [docs/README.md](docs/README.md) — documentation index
- [examples/](examples/) — runnable examples, each covering one concept