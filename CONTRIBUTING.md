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

## Releasing (maintainers)

The release is reproducible by anyone with PyPI rights — no hidden steps:

1. Bump `power_loop.__version__` in `power_loop/__init__.py` (the version is read from
   there; `pyproject.toml` has no separate version string).
2. Move the `CHANGELOG.md` `[Unreleased]` entries under a new `## [X.Y.Z] — DATE` heading;
   leave a fresh empty `[Unreleased]`. Call out any breaking change explicitly and list the
   affected Public API (post-1.0, a STABLE-API break requires a major bump).
3. Commit, then `git tag -a vX.Y.Z -m "…"` and push `main` + the tag.
4. Build clean and publish:
   ```bash
   rm -rf build dist *.egg-info
   python -m build
   twine check dist/*
   twine upload dist/power_loop-X.Y.Z*
   ```
   The wheel must ship **only** the `power_loop` package (no bare `llm_client`/`bench`) —
   `tests/unit/test_packaging.py` and the build guard this.
5. CI (`.github/workflows/ci.yml`) gates every push: ruff, mypy, and `pytest --no-real`
   with the coverage floor. The `import-without-extras` job proves the core stays
   dependency-free; the `bench` workflow runs the SCALE-1 smoke (non-blocking).

SemVer (post-1.0): a break to the **STABLE** API is a **major** bump (`2.0.0`); additive/new
surface is a minor; a pure fix is a patch. Provisional symbols may still change in a minor.

## Where to Start

- [ROADMAP_1.0.md](ROADMAP_1.0.md) — the road to 1.0 (current); [ROADMAP.md](ROADMAP.md) — historical M0–M3 plan
- [docs/README.md](docs/README.md) — documentation index
- [examples/](examples/) — runnable examples, each covering one concept
- [SECURITY.md](SECURITY.md) — security model + how to report a vulnerability