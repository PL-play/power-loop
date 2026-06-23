# CLAUDE.md — power-loop

Zero-dependency-core agentic loop library (PyPI `power-loop`). Consumed by DeepTalk's
`agent` + `dispatcher` services. Public **STABLE** API is the `STABLE_API` tuple in
`power_loop/__init__.py` (a SemVer guard test enforces it; breaking a STABLE symbol
requires a major bump + CHANGELOG entry).

## Environment / commands
- venv: `power-loop/.venv` — run everything through it (`.venv/bin/python`, `.venv/bin/ruff`).
- Tests: `.venv/bin/python -m pytest tests/unit -q -p no:cacheprovider` (fast).
- Lint: `.venv/bin/ruff check --no-cache <paths>` (the `.ruff_cache`/`.pytest_cache` dirs
  aren't writable here — always pass `--no-cache` / `-p no:cacheprovider`).

## Real-LLM tests — available, use them
`.env` (in this repo) holds a **working real LLM config** (`POWER_LOOP_API_KEY` /
`POWER_LOOP_BASE_URL` / `POWER_LOOP_MODEL` / `POWER_LOOP_PROVIDER` + `POWER_LOOP_SUPPORTS_*`).
`tests/conftest.py` auto-loads it (`load_dotenv`) and gates the real suite on
`REAL_LLM_ENV_GROUPS`. So **`tests/real/` runs against a live provider** — use it to
validate real behavior; don't mock or skip when the point is real validation.
`.venv/bin/python -m pytest tests/real -q`. Gotchas: `rm` any stale `./power_loop_sessions.db`
first; some real tests have pre-existing cross-test async-teardown flakiness (pass in isolation).

## Publishing to PyPI
Token at **`/home/ubuntu/deeptalk/pypi_token`** (workspace root; `._pypi_token` is a macOS
sidecar — ignore). Version = `power_loop.__version__` (pyproject `dynamic`); bump it + add a
CHANGELOG entry, then:
`.venv/bin/python -m build && .venv/bin/python -m twine upload -u __token__ -p "$(cat /home/ubuntu/deeptalk/pypi_token)" dist/*`.
Publishing is irreversible/outward — only when the user explicitly asks to release. After a
release, repin DeepTalk `agent`+`dispatcher` (`power-loop[all]>=X`) and rebuild.

## Architecture pointers
- `core/pipeline.py` `AgentPipeline.run()` — the loop; hook orchestration + phases.
- `core/hooks.py` `AgentHooks` — `register(name=,replace=)/replace/remove/has`; built-in
  hooks register under `builtin.*` names (overridable by hosts). e.g. `runtime/memory.py`
  `MemoryRecallHook` (LLM_BEFORE, ephemeral tail injection) auto-registered by the loop.
- `agent/types.py` `AgentLoopConfig` — config-pluggable seams: `representation`,
  `fold_strategy`, `memory`, `compactor`, `runtime_projectors`, `microcompact_*` (default OFF).
- `agent/system_prompt.py` `resolve_runtime_system_prompt()` — single source of truth for the
  system-prompt assembly (shared by the live pipeline + the `resolve_system_prompt` preview).
- `runtime/store/` — pluggable async store (SQLite / Postgres / MySQL), `pl_`-prefixed tables.
