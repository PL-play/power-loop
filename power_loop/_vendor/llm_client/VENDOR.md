# Vendored: `llm_client`

This directory is a **vendored copy** of the project's `llm_client` package — the
SDK-agnostic LLM transport layer (OpenAI-compatible + Anthropic Messages, the
`LLMService` Protocol, tooling/multimodal helpers). It is vendored under
`power_loop._vendor.llm_client` so power-loop ships a single self-contained
`power_loop` top-level package with **no bare, squat-prone `llm_client`** on the import
path and no external dependency on a separately-published `llm_client` distribution.

> All imports are rewritten to `power_loop._vendor.llm_client.*`. Do not add new
> external imports of a top-level `llm_client` — always go through the vendored path.

## Provenance

| Field | Value |
|---|---|
| Upstream | the project's own `llm_client` package (internal; same authorship as power-loop) |
| License | MIT (same as power-loop — see top-level `LICENSE`) |
| Vendored | 2026-06 (power-loop 0.13.x; cleaned in 0.18.0) |
| Local modifications | (1) all imports rewritten to `power_loop._vendor.llm_client.*`; (2) **removed `qwen_image.py` and `web_search.py`** in 0.18.0 — neither was imported by power-loop, and `qwen_image.py` was the *sole* importer of `certifi`, so dropping it made the core literally zero-dependency. |

> ⚠️ Upstream version/commit: this is an internally-maintained package, not a tagged
> external release — record the exact source commit here when re-syncing from a VCS.
> If the upstream license ever diverges from MIT, update this file and reassess
> compatibility before shipping.

## Re-syncing

Use `scripts/sync_vendor.sh` (repo root) to re-vendor from a source checkout — it copies
the package in, strips the unused modules, and rewrites imports to the vendored path.
After running it, re-check: `ruff check`, `mypy power_loop`, and the
`import-without-extras` CI job (the core must stay dependency-free).
