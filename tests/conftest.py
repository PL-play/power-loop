"""Pytest configuration.

Test-tier policy (matches ROADMAP §四条护栏 4):

- ``unit``         : pure control-flow / contract tests with fake LLM. Always on.
- ``integration``  : multi-component scenarios with fake LLM. Always on.
- ``real_llm``     : hits the real configured LLM provider.
                     - Default ON locally.
                     - ``--no-real`` reverses to skip.
                     - Automatically skipped if required env vars are missing
                       (so CI without secrets stays green).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv is a hard dep, but be defensive.
    pass


REAL_LLM_ENV_GROUPS = (
    ("POWER_LOOP_BASE_URL", "POWER_LOOP_API_KEY", "POWER_LOOP_MODEL"),
    ("OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_MODEL"),
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-real",
        action="store_true",
        default=False,
        help="Skip tests marked @pytest.mark.real_llm (which are ON by default).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    no_real = config.getoption("--no-real")
    has_real_env = any(all(os.environ.get(key) for key in group) for group in REAL_LLM_ENV_GROUPS)

    if no_real:
        skip_reason = "skipped via --no-real"
    elif not has_real_env:
        skip_reason = (
            "real_llm skipped: missing POWER_LOOP_* or OPENAI_COMPAT_* provider env. "
            "Set one complete group in .env (locally) or as repo secrets (CI)."
        )
    else:
        skip_reason = None

    if skip_reason is None:
        return

    skip_marker = pytest.mark.skip(reason=skip_reason)
    for item in items:
        if "real_llm" in item.keywords:
            item.add_marker(skip_marker)
