"""Auto-apply the `real_llm` marker to every test under tests/real/.

This relies on the parent ``tests/conftest.py`` to handle the actual
skip-when-missing-env logic.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    real_marker = pytest.mark.real_llm
    for item in items:
        if "tests/real/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(real_marker)
