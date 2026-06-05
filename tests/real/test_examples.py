"""Real-LLM contract tests for examples/.

These tests double as living documentation: if a numbered example breaks,
either the example needs an update or a public API regressed. Each example
must remain runnable as ``python examples/NN_*.py`` against real DashScope.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_example(filename: str):
    spec = importlib.util.spec_from_file_location(
        f"example_{filename.removesuffix('.py')}", EXAMPLES_DIR / filename
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_00_minimal_runs() -> None:
    """examples/00_minimal.py — single-shot reply via real LLM."""
    module = _load_example("00_minimal.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str)
    assert len(final_text.strip()) > 0, "model returned an empty reply"
