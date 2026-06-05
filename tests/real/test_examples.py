"""Real-LLM contract tests for ``examples/``.

These double as living documentation: if a numbered example breaks, either
the example needs an update or a public API regressed. Each example must
remain runnable as ``python examples/NN_*.py`` against real DashScope.
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
    module = _load_example("00_minimal.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str)
    assert final_text.strip(), "model returned an empty reply"


def test_example_01_tool_use_runs() -> None:
    module = _load_example("01_tool_use.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str) and final_text.strip()
    # The lookup tool only knows "pad thai" for Bangkok; if the loop
    # routed correctly through the tool, the answer mentions it.
    assert "pad thai" in final_text.lower(), (
        f"expected the answer to surface the tool result; got: {final_text!r}"
    )


def test_example_02_subagent_runs() -> None:
    module = _load_example("02_subagent.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str) and final_text.strip()
    assert "tokyo" in final_text.lower() or "东京" in final_text, (
        f"sub-agent answer did not surface 'Tokyo': {final_text!r}"
    )


def test_example_03_compaction_runs() -> None:
    module = _load_example("03_compaction.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str) and final_text.strip()
    assert "jupiter" in final_text.lower(), (
        f"compacted-history answer did not name Jupiter: {final_text!r}"
    )
