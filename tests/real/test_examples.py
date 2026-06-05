"""Real-LLM contract tests for ``examples/``.

These double as living documentation: if a numbered example breaks, either
the example needs an update or a public API regressed. Each example must
remain runnable as ``python examples/NN_*.py`` against real DashScope.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# Examples use ``from _helpers import …`` — they're not a package, just a
# folder of scripts. Putting examples/ on sys.path makes that import work
# both for ``python examples/NN_*.py`` (auto) and for our test loader.
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


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
    assert isinstance(final_text, str) and final_text.strip()


def test_example_01_multi_turn_runs() -> None:
    module = _load_example("01_multi_turn.py")
    final_text = asyncio.run(module.main())
    # The fact established in turn 1 must surface in turn 2.
    assert "teal" in final_text.lower(), (
        f"multi-turn answer did not recall 'teal': {final_text!r}"
    )


def test_example_02_tool_use_runs() -> None:
    module = _load_example("02_tool_use.py")
    final_text = asyncio.run(module.main())
    assert "pad thai" in final_text.lower(), (
        f"expected the answer to surface the tool result; got: {final_text!r}"
    )


def test_example_03_subagent_runs() -> None:
    module = _load_example("03_subagent.py")
    final_text = asyncio.run(module.main())
    assert "tokyo" in final_text.lower() or "东京" in final_text, (
        f"sub-agent answer did not surface 'Tokyo': {final_text!r}"
    )


def test_example_04_compaction_runs() -> None:
    module = _load_example("04_compaction.py")
    final_text = asyncio.run(module.main())
    assert "jupiter" in final_text.lower(), (
        f"compacted-history answer did not name Jupiter: {final_text!r}"
    )


def test_example_05_pending_resume_runs() -> None:
    module = _load_example("05_pending_resume.py")
    final_text = asyncio.run(module.main())
    assert "hypertext" in final_text.lower().replace("-", "").replace(" ", ""), (
        f"post-abort send did not produce the expected answer: {final_text!r}"
    )


def test_example_06_declarative_subagent_runs() -> None:
    module = _load_example("06_declarative_subagent.py")
    final_text = asyncio.run(module.main())
    # The orchestrator delegates (17+25)*3 = 126; the answer should surface
    # that number (subagent may include extra prose).
    assert "126" in final_text, (
        f"declarative subagent answer missing '126': {final_text!r}"
    )
