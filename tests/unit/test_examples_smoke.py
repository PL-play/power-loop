"""H2.9: import every examples/NN_*.py so a public-API rename breaks CI per-PR.

This only IMPORTS each example (executing its top-level imports + definitions) — it
does not run main(), which needs a live model. Semantic validation stays in the
nightly real-LLM suite (tests/real/test_examples.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# Examples import `from _helpers import ...`; put examples/ on sys.path.
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

_EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("[0-9]*.py"))


def test_there_are_examples() -> None:
    assert _EXAMPLE_FILES, "no numbered examples found — is the path right?"


@pytest.mark.parametrize("filename", _EXAMPLE_FILES)
def test_example_imports_cleanly(filename: str) -> None:
    """Importing must not raise (catches renamed/removed public symbols) and each
    example must expose a `main` entry point."""
    mod_name = f"smoke_example_{filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(mod_name, EXAMPLES_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(mod_name, None)
    assert hasattr(module, "main"), f"{filename} has no main() entry point"
