from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_PARENT_HELPER = Path(__file__).resolve().parents[1] / "_helpers.py"
_SPEC = importlib.util.spec_from_file_location("_power_loop_examples_helpers", _PARENT_HELPER)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load examples helper: {_PARENT_HELPER}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

make_llm: Any = _MODULE.make_llm

__all__ = ["make_llm"]
