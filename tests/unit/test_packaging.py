"""Packaging guarantees that are easy to regress silently."""

from __future__ import annotations

import pathlib

import pytest

import power_loop

pytestmark = pytest.mark.unit


def test_py_typed_marker_is_shipped() -> None:
    """PEP 561: the inline-types marker must sit next to __init__ so downstream
    type-checkers pick up power-loop's annotations (H3.3)."""
    marker = pathlib.Path(power_loop.__file__).parent / "py.typed"
    assert marker.is_file(), "power_loop/py.typed is missing — downstream mypy/pyright lose all types"
