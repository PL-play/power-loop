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


# ── H3.6: STABLE_API is the single source of truth + a SemVer guard ─────────


def test_stable_api_symbols_are_exported_and_real() -> None:
    """Every STABLE_API name must be in __all__ and a real module attribute, so the
    advertised stable surface always resolves."""
    missing_attr = [n for n in power_loop.STABLE_API if not hasattr(power_loop, n)]
    missing_all = [n for n in power_loop.STABLE_API if n not in power_loop.__all__]
    assert not missing_attr, f"STABLE_API names with no module attribute: {missing_attr}"
    assert not missing_all, f"STABLE_API names not in __all__: {missing_all}"


# A FROZEN snapshot of the v0 STABLE tier. Adding to STABLE_API is fine; REMOVING or
# RENAMING any of these without a major version bump must fail this test — that is the
# SemVer promise the module docstring makes, made enforceable.
_FROZEN_STABLE_V0 = frozenset({
    "StatefulAgentLoop", "StatefulResult", "FollowUpQueued",
    "AgentLoopConfig", "AgentLoopResult", "SessionStore", "SubagentLifecycle",
    "PowerLoopError", "SessionPendingError", "SessionNotFoundError",
    "LLMTimeout", "LLMRetryExhausted", "CancellationRequested",
    "LLMRetryPolicy", "CancellationToken",
    "AgentHooks", "AgentEventBus", "HookPoint", "HookDirective",
    "ToolRegistry", "ToolDefinition",
})


def test_stable_api_semver_guard() -> None:
    """No STABLE symbol may be dropped/renamed without a major bump (and a conscious
    edit to this frozen baseline)."""
    current = set(power_loop.STABLE_API)
    removed = _FROZEN_STABLE_V0 - current
    assert not removed, (
        f"STABLE_API dropped/renamed {sorted(removed)} — that's a breaking change "
        "requiring a MAJOR version bump. If intentional, update _FROZEN_STABLE_V0."
    )
