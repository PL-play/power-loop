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


# A FROZEN snapshot of the 1.0 STABLE tier. Adding to STABLE_API is fine; REMOVING or
# RENAMING any of these without a MAJOR version bump (→ 2.0.0) must fail this test — that
# is the SemVer promise the module docstring makes, made enforceable. The LLM-contract
# symbols joined the frozen surface at 1.0 to give StatefulAgentLoop construction closure.
_FROZEN_STABLE_V0 = frozenset({
    "StatefulAgentLoop", "StatefulResult", "FollowUpQueued",
    "AgentLoopConfig", "AgentLoopResult", "SessionStore", "SubagentLifecycle",
    "PowerLoopError", "SessionPendingError", "SessionNotFoundError",
    "LLMTimeout", "LLMRetryExhausted", "CancellationRequested",
    "LLMRetryPolicy", "CancellationToken",
    "AgentHooks", "AgentEventBus", "HookPoint", "HookDirective",
    "ToolRegistry", "ToolDefinition",
    # construction closure (1.0): build + use + implement a provider from STABLE symbols
    "LLMService", "LLMRequest", "LLMResponse", "LLMStreamChunk",
    "LLMProviderConfig", "create_llm_service_from_env", "create_llm_service_from_config",
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


# The error `.code` dotted strings are an advertised machine-readable contract
# (errors.py: "branch on exc.code, not class identity"). At 1.0 they are FROZEN: a
# changed code string is a breaking change requiring a MAJOR bump — so pin them here.
_FROZEN_ERROR_CODES = {
    "PowerLoopError": "power_loop.error",
    "SessionPendingError": "session.pending",
    "SessionNotFoundError": "session.not_found",
    "LLMTimeout": "llm.timeout",
    "LLMRetryExhausted": "llm.retry_exhausted",
    "CancellationRequested": "cancelled",
}


def test_stable_error_codes_are_frozen() -> None:
    """Each STABLE exception's machine-readable `.code` is part of the 1.0 contract;
    changing one is a MAJOR-bump event (update this baseline consciously if so)."""
    for cls_name, expected in _FROZEN_ERROR_CODES.items():
        assert cls_name in power_loop.STABLE_API, f"{cls_name} must stay a STABLE export"
        cls = getattr(power_loop, cls_name)
        assert cls.code == expected, (
            f"{cls_name}.code changed {expected!r} -> {cls.code!r}: that breaks the "
            "documented exc.code contract — requires a MAJOR version bump."
        )
