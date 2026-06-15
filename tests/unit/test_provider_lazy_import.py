"""H3.1: transports import lazily; the core is featherweight (no SDK at import)."""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

from power_loop.runtime.provider import (
    LLMProviderConfig,
    create_llm_service_from_config,
)

pytestmark = pytest.mark.unit


def test_importing_power_loop_does_not_pull_vendor_sdks() -> None:
    """`import power_loop` must not eagerly import anthropic/openai — they are heavy
    vendor SDKs and optional extras. Run in a clean subprocess so a SDK already
    imported by this test process can't mask a regression."""
    code = (
        "import sys, power_loop; "
        "print('anthropic' in sys.modules, 'openai' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False False", f"a vendor SDK was imported eagerly: {out}"


@pytest.mark.parametrize(
    ("provider", "module", "extra"),
    [
        ("openai", "llm_client.llm_factory", "openai"),
        ("anthropic", "llm_client.anthropic_factory", "anthropic"),
    ],
)
def test_missing_sdk_raises_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch, provider: str, module: str, extra: str
) -> None:
    """When the chosen transport's SDK is absent, construction must fail with an
    actionable ImportError naming the right extra — not a raw ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == module:
            raise ImportError(f"No module named {extra!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg = LLMProviderConfig(base_url="x", api_key="y", model="m", provider=provider)
    with pytest.raises(ImportError, match=rf"power-loop\[{extra}\]"):
        create_llm_service_from_config(cfg)
