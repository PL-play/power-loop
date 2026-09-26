"""Real-provider cover for design/125: the probe reaches the right verdicts on real models, and
view_image lets a loop look at a workspace file — or says plainly that it cannot.

Runs on the endpoint in ``.env`` (DeepSeek, ``deepseek-flash``: sees images, rejects native
json_schema). ``deepseek-v4-pro`` on the same endpoint does NOT see images (2026-09-26: it answers
"无法确定") — it is the real "no" here; if DeepSeek ever gives it vision, that assertion flags it.
A bogus key (401) and a wrong model name (DeepSeek answers 400 "The supported API model names are
…, but you passed …") must come back inconclusive with ``error_kind`` auth / model_missing.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import struct
import zlib

import pytest

from power_loop import AgentLoopConfig, LLMProviderConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.capability_probe import _COLOURS, probe_capabilities
from power_loop.tools import create_default_tool_registry

from ._llm import make_llm

pytestmark = pytest.mark.skipif(
    not os.environ.get("POWER_LOOP_API_KEY"), reason="needs the real provider in .env")


def _probe(model: str | None = None):
    cfg = LLMProviderConfig.from_env()
    if model:
        cfg = dataclasses.replace(cfg, model=model)
    return asyncio.run(probe_capabilities(cfg))


def test_probe_on_a_vision_model_that_rejects_native_json_schema() -> None:
    rep = _probe()
    checks = {k: (v.status, v.evidence) for k, v in rep.checks.items()}
    assert rep.checks["image_input"].status == "yes", checks
    assert rep.checks["json_schema"].status == "no", checks
    assert "response_format" in rep.checks["json_schema"].evidence, checks
    assert rep.checks["tools"].status == "yes", checks
    assert rep.checks["thinking"].status == "yes", checks
    assert rep.capabilities() == {"supports_image_input": True, "supports_json_schema": False}


def test_probe_on_a_model_that_cannot_see() -> None:
    rep = _probe("deepseek-v4-pro")
    img = rep.checks["image_input"]
    assert img.status == "no", (
        f"deepseek-v4-pro judged {img.status!r} ({img.evidence}) — if it gained vision, update this test")


# ── a broken configuration says what is broken ───────────────────────────────


def test_probe_with_a_bogus_key_is_inconclusive_auth_everywhere() -> None:
    cfg = dataclasses.replace(LLMProviderConfig.from_env(), api_key="sk-bogus0000000000000000000000")
    rep = asyncio.run(probe_capabilities(cfg))
    got = {k: (v.status, v.error_kind, v.evidence) for k, v in rep.checks.items()}
    assert {k: v[:2] for k, v in got.items()} == {
        k: ("inconclusive", "auth") for k in ("image_input", "json_schema", "tools", "thinking")}, got
    assert rep.capabilities() == {}
    assert all("sk-bogus0000" not in v.evidence for v in rep.checks.values()), got


def test_probe_with_a_nonexistent_model_is_inconclusive_model_missing() -> None:
    rep = _probe("deepseek-no-such-model")
    got = {k: (v.status, v.error_kind, v.evidence) for k, v in rep.checks.items()}
    assert {k: v[:2] for k, v in got.items()} == {
        k: ("inconclusive", "model_missing") for k in ("image_input", "json_schema", "tools", "thinking")}, got
    assert rep.capabilities() == {}


# ── view_image through a real loop ───────────────────────────────────────────


def _two_colour_png(left, right, size: int = 64) -> bytes:
    half = size // 2
    raw = (b"\x00" + bytes(left) * half + bytes(right) * (size - half)) * size

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _ask_about_file(tmp_path, capabilities) -> tuple[str, tuple, tuple, list[str]]:
    left, right = random.Random(11).sample(_COLOURS, 2)
    (tmp_path / "shot.png").write_bytes(_two_colour_png(left[2], right[2]))
    tools_called: list[str] = []

    async def _run() -> str:
        store = await SessionStore.open(":memory:")
        try:
            loop = StatefulAgentLoop(
                llm=make_llm(max_tokens=4096, temperature=0.0, capabilities=capabilities),
                store=store,
                tool_registry=create_default_tool_registry(include=["view_image"], workspace_dir=tmp_path),
                config=AgentLoopConfig(system_prompt="You can look at workspace images with view_image.",
                                       max_rounds=4, max_tokens=4096),
            )
            from power_loop import AgentEventType

            loop.event_bus.subscribe(AgentEventType.TOOL_CALL_STARTED,
                                     lambda e: tools_called.append(e.data.name))
            sid = await loop.new_session()
            r = await loop.send("用 view_image 看 shot.png，告诉我左半边和右半边各是什么颜色。只回颜色。", sid)
            return r.final_text or ""
        finally:
            await store.close()

    return asyncio.run(_run()), left, right, tools_called


def test_view_image_lets_a_vision_model_see_a_workspace_file(tmp_path) -> None:
    answer, left, right, called = _ask_about_file(tmp_path, {"supports_image_input": True})
    assert "view_image" in called, called
    low = answer.lower()
    assert any(w in low for w in (left[0], *left[1])) and any(w in low for w in (right[0], *right[1])), (
        f"expected {left[0]}/{right[0]}: {answer!r}")


def test_view_image_on_a_model_declared_blind_does_not_make_it_up(tmp_path) -> None:
    answer, left, right, called = _ask_about_file(tmp_path, None)
    assert "view_image" in called, called
    low = answer.lower()
    both = any(w in low for w in (left[0], *left[1])) and any(w in low for w in (right[0], *right[1]))
    assert not both, f"model named both colours of a picture it was never shown: {answer!r}"
