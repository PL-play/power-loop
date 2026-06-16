"""Per-send tool/system_prompt overrides, ToolRegistry.subset, unbound registry,
ShellBackend.session_key — power-loop 0.7.0 additions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from power_loop import (
    AgentLoopConfig,
    LocalShellBackend,
    RuntimeEnv,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    runtime_env_context,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk


@dataclass
class _Recorder(LLMService):
    """Records each request's tool names + system prompt; replies with plain text."""

    seen_tools: list[list[str]] = field(default_factory=list)
    seen_system: list[str | None] = field(default_factory=list)

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        names = [
            (t.get("function") or {}).get("name")
            for t in (request.tools or [])
        ]
        self.seen_tools.append([n for n in names if n])
        self.seen_system.append(request.system_prompt)
        return LLMResponse(raw_text="ok")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _noop(**_: Any) -> str:
    return "ok"


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    for name in ("alpha", "beta", "gamma"):
        r.register(ToolDefinition(name=name, description=name), _noop)
    return r


def _loop(llm: _Recorder, *, system_prompt: str = "BASE") -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm,
        db_path=":memory:",
        tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt=system_prompt, max_rounds=1, compactor=None),
    )


# ── ToolRegistry.subset ──────────────────────────────────────────────────


def test_subset_keeps_only_named_tools():
    sub = _registry().subset(["alpha", "gamma", "missing"])
    assert sorted(sub.names()) == ["alpha", "gamma"]


# ── per-send tools override ──────────────────────────────────────────────


async def test_send_tools_allowlist_limits_what_model_sees():
    llm = _Recorder()
    loop = _loop(llm)
    sid = await loop.new_session()
    await loop.send("hi", sid, tools=["alpha"])
    assert llm.seen_tools[-1] == ["alpha"]


async def test_send_without_override_sees_all_tools():
    llm = _Recorder()
    loop = _loop(llm)
    sid = await loop.new_session()
    await loop.send("hi", sid)
    assert sorted(llm.seen_tools[-1]) == ["alpha", "beta", "gamma"]


async def test_send_tools_accepts_a_registry():
    llm = _Recorder()
    loop = _loop(llm)
    sid = await loop.new_session()
    only_beta = _registry().subset(["beta"])
    await loop.send("hi", sid, tools=only_beta)
    assert llm.seen_tools[-1] == ["beta"]


# ── per-send system_prompt override ──────────────────────────────────────


async def test_send_system_prompt_override():
    llm = _Recorder()
    loop = _loop(llm, system_prompt="BASE")
    sid = await loop.new_session()
    await loop.send("hi", sid, system_prompt="OVERRIDE-PROMPT")
    assert llm.seen_system[-1] is not None
    assert "OVERRIDE-PROMPT" in llm.seen_system[-1]
    assert "BASE" not in llm.seen_system[-1]


async def test_send_default_uses_config_prompt():
    llm = _Recorder()
    loop = _loop(llm, system_prompt="BASE-PROMPT")
    sid = await loop.new_session()
    await loop.send("hi", sid)
    assert "BASE-PROMPT" in (llm.seen_system[-1] or "")


def test_send_sync_forwards_per_call_overrides():
    llm = _Recorder()
    loop = _loop(llm, system_prompt="BASE")
    sid = asyncio.run(loop.new_session())
    loop.send_sync("hi", sid, tools=["gamma"], system_prompt="SYNC-PROMPT")
    assert llm.seen_tools[-1] == ["gamma"]
    assert "SYNC-PROMPT" in (llm.seen_system[-1] or "")


def test_follow_up_sync_forwards_overrides_when_idle():
    llm = _Recorder()
    loop = _loop(llm, system_prompt="BASE")
    sid = asyncio.run(loop.new_session())
    loop.follow_up_sync("hi", sid, tools=["beta"], system_prompt="FOLLOW-UP-PROMPT")
    assert llm.seen_tools[-1] == ["beta"]
    assert "FOLLOW-UP-PROMPT" in (llm.seen_system[-1] or "")


# ── unbound registry + session_key ───────────────────────────────────────


def test_create_unbound_registry_needs_no_workspace():
    # bind=False must not raise even without a workspace configured.
    reg = create_default_tool_registry(preset="core", bind=False)
    assert "bash" in reg.names()


async def test_unbound_registry_resolves_runtime_env_at_invocation(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "tenant.txt").write_text("alpha", encoding="utf-8")
    (second / "tenant.txt").write_text("beta", encoding="utf-8")
    reg = create_default_tool_registry(include=["read_file"], bind=False)

    with runtime_env_context(RuntimeEnv(workspace_dir=first)):
        first_result = await reg.invoke_async("read_file", {"path": "tenant.txt"})
    with runtime_env_context(RuntimeEnv(workspace_dir=second)):
        second_result = await reg.invoke_async("read_file", {"path": "tenant.txt"})

    assert "alpha" in first_result
    assert "beta" in second_result


def test_local_backend_session_key_is_stable_per_workspace():
    b = LocalShellBackend()
    assert b.session_key(Path("/w")) == b.session_key(Path("/w"))
    assert b.session_key(Path("/w")) != b.session_key(Path("/other"))


class _TargetBackend(LocalShellBackend):
    def __init__(self, target: str):
        self.target = target

    def session_key(self, workspace_dir: Path):
        return ("target", self.target, str(workspace_dir))


def test_bash_sessions_are_cached_by_execution_target(tmp_path, monkeypatch):
    from power_loop.tools import default_tools

    class _FakeBashSession:
        def __init__(self, cwd, backend):
            self.cwd = cwd
            self.backend = backend

        def close(self):
            return None

    monkeypatch.setattr(default_tools, "BashSession", _FakeBashSession)
    sessions = default_tools._BASH_SESSIONS

    saved = dict(sessions)
    sessions.clear()
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=_TargetBackend("a"))):
            first = default_tools._bash_for_current_workspace()
            again = default_tools._bash_for_current_workspace()
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=_TargetBackend("b"))):
            second = default_tools._bash_for_current_workspace()

        assert first is again
        assert first is not second
        assert first.backend.session_key(tmp_path) != second.backend.session_key(tmp_path)
    finally:
        for session in sessions.values():
            session.close()
        sessions.clear()
        sessions.update(saved)
