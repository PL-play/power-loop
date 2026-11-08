"""Per-send tool/system_prompt overrides, ToolRegistry.subset, unbound registry,
ShellBackend.session_key — power-loop 0.7.0 additions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from power_loop import (
    AgentLoopConfig,
    LocalShellBackend,
    RequestOptions,
    RuntimeEnv,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    runtime_env_context,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.pricing import Free


@dataclass
class _Recorder(LLMService):
    """Records each request's tool names + system prompt; replies with plain text."""

    seen_tools: list[list[str]] = field(default_factory=list)
    seen_system: list[str | None] = field(default_factory=list)
    seen_format: list[Any] = field(default_factory=list)

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        names = [
            (t.get("function") or {}).get("name")
            for t in (request.tools or [])
        ]
        # pass_turn 永远在（design/188，单独测）：这里只看业务工具
        self.seen_tools.append([n for n in names if n and n != "pass_turn"])
        self.seen_system.append(request.system_prompt)
        self.seen_format.append(request.response_format)
        return LLMResponse(raw_text='{"ok":true}' if request.response_format else "ok")

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
        r.register(ToolDefinition(name=name, description=name, pricing=Free("test_fixture")), _noop)
    return r


async def _loop(runtime_agent, llm: _Recorder, *, system_prompt: str = "BASE"):
    return await runtime_agent(
        model=llm, registry=_registry(),
        config=AgentLoopConfig(system_prompt=system_prompt, max_rounds=1, compactor=None),
    )


# ── ToolRegistry.subset ──────────────────────────────────────────────────


def test_subset_keeps_only_named_tools():
    sub = _registry().subset(["alpha", "gamma", "missing"])
    assert sorted(sub.names()) == ["alpha", "gamma"]


# ── per-send tools override ──────────────────────────────────────────────


async def test_send_tools_allowlist_limits_what_model_sees(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    await loop.send("hi", options=RequestOptions(tools=["alpha"]))
    assert llm.seen_tools[-1] == ["alpha"]


async def test_send_without_override_sees_all_tools(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    await loop.send("hi")
    assert sorted(llm.seen_tools[-1]) == ["alpha", "beta", "gamma"]


async def test_send_tools_accepts_a_registry(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    only_beta = _registry().subset(["beta"])
    await loop.send("hi", options=RequestOptions(tools=list(only_beta.names())))
    assert llm.seen_tools[-1] == ["beta"]


# ── per-send system_prompt override ──────────────────────────────────────


async def test_send_system_prompt_override(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm, system_prompt="BASE")
    await loop.send("hi", options=RequestOptions(system_prompt="OVERRIDE-PROMPT"))
    assert llm.seen_system[-1] is not None
    assert "OVERRIDE-PROMPT" in llm.seen_system[-1]
    assert "BASE" not in llm.seen_system[-1]


async def test_send_default_uses_config_prompt(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm, system_prompt="BASE-PROMPT")
    await loop.send("hi")
    assert "BASE-PROMPT" in (llm.seen_system[-1] or "")


async def test_send_forwards_combined_overrides(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm, system_prompt="BASE")
    await loop.send("hi", options=RequestOptions(tools=["gamma"], system_prompt="SYNC-PROMPT"))
    assert llm.seen_tools[-1] == ["gamma"]
    assert "SYNC-PROMPT" in (llm.seen_system[-1] or "")


async def test_idle_input_forwards_combined_overrides(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm, system_prompt="BASE")
    await loop.send("hi", options=RequestOptions(tools=["beta"], system_prompt="FOLLOW-UP-PROMPT"))
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


# ── per-send response_format ──────────────────────────────────────────────
#
# 一个 loop 常常被**多个调用方共用**（宿主按 agent 定义缓存一个 loop，很多会话跑在上面）。
# 想要 JSON 的那一次 send 如果去改 loop 的 config，就会把所有并发的 send 一起翻成 JSON 模式。
# 所以这必须是 per-run 覆盖，且不能污染 self.config。


async def test_send_response_format_reaches_the_request(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    await loop.send("hi", options=RequestOptions(response_format={"type": "json_object"}))
    assert llm.seen_format[-1] == {"type": "json_object"}


async def test_send_without_response_format_sends_none(runtime_agent):
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    await loop.send("hi")
    assert llm.seen_format[-1] is None


async def test_response_format_does_not_leak_into_shared_config(runtime_agent):
    """下一次不带它的 send 必须干净 —— 否则一次 JSON 请求会污染整个 loop。"""
    llm = _Recorder()
    loop = await _loop(runtime_agent, llm)
    await loop.send("hi", options=RequestOptions(response_format={"type": "json_object"}))
    loop.target = await loop.loop.new_session(loop.descriptor, actor=loop.actor)
    await loop.send("hi")
    assert llm.seen_format == [{"type": "json_object"}, None]
    assert loop.config.response_format is None


async def test_config_level_response_format_still_applies(runtime_agent):
    """没传 per-send 时,配置里的那份照常生效（不许被覆盖成 None）。"""
    llm = _Recorder()
    loop = await runtime_agent(
        model=llm, registry=_registry(),
        config=AgentLoopConfig(system_prompt="B", max_rounds=1, compactor=None,
                               response_format={"type": "json_object"}),
    )
    await loop.send("hi")
    assert llm.seen_format[-1] == {"type": "json_object"}
