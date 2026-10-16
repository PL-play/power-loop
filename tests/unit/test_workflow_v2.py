"""workflow v2（DeepTalk design/170）：节点输入里的 {{node.key}} 引用上游结构化输出。"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.workflow import create_workflow

pytestmark = pytest.mark.unit


@dataclass
class _EchoLLM(LLMService):
    """classify → JSON；其余原样回显最后一条用户消息（看下游实际收到了什么）。"""

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        sp = (request.system_prompt or "").lower()
        last = next((str(m.get("content") or "") for m in reversed(request.messages or []) if m.get("role") == "user"), "")
        txt = json.dumps({"label": "urgent", "items": ["a", "b"]}) if "classify" in sp else f"echo::{last}"
        return LLMResponse(raw_text=txt, content_text=txt)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(llm=_EchoLLM(), db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3, compactor=None))


async def test_dotted_ref_renders_upstream_payload():
    spec = {
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "classify-1", "spec": {"name": "c", "system_prompt": "classify it"},
             "output_schema": {"name": "C", "schema": {"type": "object", "properties": {
                 "label": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "agent", "id": "use", "spec": {"name": "u", "system_prompt": "use it"},
             "input": "label={{classify-1.label}} items={{classify-1.items}} miss={{classify-1.nope}}"},
        ]},
    }
    res = await create_workflow(spec, parent_loop=_loop()).run()
    assert res.status == "completed"
    text = res.results["use"].text
    assert "label=urgent" in text
    assert 'items=["a", "b"]' in text
    assert "miss=[取不到 classify-1.nope" in text
    assert "{{" not in text


# ── background_run action=wait（叶子等自己的后台任务）─────────────────────────


def test_background_wait_blocks_until_done(tmp_path):
    import asyncio

    from power_loop.core.agent_context import reset_current_loop, reset_session_id, set_current_loop, set_session_id
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.store.store import SessionStore
    from power_loop.tools.default_tools import BackgroundManager

    class _FakeLoop:
        def __init__(self, store):
            self.store, self.config = store, None

    async def go() -> tuple[str, str, str]:
        store = await SessionStore.open(str(tmp_path / "bg.db"))
        sid = await store.create_session()
        mgr = BackgroundManager()
        tl, ts = set_current_loop(_FakeLoop(store)), set_session_id(sid)
        try:
            with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
                started = await mgr.run("sleep 0.4; echo finished-ok")
                tid = started.split()[2]
                done = await mgr.wait(None, timeout_s=10)
                none = await mgr.wait(None, timeout_s=1)
                return tid, done, none
        finally:
            reset_current_loop(tl)
            reset_session_id(ts)

    tid, done, none = asyncio.run(go())
    assert done.startswith("都做完了") and tid in done and "finished-ok" in done
    assert none.startswith("没有在跑的后台任务")


def test_background_wait_times_out_without_killing(tmp_path):
    import asyncio

    from power_loop.core.agent_context import reset_current_loop, reset_session_id, set_current_loop, set_session_id
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.store.store import SessionStore
    from power_loop.tools.default_tools import BackgroundManager

    class _FakeLoop:
        def __init__(self, store):
            self.store, self.config = store, None

    async def go() -> tuple[str, str]:
        store = await SessionStore.open(str(tmp_path / "bg2.db"))
        sid = await store.create_session()
        mgr = BackgroundManager()
        tl, ts = set_current_loop(_FakeLoop(store)), set_session_id(sid)
        try:
            with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
                started = await mgr.run("sleep 2; echo late")
                tid = started.split()[2]
                early = await mgr.wait([tid], timeout_s=1)
                later = await mgr.wait([tid], timeout_s=10)
                return early, later
        finally:
            reset_current_loop(tl)
            reset_session_id(ts)

    early, later = asyncio.run(go())
    assert early.startswith("还有 1 件没做完") and "[running]" in early
    assert later.startswith("都做完了") and "late" in later


# ── 子 agent 的用量并进调用它的那一层（_USAGE_SINK）───────────────────────────


@dataclass
class _CostLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        from power_loop._vendor.llm_client.interface import LLMTokenUsage

        return LLMResponse(raw_text="ok", content_text="ok",
                           token_usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=10))

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


async def test_child_usage_flows_into_enclosing_sink():
    from power_loop.runtime.spec import _USAGE_SINK, run_agent_spec

    from power_loop.runtime.store.store import SessionStore

    store = await SessionStore.open(tempfile.mktemp(suffix=".db"))
    parent = StatefulAgentLoop(llm=_CostLLM(), store=store,
                               config=AgentLoopConfig(system_prompt="p", max_rounds=2, compactor=None))
    outer: dict[str, int] = {}
    tok = _USAGE_SINK.set(outer)
    try:
        raw = await run_agent_spec({"name": "kid", "system_prompt": "do it", "max_rounds": 2}, "go", parent_loop=parent)
    finally:
        _USAGE_SINK.reset(tok)
    assert raw["status"] == "completed"
    assert raw["usage"].get("prompt_tokens", 0) >= 100
    assert outer.get("prompt_tokens") == raw["usage"]["prompt_tokens"]
