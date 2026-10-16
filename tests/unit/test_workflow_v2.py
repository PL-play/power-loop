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


# ── 第二批：when / 超时 / 单节点取消 / 叶子会话回调 / 重启接着跑 / 运行中加轮数 ─────────


@dataclass
class _SlowLLM(LLMService):
    """system_prompt 里带 slow 的叶子每次调用等 delay 秒；classify 交 JSON；其余回显。"""

    delay: float = 3.0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        import asyncio

        sp = (request.system_prompt or "").lower()
        if "slow" in sp:
            await asyncio.sleep(self.delay)
        last = next((str(m.get("content") or "") for m in reversed(request.messages or []) if m.get("role") == "user"), "")
        txt = json.dumps({"pass": False, "n": 2}) if "classify" in sp else f"echo::{last[:200]}"
        return LLMResponse(raw_text=txt, content_text=txt)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _slow_loop(delay: float = 3.0) -> StatefulAgentLoop:
    return StatefulAgentLoop(llm=_SlowLLM(delay=delay), db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3, compactor=None))


_CLASSIFY = {"type": "agent", "id": "review", "spec": {"name": "r", "system_prompt": "classify it"},
             "output_schema": {"name": "R", "schema": {"type": "object", "properties": {
                 "pass": {"type": "boolean"}, "n": {"type": "integer"}}}}}


async def test_when_skips_node_and_run_still_completes():
    spec = {"name": "w", "input": "x", "root": {"type": "sequence", "steps": [
        _CLASSIFY,
        {"type": "agent", "id": "fix", "spec": {"name": "f", "system_prompt": "fix"},
         "when": {"ref": "review.pass", "op": "eq", "value": False}, "input": "fix it"},
        {"type": "agent", "id": "celebrate", "spec": {"name": "c", "system_prompt": "party"},
         "when": {"ref": "review.pass"}, "input": "party"},
        {"type": "agent", "id": "after", "spec": {"name": "a", "system_prompt": "after"},
         "inputs_from": ["celebrate"], "input": "go"},
    ]}}
    res = await create_workflow(spec, parent_loop=_slow_loop()).run()
    assert res.status == "completed", res.errors
    assert res.results["fix"].status == "completed"
    assert res.results["celebrate"].status == "skipped" and "条件不成立" in (res.results["celebrate"].error or "")
    assert "条件不成立，跳过了" in res.results["after"].text


def test_when_validation():
    from power_loop.workflow import WorkflowSpecError

    bad = {"name": "w", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "x"},
         "when": {"ref": "b.pass", "op": "eq", "value": True}},
        {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "x"}},
    ]}}
    with pytest.raises(WorkflowSpecError) as exc:
        create_workflow(bad, parent_loop=_slow_loop())
    assert any("when references 'b'" in p for p in exc.value.problems)
    with pytest.raises(WorkflowSpecError):
        create_workflow({"name": "w", "root": {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "x"},
                                               "when": {"ref": "a", "op": "between"}}}, parent_loop=_slow_loop())


async def test_run_timeout_stops_and_fails():
    import time

    spec = {"name": "w", "input": "x", "timeout_s": 1,
            "root": {"type": "agent", "id": "s", "spec": {"name": "s", "system_prompt": "slow worker"}}}
    t0 = time.monotonic()
    res = await create_workflow(spec, parent_loop=_slow_loop(delay=8)).run()
    assert time.monotonic() - t0 < 6
    assert res.status == "failed" and any("超时" in e for e in res.errors)


async def test_cancel_one_node_others_continue():
    import asyncio

    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.spec import WorkflowSpec

    spec = WorkflowSpec.from_json({"name": "w", "input": "x", "root": {"type": "parallel", "branches": [
        {"type": "agent", "id": "slow-a", "spec": {"name": "a", "system_prompt": "slow a"}},
        {"type": "agent", "id": "slow-b", "spec": {"name": "b", "system_prompt": "slow b"}},
    ]}})
    eng = WorkflowEngine(_slow_loop(delay=1.5))
    task = asyncio.create_task(eng.run(spec))
    await asyncio.sleep(0.5)
    assert eng.cancel_node("slow-a", "不要这一路了") == 1
    res = await task
    assert res.results["slow-a"].status == "cancelled" and "不要这一路了" in (res.results["slow-a"].error or "")
    assert res.results["slow-b"].status == "completed"
    assert res.status == "completed" and "slow-a" in res.tolerated


async def test_node_session_callback_and_resume_in_same_session():
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.spec import WorkflowSpec

    seen: list[tuple[str, str]] = []

    async def on_sess(nid: str, sid: str) -> None:
        seen.append((nid, sid))

    spec = WorkflowSpec.from_json({"name": "w", "input": "x",
                                   "root": {"type": "agent", "id": "leaf", "spec": {"name": "l", "system_prompt": "work"}}})
    loop = _slow_loop()
    res = await WorkflowEngine(loop, on_node_session=on_sess).run(spec)
    assert res.status == "completed" and seen and seen[0][0] == "leaf" and seen[0][1] == res.results["leaf"].session_id

    class _Exec:
        def __init__(self):
            self.ran, self.cont = 0, []

        async def run_agent(self, *a, **k):
            self.ran += 1
            return {"status": "completed", "final_text": "fresh"}

        async def continue_agent(self, sid, prompt, **k):
            self.cont.append((sid, prompt))
            return {"status": "completed", "final_text": "continued", "rounds": 1, "usage": {}}

    ex = _Exec()
    res2 = await WorkflowEngine(loop, executor=ex, resume_sessions={"leaf": "sess-old"}).run(spec)
    assert ex.ran == 0 and ex.cont[0][0] == "sess-old" and "进程重启" in ex.cont[0][1]
    assert res2.results["leaf"].text == "continued" and res2.results["leaf"].session_id == "sess-old"


def test_extend_rounds_only_for_running_session():
    import asyncio

    from power_loop.agent.stateful_loop import _session_sync, take_round_bonus

    loop = _slow_loop()
    assert loop.extend_rounds("sess-x", 5) is False          # 没在跑

    async def go() -> tuple[bool, int, int]:
        entry = _session_sync("sess-y", owner=id(loop))
        async with entry.lock:
            ok = loop.extend_rounds("sess-y", 7)
            return ok, take_round_bonus("sess-y"), take_round_bonus("sess-y")

    ok, first, second = asyncio.run(go())
    assert ok and first == 7 and second == 0
