"""G3 regression: a background task's terminal status must never be lost when its owning
event loop is gone.

The daemon thread persists the final status via ``run_coroutine_threadsafe`` on the loop
captured at submit time (the store's lock / a PG-MySQL pool are loop-bound). Pre-fix, if
that loop was closed (a per-call ``asyncio.run`` that already returned, or shutdown), the
write-back was silently swallowed and the durable row stayed at ``running`` forever. Now an
undeliverable write-back is deferred and recovered by the next ``check`` or by ``aclose``.
"""

from __future__ import annotations

import asyncio

import pytest

from power_loop.agent.stateful_loop import StatefulAgentLoop
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.stub_provider import EchoLLMService
from power_loop.tools.default_tools import BackgroundManager

pytestmark = pytest.mark.unit


class _FakeLoop:
    """Minimal stand-in so get_tool_runtime_context() resolves a store + session."""

    def __init__(self, store: object) -> None:
        self.store = store
        self.config = None


def _run_bg(mgr: BackgroundManager, store: object, sid: str, workspace, command: str) -> str:
    """Fire a background task on a throwaway loop, then let that loop CLOSE — reproducing a
    captured loop that is dead by the time the daemon thread writes the terminal status."""

    async def fire() -> str:
        tok_l = set_current_loop(_FakeLoop(store))
        tok_s = set_session_id(sid)
        try:
            with runtime_env_context(RuntimeEnv(workspace_dir=workspace)):
                return await mgr.run(command)
        finally:
            reset_current_loop(tok_l)
            reset_session_id(tok_s)

    return asyncio.run(fire())


def test_orphaned_writeback_recovered_on_check(tmp_path) -> None:
    store = asyncio.run(SessionStore.open(str(tmp_path / "bg.db")))
    sid = asyncio.run(store.create_session())
    mgr = BackgroundManager()

    # The sleep makes the subprocess outlive fire()'s loop, so the write-back targets a
    # closed loop.
    started = _run_bg(mgr, store, sid, tmp_path, "sleep 0.3; echo done")
    task_id = started.split()[2]

    assert mgr.join_pending(timeout=5.0) == 0  # daemon thread (and its write-back attempt) finished

    # Could not reach the closed loop → deferred, NOT silently dropped; row still 'running'.
    row = asyncio.run(store.get_background_task(sid, task_id))
    assert row is not None and row.status == "running"
    assert any(o["task_id"] == task_id for o in mgr._orphaned)

    # A later background_run action=check on a live loop self-heals the durable row.
    async def chk() -> str:
        tok_l = set_current_loop(_FakeLoop(store))
        tok_s = set_session_id(sid)
        try:
            return await mgr.check(task_id)
        finally:
            reset_current_loop(tok_l)
            reset_session_id(tok_s)

    out = asyncio.run(chk())
    assert "completed" in out
    row2 = asyncio.run(store.get_background_task(sid, task_id))
    assert row2.status == "completed" and "done" in (row2.output_tail or "")
    assert mgr._orphaned == []
    asyncio.run(store.close())


async def test_aclose_drains_background_writeback(tmp_path) -> None:
    """aclose() must drain in-flight background tasks so their terminal status lands on the
    still-open store before it is closed (the second G3 trigger)."""
    loop = StatefulAgentLoop(llm=EchoLLMService(), db_path=str(tmp_path / "x.db"))
    sid = await loop.new_session()

    tok_l = set_current_loop(loop)
    tok_s = set_session_id(sid)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            started = await _fire(loop, "echo done")
    finally:
        reset_current_loop(tok_l)
        reset_session_id(tok_s)
    task_id = started.split()[2]

    await loop.aclose()  # drains + persists before closing the store

    verify = await SessionStore.open(str(tmp_path / "x.db"))
    row = await verify.get_background_task(sid, task_id)
    assert row is not None and row.status == "completed"
    await verify.close()


async def _fire(loop: StatefulAgentLoop, command: str) -> str:
    from power_loop.tools.default_tools import BG

    return await BG.run(command)


# ── 6.8.0 action=tool：async_capable 工具后台化 ──────────────────────────────────

class _ToolTaskReg:
    def __init__(self, defs):
        self._defs = {d.name: d for d in defs}
        self.calls = []

    def get(self, name):
        d = self._defs.get(name)
        if d is None:
            return None
        class _R:  # RegisteredTool 形状：.definition
            definition = d
        return _R()

    def definitions(self):
        return list(self._defs.values())

    async def invoke_async(self, name, args):
        self.calls.append((name, dict(args)))
        await asyncio.sleep(0.01)
        if args.get("boom"):
            raise RuntimeError("provider down")
        return f"image saved: {args.get('prompt', '')[:20]}"


class _ToolTaskLoop:
    def __init__(self, reg):
        self.tool_registry = reg


@pytest.mark.asyncio
async def test_action_tool_runs_async_capable_tool_and_notifies(monkeypatch):
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.core import agent_context
    from power_loop.tools import default_tools as dt

    reg = _ToolTaskReg([
        ToolDefinition(name="generate_image", description="d", async_capable=True),
        ToolDefinition(name="write_file", description="d"),
        ToolDefinition(name="background_run", description="d"),
    ])
    monkeypatch.setattr(agent_context, "get_current_loop", lambda: _ToolTaskLoop(reg))
    monkeypatch.setattr(dt, "_current_store_and_session", lambda: (None, None))
    done = []

    async def _cb(sid, task_id, status):
        done.append((task_id, status))

    dt.register_tool_task_callback(_cb)
    try:
        # 未标记的工具拒绝
        with pytest.raises(ValueError, match="async_capable"):
            await dt.BG.run_tool("write_file", {})
        # 递归入口拒绝
        with pytest.raises(ValueError, match="不允许后台化"):
            await dt.BG.run_tool("background_run", {})
        # happy path：立即返回 task_id，完成后回调 + check 可取结果
        out = await dt.BG.run_tool("generate_image", {"prompt": "a cat"})
        assert "task_id=" in out
        tid = out.split("task_id=", 1)[1].split("（", 1)[0]
        await asyncio.sleep(0.05)
        assert done and done[0] == (None, "completed") or done[0][1] == "completed"
        got = await dt.BG.check(tid)
        assert "[completed]" in got and "image saved" in got
        # 失败是任务结果不是崩溃
        out2 = await dt.BG.run_tool("generate_image", {"boom": True})
        tid2 = out2.split("task_id=", 1)[1].split("（", 1)[0]
        await asyncio.sleep(0.05)
        assert "[failed]" in await dt.BG.check(tid2)
        assert "provider down" in await dt.BG.check(tid2)
    finally:
        dt.register_tool_task_callback(None)


def test_async_capable_suffix_only_with_background_run_mounted():
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.tools.registry import ToolRegistry

    async def _h(**kw):
        return "x"

    def _mk(*defs):
        r = ToolRegistry()
        for d in defs:
            r.register(d, _h)
        return r

    gen = ToolDefinition(name="generate_image", description="生成图片。", async_capable=True)
    bg = ToolDefinition(name="background_run", description="bg。")
    plain = ToolDefinition(name="write_file", description="写文件。")

    with_bg = {t["function"]["name"]: t["function"]["description"]
               for t in _mk(gen, bg, plain).to_openai_tools()}
    assert "可异步" in with_bg["generate_image"]
    assert "可异步" not in with_bg["write_file"], "未标记的工具不加后缀"
    assert "可异步" not in with_bg["background_run"]

    without_bg = {t["function"]["name"]: t["function"]["description"]
                  for t in _mk(gen, plain).to_openai_tools()}
    assert "⏳" not in without_bg["generate_image"], "background_run 不在场就别教模型调它"
