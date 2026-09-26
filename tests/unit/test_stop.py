"""Stopping (design/124 §8): a stop reaches exactly what it should, winds it down within a bounded
time, and never starts new work or wakes the agent afterwards.

- the stop tree: child / any_of / reason / wait;
- a stop during the model call ends the run at once (output discarded, usage estimated);
- a stop during a tool follows the tool's interrupt mode, bounded by the StopPolicy;
- a sub-agent (run_agent_spec from a tool) and an AWAITED workflow stop with their caller;
  a DETACHED workflow does not — it is stopped by id and then doesn't wake the agent;
- background tool tasks and background shell commands stop by id;
- list_jobs shows what's running (with the request it came from); stop_job stops one of them.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    AgentSpec,
    CancellationToken,
    SessionStore,
    StatefulAgentLoop,
    StopPolicy,
    run_agent_spec,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_cancel_token
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.default_tools import BG, BackgroundManager, register_tool_task_callback
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow
from power_loop.workflow.runner import get_run_handle, stop_run

USAGE = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)

# ── the stop tree ─────────────────────────────────────────────────────────────────────────


def test_child_tokens_form_a_tree() -> None:
    root = CancellationToken()
    a, b = root.child(), root.child()
    a1 = a.child()
    a.cancel("stop research")
    assert a.is_cancelled() and a1.is_cancelled() and a1.reason == "stop research"
    assert not root.is_cancelled() and not b.is_cancelled()
    root.cancel("user stop")
    assert b.is_cancelled() and b.reason == "user stop"


def test_any_of_and_never() -> None:
    own, caller = CancellationToken(), CancellationToken()
    both = CancellationToken.any_of(own, caller)
    assert not both.is_cancelled()
    caller.cancel("caller stopped")
    assert both.is_cancelled() and both.reason == "caller stopped"
    assert CancellationToken.any_of(own, None) is own
    assert CancellationToken.never().is_never and not CancellationToken().is_never


@pytest.mark.asyncio
async def test_token_wait() -> None:
    t = CancellationToken()
    asyncio.get_running_loop().call_later(0.05, t.cancel)
    await asyncio.wait_for(t.wait(poll_s=0.01), 1)


def test_stop_policy_rejects_negative() -> None:
    with pytest.raises(ValueError):
        StopPolicy(subtask_s=-1)


# ── helpers ───────────────────────────────────────────────────────────────────────────────


class _LLM(LLMService):
    """By system prompt: a prompt containing HANG blocks (a long generation); otherwise the
    scripted steps."""

    def __init__(self, steps: list[Any] | None = None) -> None:
        self.steps = list(steps or [])
        self.hanging = asyncio.Event()
        self.cancelled = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None, **kw: Any):
        if "HANG" in str(request.system_prompt or ""):
            if on_chunk_delta_text:
                on_chunk_delta_text("thinking hard…")
            self.hanging.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, LLMResponse):
            return step
        r = LLMResponse(raw_text=str(step))
        r.token_usage = USAGE
        return r

    async def close(self) -> None:
        return None


def _call(name: str) -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[{"id": "c1", "type": "function",
                                             "function": {"name": name, "arguments": "{}"}}])
    r.token_usage = USAGE
    return r


def _loop(store: SessionStore, llm: LLMService, reg: ToolRegistry | None = None, *,
          system: str = "S", policy: StopPolicy | None = None,
          bus: AgentEventBus | None = None) -> StatefulAgentLoop:
    return StatefulAgentLoop(llm=llm, store=store, tool_registry=reg, event_bus=bus,
                             config=AgentLoopConfig(system_prompt=system, max_rounds=4,
                                                    compactor=None, retry_policy=None,
                                                    stop_policy=policy or StopPolicy()))


@pytest.fixture
async def store():
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── the model call ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_during_the_model_call_ends_the_run_at_once(store) -> None:
    bus = AgentEventBus()
    calls: list[dict] = []
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: calls.append(dict(e.payload or {})))
    llm = _LLM()
    loop = _loop(store, llm, system="HANG", bus=bus)
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("write a novel", sid, stop_event=tok))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    t0 = time.monotonic()
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled" and time.monotonic() - t0 < 1.0
    assert llm.cancelled == 1
    assert calls and calls[0]["outcome"] == "aborted" and calls[0]["estimated"] is True
    rows = await store.load_active_messages(sid)
    assert not any(r.role == "assistant" for r in rows)


# ── tools ─────────────────────────────────────────────────────────────────────────────────


def _tool_loop(store, *, interrupt: str, work: Any, policy: StopPolicy | None = None):
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="t", description="t",
                                input_schema={"type": "object", "properties": {}},
                                interrupt=interrupt), work)
    llm = _LLM([_call("t"), "after"])
    return _loop(store, llm, reg, policy=policy), llm


@pytest.mark.asyncio
async def test_stop_aborts_an_abort_mode_tool(store) -> None:
    started, state = asyncio.Event(), {}

    async def work(**kw: Any) -> str:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return "never"

    loop, _ = _tool_loop(store, interrupt="abort", work=work)
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("go", sid, stop_event=tok))
    await asyncio.wait_for(started.wait(), 5)
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled" and state.get("cancelled")
    tool = next(r for r in await store.load_active_messages(sid) if r.role == "tool")
    assert tool.content.startswith("[stopped:") and "已中止" in tool.content


@pytest.mark.asyncio
async def test_stop_lets_a_finish_mode_tool_finish_within_the_limit(store) -> None:
    started = asyncio.Event()

    async def work(**kw: Any) -> str:
        started.set()
        await asyncio.sleep(0.2)
        return "real result"

    loop, _ = _tool_loop(store, interrupt="finish", work=work,
                         policy=StopPolicy(finish_tool_s=5))
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("go", sid, stop_event=tok))
    await asyncio.wait_for(started.wait(), 5)
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled"
    tool = next(r for r in await store.load_active_messages(sid) if r.role == "tool")
    assert "real result" in tool.content and tool.content.startswith("[stopped:")


@pytest.mark.asyncio
async def test_stop_cuts_a_finish_mode_tool_at_the_limit(store) -> None:
    started = asyncio.Event()

    async def work(**kw: Any) -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    loop, _ = _tool_loop(store, interrupt="finish", work=work,
                         policy=StopPolicy(finish_tool_s=0.2, abort_tool_s=0.2))
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("go", sid, stop_event=tok))
    await asyncio.wait_for(started.wait(), 5)
    t0 = time.monotonic()
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled" and time.monotonic() - t0 < 2
    tool = next(r for r in await store.load_active_messages(sid) if r.role == "tool")
    assert "强制中止" in tool.content


@pytest.mark.asyncio
async def test_a_sub_agent_stops_with_its_caller(store) -> None:
    """A tool that runs a sub-agent passes nothing explicit — run_agent_spec picks up the calling
    tool's stop token — and the parent's stop reaches the child mid-generation."""
    llm = _LLM([_call("delegate"), "after"])
    child: dict[str, Any] = {}

    async def delegate(**kw: Any) -> str:
        from power_loop.core.agent_context import get_current_loop

        child["token_seen"] = get_current_cancel_token() is not None
        out = await run_agent_spec(AgentSpec(name="kid", system_prompt="HANG"), "research",
                                   parent_loop=get_current_loop())
        child["status"] = out.get("status")
        return f"child ended: {out.get('status')}"

    reg = ToolRegistry()
    reg.register(ToolDefinition(name="delegate", description="d",
                                input_schema={"type": "object", "properties": {}},
                                interrupt="background"), delegate)
    loop = _loop(store, llm, reg, policy=StopPolicy(subtask_s=5))
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("go", sid, stop_event=tok))
    await asyncio.wait_for(llm.hanging.wait(), 5)       # the CHILD is generating
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled"
    assert child["token_seen"] and child["status"] == "cancelled"
    assert llm.cancelled == 1, "the child's model call was not interrupted"


# ── workflows ─────────────────────────────────────────────────────────────────────────────

HANG_WF = {"name": "hang", "root": {"type": "agent", "id": "a",
                                    "spec": {"name": "a", "system_prompt": "HANG"}}}


@pytest.mark.asyncio
async def test_an_awaited_workflow_stops_with_its_caller(store) -> None:
    llm = _LLM([_call("wf"), "after"])
    out: dict[str, Any] = {}

    async def run_wf(**kw: Any) -> str:
        from power_loop.core.agent_context import get_current_loop, get_session_id

        wf = create_workflow(HANG_WF, parent_loop=get_current_loop(),
                             parent_session_id=get_session_id())
        res = await wf.run()
        out["status"] = res.status
        return res.status

    reg = ToolRegistry()
    reg.register(ToolDefinition(name="wf", description="w",
                                input_schema={"type": "object", "properties": {}},
                                interrupt="background"), run_wf)
    loop = _loop(store, llm, reg, policy=StopPolicy(subtask_s=5))
    sid = await loop.new_session()
    tok = CancellationToken()
    run = asyncio.create_task(loop.send("go", sid, stop_event=tok))
    await asyncio.wait_for(llm.hanging.wait(), 5)
    tok.cancel("user stop")
    res = await asyncio.wait_for(run, 5)
    assert res.status == "cancelled" and out["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_detached_workflow_outlives_a_foreground_stop_and_stops_by_id() -> None:
    llm = _LLM()
    loop = StatefulAgentLoop(llm=llm, db_path=tempfile.mktemp(suffix=".db"),
                             config=AgentLoopConfig(system_prompt="S", max_rounds=3,
                                                    compactor=None, retry_policy=None))
    psid = await loop.new_session()
    woke: list[Any] = []

    async def on_complete(completion: Any) -> None:
        woke.append(completion)

    wf = create_workflow(HANG_WF, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True, on_complete=on_complete)
    await asyncio.wait_for(llm.hanging.wait(), 5)
    assert get_run_handle(handle.run_id) is handle
    jobs = await loop.list_jobs(psid)
    assert any(j["id"] == f"wf:{handle.run_id}" and j["status"] == "running" for j in jobs)
    res = await loop.stop_job(psid, f"wf:{handle.run_id}")
    assert res["result"] in ("stopped", "forced")
    await asyncio.sleep(0.05)
    assert woke == [], "a workflow the user stopped must not wake the agent"
    assert get_run_handle(handle.run_id) is None
    assert await stop_run(handle.run_id) == "unknown"
    await loop.aclose()


# ── background tasks by id ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_background_tool_task_stops_gracefully_and_does_not_wake(store) -> None:
    woke: list[tuple[str, str]] = []

    async def on_done(sid: str, task_id: str, status: str) -> None:
        woke.append((task_id, status))

    register_tool_task_callback(on_done)
    try:
        started = asyncio.Event()

        async def watcher(**kw: Any) -> str:
            tok = get_current_cancel_token()
            started.set()
            while not tok.is_cancelled():       # a well-behaved long task checks its token
                await asyncio.sleep(0.01)
            return "wound down"

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="watch", description="w",
                                    input_schema={"type": "object", "properties": {}},
                                    async_capable=True), watcher)
        loop = _loop(store, _LLM(), reg)
        sid = await loop.new_session()
        from power_loop.core.agent_context import (
            reset_current_loop,
            reset_session_id,
            set_current_loop,
            set_session_id,
        )

        lt, st = set_current_loop(loop), set_session_id(sid)
        try:
            msg = await BG.run_tool("watch", {})
        finally:
            reset_session_id(st)
            reset_current_loop(lt)
        task_id = msg.split("task_id=")[1].split("（")[0]
        await asyncio.wait_for(started.wait(), 5)
        res = await loop.stop_job(sid, f"bg:{task_id}")
        assert res["result"] == "stopped"
        row = await store.get_background_task(sid, task_id)
        assert row.status == "cancelled"
        await asyncio.sleep(0.05)
        assert woke == []
    finally:
        register_tool_task_callback(None)


@pytest.mark.asyncio
async def test_background_shell_command_is_killed(tmp_path) -> None:
    mgr = BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        started = await mgr.run("sleep 30; echo never")
    task_id = started.split()[2]
    for _ in range(200):
        with mgr._lock:
            if mgr.tasks[task_id].get("proc") is not None:
                break
        await asyncio.sleep(0.01)
    t0 = time.monotonic()
    res = await mgr.cancel_task(task_id, shell_term_s=2)
    assert res == "stopped" and time.monotonic() - t0 < 3
    for _ in range(300):
        with mgr._lock:
            if mgr.tasks[task_id]["status"] != "running":
                break
        await asyncio.sleep(0.01)
    assert mgr.tasks[task_id]["status"] == "cancelled"
    assert "never" not in (mgr.tasks[task_id]["result"] or "")


# ── jobs ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_jobs_shows_origin_and_stop_job_cancels_a_timer(store) -> None:
    loop = _loop(store, _LLM(["ok"]))
    sid = await loop.new_session()
    await loop.send("10 分钟后提醒我吃药", sid)
    await loop.schedule_timer(sid, delay_s=600, note="提醒吃药")
    jobs = await loop.list_jobs(sid)
    timers = [j for j in jobs if j["kind"] == "timer"]
    assert timers and timers[0]["title"] == "提醒吃药"
    assert "提醒我吃药" in timers[0]["origin"]
    res = await loop.stop_job(sid, timers[0]["id"])
    assert res["result"] == "stopped"
    assert not [j for j in await loop.list_jobs(sid) if j["kind"] == "timer"]
    assert (await loop.stop_job(sid, "send"))["result"] == "not_here"
    assert (await loop.stop_job(sid, "bg:nope"))["result"] == "unknown"


def _tagged_pids(task_id: str) -> list[int]:
    import os

    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/environ", "rb") as fh:
                if f"PL_BG_TAG={task_id}".encode() in fh.read().split(b"\0"):
                    out.append(int(p))
        except OSError:
            continue
    return out


@pytest.mark.asyncio
async def test_background_shell_stop_reaches_processes_that_left_the_group(tmp_path) -> None:
    """A process that escaped the task's process group (setsid — as a daemon, or a command run
    inside a sandbox, where the local process is only the sandbox client) is still found and
    stopped by its task tag."""
    mgr = BackgroundManager()
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        started = await mgr.run("setsid sleep 60 & echo started; wait")
    task_id = started.split()[2]
    for _ in range(300):
        if _tagged_pids(task_id):
            break
        await asyncio.sleep(0.01)
    assert _tagged_pids(task_id), "the task's processes were not tagged"
    res = await mgr.cancel_task(task_id, shell_term_s=2)
    assert res in ("stopped", "forced")
    for _ in range(300):
        if not _tagged_pids(task_id):
            break
        await asyncio.sleep(0.01)
    assert _tagged_pids(task_id) == [], "an escaped process survived the stop"
