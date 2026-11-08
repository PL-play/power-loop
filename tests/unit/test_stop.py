"""Cancellation through current RuntimeClient reaches the original execution."""
import asyncio
import json
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    CancellationToken,
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StopPolicy,
    ToolDefinition,
    create_default_tool_registry,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']
pytestmark = pytest.mark.unit
USAGE = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)

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



@pytest.mark.asyncio
async def test_stop_during_the_model_call_ends_the_run_at_once(registry_store):
    store, _ = registry_store
    model = _LLM()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(system_prompt='HANG', compactor=None)).start()
    try:
        receipt = await runner.runtime.client.send(runner.target, 'write a novel', message_id='run', actor=runner.actor)
        await asyncio.wait_for(model.hanging.wait(), 10)
        await runner.runtime.client.cancel(receipt.input_ref, actor=runner.actor)
        result = await runner.runtime.client.wait(receipt.input_ref, actor=runner.actor, deadline_ms=now_ms()+5000)
        assert result.status == 'cancelled' and model.cancelled == 1
        rows = await store.load_all_messages(runner.target.session_id)
        assert not any(row.role == 'assistant' and row.content for row in rows)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,finishes', [('abort',False),('finish',True),('finish',False)])
async def test_stop_respects_tool_interrupt_policy(registry_store, mode, finishes):
    store, _ = registry_store
    began, release = asyncio.Event(), asyncio.Event()
    cancelled = []
    async def work():
        began.set()
        try:
            if finishes:
                from power_loop.core.agent_context import get_current_cancel_token
                await get_current_cancel_token().wait()
                await asyncio.sleep(.05)
            else:
                await release.wait()
            return 'real result'
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    registry = ToolRegistry()
    registry.register(ToolDefinition('t','tool',interrupt=mode,pricing=Free('fixture')),work)
    runner = await RunningAgent(store, _LLM([_call('t')]), registry,
        AgentLoopConfig(compactor=None, stop_policy=StopPolicy(finish_tool_s=.3,abort_tool_s=.1))).start()
    try:
        receipt = await runner.runtime.client.send(runner.target,'go',message_id='run',actor=runner.actor)
        await asyncio.wait_for(began.wait(),10)
        await runner.runtime.client.cancel(receipt.input_ref,actor=runner.actor)
        result = await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+5000)
        assert result.status == 'cancelled'
        rows = await store.load_all_messages(runner.target.session_id)
        output = next(row.content for row in rows if row.role == 'tool')
        assert 'real result' in output if finishes else cancelled == [True]
        assert output.startswith('[stopped:')
    finally:
        release.set()
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['subagent','workflow'])
async def test_subtree_cancel_reaches_child_model(registry_store, kind):
    store, _ = registry_store
    model = _LLM()
    registry = ToolRegistry()
    register_spawn_agent(registry)
    if kind == 'subagent':
        model.steps = [_tool_resp('spawn','spawn_agent','{"task":"research","system_prompt":"HANG","tools":[]}')]
    runner = await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    try:
        if kind == 'subagent':
            parent = await runner.send('delegate')
            reference = parent.job_ref
        else:
            handle = await runner.runtime.client.workflow(runner.target,
                {'name':'hang','root':{'type':'agent','id':'a','spec':{'name':'a','system_prompt':'HANG','tools':[]}}},
                message_id='workflow',actor=runner.actor)
            reference = handle.input_ref
        await asyncio.wait_for(model.hanging.wait(),10)
        await runner.runtime.client.cancel(reference,actor=runner.actor,scope='subtree')
        await asyncio.wait_for(runner.drain(),10)
        assert model.cancelled == 1
        rows = await store._db.fetchall(f"SELECT outcome FROM {store.table_prefix}jobs WHERE kind=?",(kind,))
        assert rows and all(row['outcome']=='cancelled' for row in rows)
    finally:
        await runner.aclose()


def _tagged_pids(task_id: str) -> list[int]:
    import os

    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/environ", "rb") as fh:
                if f"POWER_LOOP_BACKGROUND_JOB={task_id}".encode() in fh.read().split(b"\0"):
                    out.append(int(p))
        except OSError:
            continue
    return out



@pytest.mark.asyncio
async def test_background_shell_stop_reaches_processes_that_left_the_group(registry_store,tmp_path):
    store, _ = registry_store
    model = _Scripted([_tool_resp('bg','background_run',json.dumps({
        'action':'run','command':'setsid sleep 60 & echo started; wait'}))])
    runner = await RunningAgent(store,model,create_default_tool_registry(workspace_dir=tmp_path)).start()
    try:
        await runner.send('launch')
        rows = await store.load_all_messages(runner.target.session_id)
        ref = JobRef(**json.loads(next(row.content for row in rows if row.tool_call_id=='bg'))['job_ref'])
        tag = ref.job_id+'_1'
        async with asyncio.timeout(10):
            while not _tagged_pids(tag):
                await asyncio.sleep(.02)
        status = await runner.runtime.client.inspect(ref,actor=runner.actor)
        await runner.runtime.client.cancel(status.job_ref,actor=runner.actor)
        result = await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status == 'cancelled'
        async with asyncio.timeout(5):
            while _tagged_pids(tag):
                await asyncio.sleep(.02)
        assert _tagged_pids(tag) == []
    finally:
        await runner.aclose()
