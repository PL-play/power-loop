"""S2 (3.14): per-send tool allowlist propagates to child spawns.

``send(tools=...)`` publishes the run's effective allowlist via
``agent_context.get_effective_tools``; ``run_agent_spec`` intersects the child's
registry with it (``inherit_send_filter``); workflows capture the set at
submission and clamp every leaf at the SPEC level (executor-agnostic), journal
it, and reapply it on resume.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, RequestOptions, SessionStore
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import (
    get_effective_tools,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow


@dataclass
class _Capturing(LLMService):
    """Scripted LLM that records each request's advertised tool names."""

    responses: list[LLMResponse] = field(default_factory=list)
    seen_tools: list[list[str]] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        # pass_turn 永远在（design/188，单独测）：这里只看业务工具
        self.seen_tools.append(
            sorted(t["function"]["name"] for t in (request.tools or []) if t["function"]["name"] != "pass_turn")
        )
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name, description=name,
        input_schema={"type": "object", "properties": {}},
    pricing=Free("test_fixture"))


def _registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(_tool_def(n), lambda **_kw: "ok")
    return reg


def _tool_call(call_id: str, name: str, args: dict | None = None) -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }],
    )


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── contextvar published by _run_loop ────────────────────────────────────


@pytest.mark.asyncio
async def test_send_with_tools_publishes_effective_set(store: SessionStore, runtime_agent) -> None:
    seen: list[frozenset[str] | None] = []

    def probe(**_kw: Any) -> str:
        seen.append(get_effective_tools())
        return "probed"

    reg = _registry("a", "b")
    reg.register(_tool_def("probe"), probe)
    llm = _Capturing(responses=[
        _tool_call("t1", "probe"),
        LLMResponse(raw_text="fin"),
    ])
    loop = await runtime_agent(model=llm, store=store, registry=reg, config=AgentLoopConfig(max_rounds=3))
    r = await loop.send("go", options=RequestOptions(tools=["probe", "a"]))
    assert r.status == "completed"
    assert seen == [frozenset({"probe", "a"})]
    # And it is reset once the run finishes.
    assert get_effective_tools() is None


@pytest.mark.asyncio
async def test_send_without_tools_publishes_none(store: SessionStore, runtime_agent) -> None:
    seen: list[frozenset[str] | None] = []

    def probe(**_kw: Any) -> str:
        seen.append(get_effective_tools())
        return "probed"

    reg = _registry("a")
    reg.register(_tool_def("probe"), probe)
    llm = _Capturing(responses=[
        _tool_call("t1", "probe"),
        LLMResponse(raw_text="fin"),
    ])
    loop = await runtime_agent(model=llm, store=store, registry=reg, config=AgentLoopConfig(max_rounds=3))
    await loop.send("go")
    assert seen == [None]




@pytest.mark.parametrize('declared,grant,expected', [
    (None, ['a', 'c'], ['a', 'c']),
    (['b'], ['a', 'b'], ['b']),
    (None, ['a', 'b', 'c'], ['a', 'b', 'c']),
    (['b', 'c'], ['a', 'b'], None),
])
async def test_child_uses_frozen_grant_and_cannot_widen_it(store, runtime_agent, declared, grant, expected):
    from power_loop.core.agent_context import get_current_loop
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
    reg = _registry('a', 'b', 'c')
    async def delegate():
        return await run_agent_spec(AgentSpec(name='kid', system_prompt='CHILD', tools=declared),
                                    'task', parent_loop=get_current_loop())
    reg.register(_tool_def('delegate'), FunctionRuntimeTool(delegate,
        ToolRuntimeContract('test.delegate', allowed_roles=('main',))))
    class Model(_Capturing):
        submitted = False
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith('CHILD'):
                return await super().complete(request, **kwargs)
            if not self.submitted:
                self.submitted = True
                return _tool_call('delegation', 'delegate')
            return LLMResponse(raw_text='parent done')
    model = Model([LLMResponse(raw_text='child done')])
    runner = await runtime_agent(model=model, store=store, registry=reg, config=AgentLoopConfig(compactor=None))
    assert (await runner.send('go', options=RequestOptions(tools=['delegate', *grant]))).status == 'completed'
    rows = await store.load_all_messages(runner.target.session_id)
    receipt = next(r.content for r in rows if r.tool_call_id == 'delegation')
    if expected is None:
        assert 'forbidden' in receipt and 'grant' in receipt
        assert not model.seen_tools and not await store.list_children(runner.target.session_id)
    else:
        assert model.seen_tools == [expected]
        assert json.loads(receipt)['status'] == 'completed'
    assert get_effective_tools() is None
    await runner.drain()


async def test_child_cannot_spawn_grandchild_and_filter_is_restored(store, runtime_agent):
    from power_loop.core.agent_context import get_current_loop
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
    reg = _registry('a', 'b')
    seen = []
    async def spawnling():
        seen.append(get_effective_tools())
        return await run_agent_spec(AgentSpec(name='grandkid', system_prompt='GRANDCHILD'),
                                    'task', parent_loop=get_current_loop())
    reg.register(_tool_def('spawnling'), FunctionRuntimeTool(spawnling,
        ToolRuntimeContract('test.nested', allowed_roles=('subagent',))))
    async def delegate():
        return await run_agent_spec(AgentSpec(name='kid', system_prompt='CHILD'),
                                    'task', parent_loop=get_current_loop())
    reg.register(_tool_def('delegate'), FunctionRuntimeTool(delegate,
        ToolRuntimeContract('test.delegate', allowed_roles=('main',))))
    class Model(_Capturing):
        submitted = False
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith('CHILD'):
                return await super().complete(request, **kwargs)
            if not self.submitted:
                self.submitted = True
                return _tool_call('delegate', 'delegate')
            return LLMResponse(raw_text='parent done')
    model = Model([_tool_call('nested', 'spawnling'), LLMResponse(raw_text='child done')])
    runner = await runtime_agent(model=model, store=store, registry=reg, config=AgentLoopConfig(compactor=None))
    await runner.send('go', options=RequestOptions(tools=['delegate', 'spawnling', 'a']))
    [child] = await store.list_children(runner.target.session_id)
    assert not await store.list_children(child.session_id)
    rows = await store.load_all_messages(child.session_id)
    assert any(r.tool_call_id == 'nested' and 'role' in r.content for r in rows)
    assert model.seen_tools == [['a', 'spawnling'], ['a', 'spawnling']]
    assert seen == [frozenset({'a', 'spawnling'})]
    assert get_effective_tools() is None
    await runner.drain()


@pytest.mark.parametrize('grant,declared,expected', [
    (['a'], None, ['a']), (['b'], None, ['b']), (None, None, ['a', 'b', 'c']),
    (['a', 'b'], None, ['a', 'b']), (['b', 'c'], ['a', 'b'], ['b']),
])
async def test_workflow_submission_freezes_grant_and_clamps_leaf(store, runtime_agent, grant, declared, expected):
    from power_loop.agent.runtime_store import now_ms
    model = _Capturing([LLMResponse(raw_text='leaf out')])
    runner = await runtime_agent(model=model, store=store, registry=_registry('a', 'b', 'c'),
                                 config=AgentLoopConfig(compactor=None))
    node_spec = {'name': 'leaf', 'system_prompt': 'LEAF'}
    if declared is not None:
        node_spec['tools'] = declared
    options = RequestOptions(tools=grant)
    workflow = create_workflow({'name':'filtered','root':{'type':'agent','id':'n1','spec':node_spec}},
        client=runner.runtime.client, target=runner.target, actor=runner.actor, message_id='workflow', options=options)
    handle = await workflow.start()
    if grant is not None:
        grant.append('not-registered-after-submission')
    result = await handle.wait(deadline_ms=now_ms()+10000)
    assert result.status == 'completed'
    assert model.seen_tools == [expected]
    await runner.drain()
