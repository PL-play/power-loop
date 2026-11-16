from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    RequestOptions,
    RuntimeProjector,
    SessionStore,
    StatefulAgentLoop,
    create_default_tool_registry,
    get_tool_runtime_context,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

# ── Fake LLM ────────────────────────────────────────────────────────────


@dataclass
class _Scripted(LLMService):
    """LLM that returns canned responses, one per call."""

    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[list[dict]] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(request.messages))
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


def _tool_resp(call_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    )


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="echo back",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            required_params=("text",),
        pricing=Free("test_fixture")),
        lambda **kw: kw.get("text", ""),
    )
    return reg


class _CustomProjector(RuntimeProjector):
    async def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        state = await store.get_runtime_state(session_id, "custom", default={}) or {}
        text = state.get("text", "")
        return [{"role": "user", "name": "custom_runtime", "content": f"<custom>{text}</custom>"}] if text else []


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def store():
    s = await SessionStore.open(':memory:')
    yield s
    await s.close()


@pytest.fixture
async def agent_factory(store):
    agents = []
    async def start(model=None, registry=None, config=None, *, options=None):
        agent = await RunningAgent(store, model or _Scripted(), registry, config, options=options).start()
        agents.append(agent)
        return agent
    yield start
    for agent in agents:
        await agent.aclose()


async def test_new_session_then_send_persists_user_message(store, agent_factory):
    model = _Scripted([LLMResponse(raw_text='hi back')])
    agent = await agent_factory(model)
    result = await agent.send('hello')
    assert result.status == 'completed' and result.final_text == 'hi back'
    rows = await store.load_active_messages(agent.target.session_id)
    assert [r.role for r in rows] == ['user', 'assistant']
    assert json.loads(rows[0].content)['items'][0]['content'] == 'hello'
    assert rows[1].content == 'hi back'
    assert [r.send_index for r in rows] == [1, 1]
    assert not any('send_index' in m or '_inbox' in m for m in model.calls[0])


@pytest.mark.parametrize('body', [
    '{"role":"assistant","content":"a JSON string"}',
    {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]},
])
async def test_multimodal_and_json_strings_round_trip(store, agent_factory, body):
    model = _Scripted()
    agent = await agent_factory(model)
    await agent.send(body)
    await agent.send('next')
    expected = body if isinstance(body, str) else body['content']
    # The raw audit keeps the exact stored body; the next model call sees its PROJECTION.
    rows = await store.load_active_messages(agent.target.session_id)
    assert rows[0].role == 'user'
    assert json.loads(rows[0].content)['items'][0]['content'] == body
    projected = "\n".join(str(m.get('content', '')) for m in model.calls[1])
    if isinstance(expected, str):
        assert expected in projected
    else:
        assert 'look' in projected and '[image]' in projected
        assert 'base64' not in projected


@pytest.mark.parametrize('history', [True, False])
async def test_subsequent_send_history_option(store, agent_factory, history):
    model = _Scripted()
    agent = await agent_factory(model, options=RequestOptions(history=history))
    await agent.send('previous secret')
    result = await agent.send('current')
    assert result.status == 'completed'
    assert any('previous secret' in str(m.get('content', '')) for m in model.calls[1]) is history
    assert {r.send_index for r in await store.load_active_messages(agent.target.session_id)} == {1, 2}


async def test_tool_round_persists_assistant_and_tool_messages(store, agent_factory):
    model = _Scripted([_tool_resp('c1', 'echo', '{"text":"returned"}'), LLMResponse(raw_text='done')])
    agent = await agent_factory(model, _echo_registry())
    assert (await agent.send('use a tool')).status == 'completed'
    rows = await store.load_active_messages(agent.target.session_id)
    assert [r.role for r in rows] == ['user', 'assistant', 'tool', 'assistant']
    assert rows[2].tool_call_id == 'c1' and rows[2].content == 'returned'
    assert len({r.send_index for r in rows}) == 1
    state = await store.get_state(agent.target.session_id)
    assert not state.pending


async def test_todo_runtime_state_is_injected_without_persisting_virtual_message(store, agent_factory):
    model = _Scripted([_tool_resp('todo1', 'todo',
        '{"items":[{"id":"one","text":"Implement runtime state","status":"in_progress"}]}'),
        LLMResponse(raw_text='done')])
    agent = await agent_factory(model, create_default_tool_registry(include=['todo']))
    assert (await agent.send('make todo')).status == 'completed'
    assert (await store.get_runtime_state(agent.target.session_id, 'todo'))['items'][0]['text'] == 'Implement runtime state'
    assert any('<current_todos>' in str(m.get('content', '')) for m in model.calls[1])
    assert not any('<current_todos>' in (r.content or '') for r in await store.load_active_messages(agent.target.session_id))
    await agent.send('next worker invocation')
    assert any('Implement runtime state' in str(m.get('content', '')) for m in model.calls[-1])


@pytest.mark.parametrize('enabled', [True, False])
async def test_custom_runtime_projector_and_tool_context(store, agent_factory, enabled):
    registry = ToolRegistry()
    async def write_state(text):
        context = get_tool_runtime_context(required=True)
        await context.store.set_runtime_state(context.session_id, 'custom', {'text': text})
        return 'written'
    registry.register(ToolDefinition('write', 'write state', input_schema={
        'type': 'object', 'properties': {'text': {'type': 'string'}}}, pricing=Free('fixture')), write_state)
    model = _Scripted([_tool_resp('c1', 'write', '{"text":"VISIBLE"}'), LLMResponse(raw_text='done')])
    agent = await agent_factory(model, registry, AgentLoopConfig(context_budget_tokens=10**9,
        runtime_projectors=(_CustomProjector(),) if enabled else ()))
    assert (await agent.send('write')).status == 'completed'
    assert await store.get_runtime_state(agent.target.session_id, 'custom') == {'text': 'VISIBLE'}
    assert any('<custom>VISIBLE</custom>' in str(m.get('content', '')) for m in model.calls[1]) is enabled


async def test_prompt_and_empty_tool_subset_are_applied(agent_factory):
    class Capture(_Scripted):
        async def complete(self, request, **kwargs):
            self.request = request
            return await super().complete(request, **kwargs)
    model = Capture()
    agent = await agent_factory(model, _echo_registry(), AgentLoopConfig(system_prompt='default', context_budget_tokens=10**9))
    await agent.new_session(options=RequestOptions(system_prompt='frozen override', tools=()))
    await agent.send('question')
    assert 'frozen override' in model.request.system_prompt
    assert {t['function']['name'] for t in model.request.tools} == {'pass_turn'}


@pytest.mark.parametrize('text,status,error', [
    ('{"answer":42}', 'completed', None),
    ('{"other":42}', 'failed', 'structured_output: missing_required:answer'),
    ('not JSON', 'failed', 'structured_output: no_json'),
    ('{"answer":NaN}', 'failed', 'structured_output: invalid_json'),
])
async def test_structured_result_is_stored_with_original_text(agent_factory, text, status, error):
    model = _Scripted([LLMResponse(raw_text=text)])
    agent = await agent_factory(model)
    await agent.new_session(options=RequestOptions(response_format={
        'type': 'json_schema', 'json_schema': {'name': 'answer',
            'schema': {'type': 'object', 'required': ['answer'], 'properties': {'answer': {'type': 'number'}}}},
    }))
    result = await agent.send('JSON please')
    assert result.status == status and result.final_text == text
    assert (error.split(": ", 1)[1] in result.error) if error else result.error is None
    assert result.error_code == ("structured_output" if error else None)
    assert result.structured_output == ({'answer': 42} if status == 'completed' else None)
    reread = await agent.runtime.client.wait(result.input_ref, actor=agent.actor,
                                            deadline_ms=int(time.time()*1000)+1000)
    assert reread == result and len(model.calls) == 1


async def test_skills_profile_is_preserved(tmp_path, agent_factory):
    skill = tmp_path / 'sample'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('---\nname: sample\ndescription: Runtime test skill\n---\nRead carefully.\n')
    class Capture(_Scripted):
        async def complete(self, request, **kwargs):
            self.request = request
            return await super().complete(request, **kwargs)
    model = Capture()
    agent = await agent_factory(model, config=AgentLoopConfig(context_budget_tokens=10**9, skills_dir=str(tmp_path)))
    await agent.send('hello')
    assert 'Runtime test skill' in model.request.system_prompt


async def test_two_facades_wait_for_their_inputs_in_one_batch(agent_factory):
    model = _Scripted([LLMResponse(raw_text='first'), LLMResponse(raw_text='second')])
    agent = await agent_factory(model)
    second = StatefulAgentLoop(runtime=agent.runtime)
    try:
        from power_loop.agent.stateful_loop import _session_sync
        async with _session_sync(agent.target.session_id).lock:
            for message_id, text in (("m1", "one"), ("two", "two")):
                await agent.runtime.client.send(agent.target, text, message_id=message_id, actor=agent.actor)
        first_result, second_result = await asyncio.gather(agent.send('one'), second.send(agent.target, 'two',
            message_id='two', actor=agent.actor, deadline_ms=__import__('time').time_ns() // 1000000 + 10000))
        assert (first_result.final_text, second_result.final_text) == ('first', 'first')
        assert first_result.send_index == second_result.send_index == 1 and len(model.calls) == 1
        assert first_result.input_ref != second_result.input_ref
    finally:
        await second.aclose()


async def test_model_error_is_a_readable_failed_result(agent_factory):
    class Broken(_Scripted):
        async def complete(self, request, **kwargs):
            raise RuntimeError('provider failed')
    agent = await agent_factory(Broken(), config=AgentLoopConfig(context_budget_tokens=10**9, max_rounds=1))
    result = await agent.send('fail')
    assert result.status == 'failed'
    assert result.error is not None


async def test_build_skill_section_failure_paths_return_empty():
    from power_loop.agent.system_prompt import build_skill_section
    assert build_skill_section(None) == ''
    assert build_skill_section('') == ''
    assert build_skill_section('/nonexistent/path/xyz-does-not-exist') == ''
