"""DeepSeek's tool-mode contract requires reasoning on historical assistant turns."""
from copy import deepcopy

import pytest

from power_loop import AgentLoopConfig, LLMResponse, RequestOptions
from power_loop.runtime.history_projector import IdentityProjector
from tests.unit.test_job_registry_192 import registry_store  # noqa: F401
from tests.unit.test_request_sanitize import StrictModel, assert_valid_request, call
from tests.unit.test_stateful_loop import _echo_registry


class Interrupted(Exception):
    pass


class ThinkingModel(StrictModel):
    def __init__(self, interrupt=False):
        super().__init__()
        self.interrupt = interrupt
        self.expected = {}

    async def complete(self, request, **kwargs):
        assert_valid_request(request.to_messages())
        for msg in request.messages:
            if msg['role'] == 'assistant':
                key = msg.get('content') or msg['tool_calls'][0]['id']
                if key.startswith('[degraded:'):
                    assert 'reasoning_content' not in msg
                    continue
                actual = msg.get('reasoning_content')
                assert actual is not None
                assert actual == self.expected[key]
        self.requests.append(deepcopy(request.messages))
        if self.interrupt and len(self.requests) == 2:
            raise Interrupted('stopped after tool result, before the next model response')
        text = '' if len(self.requests) == 1 else 'done'
        calls = [call('reasoning-tool', '{"text":"known"}')] if not text else []
        thinking = '  exact reasoning ' + str(len(self.requests)) + '\n'
        self.expected[text or calls[0]['id']] = thinking
        return LLMResponse(raw_text=text, tool_calls=calls, think=thinking)


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['send', 'resume'])
@pytest.mark.parametrize('projected', [False, True])
async def test_reasoning_survives_reload(registry_store, action, projected, runtime_agent):  # noqa: F811
    store, connect = registry_store
    model = ThinkingModel(interrupt=action == 'resume')
    config = AgentLoopConfig(model='deepseek-flash', compactor=None,
        history_projector=IdentityProjector() if projected else None)
    loop = await runtime_agent(model=model, store=store, registry=_echo_registry(), config=config,
        allowed_models=['deepseek-flash', 'another-model'])
    sid = loop.target.session_id
    first = await loop.send('start')
    assert first.status == ('failed' if action == 'resume' else 'completed')
    await loop.aclose()
    reopened = await runtime_agent(model=model, store=await connect(), registry=_echo_registry(), config=config,
                                   target=loop.target)
    result = await reopened.send('continue' if action == 'resume' else 'next', message_id='next')
    assert result.status == 'completed', result
    rows = await reopened.runtime.store.load_active_messages(sid)
    assert all(r.meta['reasoning_content'].startswith('  exact') for r in rows if r.role == 'assistant' and not str(r.content).startswith('[degraded:'))
    assert all(not any(k.startswith('_reasoning') for k in m) for m in model.requests[-1])
    await reopened.aclose()
    other = StrictModel()
    switched = await runtime_agent(model=other, store=reopened.runtime.store, registry=_echo_registry(),
        config=AgentLoopConfig(model='another-model', compactor=None,
            history_projector=IdentityProjector() if projected else None), target=loop.target)
    assert (await switched.send('switch', message_id='switch',
        options=RequestOptions(model='another-model'))).status == 'completed'
    assert all('reasoning_content' not in m for m in other.requests[-1])
