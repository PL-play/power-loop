"""DeepSeek's tool-mode contract requires reasoning on historical assistant turns."""
from copy import deepcopy

import pytest

from power_loop import AgentLoopConfig, LLMResponse, StatefulAgentLoop
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
async def test_reasoning_survives_reload(registry_store, action, projected):  # noqa: F811
    store, connect = registry_store
    model = ThinkingModel(interrupt=action == 'resume')
    config = AgentLoopConfig(model='deepseek-flash', compactor=None,
        history_projector=IdentityProjector() if projected else None)
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=_echo_registry(), config=config)
    sid = await loop.new_session()
    try:
        if action == 'resume':
            with pytest.raises(Interrupted):
                await loop.send('start', sid)
        else:
            assert (await loop.send('start', sid)).status == 'completed'
    finally:
        await loop.aclose()
    reopened = StatefulAgentLoop(llm=model, store=await connect(), tool_registry=_echo_registry(), config=config)
    try:
        result = await reopened.resume(sid) if action == 'resume' else await reopened.send('next', sid)
        assert result.status == 'completed'
        rows = await reopened.store.load_active_messages(sid)
        assert all(r.meta['reasoning_content'].startswith('  exact') for r in rows if r.role == 'assistant')
        assert all(not any(k.startswith('_reasoning') for k in m) for m in model.requests[-1])
        other = StrictModel()
        switched = StatefulAgentLoop(llm=other, store=reopened.store, tool_registry=_echo_registry(),
            config=AgentLoopConfig(model='another-model', compactor=None,
                history_projector=IdentityProjector() if projected else None))
        try:
            assert (await switched.send('switch', sid)).status == 'completed'
            assert all('reasoning_content' not in m for m in other.requests[-1])
        finally:
            await switched.aclose()
    finally:
        await reopened.aclose()
