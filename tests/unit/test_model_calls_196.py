"""Auxiliary model calls keep normal usage events inside the owning session."""
from types import SimpleNamespace

import pytest

from power_loop import AgentEventBus, AgentEventType, LLMRequest, LLMResponse, LLMTokenUsage
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.runtime.model_calls import ModelCalls
from power_loop.runtime.task_runtime import _SESSION_SEND, SessionSend

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_auxiliary_call_uses_normal_llm_accounting():
    bus, events = AgentEventBus(), []
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, events.append)
    class Model:
        async def complete(self, request):
            return LLMResponse(raw_text='folded', token_usage=LLMTokenUsage(
                prompt_tokens=3, completion_tokens=2, total_tokens=5))
    binding = SessionSend(None, SimpleNamespace(session_id='s'), SimpleNamespace(send_index=1), None)
    token = _SESSION_SEND.set(binding)
    try:
        response = await ModelCalls(session_id='s', bus=bus).wrap(Model()).complete(
            LLMRequest(messages=[{'role':'user','content':'compare'}], max_tokens=20))
    finally:
        _SESSION_SEND.reset(token)
    assert response.raw_text == 'folded'
    assert len(events) == 1 and events[0].data.total_tokens == 5


async def test_unowned_auxiliary_call_is_refused():
    with pytest.raises(RuntimeRequestError, match='unmanaged'):
        await ModelCalls(session_id='s').wrap(object()).complete(LLMRequest(messages=[]))


async def test_failed_fold_does_not_retry_through_strategy_fallback():
    from power_loop import TargetRef
    class Model:
        calls = 0
        async def complete(self, request):
            self.calls += 1
            raise OSError('provider outcome unknown')
    model = Model()
    calls = ModelCalls(session_id='s', purpose='fold')
    token = _SESSION_SEND.set(SessionSend(None, TargetRef('store','s'), SimpleNamespace(send_index=1), None))
    try:
        for _ in range(2):
            with pytest.raises(OSError, match='outcome unknown'):
                await calls.wrap(model).complete(LLMRequest(messages=[]))
        assert model.calls == 1
    finally:
        _SESSION_SEND.reset(token)
