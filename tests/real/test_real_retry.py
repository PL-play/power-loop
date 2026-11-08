"""Unknown provider failures do not replay; an independent next input can run."""
import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig, LLMService
from power_loop.agent.runtime_store import now_ms

from ._llm import make_llm


class _FlakyWrap(LLMService):
    def __init__(self, inner, *, fail_first):
        self.inner, self.fail_first, self.calls = inner, fail_first, 0
    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f'injected unknown provider outcome #{self.calls}')
        return await self.inner.complete(request, **kwargs)
    async def close(self):
        await self.inner.close()


@pytest.mark.asyncio
async def test_real_new_input_can_continue_after_unknown_without_replaying_original(runtime_agent):
    model = _FlakyWrap(make_llm(max_tokens=2048, temperature=0), fail_first=1)
    bus, events = AgentEventBus(), []
    bus.subscribe(None, events.append)
    agent = await runtime_agent(model=model, event_bus=bus,
        config=AgentLoopConfig(system_prompt='Reply in one short sentence.',
            max_rounds=1, max_tokens=2048, temperature=0, compactor=None))
    original = await agent.send('Reply OK.', message_id='original')
    assert original.status == 'failed' and model.calls == 1
    reread = await agent.runtime.client.wait(original.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
    assert reread == original and model.calls == 1
    result = await agent.send('This is a new request. Reply OK.', message_id='new')
    assert result.status == 'completed' and result.final_text.strip() and model.calls == 2
    assert not [e for e in events if e.type == AgentEventType.LLM_RETRY_ATTEMPTED]


@pytest.mark.asyncio
async def test_unknown_provider_outcome_is_not_synthetic_success(runtime_agent):
    model = _FlakyWrap(make_llm(max_tokens=2048, temperature=0), fail_first=100)
    agent = await runtime_agent(model=model, config=AgentLoopConfig(max_rounds=1, compactor=None))
    result = await agent.send('anything')
    assert result.status == 'failed' and model.calls == 1
    assert result.error and not result.final_text
    assert (await agent.runtime.client.wait(result.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)) == result
    assert model.calls == 1
