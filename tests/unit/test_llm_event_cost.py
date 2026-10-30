"""7.2: LLM_CALL_COMPLETED carries the endpoint, the provider-reported cost and image tokens."""
from __future__ import annotations

from types import SimpleNamespace

from power_loop import AgentEventBus, AgentHooks, AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.events import AgentEventType
from power_loop.core.pipeline import _llm_endpoint
from power_loop.llm_metrics import ProviderCost
from power_loop.runtime.store.store import SessionStore

from .test_stateful_loop import _Scripted


async def test_completed_event_carries_cost_endpoint_and_image_tokens():
    llm = _Scripted([LLMResponse(raw_text="ok", token_usage=LLMTokenUsage(
        prompt_tokens=100, completion_tokens=5, total_tokens=105, prompt_image_tokens=40), provider_cost=ProviderCost('0.0021','USD'))])
    llm._cfg = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(system_prompt="S", max_tokens=50),
                                 hooks=AgentHooks(), event_bus=AgentEventBus())
        seen = []
        loop.event_bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, seen.append)
        await loop.send("hi", session_id=await loop.new_session())
        (ev,) = seen
        assert (ev.data.endpoint, ev.data.provider_cost, ev.data.prompt_image_tokens) == (
            "https://openrouter.ai/api/v1", '0.0021', 40)
        assert ev.data.provider_cost_currency == 'USD'
    finally:
        await store.close()


def test_endpoint_unwraps_wrappers():
    inner = SimpleNamespace(_cfg=SimpleNamespace(base_url="https://api.deepseek.com/v1"))
    assert _llm_endpoint(SimpleNamespace(inner=inner)) == "https://api.deepseek.com/v1"
    assert _llm_endpoint(object()) == ""
