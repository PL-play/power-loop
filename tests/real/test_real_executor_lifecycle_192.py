"""Actual provider work under Runtime startup and bounded worker shutdown."""

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_shutdown_retains_known_usage_and_rejects_new_work(runtime_agent, record_property):
    provider = BoundedProvider(limit=3)
    agent = await runtime_agent(model=provider,
        config=AgentLoopConfig(max_rounds=2, max_tokens=2048, context_budget_tokens=10**9))
    try:
        result = await agent.send('Reply exactly 42 in plain assistant text, without tools or pass_turn.')
        assert result.final_text.strip() == '42' and provider.calls == 1
        assert result.usage['total_tokens'] > 0
        stopped = await agent.runtime.worker.shutdown(timeout_s=5)
        assert stopped.drained
        with pytest.raises(RuntimeRequestError, match='session_closed'):
            await agent.send('Must not dispatch.')
        saved = await agent.runtime.client.wait(result.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)
        assert saved == result and provider.calls == 1
    finally:
        provider.report(record_property)


