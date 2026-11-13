"""Real provider usage and cancellation use the unified session runtime."""
from __future__ import annotations

import asyncio
import secrets

import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.tools.registry import ToolRegistry

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.fixture
async def model_runtime(tmp_path, record_property, runtime_agent):
    store = await SessionStore.open(str(tmp_path / "model-attempts.db"))
    provider = loop = None
    try:
        provider = BoundedProvider(limit=6)
        loop = await runtime_agent(
            model=provider,
            store=store,
            registry=ToolRegistry(),
            config=AgentLoopConfig(
                max_rounds=3,
                max_tokens=2048,
                compactor=None,
                retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0),
                system_prompt=(
                    "Follow the user's request exactly. Do not create, stop or repeat any work. "
                    "Reply in final assistant text without pass_turn."
                ),
            ),
        )
        yield store, provider, loop, loop.target.session_id
    finally:
        if provider is not None:
            provider.report(record_property)
        if loop is not None:
            await loop.aclose()
        if provider is not None:
            await provider.close()
        await store.close()


async def test_real_response_preserves_usage(model_runtime):
    _, provider, loop, _ = model_runtime
    code = "PROOF-" + secrets.token_hex(6)
    async with asyncio.timeout(120):
        result = await loop.send(f"Reply exactly {code}. Do not use any tools.")
        assert result.status == "completed" and code in result.final_text
        assert result.input_ref.session_id == loop.target.session_id
        assert result.send_index == 1 and provider.calls == 1
        assert result.usage["total_tokens"] > 0


async def test_real_client_cancellation_keeps_uncertain_request_and_does_not_retry(
    model_runtime, monkeypatch, record_property,
):
    _, provider, loop, _ = model_runtime
    entered = asyncio.Event()
    complete = provider.complete

    async def started(request, **kwargs):
        work = asyncio.create_task(complete(request, **kwargs))
        while provider.calls == 0:
            await asyncio.sleep(0)
        entered.set()
        return await work

    monkeypatch.setattr(provider, "complete", started)
    receipt = await loop.runtime.client.send(
        loop.target,
        "Write a long arithmetic explanation without tools.",
        message_id="cancel-original",
        actor=loop.actor,
    )
    async with asyncio.timeout(120):
        await entered.wait()
        await loop.runtime.client.cancel(receipt.input_ref, actor=loop.actor, reason="user stop")
        result = await loop.runtime.client.wait(
            receipt.input_ref, actor=loop.actor, deadline_ms=now_ms() + 10000,
        )
        await loop.drain()
        assert result.status == "cancelled" and result.send_index == 1
        assert provider.calls == 1
        record_property("provider_server_receipt_confirmed", False)
