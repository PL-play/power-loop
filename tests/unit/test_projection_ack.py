"""Runtime projections are consumed only when the model actually saw them (design/124 Z7).

``<background_updates>`` used to be marked seen while the request was being BUILT, so an LLM call
that then failed (degraded after retries — or, with steering, an aborted stream) swallowed the
news for good: the next request never mentioned the finished task.
"""

from __future__ import annotations

from typing import Any

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.runtime.retry import LLMRetryPolicy


class _FailThenOk(LLMService):
    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self.calls.append([dict(m) for m in request.messages])
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("provider exploded")
        return LLMResponse(raw_text="ok")

    async def close(self) -> None:
        return None


def _mentions_task(call: list[dict[str, Any]]) -> bool:
    return any("bg-7" in str(m.get("content") or "") for m in call)


@pytest.mark.asyncio
async def test_failed_call_does_not_swallow_background_update() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _FailThenOk(fail_first=1)
        loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(
            system_prompt="S", max_rounds=2, compactor=None,
            retry_policy=LLMRetryPolicy(max_attempts=1),
        ))
        sid = await loop.new_session()
        await store.upsert_background_task(sid, task_id="bg-7", command="build",
                                           status="done", output_tail="BUILD OK")
        r1 = await loop.send("hi", sid)
        assert r1.status == "degraded"
        assert _mentions_task(llm.calls[0])
        r2 = await loop.send("again", sid)
        assert r2.status == "completed"
        assert _mentions_task(llm.calls[1]), "the update vanished with the failed call"
        # the successful call acknowledged it → gone from later requests
        await loop.send("third", sid)
        assert not _mentions_task(llm.calls[2])
        for call in llm.calls:
            assert all("_ack" not in m for m in call), "private marker leaked to the provider"
    finally:
        await store.close()
