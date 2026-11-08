"""Bounded real model provider shared by current-runtime tests."""
import asyncio

from power_loop import LLMService
from power_loop.core.agent_context import get_session_id

from ._llm import make_llm


class BoundedProvider(LLMService):
    """Keep the provider real; gates only control when a child may start its request."""

    def __init__(self, *, limit=8):
        self.inner = make_llm(max_tokens=2048, temperature=0)
        self.limit = limit
        self.calls = 0
        self.refused_calls = 0
        self.responses = []
        self.root_sid = None
        self.hold_children = False
        self.child_entered = asyncio.Event()
        self.release_children = asyncio.Event()

    async def complete(self, request, **kwargs):
        if self.hold_children and get_session_id() != self.root_sid:
            self.child_entered.set()
            await self.release_children.wait()
        if self.calls >= self.limit:
            self.refused_calls += 1
            raise AssertionError(f"real-provider test exceeded {self.limit} requests")
        self.calls += 1
        response = await asyncio.wait_for(self.inner.complete(request, **kwargs), 60)
        self.responses.append(response)
        return response

    def stream(self, request):
        return self.inner.stream(request)

    async def close(self):
        await self.inner.close()

    def report(self, record_property):
        record_property("real_provider_requests", self.calls)
        record_property("local_provider_calls_refused", self.refused_calls)
        record_property("reported_tokens", sum(
            getattr(getattr(r, "token_usage", None), "total_tokens", 0) or 0
            for r in self.responses))

