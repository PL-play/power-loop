"""Actual provider requests cannot be caused by withdrawn inbox claims."""
from __future__ import annotations

import json

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.agent.follow_up import InboxItem

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("phase,keep", [("idle", True), ("terminal", False), ("terminal", True)])
async def test_real_retracted_batch_never_reaches_request_or_adds_paid_round(
        tmp_path, record_property, monkeypatch, phase, keep):
    store = await SessionStore.open(str(tmp_path / "original-inbox.db"))
    requests, changed = [], False

    class Provider(BoundedProvider):
        async def complete(self, request, **kwargs):
            requests.append([dict(m) for m in request.messages])
            response = await super().complete(request, **kwargs)
            if self.calls == 1 and phase == "terminal":
                await store.inbox_put(sid, [{"item_id": "drop", "content": "WITHDRAWN-DO-NOT-READ"},
                    *([{"item_id": "keep", "content": "Return READY again."}] if keep else [])])
                loop._note_committed_inbox(sid)
            return response

    provider = Provider(limit=3)
    loop = StatefulAgentLoop(llm=provider, store=store,
        config=AgentLoopConfig(system_prompt="Return READY as plain text. Do not call tools.",
            max_rounds=3, max_tokens=2048, compactor=None))
    sid = await loop.new_session()
    original = type(store).append_message

    async def append(self, session_id, **kwargs):
        nonlocal changed
        if self is store and kwargs.get("inbox") and not changed:
            changed = True
            assert await store.inbox_void(sid, ["drop"]) == 1
        return await original(self, session_id, **kwargs)

    async def no_tools(ctx):
        ctx.tools = []

    monkeypatch.setattr(type(store), "append_message", append)
    loop.hooks.register(HookPoint.LLM_BEFORE, no_tools, name="test.inbox-response-shape")
    try:
        if phase == "idle":
            result = await loop.deliver([InboxItem("WITHDRAWN-DO-NOT-READ", item_id="drop"),
                InboxItem("Return READY.", item_id="keep")], sid)
        else:
            result = await loop.send("Return READY.", sid)
        assert result.status == "completed" and result.final_text
        assert changed and provider.calls == (2 if phase == "terminal" and keep else 1)
        assert "WITHDRAWN-DO-NOT-READ" not in json.dumps(requests)
        assert all("_inbox" not in m and "inbox" not in m for req in requests for m in req)
        assert "WITHDRAWN-DO-NOT-READ" not in json.dumps([r.content for r in await store.load_all_messages(sid)])
        rows = {r["item_id"]: r for r in await store.inbox_items(sid)}
        assert rows["drop"]["status"] == "void"
        if keep:
            assert rows["keep"]["status"] == "delivered"
        assert (await loop.inbox_pending(sid))["pending"] == 0
    finally:
        provider.report(record_property)
        await loop.aclose()
        await provider.close()
        await store.close()
