"""Node cancellation stops its own operation, never a shared parent task."""

from __future__ import annotations

import json

import pytest

from power_loop import LLMResponse
from power_loop.core.agent_context import get_session_id

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def issue(loop, model, sid, monkeypatch, name, arguments):
    original, issued = model.complete, False

    async def control(request, **kwargs):
        nonlocal issued
        if get_session_id() == sid and not issued:
            issued = True
            return LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "node-control",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            )
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", control)
    await loop.send("Stop only the selected node", sid)
    assert issued
    store = await loop._ensure_store()
    records = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == name]
    return records[-1], json.loads(records[-1].content)
