"""Actual background receipt -> wait -> same-session evidence consumption."""
from __future__ import annotations

import asyncio
import secrets

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition, create_default_tool_registry
from power_loop.runtime.completion import CompletionWindow, collect_required
from power_loop.tools import default_tools

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_real_common_collector_waits_for_background_and_reentry_makes_no_new_request(tmp_path, monkeypatch, record_property, ending):
    store = await SessionStore.open(str(tmp_path / "completion.db"))
    provider = BoundedProvider(limit=8)
    manager = default_tools.BackgroundManager()
    monkeypatch.setattr(default_tools, "BG", manager)
    monkeypatch.setattr(default_tools, "_TOOL_TASK_ON_COMPLETE", None)
    registry = create_default_tool_registry(include=["background_run"], workspace_dir=tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    proof = "BACKGROUND-" + secrets.token_hex(8)
    async def evidence():
        entered.set()
        await release.wait()
        return proof
    registry.register(ToolDefinition(name="local_evidence", description="Read a local result code.",
        input_schema={"type": "object", "properties": {}}, async_capable=True), evidence)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=4, max_tokens=2048,
            system_prompt="Follow the requested tool sequence. A background acceptance receipt is not a result. "
                "When final background evidence arrives, report its exact BACKGROUND code in assistant text, "
                "without any tool call, polling or pass_turn. Never start replacement work."))
    sid = await loop.new_session()
    collection = None
    try:
        async with asyncio.timeout(180):
            ending_instruction = "then call pass_turn to wait" if ending == "pass_turn" else "then reply WAITING without tools or pass_turn"
            first = await loop.send("Use background_run action=tool tool=local_evidence args={} exactly once, "
                + ending_instruction + ". Do not call local_evidence directly, check, wait, or list.", sid)
            assert first.status == "completed" and first.passed == (ending == "pass_turn")
            await entered.wait()
            raw = {"session_id": sid, "status": first.status, "final_text": first.final_text,
                "rounds": first.rounds, "usage": first.usage}
            window = CompletionWindow(60, 3)
            async def original_loop(owner, session):
                # This is a main-session collector, not a workflow leaf. Its
                # trusted resolver preserves the actual original loop/config.
                assert owner is loop and session == sid
                return loop
            before = provider.calls
            collection = asyncio.create_task(collect_required(loop, raw, window=window, resolve_loop=original_loop))
            await asyncio.sleep(0.02)
            assert not collection.done() and provider.calls == before
            release.set()
            result = await collection
            assert result["status"] == "completed" and proof in result["final_text"], result
            assert result["usage"]["total_tokens"] == sum(r.token_usage.total_tokens for r in provider.responses)
            calls, deadline, remaining = provider.calls, window.deadline, window.remaining
            again = await collect_required(loop, result, window=window, resolve_loop=original_loop)
            assert again == result and provider.calls == calls
            assert window.deadline == deadline and window.remaining == remaining
            rows = await store.load_all_messages(sid)
            assert any(proof in str(row.content) for row in rows if row.role == "assistant")
            assert len(await store.list_background_tasks(sid)) == 1
    finally:
        release.set()
        if collection is not None:
            await asyncio.wait_for(asyncio.gather(collection, return_exceptions=True), 10)
        if manager._workers:
            await asyncio.wait_for(asyncio.gather(*list(manager._workers.values()), return_exceptions=True), 10)
        provider.report(record_property)
        await provider.close()
        await store.close()
