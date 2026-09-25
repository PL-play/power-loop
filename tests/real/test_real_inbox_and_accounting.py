"""Real-provider cover for design/124 phase 1–2: the inbox, give-up-on-permanent-errors, and
usage accounting for calls that never reported usage.

Runs against the endpoint in ``.env`` (DeepSeek at the time of writing). What only a live provider
can prove:
- steering parked in the inbox across a CANCEL reaches the real model exactly once;
- a user message and a system wake-up delivered together arrive as two messages the model
  answers separately; an image survives the round trip through the inbox;
- a bad key (401) costs ONE request, not attempts × transport retries;
- a streamed call still reports the provider's own usage (U1 removed the "borrow the last call's
  usage" fallback — this is the guard that real streams didn't depend on it);
- our prompt estimate is within tolerance of the provider's count, and an aborted stream is
  recorded with an estimate rather than as free.
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
import zlib
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    FollowUpQueued,
    InboxItem,
    LLMProviderConfig,
    LLMRetryPolicy,
    SessionStore,
    StatefulAgentLoop,
    create_llm_service_from_config,
)
from power_loop._vendor.llm_client.multimodal import create_attachment_ref
from power_loop.agent.follow_up import FOLLOW_UP_MESSAGE_NAME
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.usage_estimate import estimate_prompt_tokens, estimate_text_tokens
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes


def _wait_registry(seconds: float = 2.0) -> ToolRegistry:
    reg = ToolRegistry()

    async def wait_tool(**kwargs: Any) -> str:
        await asyncio.sleep(seconds)
        return "waited"

    reg.register(ToolDefinition(name="wait", description="Wait briefly before continuing.",
                                input_schema={"type": "object", "properties": {}}), wait_tool)
    return reg


async def _until_in_flight(llm_events: list[Any], loop: StatefulAgentLoop, sid: str) -> None:
    for _ in range(1000):
        if loop._lock_for(sid).locked() and llm_events:
            return
        await asyncio.sleep(0.01)
    pytest.fail("the run never reached its first LLM call")


def _user_rows_text(rows: list[Any]) -> str:
    return "\n".join(str(r.content or "") for r in rows if r.role == "user")


# ── the inbox against a live model ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_real_cancelled_run_leaves_steering_that_reaches_the_model_once() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store, event_bus=bus,
            tool_registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="First call the `wait` tool exactly once, then answer the user "
                              "briefly in English.",
                max_rounds=4, max_tokens=2048, temperature=0, compactor=None))
        sid = await loop.new_session()
        tok = CancellationToken()
        run = asyncio.create_task(loop.deliver(
            InboxItem("Name a fruit.", item_id="m1"), sid, stop_event=tok))
        await _until_in_flight(started, loop, sid)
        q = await loop.deliver(InboxItem("Also: the fruit must be yellow.", item_id="m2"), sid)
        assert isinstance(q, FollowUpQueued)
        tok.cancel("user stop")
        first = await run
        assert first.status == "cancelled"
        await loop.abort_pending(sid, reason="cancelled by user")
        assert (await loop.inbox_pending(sid))["pending"] == 1, "steering lost with the cancel"

        # the host re-sends the whole batch (as DeepTalk's re-read does) plus a new line
        res = await loop.deliver([
            InboxItem("Name a fruit.", item_id="m1"),
            InboxItem("Also: the fruit must be yellow.", item_id="m2"),
            InboxItem("Reply with just the fruit name.", item_id="m3"),
        ], sid)
        assert res.status == "completed"
        text = _user_rows_text(await store.load_active_messages(sid))
        assert text.count("Name a fruit.") == 1
        assert text.count("the fruit must be yellow") == 1
        assert text.count("Reply with just the fruit name.") == 1
        await assert_passes(
            question="Name a fruit. The fruit must be yellow. Reply with just the fruit name.",
            answer=res.final_text or "",
            rubric="The reply names a fruit that is typically yellow (e.g. banana, lemon, "
                   "pineapple). A non-yellow fruit FAILS — the steering did not arrive.",
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_user_message_and_task_wakeup_arrive_separately_and_both_land() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store, event_bus=bus,
            tool_registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="You are an assistant. First call the `wait` tool exactly once, "
                              "then reply to everything you have been told, in English.",
                max_rounds=4, max_tokens=2048, temperature=0, compactor=None))
        sid = await loop.new_session()
        run = asyncio.create_task(loop.send("Hi, I'm starting a session.", sid))
        await _until_in_flight(started, loop, sid)
        await loop.deliver([
            InboxItem("User says: what is 17 + 25?", item_id="u1"),
            InboxItem("[system] Your background task `build-42` finished: BUILD FAILED "
                      "(missing dependency libfoo).", kind="task_done", item_id="t1"),
        ], sid)
        res = await run
        assert res.status == "completed"
        rows = [r for r in await store.load_active_messages(sid) if r.name == FOLLOW_UP_MESSAGE_NAME]
        assert len(rows) == 2, "user words and the wake-up must not share a message"
        assert "17 + 25" in rows[0].content and "build-42" not in rows[0].content
        assert "build-42" in rows[1].content
        await assert_passes(
            question="(user) what is 17 + 25?  +  (system) background task build-42 failed: "
                     "missing dependency libfoo",
            answer=res.final_text or "",
            rubric="The reply (1) gives 42 as the answer to 17 + 25 AND (2) mentions that the "
                   "build / background task failed (e.g. missing dependency). Both required.",
        )
    finally:
        await store.close()


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return (struct.pack(">I", len(data)) + payload
                + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


@pytest.mark.asyncio
async def test_real_image_survives_the_inbox(tmp_path: Any) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0,
                         capabilities={"supports_image_input": True}),
            store=store, event_bus=bus, tool_registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="First call the `wait` tool exactly once, then answer the user's "
                              "latest question.", max_rounds=4, max_tokens=2048, temperature=0,
                compactor=None))
        sid = await loop.new_session()
        run = asyncio.create_task(loop.send("I'll send you a picture in a moment.", sid))
        await _until_in_flight(started, loop, sid)
        path = tmp_path / "blue.png"
        path.write_bytes(_solid_png(64, 64, (20, 90, 220)))
        await loop.deliver(InboxItem({"role": "user", "content": [
            {"type": "text", "text": "这张图是什么纯色？只回颜色名。"},
            {"type": "attachment", "attachment": create_attachment_ref(str(path))},
        ]}, item_id="img"), sid)
        res = await run
        assert res.status == "completed"
        answer = (res.final_text or "").lower()
        assert "蓝" in answer or "blue" in answer, answer
    finally:
        await store.close()


# ── give up on permanent errors ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_real_bad_key_costs_one_request() -> None:
    cfg = LLMProviderConfig.from_env()
    cfg.api_key = "sk-invalid-design124-probe"
    svc = create_llm_service_from_config(cfg)
    client = svc._ensure_client()
    creates = 0
    real_create = client.chat.completions.create

    async def counting_create(*a: Any, **kw: Any) -> Any:
        nonlocal creates
        creates += 1
        return await real_create(*a, **kw)

    client.chat.completions.create = counting_create
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        degraded: list[dict] = []
        bus.subscribe(AgentEventType.LLM_DEGRADED, lambda e: degraded.append(dict(e.payload or {})))
        loop = StatefulAgentLoop(llm=svc, store=store, event_bus=bus, config=AgentLoopConfig(
            system_prompt="S", max_rounds=1, compactor=None,
            retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=1.0, total_timeout=None)))
        sid = await loop.new_session()
        t0 = time.monotonic()
        res = await loop.send("hello", sid)
        elapsed = time.monotonic() - t0
        assert res.status == "degraded"
        assert degraded and degraded[0]["reason"] == "non_retryable", degraded
        # The SDK's own client may retry transport-level failures, but never a 401: one request.
        assert creates == 1, f"{creates} requests for a permanent 401"
        assert elapsed < 15, f"gave up only after {elapsed:.1f}s"
    finally:
        await store.close()


# ── usage accounting against the provider's own numbers ──────────────────────────────────


_PROMPT = ("请用三句话介绍一下杭州的西湖，并说明最适合游览的季节。"
           "Then add one English sentence about the best time of day to visit.")


@pytest.mark.asyncio
async def test_real_streamed_call_reports_provider_usage_and_estimate_is_close() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        done: list[dict] = []
        bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: done.append(dict(e.payload or {})))
        system = "You are a concise travel guide. " * 20
        loop = StatefulAgentLoop(llm=make_llm(max_tokens=2048, temperature=0), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt=system, max_rounds=1, compactor=None))
        sid = await loop.new_session()
        res = await loop.send(_PROMPT * 8, sid)
        assert res.status == "completed"
        ev = done[-1]
        assert ev["success"] and ev["estimated"] is False, "the stream reported no usage"
        real_prompt = ev["prompt_tokens"]
        est = estimate_prompt_tokens(messages=[{"role": "user", "content": _PROMPT * 8}],
                                     system_prompt=system)
        ratio = est / real_prompt
        print(f"prompt: provider={real_prompt} estimate={est} ratio={ratio:.2f}")
        real_completion = ev["completion_tokens"]
        est_c = estimate_text_tokens(res.final_text or "")
        print(f"completion: provider={real_completion} estimate(text only)={est_c}")
        assert 0.6 <= ratio <= 1.5, f"prompt estimate off by {ratio:.2f}×"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_aborted_stream_is_recorded_with_an_estimate() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        done: list[dict] = []
        chars = [0]
        bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: done.append(dict(e.payload or {})))
        def _count(e: Any) -> None:
            chars[0] += len((e.payload or {}).get("text", ""))

        # thinking counts too: a thinking model streams reasoning before any answer text, and
        # both are billed output
        bus.subscribe(AgentEventType.STREAM_DELTA, _count)
        bus.subscribe(AgentEventType.STREAM_THINK_DELTA, _count)
        loop = StatefulAgentLoop(llm=make_llm(max_tokens=4000, temperature=0.7), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt="S", max_rounds=1, compactor=None,
                                     retry_policy=None))
        sid = await loop.new_session()
        task = asyncio.create_task(loop.send("请写一篇至少 3000 字的连续小说，一直写下去。", sid))
        for _ in range(3000):
            if chars[0] >= 200:
                break
            await asyncio.sleep(0.01)
        assert chars[0] >= 200, "the stream never produced output"
        task.cancel()   # what a steer / stop abort does to the in-flight call
        with pytest.raises(asyncio.CancelledError):
            await task
        await bus.drain(timeout=2)
        assert done, "no LLM_CALL_COMPLETED for the aborted call"
        ev = done[-1]
        assert ev["success"] is False and ev["outcome"] == "aborted" and ev["estimated"] is True
        assert ev["prompt_tokens"] > 0
        assert ev["completion_tokens"] >= estimate_text_tokens("x" * 200) * 0.5
        print(f"aborted after ~{chars[0]} chars: estimate prompt={ev['prompt_tokens']} "
              f"completion={ev['completion_tokens']}")
    finally:
        await store.close()


pytestmark = pytest.mark.skipif(
    not os.environ.get("POWER_LOOP_API_KEY"), reason="needs the real provider in .env")
