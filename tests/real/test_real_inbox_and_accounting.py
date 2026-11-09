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
    LLMProviderConfig,
    LLMRetryPolicy,
    SessionStore,
    create_llm_service_from_config,
)
from power_loop._vendor.llm_client.multimodal import create_attachment_ref
from power_loop.agent.runtime_store import now_ms
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
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
                                input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), wait_tool)
    return reg


async def _until_in_flight(llm_events: list[Any], loop, sid: str) -> None:
    for _ in range(1000):
        if llm_events:
            return
        await asyncio.sleep(0.01)
    pytest.fail("the run never reached its first LLM call")


def _user_rows_text(rows: list[Any]) -> str:
    return "\n".join(str(r.content or "") for r in rows if r.role == "user")


# ── the inbox against a live model ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_real_cancelled_input_does_not_erase_the_next_accepted_input(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = await runtime_agent(
            model=make_llm(max_tokens=2048, temperature=0), store=store, event_bus=bus,
            registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="First call the `wait` tool exactly once, then answer the user "
                              "briefly in English.",
                max_rounds=4, max_tokens=2048, temperature=0, compactor=None))
        sid = loop.target.session_id
        first = await loop.runtime.client.send(loop.target, 'Name a fruit.', message_id='m1', actor=loop.actor)
        await _until_in_flight(started, loop, sid)
        queued = await loop.runtime.client.send(loop.target, 'Also: the fruit must be yellow.', message_id='m2', actor=loop.actor)
        assert queued.newly_accepted
        await loop.runtime.client.cancel(first.input_ref, actor=loop.actor, reason='user stop')
        cancelled = await loop.runtime.client.wait(first.input_ref, actor=loop.actor, deadline_ms=now_ms()+60000)
        assert cancelled.status == 'cancelled'
        for mid, body in [('m1','Name a fruit.'), ('m2','Also: the fruit must be yellow.')]:
            duplicate = await loop.runtime.client.send(loop.target, body, message_id=mid, actor=loop.actor)
            assert not duplicate.newly_accepted
        assert (await loop.runtime.client.wait(queued.input_ref, actor=loop.actor, deadline_ms=now_ms()+60000)).status == 'completed'
        res = await loop.send('Reply with just the fruit name.', message_id='m3')
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
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_real_user_message_and_task_wakeup_arrive_separately_and_both_land(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = await runtime_agent(
            model=make_llm(max_tokens=2048, temperature=0), store=store, event_bus=bus,
            registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="You are an assistant. First call the `wait` tool exactly once, "
                              "then reply to everything you have been told, in English.",
                max_rounds=4, max_tokens=2048, temperature=0, compactor=None))
        sid = loop.target.session_id
        run = asyncio.create_task(loop.send("Hi, I'm starting a session."))
        await _until_in_flight(started, loop, sid)
        for mid, body in [('u1','User says: what is 17 + 25?'),
                          ('t1','[system] Your background task build-42 finished: BUILD FAILED (missing dependency libfoo).')]:
            await loop.runtime.client.send(loop.target, body, message_id=mid, actor=loop.actor)
        res = await run
        assert res.status == "completed"
        rows = [r for r in await store.load_active_messages(sid) if r.role == 'user' and ('17 + 25' in str(r.content) or 'build-42' in str(r.content))]
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
        await loop.aclose()
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
async def test_real_image_survives_the_inbox(tmp_path: Any, runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        started: list[Any] = []
        bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e))
        loop = await runtime_agent(
            model=make_llm(max_tokens=2048, temperature=0,
                         capabilities={"supports_image_input": True}),
            store=store, event_bus=bus, registry=_wait_registry(),
            config=AgentLoopConfig(
                system_prompt="First call the `wait` tool exactly once, then answer the user's "
                              "latest question.", max_rounds=4, max_tokens=2048, temperature=0,
                compactor=None))
        sid = loop.target.session_id
        run = asyncio.create_task(loop.send("I'll send you a picture in a moment."))
        await _until_in_flight(started, loop, sid)
        path = tmp_path / "blue.png"
        path.write_bytes(_solid_png(64, 64, (20, 90, 220)))
        await loop.runtime.client.send(loop.target, {'role':'user', 'content':[
            {'type':'text', 'text':'这张图是什么纯色？只回颜色名。'},
            {'type':'attachment', 'attachment':create_attachment_ref(str(path))},
        ]}, message_id='img', actor=loop.actor)
        res = await run
        assert res.status == "completed"
        answer = (res.final_text or "").lower()
        assert "蓝" in answer or "blue" in answer, answer
    finally:
        await loop.aclose()
        await store.close()


# ── give up on permanent errors ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_real_bad_key_costs_one_request(runtime_agent) -> None:
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
        loop = await runtime_agent(model=svc, store=store, event_bus=bus, config=AgentLoopConfig(
            system_prompt="S", max_rounds=1, compactor=None,
            retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=1.0, total_timeout=None)))
        t0 = time.monotonic()
        res = await loop.send("hello")
        elapsed = time.monotonic() - t0
        assert res.status == "failed"
        assert degraded and degraded[0]["reason"] == "non_retryable", degraded
        # The SDK's own client may retry transport-level failures, but never a 401: one request.
        assert creates == 1, f"{creates} requests for a permanent 401"
        assert elapsed < 15, f"gave up only after {elapsed:.1f}s"
    finally:
        await loop.aclose()
        await store.close()


# ── usage accounting against the provider's own numbers ──────────────────────────────────


_PROMPT = ("请用三句话介绍一下杭州的西湖，并说明最适合游览的季节。"
           "Then add one English sentence about the best time of day to visit.")


@pytest.mark.asyncio
async def test_real_streamed_call_reports_provider_usage_and_estimate_is_close(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        done: list[dict] = []
        bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: done.append(dict(e.payload or {})))
        system = "You are a concise travel guide. " * 20
        loop = await runtime_agent(model=make_llm(max_tokens=2048, temperature=0), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt=system, max_rounds=1, compactor=None))
        res = await loop.send(_PROMPT * 8)
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
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_real_aborted_stream_is_recorded_with_an_estimate(runtime_agent) -> None:
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
        loop = await runtime_agent(model=make_llm(max_tokens=4000, temperature=0.7), store=store,
                                 event_bus=bus, config=AgentLoopConfig(
                                     system_prompt="S", max_rounds=1, compactor=None,
                                     retry_policy=None))
        receipt = await loop.runtime.client.send(loop.target, '请写一篇至少 3000 字的连续小说，一直写下去。', message_id='stream', actor=loop.actor)
        for _ in range(3000):
            if chars[0] >= 200:
                break
            await asyncio.sleep(0.01)
        assert chars[0] >= 200, "the stream never produced output"
        await loop.runtime.client.cancel(receipt.input_ref, actor=loop.actor, reason='user stop')
        result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+60000)
        assert result.status == 'cancelled'
        await bus.drain(timeout=2)
        assert done, "no LLM_CALL_COMPLETED for the aborted call"
        ev = done[-1]
        assert ev["success"] is False and ev["outcome"] == "aborted" and ev["estimated"] is True
        assert ev["prompt_tokens"] > 0
        assert ev["completion_tokens"] >= estimate_text_tokens("x" * 200) * 0.5
        print(f"aborted after ~{chars[0]} chars: estimate prompt={ev['prompt_tokens']} "
              f"completion={ev['completion_tokens']}")
    finally:
        await loop.aclose()
        await store.close()


pytestmark = pytest.mark.skipif(
    not os.environ.get("POWER_LOOP_API_KEY"), reason="needs the real provider in .env")
