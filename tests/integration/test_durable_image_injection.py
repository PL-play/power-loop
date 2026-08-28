"""图片注入落库成真实 user 行，并在跨 send 时蒸馏成引用（design/75）。

这条链路有三段，任何一段断了都不会报错、只会静默退化：
  工具入队 → pipeline 落库 + 进本轮请求 → 跨 send 投影蒸馏成 `[image: … · file_uuid=…]`
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import AsyncIterator, Callable

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.runtime.image_recall import discard_queued_images, queue_image_for_next_round


def _png(size: int = 8) -> bytes:
    raw = b"".join(b"\x00" + bytes((200, 40, 40)) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(
            ">I", zlib.crc32(payload) & 0xFFFFFFFF
        )

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class _ToolThenDone(LLMService):
    """第 1 轮调一次工具，第 2 轮收尾——模拟 agent「我要看这张图」。"""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_rounds: list[list[dict]] = []

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable | None = None,
                       on_chunk_think: Callable | None = None,
                       on_stream_end: Callable | None = None) -> LLMResponse:
        self.calls += 1
        self.seen_rounds.append(list(request.messages or []))
        if self.calls == 1:
            r = LLMResponse(raw_text="")
            r.tool_calls = [{"id": "c1", "type": "function",
                             "function": {"name": "look", "arguments": "{}"}}]
            return r
        return LLMResponse(raw_text="看到了，是一张红色的图。")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()


@pytest.mark.asyncio
async def test_durable_injection_becomes_a_real_user_row(tmp_path) -> None:
    img = tmp_path / "shot.png"
    img.write_bytes(_png())
    store = await SessionStore.open(":memory:")
    try:
        llm = _ToolThenDone()
        sid_holder: dict[str, str] = {}

        async def _look(**_kw) -> str:
            queue_image_for_next_round(sid_holder["sid"], path=str(img),
                                       ref="file_uuid=u-1", note="（你要看的图）")
            return "这 1 张已放到你眼前 ↓"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="look", description="look at an image",
                           input_schema={"type": "object", "properties": {}}),
            _look,
        )
        loop = StatefulAgentLoop(
            llm=llm, store=store, tool_registry=registry,
            config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
        )
        sid = await loop.new_session()
        sid_holder["sid"] = sid
        discard_queued_images(sid)
        await loop.send("看看这张图", session_id=sid)

        rows = await store.load_all_messages(sid)
        # ① 落库：图成了一条真实的 user 行，排在宣告它的 tool 结果之后
        user_rows = [r for r in rows if r.role == "user"]
        assert len(user_rows) == 2, [r.role for r in rows]      # 原始输入 + 注入的图
        injected = user_rows[-1]
        assert (injected.meta or {}).get("content_encoding") == "json"
        assert rows.index(injected) > next(i for i, r in enumerate(rows) if r.role == "tool")

        # ② 本轮请求：第 2 轮的 LLM 确实看到了它
        assert any(isinstance(m.get("content"), list) for m in llm.seen_rounds[1])

        # ③ 跨 send 蒸馏：图变成可回取的引用，而不是一坨结构
        from power_loop.runtime.representation import distill_multimodal_text

        distilled = distill_multimodal_text(injected.content, injected.meta)
        assert "[image: shot.png · file_uuid=u-1]" in distilled
        assert "base64" not in distilled
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_ephemeral_injection_is_not_stored(tmp_path) -> None:
    img = tmp_path / "shot.png"
    img.write_bytes(_png())
    store = await SessionStore.open(":memory:")
    try:
        llm = _ToolThenDone()
        sid_holder: dict[str, str] = {}

        async def _look(**_kw) -> str:
            queue_image_for_next_round(sid_holder["sid"], path=str(img), durable=False)
            return "看一眼就好"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="look", description="look",
                           input_schema={"type": "object", "properties": {}}),
            _look,
        )
        loop = StatefulAgentLoop(
            llm=llm, store=store, tool_registry=registry,
            config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
        )
        sid = await loop.new_session()
        sid_holder["sid"] = sid
        discard_queued_images(sid)
        await loop.send("看看", session_id=sid)

        rows = await store.load_all_messages(sid)
        assert len([r for r in rows if r.role == "user"]) == 1   # 只有原始输入
        assert any(isinstance(m.get("content"), list) for m in llm.seen_rounds[1])  # 但这一轮看得见
    finally:
        await store.close()
