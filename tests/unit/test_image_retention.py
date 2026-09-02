"""6.12.0 图片看过即撤：attachment 块只在入上下文后的 N 轮内以原图参与请求。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime import image_recall
from power_loop.tools.registry import ToolRegistry


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _idx: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text: Callable[[str], Any] | None = None,
                       on_chunk_think: Callable[[str], Any] | None = None,
                       on_stream_end: Callable[[LLMResponse], Any] | None = None) -> LLMResponse:
        import copy
        self.seen.append(copy.deepcopy([m for m in (getattr(request, "messages", None) or []) if isinstance(m, dict)]))
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _resp(text: str) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text)
    r.token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return r


def _tool(cid: str) -> LLMResponse:
    r = LLMResponse(raw_text="", tool_calls=[{"id": cid, "type": "function", "function": {"name": "look", "arguments": "{}"}}])
    r.token_usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return r


def _attachment_rows(msgs: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            kinds = [str(b.get("type")) + (":" + str(b.get("text"))[:22] if b.get("type") == "text" else "") for b in c if isinstance(b, dict)]
            if any(k.startswith("attachment") or "image retired" in k for k in kinds):
                out.append(kinds)
    return out


def _registry(tmp_path) -> ToolRegistry:
    img = tmp_path / "shot.png"
    img.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + b"0" * 64)
    reg = ToolRegistry()

    async def look(**kw):
        from power_loop.core.agent_context import get_session_id
        image_recall.queue_image_for_next_round(get_session_id(), path=str(img), note="看这张")
        return "queued"

    reg.register(ToolDefinition(name="look", description="look", input_schema={"type": "object", "properties": {}}), look)
    return reg


@pytest.mark.asyncio
async def test_images_are_retired_after_retention_rounds(tmp_path):
    llm = _Scripted(responses=[_tool("c1"), _tool("c2"), _tool("c3"), _resp("done")])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=8, image_retention_rounds=1),
                             tool_registry=_registry(tmp_path))
    sid = await loop.new_session()
    res = await loop.send("hi", session_id=sid)
    assert res.status == "completed"
    # 第 1 次 look 在 round 0 入图 → round 1 请求里是原图；round 2 请求里已换成占位
    r1 = _attachment_rows(llm.seen[1])
    r2 = _attachment_rows(llm.seen[2])
    r3 = _attachment_rows(llm.seen[3])
    assert r1 and any(k == "attachment" for k in r1[0])
    assert r2 and not any(k == "attachment" for k in r2[0]) and any("image retired" in k for k in r2[0])
    # 新入的图（round 1 的 look）在 round 2 仍是原图
    assert len(r2) >= 2 and any(k == "attachment" for k in r2[1])
    assert all(not any(k == "attachment" for k in row) for row in r3[:2])
    await loop.aclose()


@pytest.mark.asyncio
async def test_retention_off_keeps_images(tmp_path):
    llm = _Scripted(responses=[_tool("c1"), _tool("c2"), _resp("done")])
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path / "s.db"),
                             config=AgentLoopConfig(system_prompt="t", max_rounds=8, image_retention_rounds=0),
                             tool_registry=_registry(tmp_path))
    sid = await loop.new_session()
    await loop.send("hi", session_id=sid)
    rows = _attachment_rows(llm.seen[2])
    assert rows and all(any(k == "attachment" for k in row) for row in rows)
    await loop.aclose()
