"""LLM_BEFORE ``persist_messages``: a hook can inject a DURABLE turn (real history/store row,
stamped with the round's send_index) AND have it join this round's request — as opposed to the
ephemeral, request-only edits to ``ctx.messages``. Powers periodic injected reminders."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from power_loop import AgentHooks, AgentLoopConfig, HookPoint, LlmBeforeCtx, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)


@dataclass
class _Capturing(LLMService):
    """Fake LLM that records the messages of every request it receives."""

    seen: list[list[dict]] = field(default_factory=list)

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.seen.append([dict(m) for m in request.messages])
        return LLMResponse(raw_text="done", content_text="done")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _loop(tmp_path, llm, hooks=None):
    return StatefulAgentLoop(
        llm=llm,
        db_path=str(tmp_path / "s.db"),
        config=AgentLoopConfig(system_prompt="t", max_rounds=2),
        hooks=hooks,
    )


async def test_persist_message_is_durable_and_joins_request(tmp_path) -> None:
    hooks = AgentHooks()

    def inject(ctx: LlmBeforeCtx) -> None:
        if ctx.round_index == 0:
            ctx.persist_messages.append({"role": "user", "content": "REMINDER: call note_add"})

    hooks.register(HookPoint.LLM_BEFORE, inject, name="test.persist")
    llm = _Capturing()
    loop = _loop(tmp_path, llm, hooks)
    sid = await loop.new_session()
    await loop.send("hi", sid)

    # (1) durable: the reminder is a REAL persisted turn (not ephemeral)
    rows = await loop.get_messages(sid)
    reminders = [m for m in rows if m["role"] == "user" and "REMINDER" in (m.get("content") or "")]
    assert len(reminders) == 1, rows

    # (2) it was in that round's actual request (the LLM saw it)
    assert any("REMINDER" in (m.get("content") or "") for m in llm.seen[0])

    # (3) ordering: the reminder precedes the assistant's response in history
    idx = next(
        i for i, m in enumerate(rows)
        if m["role"] == "user" and "REMINDER" in (m.get("content") or "")
    )
    assert any(m["role"] == "assistant" for m in rows[idx + 1:])


async def test_empty_persist_messages_is_noop(tmp_path) -> None:
    llm = _Capturing()
    loop = _loop(tmp_path, llm)  # no hook
    sid = await loop.new_session()
    await loop.send("hi", sid)
    rows = await loop.get_messages(sid)
    assert not [m for m in rows if "REMINDER" in (m.get("content") or "")]
