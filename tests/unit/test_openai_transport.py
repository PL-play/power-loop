"""Unit tests for the OpenAI-compatible streaming transport (the DEFAULT provider).

Mirrors ``test_anthropic_transport.py`` with a fake ``AsyncOpenAI`` that yields a
scripted async stream of chat-completion chunks. This moves the default transport
from nightly-only real-LLM coverage to per-PR unit coverage (H2.2).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from power_loop._vendor.llm_client.interface import LLMRequest, OpenAICompatibleChatConfig
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService

pytestmark = pytest.mark.unit


# ── Fake AsyncOpenAI streaming surface ────────────────────────────────────────
# The transport calls `await client.chat.completions.create(messages=..., stream=True)`
# and `async for event in <result>`. Each event mimics a ChatCompletionChunk: a
# `.choices[0].delta` carrying `.content` and/or `.tool_calls`, and a trailing
# usage-only event with empty `.choices` and `.usage`. SimpleNamespace works because
# the transport reads via attributes / `__dict__`.


def _content_event(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _tool_event(index: int, *, call_id: str | None = None, name: str | None = None,
                args: str | None = None, type_: str | None = None) -> Any:
    fn = SimpleNamespace(name=name, arguments=args)
    tc = SimpleNamespace(index=index, id=call_id, type=type_, function=fn)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))])


def _usage_event(prompt: int, completion: int, total: int) -> Any:
    return SimpleNamespace(
        choices=[],
        usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total},
    )


class _FakeCompletions:
    """Each `create()` call consumes the next scripted stream from ``streams``.

    A stream entry is a list of events; if an entry ends with an Exception class/
    instance, iteration raises it after yielding the preceding events (to exercise
    the resume path).
    """

    def __init__(self, streams: list[list[Any]]):
        self._streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        events = self._streams.pop(0)

        async def _gen() -> Any:
            for e in events:
                if isinstance(e, BaseException) or (isinstance(e, type) and issubclass(e, BaseException)):
                    raise (e() if isinstance(e, type) else e)
                yield e

        return _gen()


class _FakeClient:
    def __init__(self, streams: list[list[Any]]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(streams))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _service(streams: list[list[Any]], **cfg_overrides: Any) -> tuple[OpenAICompatibleChatLLMService, _FakeClient]:
    cfg = OpenAICompatibleChatConfig(
        base_url="https://openai.example/v1",
        api_key="sk-test",
        model="qwen-test",
        max_tokens=256,
        temperature=0.2,
        **cfg_overrides,
    )
    svc = OpenAICompatibleChatLLMService(cfg)
    client = _FakeClient(streams)
    svc._client = cast(Any, client)  # bypass real AsyncOpenAI construction
    return svc, client


@pytest.mark.asyncio
async def test_streaming_ambiguous_tool_calls_not_merged() -> None:
    # llm-transport-5: two distinct tool calls streamed with NO id and NO index (each opened by a
    # function NAME in a separate event) must NOT collide into one entry. An arguments-only delta
    # continues the most-recent call.
    stream = [
        _tool_event(None, name="first", args='{"a": 1}'),    # call 1 (ambiguous)
        _tool_event(None, name="second", args='{"b":'),      # call 2 — distinct, must not merge
        _tool_event(None, args=' 2}'),                       # args-only → continues call 2
        _usage_event(1, 1, 2),
    ]
    svc, _ = _service([stream])
    result = await svc.complete(LLMRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "first", "parameters": {}}}],
    ))
    calls = result.tool_calls
    assert len(calls) == 2
    assert [c["function"]["name"] for c in calls] == ["first", "second"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1}
    assert json.loads(calls[1]["function"]["arguments"]) == {"b": 2}  # continuation appended


@pytest.mark.asyncio
async def test_complete_accumulates_text_tool_calls_and_usage() -> None:
    stream = [
        _content_event("Hello, "),
        _content_event("world!"),
        # one tool call streamed across two delta chunks (name first, args split)
        _tool_event(0, call_id="call_abc", type_="function", name="lookup", args='{"q":'),
        _tool_event(0, args=' "teal"}'),
        _usage_event(11, 7, 18),
    ]
    svc, client = _service([stream])

    result = await svc.complete(
        LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        )
    )

    # text deltas accumulate in order
    assert result.raw_text == "Hello, world!"

    # the split tool-call is reassembled into one call with the full argument string
    calls = result.tool_calls
    assert len(calls) == 1
    assert calls[0]["id"] == "call_abc"
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "teal"}

    # usage is taken from the trailing usage-only event
    assert result.token_usage.prompt_tokens == 11
    assert result.token_usage.completion_tokens == 7
    assert result.token_usage.total_tokens == 18

    # _request_kwargs mapping reached the SDK: model/temperature/max_tokens/tools + streaming
    sent = client.chat.completions.calls[0]
    assert sent["model"] == "qwen-test"
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 256
    assert sent["tools"][0]["function"]["name"] == "lookup"
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][-1]["content"] == "hi" or sent["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_stream_resumes_after_mid_stream_error() -> None:
    """A stream that drops mid-flight is resumed (a fresh create with the partial
    text + resume instruction); the final text stitches both halves together."""
    streams = [
        [_content_event("par"), RuntimeError("stream dropped")],   # first attempt: partial then error
        [_content_event("tial done"), _usage_event(5, 3, 8)],       # resume completes it
    ]
    svc, client = _service(streams, stream_resume_on_error=True, stream_max_restarts=1)

    result = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "go"}]))

    assert result.raw_text == "partial done"
    assert result.token_usage.total_tokens == 8
    # two underlying calls: the original + one resume
    assert len(client.chat.completions.calls) == 2
    # the resume request carries the partial assistant text + the resume instruction
    resume_msgs = client.chat.completions.calls[1]["messages"]
    assert any(m.get("role") == "assistant" and "par" in str(m.get("content", "")) for m in resume_msgs)


@pytest.mark.asyncio
async def test_stream_error_propagates_when_resume_disabled() -> None:
    """With resume off (the default), a mid-stream error surfaces to the caller
    instead of being silently swallowed."""
    streams = [[_content_event("oops"), RuntimeError("boom")]]
    svc, _ = _service(streams)  # stream_resume_on_error defaults False

    with pytest.raises(RuntimeError, match="boom"):
        await svc.complete(LLMRequest(messages=[{"role": "user", "content": "go"}]))


# ── C17: `reason` is opt-in, not injected by default ─────────────────────────


@pytest.mark.asyncio
async def test_reason_flag_not_injected_by_default() -> None:
    """A default LLMRequest must NOT carry extra_body={"reason": ...} — it's a
    non-standard flag that strict OpenAI-compatible servers reject and reasoning
    gateways silently honor."""
    svc, client = _service([[_content_event("hi"), _usage_event(1, 1, 2)]])
    await svc.complete(LLMRequest(messages=[{"role": "user", "content": "go"}]))
    sent = client.chat.completions.calls[0]
    assert "reason" not in (sent.get("extra_body") or {})


@pytest.mark.asyncio
async def test_reason_flag_sent_when_explicitly_enabled() -> None:
    """Opt-in still works: reason=True injects extra_body={"reason": True}."""
    svc, client = _service([[_content_event("hi"), _usage_event(1, 1, 2)]])
    await svc.complete(LLMRequest(messages=[{"role": "user", "content": "go"}], reason=True))
    sent = client.chat.completions.calls[0]
    assert (sent.get("extra_body") or {}).get("reason") is True


@pytest.mark.asyncio
async def test_stream_resume_discards_partial_tool_call() -> None:
    """U3 guard: a tool call only partially streamed before a mid-stream error must
    NOT survive into the resumed attempt's result — otherwise the final tool_calls
    carries both the truncated (invalid-args) call and the resume's complete one."""
    streams = [
        # attempt 1: a tool call streamed only halfway (args cut off), then the stream drops
        [_tool_event(0, call_id="call_partial", type_="function", name="write", args='{"path":"a"'),
         RuntimeError("stream dropped")],
        # resume: the model re-issues the call in full (a fresh id) + usage
        [_tool_event(0, call_id="call_full", type_="function", name="write",
                     args='{"path":"a","content":"b"}'),
         _usage_event(5, 3, 8)],
    ]
    svc, client = _service(streams, stream_resume_on_error=True, stream_max_restarts=1)

    result = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "go"}]))

    # exactly ONE tool call survives — the resume's complete one, not the partial
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_full"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "a", "content": "b"}
