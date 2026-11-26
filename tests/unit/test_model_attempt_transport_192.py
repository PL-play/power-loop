"""Actual localhost HTTP checks both SDK and service/stream hidden retries."""
from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from power_loop import AgentEventBus, AgentEventType, AgentLoopConfig
from power_loop._vendor.llm_client.anthropic_factory import AnthropicMessagesLLMService
from power_loop._vendor.llm_client.attempt_scope import single_attempt
from power_loop._vendor.llm_client.interface import AnthropicChatConfig, LLMRequest, OpenAICompatibleChatConfig
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.retry import LLMRetryPolicy

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["http_status", "stream_interrupted"])
async def test_unknown_provider_result_is_not_retried_by_any_transport_layer(runtime_agent, fault):
    import openai
    requests, handlers = [], set()
    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            fields = dict(line.split(b":", 1) for line in header.split(b"\r\n")[1:] if b":" in line)
            length = int(next((v for k, v in fields.items() if k.lower() == b"content-length"), b"0"))
            await reader.readexactly(length)
            requests.append(True)
            if fault == "http_status":
                body = json.dumps({"error": {"message": "test: transient failure", "type": "server_error", "code": "unavailable"}}).encode()
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            else:
                chunk = {"id": "local-response", "object": "chat.completion.chunk", "created": 0, "model": "local-test",
                    "choices": [{"index": 0, "delta": {"content": "PARTIAL-RESPONSE"}, "finish_reason": None}]}
                body = b"data: " + json.dumps(chunk).encode() + b"\n\n"
                # Break a real HTTP body after a valid SSE event. This tests
                # the native stream-resume loop, not only an SDK method mock.
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: " + str(len(body) + 1000).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)
    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", model="local-test", timeout_s=5,
        max_retries=1, retry_base_delay_s=0, stream_resume_on_error=True, stream_max_restarts=2))
    service._client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", timeout=5,
        max_retries=1, http_client=openai.DefaultAsyncHttpxClient(trust_env=False))
    agent = await runtime_agent(model=service,
        config=AgentLoopConfig(max_rounds=2, max_tokens=100, context_budget_tokens=10**9,
            retry_policy=LLMRetryPolicy(max_attempts=1, backoff_initial=0)))
    try:
        async with asyncio.timeout(30):
            result = await agent.send('Return one test result.')
        assert result.status == 'failed'
        assert len(requests) == 1
        assert single_attempt.get() is False
        assert result.usage['prompt_tokens'] > 0
        if fault == 'stream_interrupted':
            assert result.usage['completion_tokens'] > 0
    finally:
        await agent.aclose()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


def _sse_ok_response(text: str) -> bytes:
    chunk = {
        "id": "local-ok", "object": "chat.completion.chunk", "created": 0, "model": "local-test",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
    }
    usage = {
        "id": "local-ok", "object": "chat.completion.chunk", "created": 0, "model": "local-test",
        "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    body = (
        b"data: " + json.dumps(chunk).encode() + b"\n\n"
        b"data: " + json.dumps(usage).encode() + b"\n\n"
        b"data: [DONE]\n\n"
    )
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
        + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
    )


def _http_error_response(status_line: bytes, message: str, *, err_type: str = "server_error", code: str = "unavailable") -> bytes:
    body = json.dumps({"error": {"message": message, "type": err_type, "code": code}}).encode()
    return (
        b"HTTP/1.1 " + status_line + b"\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
    )


def _broken_sse_response() -> bytes:
    chunk = {
        "id": "local-partial", "object": "chat.completion.chunk", "created": 0, "model": "local-test",
        "choices": [{"index": 0, "delta": {"content": "PARTIAL-RESPONSE"}, "finish_reason": None}],
    }
    body = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
        + str(len(body) + 1000).encode() + b"\r\nConnection: close\r\n\r\n" + body
    )


def _local_service(port: int) -> OpenAICompatibleChatLLMService:
    import openai
    service = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", model="local-test", timeout_s=5,
        max_retries=1, retry_base_delay_s=0, stream_resume_on_error=True, stream_max_restarts=2))
    service._client = openai.AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", timeout=5,
        max_retries=1, http_client=openai.DefaultAsyncHttpxClient(trust_env=False))
    return service


async def _read_http_request(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readuntil(b"\r\n\r\n")
    fields = dict(line.split(b":", 1) for line in header.split(b"\r\n")[1:] if b":" in line)
    length = int(next((v for k, v in fields.items() if k.lower() == b"content-length"), b"0"))
    return await reader.readexactly(length)


@pytest.mark.parametrize("fault", ["http_status", "stream_interrupted"])
async def test_upper_retry_policy_sends_one_http_request_per_attempt(runtime_agent, fault):
    requests, handlers = [], set()
    started, completed, retried, stream_started, stream_completed = [], [], [], [], []
    scope_on_complete: list[bool] = []

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await _read_http_request(reader)
            requests.append(True)
            if len(requests) == 1:
                payload = (
                    _http_error_response(b"503 Service Unavailable", "test: transient failure")
                    if fault == "http_status" else _broken_sse_response()
                )
            else:
                payload = _sse_ok_response("RECOVERED")
            writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e.data))
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: (completed.append(e.data), scope_on_complete.append(single_attempt.get())))
    bus.subscribe(AgentEventType.LLM_RETRY_ATTEMPTED, lambda e: retried.append(e.data))
    bus.subscribe(AgentEventType.STREAM_STARTED, lambda e: stream_started.append(e.data))
    bus.subscribe(AgentEventType.STREAM_COMPLETED, lambda e: stream_completed.append(e.data))
    agent = await runtime_agent(
        model=_local_service(port),
        event_bus=bus,
        config=AgentLoopConfig(
            max_rounds=2, max_tokens=100, context_budget_tokens=10**9,
            retry_policy=LLMRetryPolicy(max_attempts=2, backoff_initial=0),
        ),
    )
    try:
        async with asyncio.timeout(30):
            result = await agent.send("Retry once on transient failure.")
        assert result.status == "completed"
        assert result.final_text == "RECOVERED"
        assert len(requests) == 2
        assert single_attempt.get() is False
        assert scope_on_complete == [False, False]
        assert len(started) == len(completed) == 2
        assert len(stream_started) == len(stream_completed) == 2
        assert len(retried) == 1
        assert [s.attempt for s in started] == [c.attempt for c in completed] == [1, 2]
        assert [s.call_id for s in started] == [c.call_id for c in completed]
        assert [c.success for c in completed] == [False, True]
    finally:
        await agent.aclose()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


async def test_request_repair_sends_one_http_request_per_attempt(runtime_agent):
    requests, handlers = [], set()
    started, completed, retried, scope_on_complete = [], [], [], []

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await _read_http_request(reader)
            requests.append(True)
            if len(requests) == 1:
                writer.write(_http_error_response(
                    b"400 Bad Request", "invalid messages: tool_call format",
                    err_type="invalid_request_error", code="invalid_messages",
                ))
            else:
                writer.write(_sse_ok_response("REPAIRED"))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e.data))
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: (completed.append(e.data), scope_on_complete.append(single_attempt.get())))
    bus.subscribe(AgentEventType.LLM_RETRY_ATTEMPTED, lambda e: retried.append(e.data))
    agent = await runtime_agent(
        model=_local_service(port),
        event_bus=bus,
        config=AgentLoopConfig(
            max_rounds=2, max_tokens=100, context_budget_tokens=10**9,
            retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0),
        ),
    )
    try:
        async with asyncio.timeout(30):
            result = await agent.send("Repair rejected request once.")
        assert result.status == "completed"
        assert result.final_text == "REPAIRED"
        assert len(requests) == 2
        assert single_attempt.get() is False
        assert scope_on_complete == [False, False]
        assert len(retried) == 0
        assert [s.attempt for s in started] == [c.attempt for c in completed] == [1, 2]
        assert [c.success for c in completed] == [False, True]
    finally:
        await agent.aclose()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


async def test_single_attempt_context_restored_after_cancellation(runtime_agent):
    requests, handlers = [], set()
    entered, release = asyncio.Event(), asyncio.Event()
    started, completed, scope_inside, scope_on_complete = [], [], [], []

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await _read_http_request(reader)
            requests.append(True)
            entered.set()
            await release.wait()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = _local_service(port)
    orig_complete = service.complete

    async def observed_complete(request, **kwargs):
        scope_inside.append(single_attempt.get())
        return await orig_complete(request, **kwargs)

    service.complete = observed_complete
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e.data))
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: (completed.append(e.data), scope_on_complete.append(single_attempt.get())))
    agent = await runtime_agent(
        model=service,
        event_bus=bus,
        config=AgentLoopConfig(
            max_rounds=2, max_tokens=100, context_budget_tokens=10**9,
            retry_policy=LLMRetryPolicy(max_attempts=2, backoff_initial=0),
        ),
    )
    try:
        receipt = await agent.runtime.client.send(agent.target, "Hold until cancelled.", message_id="r05-cancel", actor=agent.actor)
        await asyncio.wait_for(entered.wait(), 5)
        status = await agent.runtime.client.inspect(receipt.input_ref, actor=agent.actor)
        await agent.runtime.client.cancel(status.target, actor=agent.actor)
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms() + 10000)
        await agent.finish(receipt.input_ref, failed=True)
        assert result.status == "cancelled"
        assert len(requests) == 1
        assert scope_inside == [True]
        assert scope_on_complete == [False]
        assert single_attempt.get() is False
        assert len(started) == len(completed) == 1
        assert started[0].call_id == completed[0].call_id
        assert completed[0].success is False
    finally:
        release.set()
        await agent.aclose()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


def _anthropic_ok_response(text: str) -> bytes:
    body = json.dumps({
        "id": "msg_local_ok",
        "type": "message",
        "role": "assistant",
        "model": "local-claude",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }).encode()
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
    )


async def test_managed_attempt_does_not_follow_http_307_redirect(runtime_agent, monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    requests, handlers = [], set()
    started, completed, scope_on_complete = [], [], []

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request_line = await reader.readline()
            await _read_http_request(reader)
            requests.append(request_line.decode().strip())
            if len(requests) == 1:
                writer.write(
                    b"HTTP/1.1 307 Temporary Redirect\r\n"
                    b"Location: /canonical/chat/completions\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            else:
                writer.write(_sse_ok_response("REDIRECTED"))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", model="local-test", timeout_s=5,
        max_retries=1, retry_base_delay_s=0, stream_resume_on_error=True, stream_max_restarts=2,
    ))
    bus = AgentEventBus()
    bus.subscribe(AgentEventType.LLM_CALL_STARTED, lambda e: started.append(e.data))
    bus.subscribe(AgentEventType.LLM_CALL_COMPLETED, lambda e: (completed.append(e.data), scope_on_complete.append(single_attempt.get())))
    agent = await runtime_agent(
        model=service,
        event_bus=bus,
        config=AgentLoopConfig(
            max_rounds=2, max_tokens=100, context_budget_tokens=10**9,
            retry_policy=LLMRetryPolicy(max_attempts=1, backoff_initial=0),
        ),
    )
    try:
        async with asyncio.timeout(30):
            result = await agent.send("Do not follow 307 redirect.")
        assert result.status == "failed"
        assert requests == ["POST /v1/chat/completions HTTP/1.1"]
        assert single_attempt.get() is False
        assert scope_on_complete == [False]
        assert len(started) == len(completed) == 1
        assert started[0].call_id == completed[0].call_id
        assert started[0].attempt == completed[0].attempt == 1
        assert completed[0].success is False
        assert result.usage["prompt_tokens"] > 0
    finally:
        await service.close()
        await agent.aclose()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


async def test_anthropic_single_attempt_does_not_follow_http_307_redirect(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    requests, handlers = [], set()

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request_line = await reader.readline()
            await _read_http_request(reader)
            requests.append(request_line.decode().strip())
            if len(requests) == 1:
                writer.write(
                    b"HTTP/1.1 307 Temporary Redirect\r\n"
                    b"Location: /canonical/messages\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            else:
                writer.write(_anthropic_ok_response("REDIRECTED"))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = AnthropicMessagesLLMService(AnthropicChatConfig(
        base_url=f"http://127.0.0.1:{port}", api_key="local-test-only", model="local-claude", timeout_s=5,
        max_retries=1, temperature=None,
    ))
    try:
        token = single_attempt.set(True)
        try:
            with pytest.raises(Exception) as exc_info:
                await service.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=32))
            assert getattr(exc_info.value, "status_code", None) == 307
        finally:
            single_attempt.reset(token)
        assert requests == ["POST /v1/messages HTTP/1.1"]
        assert single_attempt.get() is False
    finally:
        await service.close()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_single_attempt_200_succeeds_and_preserves_unmanaged_calls(monkeypatch, provider):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    requests, handlers = [], set()

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            request_line = await reader.readline()
            await _read_http_request(reader)
            requests.append(request_line.decode().strip())
            if provider == "openai":
                writer.write(_sse_ok_response("OK"))
            else:
                writer.write(_anthropic_ok_response("OK"))
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    if provider == "openai":
        service = OpenAICompatibleChatLLMService(OpenAICompatibleChatConfig(
            base_url=f"http://127.0.0.1:{port}/v1", api_key="local-test-only", model="local-test",
            timeout_s=5, max_retries=0, temperature=None,
        ))
    else:
        service = AnthropicMessagesLLMService(AnthropicChatConfig(
            base_url=f"http://127.0.0.1:{port}", api_key="local-test-only", model="local-claude",
            timeout_s=5, max_retries=0, temperature=None,
        ))
    try:
        for managed in (True, False):
            token = single_attempt.set(managed)
            try:
                resp = await service.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=32))
            finally:
                single_attempt.reset(token)
            assert resp.raw_text == "OK"
        assert len(requests) == 2
        assert single_attempt.get() is False
    finally:
        await service.close()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*list(handlers), return_exceptions=True)
