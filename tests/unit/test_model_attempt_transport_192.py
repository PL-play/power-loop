"""Actual localhost HTTP checks both SDK and service/stream hidden retries."""
from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from power_loop import AgentLoopConfig
from power_loop._vendor.llm_client.attempt_scope import single_attempt
from power_loop._vendor.llm_client.interface import OpenAICompatibleChatConfig
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
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
        config=AgentLoopConfig(max_rounds=2, max_tokens=100, compactor=None,
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
