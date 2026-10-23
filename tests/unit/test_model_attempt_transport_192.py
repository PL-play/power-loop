"""Actual localhost HTTP checks both SDK and service/stream hidden retries."""
from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.attempt_scope import single_attempt
from power_loop._vendor.llm_client.interface import OpenAICompatibleChatConfig
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("fault", ["http_status", "stream_interrupted"])
async def test_native_ledger_has_one_wire_request_without_changing_legacy_retry_policy(tmp_path, native, fault):
    openai = pytest.importorskip("openai")
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
    store = await SessionStore.open(str(tmp_path / "http-attempt.db"), **({"schema": "tasks"} if native else {}))
    try:
        runtime = TaskRuntime(JobManager(store), tenant_id="wire-test", principal_id="host", executor_id="executor", boot_id="boot") if native else None
        loop = StatefulAgentLoop(llm=service, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=100,
                retry_policy=LLMRetryPolicy(max_attempts=1, backoff_initial=0)))
        sid = await loop.new_session()
        async with asyncio.timeout(30):
            result = await loop.send("Return one test result.", sid)
        assert result.status == "degraded"
        assert len(requests) == (1 if native else 4 if fault == "http_status" else 3)
        assert single_attempt.get() is False  # no policy leak into another send/host
        if native:
            [job] = await runtime.manager.db.fetchall(f"SELECT * FROM {runtime.manager.jobs}")
            assert job["resource_state"] == "unknown" and job["supervision_status"] == "open"
            page = await list_attempts(runtime.manager, runtime.actor(sid), job["job_id"])
            assert len(page["items"]) == 1 and page["items"][0]["status"] == "unknown"
            assert page["items"][0]["observed_usage"]["prompt_tokens"] > 0
            if fault == "stream_interrupted":
                assert page["items"][0]["observed_usage"]["completion_tokens"] > 0
    finally:
        try:
            await service.close()
        finally:
            await store.close()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*list(handlers), return_exceptions=True)
