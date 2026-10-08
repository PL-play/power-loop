"""Env-proxy client with a direct fallback when the proxy itself is unreachable.

Real sockets throughout: a local HTTP server stands in for the provider, a closed port for a
proxy that is down, and a tiny listening server for a proxy that is up.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from power_loop._vendor.llm_client import proxy_fallback
from power_loop._vendor.llm_client.anthropic_factory import AnthropicMessagesLLMService
from power_loop._vendor.llm_client.interface import AnthropicChatConfig, OpenAICompatibleChatConfig
from power_loop._vendor.llm_client.llm_factory import OpenAICompatibleChatLLMService
from power_loop._vendor.llm_client.proxy_fallback import ProxyFallbackTransport, env_proxy_http_client

_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in _PROXY_ENV:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
    monkeypatch.delenv(proxy_fallback.ENV_SWITCH, raising=False)
    proxy_fallback._down_until.clear()
    yield
    proxy_fallback._down_until.clear()


def _serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class _Provider(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"direct-ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _RefusingProxy(BaseHTTPRequestHandler):
    """A live proxy that answers every request with 403 (e.g. a rule denies it)."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def provider() -> Iterator[str]:
    yield from _serve(_Provider)


@pytest.fixture
def refusing_proxy() -> Iterator[str]:
    yield from _serve(_RefusingProxy)


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _fallback_transports(client: httpx.AsyncClient) -> list[ProxyFallbackTransport]:
    return [t for t in client._mounts.values() if isinstance(t, ProxyFallbackTransport)]


def test_no_proxy_in_env_keeps_the_sdk_default_client() -> None:
    assert env_proxy_http_client(httpx.AsyncClient) is None


def test_switch_off_keeps_the_sdk_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://gateway.invalid:8890")
    monkeypatch.setenv(proxy_fallback.ENV_SWITCH, "0")
    assert env_proxy_http_client(httpx.AsyncClient) is None


async def test_proxy_down_goes_direct_and_stays_direct_for_a_while(
    monkeypatch: pytest.MonkeyPatch, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    proxy = _closed_port_url()
    monkeypatch.setenv("HTTP_PROXY", proxy)
    client = env_proxy_http_client(httpx.AsyncClient)
    assert client is not None
    async with client:
        with caplog.at_level(logging.WARNING, logger=proxy_fallback.__name__):
            resp = await client.get(provider)
        assert (resp.status_code, resp.text) == (200, "direct-ok")
        assert proxy_fallback._down_until[proxy] > time.monotonic()
        assert "unreachable" in caplog.text

        # Within the down window the proxy is not tried again.
        (transport,) = _fallback_transports(client)
        tried: list[str] = []

        async def _must_not_be_called(request: httpx.Request) -> httpx.Response:
            tried.append(str(request.url))
            raise AssertionError("proxy tried inside the down window")

        real_send = transport._proxied.handle_async_request
        monkeypatch.setattr(transport._proxied, "handle_async_request", _must_not_be_called)
        resp = await client.get(provider)
        assert resp.status_code == 200 and tried == []

        # After the window the proxy is tried again (and, still down, falls back again).
        monkeypatch.setattr(transport._proxied, "handle_async_request", real_send)
        proxy_fallback._down_until[proxy] = time.monotonic() - 1
        resp = await client.get(provider)
        assert resp.status_code == 200
        assert proxy_fallback._down_until[proxy] > time.monotonic()


async def test_a_live_proxy_answer_stands(monkeypatch: pytest.MonkeyPatch, provider: str, refusing_proxy: str) -> None:
    monkeypatch.setenv("HTTP_PROXY", refusing_proxy)
    client = env_proxy_http_client(httpx.AsyncClient)
    assert client is not None
    async with client:
        resp = await client.get(provider)
    assert resp.status_code == 403  # the proxy's answer, not a silent direct retry
    assert proxy_fallback._down_until == {}


async def test_connect_failure_past_a_live_proxy_is_raised(
    monkeypatch: pytest.MonkeyPatch, provider: str, refusing_proxy: str
) -> None:
    monkeypatch.setenv("HTTP_PROXY", refusing_proxy)
    client = env_proxy_http_client(httpx.AsyncClient)
    assert client is not None
    (transport,) = _fallback_transports(client)

    async def _upstream_tls_failed(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream TLS handshake failed")

    monkeypatch.setattr(transport._proxied, "handle_async_request", _upstream_tls_failed)
    async with client:
        with pytest.raises(httpx.ConnectError, match="upstream TLS"):
            await client.get(provider)
    assert proxy_fallback._down_until == {}


async def test_no_proxy_hosts_never_touch_the_proxy(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    monkeypatch.setenv("HTTP_PROXY", _closed_port_url())
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    client = env_proxy_http_client(httpx.AsyncClient)
    assert client is not None
    async with client:
        resp = await client.get(provider)
    assert resp.status_code == 200
    assert proxy_fallback._down_until == {}  # went direct by NO_PROXY, not by fallback


def test_both_sdk_clients_get_the_fallback_when_a_proxy_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://gateway.invalid:8890")
    openai_svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(base_url="https://llm.example/v1", api_key="sk-test", model="m")
    )
    anthropic_svc = AnthropicMessagesLLMService(
        AnthropicChatConfig(base_url="https://anthropic.example", api_key="sk-test", model="m")
    )
    for sdk_client in (openai_svc._ensure_client(), anthropic_svc._ensure_client()):
        transports = _fallback_transports(sdk_client._client)
        assert [t.proxy_url for t in transports] == ["http://gateway.invalid:8890"]


def test_sdk_clients_stay_default_without_a_proxy() -> None:
    openai_svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(base_url="https://llm.example/v1", api_key="sk-test", model="m")
    )
    assert _fallback_transports(openai_svc._ensure_client()._client) == []
