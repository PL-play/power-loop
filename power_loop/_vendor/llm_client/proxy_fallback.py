"""Env-proxy routing for the LLM SDK clients, with a direct fallback when the proxy is down.

Hosts route outbound LLM traffic through an HTTP egress proxy with ``HTTPS_PROXY`` /
``NO_PROXY``. If that proxy goes down, every LLM call fails even though the provider is
reachable directly. :func:`env_proxy_http_client` builds the SDK's httpx client with the
same env-proxy routing httpx applies by default (``HTTP(S)_PROXY`` / ``ALL_PROXY`` /
``NO_PROXY``), plus one rule: when connecting to the proxy fails and a short TCP probe
confirms the proxy itself is unreachable, the request goes direct, and requests keep going
direct for ``DOWN_TTL_S`` seconds before the proxy is tried again.

Only "the proxy can't be reached" triggers the fallback. If the proxy answers, its answer
stands: a refused ``CONNECT`` (``httpx.ProxyError``), an error status, or a connect failure
past a live proxy (the probe succeeds) is raised exactly as before.

On by default. ``POWER_LOOP_PROXY_FALLBACK=0`` turns it off (the SDK builds its own client
and env proxies apply as before). Without a proxy in the environment nothing changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

ENV_SWITCH = "POWER_LOOP_PROXY_FALLBACK"
DOWN_TTL_S = 30.0
PROBE_TIMEOUT_S = 2.0

# The OpenAI and Anthropic SDKs use these limits for their default clients; a client built
# with an explicit transport has to pass them to the transport itself.
_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)

# proxy url -> time.monotonic() until which requests skip it; shared by every client in the
# process, so one confirmed outage is not re-detected per LLM service.
_down_until: dict[str, float] = {}


def fallback_enabled() -> bool:
    return (os.getenv(ENV_SWITCH) or "1").strip().lower() not in {"0", "false", "no", "off"}


def _keepalive_socket_options() -> list[tuple[int, int, int]]:
    """TCP keepalive like the Anthropic SDK's default transport (long streams through proxies)."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 60), ("TCP_KEEPCNT", 5)):
        opt = getattr(socket, name, None)
        if opt is not None:
            options.append((socket.IPPROTO_TCP, opt, value))
    return options


def _env_proxy_map() -> dict[str, str | None]:
    """httpx's own reading of the proxy env (pattern -> proxy url, or None for NO_PROXY)."""
    try:
        from httpx._utils import get_environment_proxies
    except ImportError:  # pragma: no cover - httpx moved it; fall back to the SDK default client
        return {}
    return get_environment_proxies()


def _redact(proxy_url: str) -> str:
    parts = urlsplit(proxy_url)
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}:{parts.port}" if parts.port else f"{parts.scheme}://{host}"


async def _proxy_reachable(proxy_url: str) -> bool:
    parts = urlsplit(proxy_url)
    if not parts.hostname:
        return False
    default_port = 443 if parts.scheme == "https" else 1080 if parts.scheme.startswith("socks") else 80
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(parts.hostname, parts.port or default_port), PROBE_TIMEOUT_S
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


class ProxyFallbackTransport(httpx.AsyncBaseTransport):
    """Send through ``proxy_url``; if the proxy is unreachable, send through ``direct``."""

    def __init__(self, proxy_url: str, direct: httpx.AsyncBaseTransport, **transport_kwargs: Any) -> None:
        self.proxy_url = proxy_url
        self._proxied = httpx.AsyncHTTPTransport(proxy=proxy_url, **transport_kwargs)
        self._direct = direct

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if time.monotonic() < _down_until.get(self.proxy_url, 0.0):
            return await self._direct.handle_async_request(request)
        try:
            return await self._proxied.handle_async_request(request)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Nothing reached the provider yet. Fall back only if the proxy itself is gone; a
            # failure past a live proxy (e.g. the upstream TLS handshake) is the proxy's answer.
            if await _proxy_reachable(self.proxy_url):
                raise
            _down_until[self.proxy_url] = time.monotonic() + DOWN_TTL_S
            logger.warning(
                "proxy %s unreachable (%s: %s); LLM requests go direct for the next %ds",
                _redact(self.proxy_url),
                type(exc).__name__,
                exc,
                int(DOWN_TTL_S),
            )
            return await self._direct.handle_async_request(request)

    async def aclose(self) -> None:
        # The direct transport is the client's own transport; the client closes it.
        await self._proxied.aclose()


def env_proxy_http_client(client_cls: type[httpx.AsyncClient]) -> httpx.AsyncClient | None:
    """An SDK http client (``client_cls`` = the SDK's ``DefaultAsyncHttpxClient``) with the fallback.

    Returns None — meaning "let the SDK build its default client" — when the fallback is
    switched off or no proxy is set in the environment.
    """
    if not fallback_enabled():
        return None
    proxy_map = _env_proxy_map()
    if not any(proxy_map.values()):
        return None
    transport_kwargs: dict[str, Any] = {"limits": _LIMITS, "socket_options": _keepalive_socket_options()}
    direct = httpx.AsyncHTTPTransport(**transport_kwargs)
    by_url: dict[str, ProxyFallbackTransport] = {}
    mounts: dict[str, httpx.AsyncBaseTransport | None] = {}
    for pattern, url in proxy_map.items():
        if url is None:
            mounts[pattern] = None  # NO_PROXY entry: the client's own (direct) transport
            continue
        if url not in by_url:
            by_url[url] = ProxyFallbackTransport(url, direct, **transport_kwargs)
        mounts[pattern] = by_url[url]
    return client_cls(transport=direct, mounts=mounts)
