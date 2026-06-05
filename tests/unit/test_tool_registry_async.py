"""Unit tests for M1.6 — ToolRegistry async-handler ergonomics.

Covers:
- Async ``async def`` handlers are auto-detected at register time.
- ``invoke()`` (sync) raises ``AsyncToolInSyncContext`` for async handlers
  with a useful message (no silent unawaited-coroutine bug).
- ``invoke_async()`` works for both sync and async same-named tools
  (across separate registries — confirms the detection isn't per-call).
- Sync handlers that *return* coroutines are still awaited by
  ``invoke_async`` (existing behaviour preserved).
"""

from __future__ import annotations

import asyncio

import pytest

from power_loop import ToolDefinition, ToolRegistry
from power_loop.tools.registry import AsyncToolInSyncContext

DEF = ToolDefinition(
    name="ping",
    description="ping",
    input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    required_params=(),
)


def test_async_handler_detected_at_register_time() -> None:
    reg = ToolRegistry()

    async def handler(x: int = 0) -> str:
        return f"async:{x}"

    reg.register(DEF, handler)
    assert reg.get("ping").is_async is True


def test_sync_handler_detected_as_sync() -> None:
    reg = ToolRegistry()
    reg.register(DEF, lambda x=0: f"sync:{x}")
    assert reg.get("ping").is_async is False


def test_invoke_sync_on_async_handler_raises_clear_error() -> None:
    reg = ToolRegistry()

    async def handler(x: int = 0) -> str:
        return f"v={x}"

    reg.register(DEF, handler)
    with pytest.raises(AsyncToolInSyncContext) as exc:
        reg.invoke("ping", {"x": 1})
    assert "invoke_async" in str(exc.value)


@pytest.mark.asyncio
async def test_invoke_async_handles_async_handler() -> None:
    reg = ToolRegistry()

    async def handler(x: int = 0) -> str:
        await asyncio.sleep(0)
        return f"async:{x}"

    reg.register(DEF, handler)
    assert await reg.invoke_async("ping", {"x": 5}) == "async:5"


@pytest.mark.asyncio
async def test_invoke_async_handles_sync_handler() -> None:
    reg = ToolRegistry()
    reg.register(DEF, lambda x=0: f"sync:{x}")
    assert await reg.invoke_async("ping", {"x": 5}) == "sync:5"


@pytest.mark.asyncio
async def test_sync_handler_returning_awaitable_is_still_awaited() -> None:
    reg = ToolRegistry()

    async def inner() -> str:
        return "inner"

    # Sync wrapper that hands back a coroutine — pre-M1.6 behaviour preserved.
    reg.register(DEF, lambda x=0: inner())
    assert await reg.invoke_async("ping", {"x": 0}) == "inner"


@pytest.mark.asyncio
async def test_callable_object_with_async_dunder_call_detected() -> None:
    reg = ToolRegistry()

    class AsyncCallable:
        async def __call__(self, x: int = 0) -> str:
            return f"obj:{x}"

    reg.register(DEF, AsyncCallable())
    assert reg.get("ping").is_async is True
    assert await reg.invoke_async("ping", {"x": 7}) == "obj:7"
