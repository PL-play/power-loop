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
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import AsyncToolInSyncContext

DEF = ToolDefinition(
    name="ping",
    description="ping",
    input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    required_params=(),
pricing=Free("test_fixture"))


def test_async_handler_detected_at_register_time() -> None:
    reg = ToolRegistry()

    async def handler(x: int = 0) -> str:
        return f"async:{x}"

    reg.register(DEF, handler)
    tool = reg.get("ping")
    assert tool is not None
    assert tool.is_async is True


def test_sync_handler_detected_as_sync() -> None:
    reg = ToolRegistry()
    reg.register(DEF, lambda x=0: f"sync:{x}")
    tool = reg.get("ping")
    assert tool is not None
    assert tool.is_async is False


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
    tool = reg.get("ping")
    assert tool is not None
    assert tool.is_async is True
    assert await reg.invoke_async("ping", {"x": 7}) == "obj:7"


@pytest.mark.parametrize('async_entry', [False, True])
@pytest.mark.asyncio
async def test_internal_type_error_never_repeats_side_effect(async_entry):
    reg = ToolRegistry()
    calls = []
    failure = TypeError('provider response could not be decoded')

    def handler(x=0):
        calls.append(x)  # represents an already accepted external request
        raise failure

    reg.register(DEF, handler)
    with pytest.raises(TypeError) as caught:
        if async_entry:
            await reg.invoke_async('ping', {'x': 7})
        else:
            reg.invoke('ping', {'x': 7})
    assert caught.value is failure
    assert calls == [7]


@pytest.mark.parametrize('async_entry', [False, True])
@pytest.mark.asyncio
async def test_legacy_mapping_handler_is_bound_before_execution(async_entry):
    reg = ToolRegistry()
    calls = []

    def handler(arguments, /):
        calls.append(arguments)
        return arguments['x']

    reg.register(DEF, handler)
    result = (await reg.invoke_async('ping', {'x': 8}) if async_entry
              else reg.invoke('ping', {'x': 8}))
    assert result == 8
    assert calls == [{'x': 8}]


@pytest.mark.asyncio
async def test_invalid_signature_does_not_execute_handler():
    reg = ToolRegistry()
    calls = []

    def handler(first, second):
        calls.append((first, second))

    reg.register(DEF, handler)
    with pytest.raises(TypeError):
        await reg.invoke_async('ping', {'x': 8})
    assert calls == []


@pytest.mark.asyncio
async def test_opaque_handler_internal_error_is_not_retried():
    reg = ToolRegistry()
    calls = []

    class OpaqueHandler:
        __signature__ = 'not inspectable'

        def __call__(self, x=0):
            calls.append(x)
            raise TypeError('already executed')

    reg.register(DEF, OpaqueHandler())
    with pytest.raises(TypeError, match='already executed'):
        await reg.invoke_async('ping', {'x': 9})
    assert calls == [9]
