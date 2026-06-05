from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from power_loop.contracts.tools import ToolDefinition, validate_tool_args

ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]


class AsyncToolInSyncContext(TypeError):
    """Raised when ``ToolRegistry.invoke`` (sync) is called for a handler
    that is a coroutine function. The caller should use ``invoke_async``
    instead — silently returning the unawaited coroutine corrupts loop
    state and is the single most common cause of "tool seemed to succeed
    but did nothing" bugs.
    """


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolCallable
    is_async: bool = False


class ToolRegistry:
    """Open tool registry for dynamic bind/add/remove operations.

    Design goals:
    - Runtime dynamic registration for library users
    - Tool schema and handler decoupled but bound by the same name
    - One execution entry (`invoke`/`invoke_async`) with built-in required-param validation
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, handler: ToolCallable, *, overwrite: bool = False) -> None:
        if not overwrite and definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        # Detect async at register time so ``invoke()`` can raise a clear
        # error without doing the call first. ``iscoroutinefunction``
        # covers ``async def``; for callable objects whose ``__call__`` is
        # async we additionally check that. Plain sync callables that
        # *happen* to return awaitables are still handled at call time by
        # ``invoke_async``.
        is_async = inspect.iscoroutinefunction(handler)
        if not is_async and not inspect.isfunction(handler) and callable(handler):
            # Callable object whose ``__call__`` is async (e.g. dataclass
            # with ``async def __call__``). Check that explicitly.
            is_async = inspect.iscoroutinefunction(handler.__call__)
        self._tools[definition.name] = RegisteredTool(
            definition=definition, handler=handler, is_async=is_async,
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [item.definition for item in self._tools.values()]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [d.to_openai_tool() for d in self.definitions()]

    def validate(self, name: str, args: Mapping[str, Any]) -> str | None:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        # Keep compatibility with zero-code required params behavior.
        err = validate_tool_args(name, args)
        if err:
            return err

        # If a definition has explicit required_params, validate as well.
        missing = [p for p in tool.definition.required_params if p not in args]
        if missing:
            return f"Error: missing required parameter(s): {', '.join(missing)}"
        return None

    def invoke(self, name: str, args: Mapping[str, Any]) -> Any:
        """Sync invocation. Raises :class:`AsyncToolInSyncContext` if the
        handler is an ``async def`` — use :meth:`invoke_async` for those.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        if tool.is_async:
            raise AsyncToolInSyncContext(
                f"Tool {name!r} has an async handler; call invoke_async() instead."
            )

        err = self.validate(name, args)
        if err:
            return err

        try:
            return tool.handler(**dict(args))
        except TypeError:
            # Backward compatibility for dict-arg handlers.
            return tool.handler(dict(args))

    async def invoke_async(self, name: str, args: Mapping[str, Any]) -> Any:
        """Universal invocation entry. Handles both sync and async handlers.

        For async handlers, calls them directly and awaits the coroutine.
        For sync handlers, calls them and awaits if the return value happens
        to be awaitable (handlers wrapping async libraries).
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        err = self.validate(name, args)
        if err:
            return err

        if tool.is_async:
            return await tool.handler(**dict(args))

        try:
            result = tool.handler(**dict(args))
        except TypeError:
            result = tool.handler(dict(args))
        if inspect.isawaitable(result):
            return await result
        return result


def build_registry(definitions: list[ToolDefinition], handlers: Mapping[str, ToolCallable]) -> ToolRegistry:
    registry = ToolRegistry()
    for definition in definitions:
        handler = handlers.get(definition.name)
        if handler is None:
            raise ValueError(f"Missing handler for tool definition: {definition.name}")
        registry.register(definition, handler)
    return registry
