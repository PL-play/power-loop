from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from power_loop.contracts.errors import ToolNotFound, ToolValidationError
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
            is_async = inspect.iscoroutinefunction(cast(Any, handler).__call__)
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

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """Return a new registry with only the named tools (definition + handler).

        Unknown names are ignored. Used for per-call tool allowlisting so the
        model only sees a permitted subset — see ``StatefulAgentLoop.send(tools=...)``.
        The new registry shares handler callables (no rebinding).
        """
        wanted = set(names)
        new = ToolRegistry()
        for name, rt in self._tools.items():
            if name in wanted:
                new.register(rt.definition, rt.handler, overwrite=True)
        return new

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [d.to_openai_tool() for d in self.definitions()]

    def validate(self, name: str, args: Mapping[str, Any]) -> str | None:
        """Validate tool name and arguments. Returns an error string or ``None``.

        This is a **legacy internal** method kept for the pipeline's
        ``execute_tool``; new code should call ``_raise_if_invalid`` or
        invoke directly and catch ``ToolNotFound`` / ``ToolValidationError``.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        err = validate_tool_args(name, args)
        if err:
            return err

        missing = [p for p in tool.definition.required_params if p not in args]
        if missing:
            return f"Error: missing required parameter(s): {', '.join(missing)}"
        return None

    def _raise_if_invalid(self, name: str, args: Mapping[str, Any]) -> None:
        """Raise :class:`ToolNotFound` / :class:`ToolValidationError` if the
        tool or its args are invalid."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(name)

        err = validate_tool_args(name, args)
        if err:
            raise ToolValidationError(name, err)

        missing = [p for p in tool.definition.required_params if p not in args]
        if missing:
            raise ToolValidationError(
                name, f"missing required parameter(s): {', '.join(missing)}",
            )

    def invoke(self, name: str, args: Mapping[str, Any]) -> Any:
        """Sync invocation. Raises :class:`ToolNotFound` if the tool is
        not registered, :class:`ToolValidationError` if args fail validation,
        and :class:`AsyncToolInSyncContext` if the handler is ``async def``.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(name)

        if tool.is_async:
            raise AsyncToolInSyncContext(
                f"Tool {name!r} has an async handler; call invoke_async() instead."
            )

        self._raise_if_invalid(name, args)

        handler = cast(Callable[..., Any], tool.handler)
        try:
            return handler(**dict(args))
        except TypeError:
            return handler(dict(args))

    async def invoke_async(self, name: str, args: Mapping[str, Any]) -> Any:
        """Universal invocation entry. Raises :class:`ToolNotFound` if the
        tool is not registered, :class:`ToolValidationError` if args fail
        validation.

        Sync handlers run in a worker thread (``asyncio.to_thread``) so a
        slow tool — a blocking subprocess, a network call — never stalls the
        event loop and every other session on it. contextvars (runtime env,
        session identity) propagate into the thread. Handlers that must run
        on the loop thread should be ``async def``.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(name)

        self._raise_if_invalid(name, args)

        handler = cast(Callable[..., Any], tool.handler)
        if tool.is_async:
            return await handler(**dict(args))

        def _call_sync() -> Any:
            try:
                return handler(**dict(args))
            except TypeError:
                return handler(dict(args))

        result = await asyncio.to_thread(_call_sync)
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
