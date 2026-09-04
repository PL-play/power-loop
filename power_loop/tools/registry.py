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



def async_capable_for(definition: Any, args: Mapping[str, Any] | None = None) -> bool:
    """这次调用可不可以异步跑（6.15.0：``async_capable`` 支持 action 粒度）。

    ``True``/``False`` 是工具级的老语义。给一组 action 名时，只有 ``args["action"]`` 命中
    这一组才算可异步——**拿不到 args 就一律判否**：同轮并发要在发起前就决定，宁可少并发一次，
    也不能把一个写 action 当成只读的并发出去。标错的代价不对称。
    """
    flag = getattr(definition, "async_capable", False)
    if isinstance(flag, bool):
        return flag
    if not flag:
        return False
    action = str((args or {}).get("action") or "").strip()
    return bool(action) and action in flag


def async_capable_actions(definition: Any) -> tuple[str, ...]:
    """标了 action 粒度时返回那几个 action 名（工具级 True/False 返回空）——只给文案用。"""
    flag = getattr(definition, "async_capable", False)
    return tuple(sorted(flag)) if not isinstance(flag, bool) and flag else ()


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
        defs = self.definitions()
        # async_capable 描述后缀（6.8.0）：仅当 background_run 真在本工具集里才追加——
        # 否则是在教模型调一个不存在的入口。文案集中在这一处，别在各工具描述里手写。
        has_bg = any(d.name == "background_run" for d in defs)
        out: list[dict[str, Any]] = []
        for d in defs:
            t = d.to_openai_tool()
            if has_bg and d.async_capable and d.name != "background_run":
                acts = async_capable_actions(d)
                scope = ("（仅 action=" + "/".join(acts) + "）") if acts else ""
                t["function"]["description"] = (
                    str(t["function"]["description"])
                    + f"\n⏳ 可异步{scope}：background_run(action=\"tool\", tool=\"{d.name}\", "
                      "args={…}) 立即返回 task_id 不阻塞；完成后会收到通知，"
                      "background_run(action=\"check\") 取结果。不需要立刻用结果时优先异步。"
                )
            out.append(t)
        return out

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
