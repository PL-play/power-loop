from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from power_loop.contracts.tools import ToolDefinition, validate_tool_args

ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolCallable


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
        self._tools[definition.name] = RegisteredTool(definition=definition, handler=handler)

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
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        err = self.validate(name, args)
        if err:
            return err

        try:
            return tool.handler(**dict(args))
        except TypeError:
            # Backward compatibility for dict-arg handlers.
            return tool.handler(dict(args))

    async def invoke_async(self, name: str, args: Mapping[str, Any]) -> Any:
        result = self.invoke(name, args)
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
