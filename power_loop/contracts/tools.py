from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    required_params: tuple[str, ...] = ()
    #: Optional self-projection for the send-context layer: ``project(args, result)`` returns
    #: a compact summary (dict or str) of one tool call for the projected history, so each
    #: tool decides what matters (a file tool → its path, bash → exit+head, …). A
    #: ``HistoryProjector`` calls it when present and falls back to truncation otherwise.
    #: ``compare=False`` keeps ToolDefinition equality/hash independent of the callable.
    project: Callable[[Mapping[str, Any], str], dict[str, Any] | str] | None = field(
        default=None, compare=False
    )

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.input_schema),
            },
        }


DEFAULT_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "write_file": ("path", "content"),
    "read_file": ("path",),
    "edit_file": ("path", "old_text", "new_text"),
    "apply_patch": ("path", "patch"),
    "bash": ("command",),
    "glob": ("pattern",),
    "grep": ("pattern",),
    "load_skill": ("name",),
    "todo": ("items",),
    "background_run": ("command",),
    "web_search": ("query",),
    "generate_image": ("prompt",),
    "edit_image": ("image_paths", "prompt"),
}


def validate_tool_args(tool_name: str, args: Mapping[str, Any]) -> str | None:
    required = DEFAULT_REQUIRED_PARAMS.get(tool_name)
    if not required:
        return None
    missing = [param for param in required if param not in args]
    if not missing:
        return None
    req = ", ".join(required)
    miss = ", ".join(missing)
    return (
        f"Error: missing required parameter(s): {miss}. "
        f"{tool_name} requires: {req}. "
        "Please provide all required parameters as a valid JSON object."
    )
