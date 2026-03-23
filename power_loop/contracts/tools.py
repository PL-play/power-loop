from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    required_params: tuple[str, ...] = ()

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.input_schema),
            },
        }


DEFAULT_REQUIRED_PARAMS: Dict[str, tuple[str, ...]] = {
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
