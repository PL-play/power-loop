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
    #: ``result`` is ``None`` when the call produced no result row (unfinished/failed) — distinct
    #: from a produced-but-empty ``""``. ``compare=False`` keeps ToolDefinition equality/hash
    #: independent of the callable.
    project: Callable[[Mapping[str, Any], str | None], dict[str, Any] | str] | None = field(
        default=None, compare=False
    )
    #: 6.8.0：标记本工具可被 ``background_run(action="tool")`` 异步执行（无副作用、可安全
    #: 并发/重跑的长耗时调用，如生成图像、抓网页）。标记后，只要 background_run 同在
    #: 工具集里，渲染给模型的描述会自动追加「可异步」用法后缀（registry.to_openai_tools）。
    async_capable: bool = False

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
    "background_run": ("action",),
    # schedule_wakeup 的 action 不再必填（6.13.0）：模型按「给了 delay_seconds+note 就是排闹钟」的
    # 直觉省略它，被硬拒后每次白烧一轮（conv-222/223/224/226 实测 21 次调用 7 次因此失败，且同族的
    # schedule_followup 的 operation 本来就可省略）。改由 handler 从参数形状推断，见 default_tools。
    "note": ("action",),
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
