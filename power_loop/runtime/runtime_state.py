from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

RuntimeMessage = dict[str, Any]


@runtime_checkable
class RuntimeProjector(Protocol):
    """Project persisted runtime state into transient LLM messages.

    Implementations may read/write ``SessionStore`` state. Returned messages
    are sent to the next LLM call but are not persisted in the conversation
    log. This is the extension point for tools whose state must survive
    compaction, process restarts, and custom agent loop instances.
    """

    def project(
        self,
        *,
        store: Any,
        session_id: str,
        round_index: int,
        context: Any,
    ) -> list[RuntimeMessage]: ...


@dataclass(frozen=True)
class TodoRuntimeProjector(RuntimeProjector):
    state_key: str = "todo"

    def project(
        self,
        *,
        store: Any,
        session_id: str,
        round_index: int,
        context: Any,
    ) -> list[RuntimeMessage]:
        todo_state = store.get_runtime_state(session_id, self.state_key, default={}) or {}
        rendered = todo_state.get("rendered") if isinstance(todo_state, dict) else None
        if not isinstance(rendered, str) or not rendered.strip():
            return []
        return [
            {
                "role": "user",
                "name": "runtime_todo_state",
                "content": f"<current_todos>\n{rendered}\n</current_todos>",
            }
        ]


@dataclass(frozen=True)
class BackgroundRuntimeProjector(RuntimeProjector):
    mark_seen: bool = True

    def project(
        self,
        *,
        store: Any,
        session_id: str,
        round_index: int,
        context: Any,
    ) -> list[RuntimeMessage]:
        updates = store.list_unseen_background_updates(session_id)
        if not updates:
            return []
        chunks: list[str] = ["<background_updates>"]
        for task in updates:
            output = (task.output_tail or "(running)").strip()
            chunks.append(
                "\n".join(
                    [
                        f'<task id="{task.task_id}" status="{task.status}">',
                        f"command: {task.command}",
                        "output:",
                        output,
                        "</task>",
                    ]
                )
            )
        chunks.append("</background_updates>")
        if self.mark_seen:
            store.mark_background_seen(session_id, [task.task_id for task in updates])
        return [
            {
                "role": "user",
                "name": "runtime_background_updates",
                "content": "\n\n".join(chunks),
            }
        ]


def default_runtime_projectors() -> tuple[RuntimeProjector, ...]:
    return (TodoRuntimeProjector(), BackgroundRuntimeProjector())


__all__ = [
    "BackgroundRuntimeProjector",
    "RuntimeMessage",
    "RuntimeProjector",
    "TodoRuntimeProjector",
    "default_runtime_projectors",
]
