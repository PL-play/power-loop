from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from power_loop.core.agent_context import get_event_bus, get_session_id
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.runtime.env import AGENT_DIR, SKILLS_DIR, WORKSPACE_DIR, safe_path
from power_loop.runtime.skills import SKILL_LOADER
from llm_client.interface import LLMRequest, LLMResponse


TOOL_MAX_LINES = 20


class TodoManager:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def update(self, items: list[dict[str, Any]]) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated: list[dict[str, Any]] = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        result = self.render()

        # Publish todo state for any UI subscriber.
        session_id = get_session_id()
        get_event_bus().publish(
            AgentEvent(
                type=AgentEventType.TODO_UPDATED,
                payload={"text": result},
                session_id=session_id,
            )
        )
        return result

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines: list[str] = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def snapshot_for_prompt(self) -> str:
        if not self.items:
            return ""
        return f"\n<current_todos>\n{self.render()}\n</current_todos>"

    @property
    def has_in_progress(self) -> bool:
        return any(item["status"] == "in_progress" for item in self.items)


@dataclass
class ContextManager:
    """Per-session agent context: usage tracking + optional compacting."""

    role: str = "main"
    recent_files: list[str] = field(default_factory=list)
    _file_counter: int = 0

    last_input_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0

    _compact_count: int = 0
    subagent_records: list[dict[str, Any]] = field(default_factory=list)

    todo: TodoManager = field(default_factory=TodoManager)

    # Compact config
    compact_threshold: int = field(default_factory=lambda: int(os.getenv("CONTEXT_COMPACT_THRESHOLD", "50000")))
    micro_hot_tail: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MICRO_HOT_TAIL", "10")))
    micro_size_limit: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MICRO_SIZE_LIMIT", "1000")))

    cache_dir: Path = field(default_factory=lambda: (AGENT_DIR / ".cache"))

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Ensure skills loader reads from correct runtime env.
        _ = SKILL_LOADER

    def track_file(self, path: str) -> None:
        if not path:
            return
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.append(path)
        self.recent_files = self.recent_files[-5:]

    def update_usage(self, response: LLMResponse) -> dict[str, int]:
        usage = None
        if getattr(response, "token_usage", None) is not None:
            tu = response.token_usage
            usage = tu.as_dict() if hasattr(tu, "as_dict") else None
        elif getattr(response, "usage", None) is not None:
            usage = response.usage

        def _pick(dct: dict, keys: list[str]) -> int:
            for key in keys:
                val = dct.get(key)
                if isinstance(val, (int, float)) and val is not None:
                    return int(val)
            return 0

        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        reasoning = 0

        if isinstance(usage, dict):
            input_tokens = _pick(usage, ["prompt_tokens", "input_tokens"])
            output_tokens = _pick(usage, ["completion_tokens", "output_tokens"])
            cache_read = _pick(
                usage,
                [
                    "cache_read_input_tokens",
                    "cache_read_tokens",
                    "cached_tokens",
                    "cache_hit_tokens",
                    "prompt_cached_tokens",
                ],
            )
            reasoning = _pick(usage, ["completion_reasoning_tokens", "reasoning_tokens"])
        else:
            input_tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)
            cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
            reasoning = int(getattr(usage, "reasoning_tokens", 0) or 0)

        self.last_input_tokens = input_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.cache_read_tokens += cache_read
        self.reasoning_tokens += reasoning
        self.api_calls += 1
        return {
            "input": self.last_input_tokens,
            "output": output_tokens,
            "cache_read": cache_read,
            "reasoning": reasoning,
            "total_in": self.total_input_tokens,
            "total_out": self.total_output_tokens,
            "total_reasoning": self.reasoning_tokens,
            "calls": self.api_calls,
        }

    def usage_summary(self) -> str:
        return f"input={self.last_input_tokens:,} | session_in={self.total_input_tokens:,} | calls={self.api_calls}"

    def all_usage_summary(self) -> str:
        main_in = self.total_input_tokens
        main_out = self.total_output_tokens
        main_reason = self.reasoning_tokens
        main_calls = self.api_calls
        return f"IN={main_in:,} OUT={main_out:,} reason={main_reason:,} calls={main_calls}"

    def reset_usage(self) -> None:
        self.last_input_tokens = 0

    def should_compact(self) -> bool:
        return self.last_input_tokens > self.compact_threshold

    def microcompact(self, messages: list[dict[str, Any]]) -> None:
        # Keep hot tail tool outputs; summarize/cached replace for old tool outputs.
        tool_output_indices: list[int] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > self.micro_size_limit:
                    tool_output_indices.append(i)

        if len(tool_output_indices) <= self.micro_hot_tail:
            return

        cold = tool_output_indices[:-self.micro_hot_tail]
        for i in cold:
            msg = messages[i]
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if content.startswith("[tool output saved to"):
                continue
            self._file_counter += 1
            cache_path = self.cache_dir / f"tool_{self._file_counter:05d}.md"
            tool_name = str(msg.get("name") or "tool")
            tool_id = str(msg.get("tool_call_id") or "")
            md = (
                f"# Tool Call: {tool_name}\n\n"
                f"**ID**: `{tool_id}`\n\n"
                f"**Output** ({len(content)} chars):\n\n"
                f"{content}\n"
            )
            cache_path.write_text(md, encoding="utf-8")
            replaced = f"[tool output saved to {cache_path.relative_to(AGENT_DIR)}, {tool_name}, {len(content)} chars]"
            msg["content"] = replaced

    async def compact_async(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        *,
        focus: str | None = None,
    ) -> list[dict[str, Any]]:
        self._compact_count += 1
        transcript_path = self.cache_dir / f"transcript_{int(time.time())}.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")

        conversation_text = json.dumps(messages, default=str, ensure_ascii=False)[:80000]
        focus_instruction = f"\nFocus especially on: {focus}\n" if focus else ""

        compact_messages = [
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for continuity. Include:\n"
                    "1) What was accomplished\n"
                    "2) Current state of the codebase and any in-progress work\n"
                    "3) Key technical decisions made and why\n"
                    "4) Open tasks and next steps\n"
                    "5) Errors encountered and how they were resolved\n"
                    "6) Files touched and why they matter — use workspace-relative paths\n"
                    "7) IMPORTANT: Preserve all file paths mentioned. Note that workspace is "
                    f"{WORKSPACE_DIR} and agent home is {AGENT_DIR}. "
                    "All user project files are in workspace.\n"
                    f"{focus_instruction}"
                    "Be concise but preserve critical details needed to continue without re-asking.\n\n"
                    + conversation_text
                ),
            }
        ]

        summary_response = await llm.complete(
            LLMRequest(messages=compact_messages, max_tokens=8000, temperature=0)
        )
        summary = getattr(summary_response, "raw_text", "") or getattr(summary_response, "content_text", "")
        summary = str(summary).strip()

        todo_state = self.todo.render()
        parts: list[str] = [
            "This session is being continued from a previous conversation that ran out of context.",
            f"Transcript saved to: {transcript_path.relative_to(AGENT_DIR)}",
            "",
            "<path_reminder>",
            f"WORKSPACE (your working directory): {WORKSPACE_DIR}",
            f"AGENT HOME (power-loop installation): {AGENT_DIR}",
            f"SKILLS: {SKILLS_DIR}",
            "All relative file paths resolve to WORKSPACE. Files you created earlier in this session are in WORKSPACE.",
            "</path_reminder>",
            "",
            summary,
        ]
        if todo_state and todo_state != "No todos.":
            parts.append(f"\n<current_todos>\n{todo_state}\n</current_todos>")
        if self.recent_files:
            parts.append(f"\nRecently accessed files (workspace-relative): {', '.join(self.recent_files)}")
        parts.append("\nPlease continue from where we left off without asking the user any further questions.")

        return [
            {"role": "user", "content": "\n".join(parts)},
            {"role": "assistant", "content": "Understood. I have the context from the summary and will continue the task."},
        ]

