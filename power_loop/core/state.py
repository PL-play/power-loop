from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.event_payloads import TodoUpdatedPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.core.agent_context import get_event_bus, get_session_id
from power_loop.runtime.env import get_runtime_env

TOOL_MAX_LINES = 20


class TodoManager:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def update(self, items: list[dict[str, Any]]) -> str:
        from power_loop.runtime.todos import validate_todos

        # 与工具层同一个校验口径（6.16.0）——这里原来是一份**逐字重复**的实现，
        # 改一处漏一处，清单就会在工具层和上下文层各说各话。
        self.items = validate_todos(items)
        result = self.render()
        done = sum(1 for t in self.items if t["status"] == "completed")

        # Publish todo state for any UI subscriber.
        session_id = get_session_id()
        get_event_bus().publish(
            AgentEvent(
                type=AgentEventType.TODO_UPDATED,
                data=TodoUpdatedPayload(
                    items=[dict(x) for x in self.items],
                    counts={"total": len(self.items), "completed": done},
                    rendered=result,
                    text=result,
                ),
                session_id=session_id,
            )
        )
        return result

    def render(self) -> str:
        from power_loop.runtime.todos import render_todos

        return render_todos(self.items)

    def snapshot_for_prompt(self) -> str:
        if not self.items:
            return ""
        return f"\n<current_todos>\n{self.render()}\n</current_todos>"

    @property
    def has_in_progress(self) -> bool:
        return any(item["status"] == "in_progress" for item in self.items)


@dataclass
class ContextManager:
    """Per-session agent context: usage tracking + microcompact + todo state.

    LLM-summary compaction lives in :mod:`power_loop.runtime.compact`
    (configured via ``AgentLoopConfig.compactor``). This class only owns
    the orthogonal "spill large tool outputs to disk" path and the
    telemetry-friendly :meth:`update_usage` parser.
    """

    role: str = "main"
    recent_files: list[str] = field(default_factory=list)
    _file_counter: int = 0

    #: Usage of the **last** LLM call only (overwritten by every
    #: :meth:`update_usage`). For whole-run totals use :attr:`usage_totals`.
    token_usage: dict[str, Any] = field(default_factory=dict)
    #: Cumulative usage across every LLM call of this context's lifetime
    #: (one context = one ``send``/run): prompt_tokens / completion_tokens /
    #: cache_read_tokens / reasoning_tokens / total_tokens / calls.
    #: Surfaced on ``AgentLoopResult.usage`` and ``StatefulResult.usage``.
    usage_totals: dict[str, int] = field(default_factory=dict)
    #: 可选计数器，供测试/扩展使用；**不会**被 ``update_usage`` 自动递增
    api_calls: int = 0
    #: 本次 run 已执行的工具调用次数（pipeline.execute_tool 递增）。
    tool_calls: int = 0

    subagent_records: list[dict[str, Any]] = field(default_factory=list)

    todo: TodoManager = field(default_factory=TodoManager)

    # Microcompact (large tool-output spill-to-disk) config — orthogonal to
    # the LLM-summary Compactor in runtime/compact.py.
    micro_hot_tail: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MICRO_HOT_TAIL", "10")))
    micro_size_limit: int = field(default_factory=lambda: int(os.getenv("CONTEXT_MICRO_SIZE_LIMIT", "1000")))

    cache_dir: Path | None = None

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
        total_tokens = 0

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
            total_tokens = _pick(usage, ["total_tokens"])
            if total_tokens == 0 and (input_tokens or output_tokens):
                # Derive total when the provider omits it (the bundled OpenAI/
                # Anthropic factories already do this). Otherwise SharedBudget —
                # which charges total_tokens — never advances for a vendor-neutral
                # custom LLMService that reports only prompt/completion.
                total_tokens = input_tokens + output_tokens

        usage_out: dict[str, int] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "reasoning_tokens": reasoning,
            "total_tokens": total_tokens,
            # 与常见 OpenAI usage 字段对齐的别名
            "input": input_tokens,
            "output": output_tokens,
            "cache_read": cache_read,
            "reasoning": reasoning,
        }
        self.token_usage = usage_out
        totals = self.usage_totals
        totals["calls"] = totals.get("calls", 0) + 1
        for key in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                    "reasoning_tokens", "total_tokens"):
            totals[key] = totals.get(key, 0) + usage_out[key]
        return usage_out

    def microcompact(
        self,
        messages: list[dict[str, Any]],
        *,
        size_limit: int | None = None,
        hot_tail: int | None = None,
        spill_dir: str | None = None,
    ) -> None:
        # Knobs come from the caller (AgentLoopConfig) when provided; otherwise
        # fall back to this context's own fields (env-defaulted) — keeps direct
        # ``ctx.microcompact(msgs)`` callers (tests) working.
        size_limit = self.micro_size_limit if size_limit is None else size_limit
        hot_tail = self.micro_hot_tail if hot_tail is None else hot_tail
        # Keep hot tail tool outputs; summarize/cached replace for old tool outputs.
        tool_output_indices: list[int] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > size_limit:
                    tool_output_indices.append(i)

        if len(tool_output_indices) <= hot_tail:
            return

        # Keep the last ``hot_tail`` tool outputs hot; spill the rest. Avoid
        # ``[:-n]`` — it is ``[:0]`` (spills NOTHING) when hot_tail==0, the
        # opposite of "keep zero hot". hot_tail=0 is a valid config.
        cold = tool_output_indices[: len(tool_output_indices) - hot_tail]
        for i in cold:
            msg = messages[i]
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if content.startswith("[tool output saved to"):
                continue
            self._file_counter += 1
            if self.cache_dir is None:
                self.cache_dir = (
                    Path(spill_dir) if spill_dir
                    else get_runtime_env().require_home_dir() / ".cache"
                )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
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
            replaced = f"[tool output saved to {cache_path}, {tool_name}, {len(content)} chars]"
            msg["content"] = replaced
