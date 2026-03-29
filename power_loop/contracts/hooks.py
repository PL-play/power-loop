from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class HookPoint(str, Enum):
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    ROUND_START = "round.start"
    ROUND_END = "round.end"
    LLM_BEFORE = "llm.before"
    LLM_AFTER = "llm.after"
    TOOLS_BATCH_BEFORE = "tools.batch.before"
    TOOLS_BATCH_AFTER = "tools.batch.after"
    TOOL_BEFORE = "tool.before"
    TOOL_AFTER = "tool.after"
    TOOL_ERROR = "tool.error"
    ROUND_DECIDE = "round.decide"
    COMPACT_BEFORE = "compact.before"
    COMPACT_AFTER = "compact.after"
    MESSAGE_APPEND = "message.append"


class HookDirective(str, Enum):
    """Control-flow directives that hooks can return to influence the agent loop.

    Not every directive is valid at every hook point. The agent loop checks
    the directive returned by ``hooks.run_async`` and acts accordingly.

    Supported combinations:
        ROUND_START        -> BREAK (end loop), SKIP (skip this round)
        LLM_BEFORE         -> SHORT_CIRCUIT (use values["response"] instead of calling LLM)
        LLM_AFTER          -> BREAK (end loop, ignore tool calls)
        ROUND_DECIDE       -> SKIP (skip tool execution), BREAK (end loop)
        TOOLS_BATCH_BEFORE -> SKIP (skip all tools this round)
        TOOL_BEFORE        -> SKIP (skip this tool, use values["tool_output"] as result)
        TOOL_ERROR         -> SKIP (swallow error, use values["tool_output"]),
                              SHORT_CIRCUIT (retry — re-invoke the tool)
        TOOL_AFTER         -> BREAK (stop executing remaining tools, proceed to next round)
        COMPACT_BEFORE     -> SKIP (skip compaction this round)
    """

    CONTINUE = "continue"
    SKIP = "skip"
    BREAK = "break"
    SHORT_CIRCUIT = "short_circuit"


@dataclass
class HookContext:
    """Mutable context passed through each hook chain."""

    values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Return value from hook execution, carrying both context and control directive."""

    context: HookContext = field(default_factory=HookContext)
    directive: HookDirective = HookDirective.CONTINUE
