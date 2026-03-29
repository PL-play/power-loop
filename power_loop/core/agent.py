"""Backward-compatible entry point for the agent loop.

The heavy lifting now lives in :class:`~power_loop.core.pipeline.AgentPipeline`.
This module provides ``agent_loop_async`` with the same signature as before,
delegating to the pipeline internally.
"""
from __future__ import annotations

import threading
from typing import Any

from llm_client.interface import LLMService

from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.core.agent_context import get_ctx, get_event_bus, get_hooks, get_session_id
from power_loop.core.pipeline import AgentPipeline
from power_loop.tools.registry import ToolRegistry


async def agent_loop_async(
    *,
    llm: LLMService,
    config: AgentLoopConfig,
    tool_registry: ToolRegistry | None,
    messages: list[LoopMessage],
    stop_event: threading.Event | None = None,
    session_id: str | None = None,
) -> AgentLoopResult:
    """Run the agent loop.  Delegates to :class:`AgentPipeline`."""
    pipeline = AgentPipeline(
        llm=llm,
        config=config,
        tool_registry=tool_registry,
        hooks=get_hooks(),
        bus=get_event_bus(),
        ctx=get_ctx(),
        session_id=session_id,
        stop_event=stop_event,
    )
    return await pipeline.run(messages)
