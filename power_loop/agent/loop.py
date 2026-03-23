from __future__ import annotations

import asyncio
import threading
from typing import List

from llm_client.interface import LLMService

from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.runner import AgentRunner
from power_loop.core.agent import agent_loop_async
from power_loop.tools.registry import ToolRegistry


class AgentLoop:
    """Robust agent loop facade.

    This class is a thin wrapper around :func:`power_loop.core.agent.agent_loop_async`.
    It keeps the original `AgentLoop(llm, config, tool_registry)` constructor shape
    so existing integrations keep working.
    """

    def __init__(
        self,
        llm: LLMService,
        config: AgentLoopConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        *,
        event_bus: AgentEventBus | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        self.llm = llm
        self.config = config if config is not None else AgentLoopConfig()
        self.tool_registry = tool_registry
        self._runner = AgentRunner(event_bus=event_bus, hooks=hooks)

    async def run(
        self,
        messages: List[LoopMessage],
        stop_event: threading.Event | None = None,
        *,
        session_id: str | None = None,
    ) -> AgentLoopResult:
        async with self._runner.session_async(session_id=session_id):
            return await agent_loop_async(
                llm=self.llm,
                config=self.config,
                tool_registry=self.tool_registry,
                messages=messages,
                stop_event=stop_event,
                session_id=session_id,
            )

    def run_sync(
        self,
        messages: List[LoopMessage],
        stop_event: threading.Event | None = None,
        *,
        session_id: str | None = None,
    ) -> AgentLoopResult:
        return asyncio.run(self.run(messages, stop_event=stop_event, session_id=session_id))
