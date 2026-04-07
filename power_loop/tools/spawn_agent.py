"""spawn_agent tool — lets the LLM create sub-agents at runtime.

The parent agent calls ``spawn_agent(task=..., ...)`` as a regular tool.
Internally a new :class:`AgentLoop` is created with an independent event bus
and session, leveraging ``contextvars`` for full isolation.  The sub-agent
runs to completion and its ``final_text`` is returned as the tool output.

Depth control
-------------
A ``max_depth`` counter is threaded through the sub-agent's system prompt
context. If the current depth exceeds the limit, the tool returns an error
instead of spawning.

Event bubbling
--------------
Sub-agent events are optionally re-published on the parent event bus with
``source="subagent:<session_id>"`` so the parent's subscribers can observe
progress.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any, Dict, Mapping

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.event_payloads import SubagentLimitPayload, SubagentTaskStartPayload, SubagentTextPayload
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_event_bus, get_session_id

# Tracks nesting depth per coroutine via contextvars.
_spawn_depth: ContextVar[int] = ContextVar("power_loop_spawn_depth", default=0)

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_ROUNDS = 20

SPAWN_AGENT_DEFINITION = ToolDefinition(
    name="spawn_agent",
    description=(
        "Spawn a sub-agent to handle a task independently. "
        "The sub-agent has its own tool set and session. "
        "Use this when a task can be delegated and worked on in isolation, "
        "for example: research, code exploration, running tests, or generating code. "
        "The sub-agent's final answer is returned as the tool result."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of the task for the sub-agent.",
            },
            "preset": {
                "type": "string",
                "enum": ["core", "explore", "full"],
                "description": "Tool preset for the sub-agent. Default: 'core'.",
            },
            "max_rounds": {
                "type": "integer",
                "description": f"Maximum rounds for the sub-agent (default: {DEFAULT_MAX_ROUNDS}).",
            },
            "system_prompt_extra": {
                "type": "string",
                "description": "Additional instructions appended to the sub-agent's system prompt.",
            },
        },
        "required": ["task"],
    },
    required_params=("task",),
)


async def run_spawn_agent(
        task: str,
        preset: str = "core",
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        system_prompt_extra: str = "",
        *,
        # Injected by the factory (not from LLM args)
        _llm: Any = None,
        _max_depth: int = DEFAULT_MAX_DEPTH,
        _bubble_events: bool = True,
) -> str:
    """Execute a sub-agent and return its final text.

    This function is designed to be called inside an active agent loop.
    It creates an independent :class:`AgentLoop` session.
    """
    # Depth guard
    current_depth = _spawn_depth.get()
    if current_depth >= _max_depth:
        return (
            f"Error: Sub-agent spawn rejected — max nesting depth ({_max_depth}) reached. "
            "Summarize what you need and handle it directly."
        )

    if _llm is None:
        return "Error: spawn_agent is not configured — no LLM service available."

    # Lazy imports to avoid circular dependencies
    from power_loop.agent.loop import AgentLoop
    from power_loop.agent.system_prompt import build_subagent_system_prompt, SystemPromptContext
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.core.events import AgentEventBus
    from power_loop.tools import create_default_tool_registry

    parent_bus = get_event_bus()
    parent_session = get_session_id()
    sub_session_id = f"sub-{uuid.uuid4().hex[:8]}"

    # Build sub-agent components
    sub_bus = AgentEventBus(suppress_subscriber_errors=True)

    # Event bubbling: re-publish sub-agent events on the parent bus
    if _bubble_events and parent_bus is not None:
        source_tag = f"subagent:{sub_session_id}"

        def _bubble(event: AgentEvent) -> None:
            parent_bus.publish(AgentEvent(
                type=event.type,
                payload={**event.payload, "source": source_tag},
                session_id=parent_session,
                round_index=event.round_index,
                stream_id=event.stream_id,
                source=source_tag,
            ))

        sub_bus.subscribe(None, _bubble)  # global subscriber

    parent_bus.publish(AgentEvent(
        type=AgentEventType.SUBAGENT_TASK_START,
        data=SubagentTaskStartPayload(
            task=task[:500], preset=preset,
            sub_session_id=sub_session_id, depth=current_depth + 1,
        ),
        session_id=parent_session,
    ))

    # Build system prompt
    from power_loop.runtime.env import WORKSPACE_DIR
    prompt = build_subagent_system_prompt(
        SystemPromptContext(model="subagent", workspace_dir=str(WORKSPACE_DIR)),
    )
    if system_prompt_extra:
        prompt += f"\n\n{system_prompt_extra}"

    # Build tool registry (exclude spawn_agent from sub-agent to prevent unbounded recursion
    # unless depth allows it)
    sub_registry = create_default_tool_registry(preset=preset)
    if current_depth + 1 < _max_depth:
        # Allow sub-agent to also spawn, but at incremented depth
        sub_registry.register(
            SPAWN_AGENT_DEFINITION,
            _make_spawn_handler(_llm, max_depth=_max_depth, bubble_events=_bubble_events),
            overwrite=True,
        )

    config = AgentLoopConfig(
        system_prompt=prompt,
        max_rounds=max(1, min(max_rounds, 50)),  # clamp
        temperature=0.0,
    )

    loop = AgentLoop(llm=_llm, config=config, tool_registry=sub_registry, event_bus=sub_bus)

    # Set increased depth for the sub-agent's coroutine
    _spawn_depth.set(current_depth + 1)
    try:
        result = await loop.run(
            messages=[{"role": "user", "content": task}],
            session_id=sub_session_id,
        )
    finally:
        _spawn_depth.set(current_depth)

    parent_bus.publish(AgentEvent(
        type=AgentEventType.SUBAGENT_TEXT,
        data=SubagentTextPayload(
            sub_session_id=sub_session_id,
            status=result.status,
            rounds=result.rounds,
            final_text=(result.final_text or "")[:2000],
        ),
        session_id=parent_session,
    ))

    if result.status == "hit_round_limit":
        parent_bus.publish(AgentEvent(
            type=AgentEventType.SUBAGENT_LIMIT,
            data=SubagentLimitPayload(sub_session_id=sub_session_id, max_rounds=config.max_rounds),
            session_id=parent_session,
        ))

    return result.final_text or "(sub-agent produced no output)"


def _make_spawn_handler(
        llm: Any,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        bubble_events: bool = True,
) -> Any:
    """Create a spawn_agent handler closure bound to the given LLM."""

    async def handler(**kwargs: Any) -> str:
        return await run_spawn_agent(
            task=kwargs["task"],
            preset=kwargs.get("preset", "core"),
            max_rounds=kwargs.get("max_rounds", DEFAULT_MAX_ROUNDS),
            system_prompt_extra=kwargs.get("system_prompt_extra", ""),
            _llm=llm,
            _max_depth=max_depth,
            _bubble_events=bubble_events,
        )

    return handler


def register_spawn_agent(
        registry: "ToolRegistry",
        llm: Any,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        bubble_events: bool = True,
        overwrite: bool = False,
) -> None:
    """Convenience: register the spawn_agent tool on an existing registry.

    Usage::

        from power_loop.tools.spawn_agent import register_spawn_agent

        registry = create_default_tool_registry(preset="core")
        register_spawn_agent(registry, llm)
    """
    from power_loop.tools.registry import ToolRegistry  # noqa: F811

    handler = _make_spawn_handler(llm, max_depth=max_depth, bubble_events=bubble_events)
    registry.register(SPAWN_AGENT_DEFINITION, handler, overwrite=overwrite)
