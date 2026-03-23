"""power-loop public API."""

from power_loop.agent.loop import AgentLoop
from power_loop.agent.system_prompt import (
	DEFAULT_AGENT_SYSTEM_PROMPT,
	DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT,
	DEFAULT_SUBAGENT_SYSTEM_PROMPT,
	SystemPromptContext,
	build_agent_system_prompt,
	build_explore_subagent_system_prompt,
	build_subagent_system_prompt,
)
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.runner import AgentRunner
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.handlers import EventHandler, HookHandler, ToolHandler, ToolHandlerResult
from power_loop.contracts.hooks import HookContext, HookPoint
from power_loop.contracts.messages import AgentMessage, MessageRole, ToolCall
from power_loop.contracts.protocols import EventBusProtocol, HookManagerProtocol, ToolArgsValidator
from power_loop.contracts.tools import ToolDefinition, validate_tool_args
from power_loop.tools import ToolRegistry, create_default_tool_registry

__all__ = [
	"AgentLoop",
	"AgentLoopConfig",
	"AgentLoopResult",
	"AgentMessage",
	"MessageRole",
	"ToolCall",
	"AgentEvent",
	"AgentEventType",
	"AgentEventBus",
	"AgentHooks",
	"AgentRunner",
	"HookContext",
	"HookPoint",
	"EventHandler",
	"HookHandler",
	"ToolHandler",
	"ToolHandlerResult",
	"EventBusProtocol",
	"HookManagerProtocol",
	"ToolArgsValidator",
	"ToolDefinition",
	"validate_tool_args",
	"ToolRegistry",
	"create_default_tool_registry",
	"DEFAULT_AGENT_SYSTEM_PROMPT",
	"DEFAULT_SUBAGENT_SYSTEM_PROMPT",
	"DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT",
	"SystemPromptContext",
	"build_agent_system_prompt",
	"build_subagent_system_prompt",
	"build_explore_subagent_system_prompt",
]
