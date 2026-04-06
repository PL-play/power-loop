"""power-loop public API."""

from power_loop.agent.loop import AgentLoop
from power_loop.agent.system_prompt import (
	BUILTIN_SECTIONS,
	DEFAULT_AGENT_SYSTEM_PROMPT,
	DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT,
	DEFAULT_SUBAGENT_SYSTEM_PROMPT,
	SystemPromptBuilder,
	SystemPromptContext,
	build_agent_system_prompt,
	build_explore_subagent_system_prompt,
	build_subagent_system_prompt,
)
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.phase import PhaseContext, PhaseResult, phase
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.runner import AgentRunner
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.handlers import EventHandler, HookHandler, ToolHandler, ToolHandlerResult
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.contracts.hook_contexts import (
	BaseHookCtx,
	CompactAfterCtx,
	CompactBeforeCtx,
	LlmAfterCtx,
	LlmBeforeCtx,
	MessageAppendCtx,
	RoundDecideCtx,
	RoundEndCtx,
	RoundStartCtx,
	SessionEndCtx,
	SessionStartCtx,
	ToolAfterCtx,
	ToolBeforeCtx,
	ToolErrorCtx,
	ToolsBatchAfterCtx,
	ToolsBatchBeforeCtx,
)
from power_loop.contracts.messages import AgentMessage, MessageRole, ToolCall
from power_loop.contracts.protocols import EventBusProtocol, HookManagerProtocol, ToolArgsValidator
from power_loop.contracts.tools import ToolDefinition, validate_tool_args
from power_loop.tools import ToolRegistry, build_registry, create_default_tool_registry
from power_loop.tools.default_manifest import (
	CORE_TOOL_NAMES,
	EXPLORE_TOOL_NAMES,
	FULL_TOOL_NAMES,
	TOOL_PRESETS,
	get_tool_definitions,
)
from power_loop.tools.spawn_agent import register_spawn_agent, SPAWN_AGENT_DEFINITION

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
	"AgentPipeline",
	"AgentRunner",
	"PhaseContext",
	"PhaseResult",
	"phase",
	"HookContext",
	"HookDirective",
	"HookPoint",
	"HookResult",
	"BaseHookCtx",
	"CompactAfterCtx",
	"CompactBeforeCtx",
	"LlmAfterCtx",
	"LlmBeforeCtx",
	"MessageAppendCtx",
	"RoundDecideCtx",
	"RoundEndCtx",
	"RoundStartCtx",
	"SessionEndCtx",
	"SessionStartCtx",
	"ToolAfterCtx",
	"ToolBeforeCtx",
	"ToolErrorCtx",
	"ToolsBatchAfterCtx",
	"ToolsBatchBeforeCtx",
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
	"build_registry",
	"create_default_tool_registry",
	"get_tool_definitions",
	"CORE_TOOL_NAMES",
	"EXPLORE_TOOL_NAMES",
	"FULL_TOOL_NAMES",
	"TOOL_PRESETS",
	"SystemPromptBuilder",
	"SystemPromptContext",
	"BUILTIN_SECTIONS",
	"DEFAULT_AGENT_SYSTEM_PROMPT",
	"DEFAULT_SUBAGENT_SYSTEM_PROMPT",
	"DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT",
	"build_agent_system_prompt",
	"build_subagent_system_prompt",
	"build_explore_subagent_system_prompt",
	"register_spawn_agent",
	"SPAWN_AGENT_DEFINITION",
]
