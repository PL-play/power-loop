"""Core contracts for agent runtime (types/protocols/constants only)."""

from power_loop.contracts.handlers import (
    EventHandler,
    HookHandler,
    ToolHandler,
    ToolHandlerResult,
)
from power_loop.contracts.hooks import HookContext, HookPoint
from power_loop.contracts.messages import (
    AgentMessage,
    MessageRole,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
)
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.protocols import EventBusProtocol, HookManagerProtocol, ToolArgsValidator
from power_loop.contracts.tools import ToolDefinition, validate_tool_args

__all__ = [
    "AgentMessage",
    "MessageRole",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "AgentEvent",
    "AgentEventType",
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
]
