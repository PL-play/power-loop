"""power-loop public API.

Stability tiers
---------------

**STABLE** — 跨 minor 版本保证向后兼容；破坏性变更必须升 minor 版本 + CHANGELOG。
业务方（如 DeepTalk `agent` 服务）只应依赖这一层。

    AgentLoop, AgentLoopConfig, AgentLoopResult,
    AgentHooks, AgentEventBus,
    HookPoint, HookDirective,
    ToolRegistry, ToolDefinition,

**PROVISIONAL** — 顶层有 re-export，但 0.x 阶段可能调整。生产代码引用前要确认。
（其余在 ``__all__`` 中、不在上方 STABLE 列表中的所有符号。）

**INTERNAL** — 不从顶层导出；直接 ``from power_loop.core.* import …`` 视为内部 API，
无版本承诺，可随时变更或删除。
"""

__version__ = "0.1.0"

from power_loop.agent.sink import MessageSink, NullSink, SQLiteSink
from power_loop.agent.stateful_loop import StatefulAgentLoop, StatefulResult
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
from power_loop.contracts.errors import (
    CancellationRequested,
    CompactionFailed,
    LLMRetryExhausted,
    LLMTimeout,
    PowerLoopError,
    SessionNotFoundError,
    SessionPendingError,
    SpecValidationError,
    ToolNotFound,
    ToolValidationError,
)
from power_loop.contracts.event_payloads import (
    AgentErrorPayload,
    AutoCompactStatusPayload,
    BaseEventPayload,
    HitRoundLimitStatusPayload,
    LlmDegradedPayload,
    LlmRetryAttemptedPayload,
    LoopCancelledPayload,
    MemoryFailedPayload,
    MemoryRecalledPayload,
    RoundCompletedPayload,
    RoundStartedPayload,
    RoundToolsPresentPayload,
    RoundUsageStatusPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    StatusChangedPayload,
    StreamCompletedPayload,
    StreamDeltaPayload,
    StreamStartedPayload,
    SubagentCompletedPayload,
    SubagentLimitPayload,
    SubagentTaskStartPayload,
    SubagentTextPayload,
    SystemLogPayload,
    TodoUpdatedPayload,
    ToolCallCompletedPayload,
    ToolCallFailedPayload,
    ToolCallStartedPayload,
    UsageUpdatedPayload,
    UserNotificationPayload,
)
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.handlers import EventHandler, HookHandler, ToolHandler, ToolHandlerResult
from power_loop.contracts.hook_contexts import (
    BaseHookCtx,
    CompactAfterCtx,
    CompactBeforeCtx,
    LlmAfterCtx,
    LlmBeforeCtx,
    MemoryRecalledCtx,
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
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.contracts.messages import AgentMessage, MessageRole, ToolCall
from power_loop.contracts.protocols import EventBusProtocol, HookManagerProtocol, ToolArgsValidator
from power_loop.contracts.tools import ToolDefinition, validate_tool_args
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.phase import PhaseContext, PhaseResult, phase
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.runner import AgentRunner
from power_loop.runtime.budget import estimate_text_tokens, estimate_tokens, trim_history
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.memory import MemoryProvider, MemorySnapshot, tag_as_memory
from power_loop.runtime.provider import (
    LLMProviderConfig,
    create_llm_service_from_config,
    create_llm_service_from_env,
)
from power_loop.runtime.retry import LLMRetryPolicy, with_retry
from power_loop.runtime.session_store import (
    DEFAULT_DB_PATH,
    MAX_SPAWN_DEPTH,
    MessageRow,
    MessageState,
    SessionKind,
    SessionRow,
    SessionStatus,
    SessionStore,
    SubagentLifecycle,
)
from power_loop.runtime.spec import AgentSpec, AgentSpecError, run_agent_spec
from power_loop.runtime.structured import (
    StructuredOutputError,
    StructuredOutputSpec,
    parse_structured,
)
from power_loop.tools import ToolRegistry, build_registry, create_default_tool_registry
from power_loop.tools.default_manifest import (
    CORE_TOOL_NAMES,
    EXPLORE_TOOL_NAMES,
    FULL_TOOL_NAMES,
    TOOL_PRESETS,
    get_tool_definitions,
)
from power_loop.tools.registry import AsyncToolInSyncContext
from power_loop.tools.spawn_agent import (
    RUN_AGENT_DEFINITION,
    SPAWN_AGENT_DEFINITION,
    register_spawn_agent,
)

STABLE_API = (
    "StatefulAgentLoop",
    "StatefulResult",
    "AgentLoopConfig",
    "AgentLoopResult",
    "SessionStore",
    "SubagentLifecycle",
    "PowerLoopError",
    "SessionPendingError",
    "SessionNotFoundError",
    "LLMTimeout",
    "LLMRetryExhausted",
    "CancellationRequested",
    "LLMRetryPolicy",
    "CancellationToken",
    "AgentHooks",
    "AgentEventBus",
    "HookPoint",
    "HookDirective",
    "ToolRegistry",
    "ToolDefinition",
)

__all__ = [
	"__version__",
	"STABLE_API",
	"StatefulAgentLoop",
	"StatefulResult",
	"AgentLoopConfig",
	"AgentLoopResult",
	"SessionStore",
	"SessionRow",
	"SessionStatus",
	"SessionKind",
	"SubagentLifecycle",
	"MessageRow",
	"MessageState",
	"MAX_SPAWN_DEPTH",
	"DEFAULT_DB_PATH",
	"PowerLoopError",
	"SessionPendingError",
	"SessionNotFoundError",
	"ToolNotFound",
	"ToolValidationError",
	"SpecValidationError",
	"LLMTimeout",
	"LLMRetryExhausted",
	"CancellationRequested",
	"CompactionFailed",
	"LLMRetryPolicy",
	"with_retry",
	"CancellationToken",
	"CancellationLike",
	"LlmRetryAttemptedPayload",
	"LlmDegradedPayload",
	"LoopCancelledPayload",
	"MemoryProvider",
	"MemorySnapshot",
	"tag_as_memory",
	"MemoryRecalledCtx",
	"MemoryRecalledPayload",
	"MemoryFailedPayload",
	"StructuredOutputSpec",
	"StructuredOutputError",
	"parse_structured",
	"estimate_tokens",
	"estimate_text_tokens",
	"trim_history",
	"LLMProviderConfig",
	"create_llm_service_from_config",
	"create_llm_service_from_env",
	"MessageSink",
	"NullSink",
	"SQLiteSink",
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
	"BaseEventPayload",
	"SessionStartedPayload",
	"SessionEndedPayload",
	"RoundStartedPayload",
	"RoundCompletedPayload",
	"RoundToolsPresentPayload",
	"StreamStartedPayload",
	"StreamDeltaPayload",
	"StreamCompletedPayload",
	"ToolCallStartedPayload",
	"ToolCallCompletedPayload",
	"ToolCallFailedPayload",
	"StatusChangedPayload",
	"AutoCompactStatusPayload",
	"RoundUsageStatusPayload",
	"HitRoundLimitStatusPayload",
	"UsageUpdatedPayload",
	"TodoUpdatedPayload",
	"UserNotificationPayload",
	"AgentErrorPayload",
	"SystemLogPayload",
	"SubagentTaskStartPayload",
	"SubagentTextPayload",
	"SubagentLimitPayload",
	"SubagentCompletedPayload",
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
	"AsyncToolInSyncContext",
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
	"RUN_AGENT_DEFINITION",
	"AgentSpec",
	"AgentSpecError",
	"run_agent_spec",
]
