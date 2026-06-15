"""power-loop public API.

Stability tiers
---------------

**STABLE** — 跨 minor 版本保证向后兼容；破坏性变更必须升 major 版本 + CHANGELOG。
业务方（如 DeepTalk `agent` 服务）只应依赖这一层。**权威清单是本模块的 ``STABLE_API``
元组**（不在 docstring 里重复罗列以免漂移；测试会校验 ``STABLE_API`` 与 ``__all__`` 一致，
并有 SemVer 守卫：未升 major 而删除/改名 STABLE 符号会让测试失败）。

**PROVISIONAL** — 顶层有 re-export，但 0.x 阶段可能调整。生产代码引用前要确认。
（其余在 ``__all__`` 中、不在上方 STABLE 列表中的所有符号。）

**INTERNAL** — 不从顶层导出；直接 ``from power_loop.core.* import …`` 视为内部 API，
无版本承诺，可随时变更或删除。
"""

__version__ = "0.17.0"

# Public LLM contract (SDK-free) re-exported so callers (e.g. writing llm.* hooks or
# a custom LLMService) don't reach into the internal vendored transport package (H3.4).
# Library logging hygiene (H4.5): attach a NullHandler to the package-root logger so
# importing power-loop never emits "No handlers could be found" noise or logs unless
# the application explicitly configures handlers. All module loggers live under this
# "power_loop.*" subtree (getLogger(__name__)), so one handler routes the whole tree.
import logging as _logging

from power_loop._vendor.llm_client.interface import (
    AnthropicChatConfig,
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
    OpenAICompatibleChatConfig,
)
from power_loop.agent.follow_up import FollowUpQueued
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
    format_tool_catalog,
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
    BudgetExceededStatusPayload,
    HitRoundLimitStatusPayload,
    LlmCallCompletedPayload,
    LlmCallStartedPayload,
    LlmDegradedPayload,
    LlmRetryAttemptedPayload,
    LoopCancelledPayload,
    MemoryFailedPayload,
    MemoryRecalledPayload,
    PhaseEventPayload,
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
    TimerFiredPayload,
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
    TimerFireCtx,
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
from power_loop.runtime.blackboard import (
    Blackboard,
    BlackboardEntry,
    BlackboardError,
    SqliteBlackboard,
    render_entries,
)
from power_loop.runtime.budget import estimate_text_tokens, estimate_tokens, trim_history
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.env import RuntimeEnv, RuntimeEnvError, runtime_env_context
from power_loop.runtime.exec_backend import (
    DEFAULT_SHELL_BACKEND,
    LocalShellBackend,
    ShellBackend,
)
from power_loop.runtime.human_input import HumanInputRequired, request_user_input
from power_loop.runtime.memory import MemoryProvider, MemorySnapshot, tag_as_memory
from power_loop.runtime.notes import (
    DEFAULT_NOTES_POLICY,
    NotesFullError,
    NotesPolicy,
    SQLiteNoteMemory,
    render_notes,
)
from power_loop.runtime.provider import (
    LLMProviderConfig,
    create_llm_service_from_config,
    create_llm_service_from_env,
)
from power_loop.runtime.retry import LLMRetryPolicy, with_retry
from power_loop.runtime.runtime_state import (
    BackgroundRuntimeProjector,
    RuntimeMessage,
    RuntimeProjector,
    TodoRuntimeProjector,
    ToolRuntimeContext,
    default_runtime_projectors,
    get_tool_runtime_context,
)
from power_loop.runtime.session_store import (
    DEFAULT_DB_PATH,
    MAX_SPAWN_DEPTH,
    MessageRow,
    MessageState,
    SessionKind,
    SessionRow,
    SessionStatsRow,
    SessionStatus,
    SessionStore,
    SubagentLifecycle,
    TimerRow,
)
from power_loop.runtime.skills import (
    LOAD_SKILL_DEFINITION,
    SkillLoader,
    get_default_loader,
    register_skill_tools,
)
from power_loop.runtime.spec import AgentSpec, AgentSpecError, run_agent_spec
from power_loop.runtime.structured import (
    StructuredOutputError,
    StructuredOutputSpec,
    parse_structured,
)
from power_loop.runtime.timers import TimerRunner
from power_loop.tools import (
    DEFAULT_TOOL_HANDLERS,
    ToolRegistry,
    build_registry,
    create_default_tool_registry,
)
from power_loop.tools.blackboard import register_blackboard_tools
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

_logging.getLogger("power_loop").addHandler(_logging.NullHandler())


# The authoritative STABLE tier (single source of truth — tests assert the module
# docstring and __all__ agree, and a SemVer guard fails if a symbol is dropped/renamed
# without a major bump). Add freely; never remove/rename within a major version.
STABLE_API = (
    "StatefulAgentLoop",
    "StatefulResult",
    "FollowUpQueued",
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
	# LLM contract (PROVISIONAL)
	"LLMService",
	"LLMRequest",
	"LLMResponse",
	"LLMStreamChunk",
	"LLMTokenUsage",
	"OpenAICompatibleChatConfig",
	"AnthropicChatConfig",
	"SessionStore",
	"SessionRow",
	"SessionStatsRow",
	"TimerRow",
	"TimerRunner",
	"TimerFireCtx",
	"TimerFiredPayload",
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
	"RuntimeMessage",
	"RuntimeProjector",
	"ToolRuntimeContext",
	"TodoRuntimeProjector",
	"BackgroundRuntimeProjector",
	"default_runtime_projectors",
	"get_tool_runtime_context",
	"CancellationToken",
	"CancellationLike",
	"RuntimeEnv",
	"RuntimeEnvError",
	"runtime_env_context",
	"ShellBackend",
	"LocalShellBackend",
	"DEFAULT_SHELL_BACKEND",
	"Blackboard",
	"BlackboardEntry",
	"BlackboardError",
	"SqliteBlackboard",
	"render_entries",
	"register_blackboard_tools",
	"HumanInputRequired",
	"request_user_input",
	"LlmCallStartedPayload",
	"LlmCallCompletedPayload",
	"LlmRetryAttemptedPayload",
	"LlmDegradedPayload",
	"LoopCancelledPayload",
	"MemoryProvider",
	"MemorySnapshot",
	"tag_as_memory",
	"NotesPolicy",
	"NotesFullError",
	"SQLiteNoteMemory",
	"DEFAULT_NOTES_POLICY",
	"render_notes",
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
	"BudgetExceededStatusPayload",
	"HitRoundLimitStatusPayload",
	"PhaseEventPayload",
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
	"DEFAULT_TOOL_HANDLERS",
	"FollowUpQueued",
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
	"format_tool_catalog",
	"register_spawn_agent",
	"SPAWN_AGENT_DEFINITION",
	"RUN_AGENT_DEFINITION",
	"LOAD_SKILL_DEFINITION",
	"SkillLoader",
	"get_default_loader",
	"register_skill_tools",
	"AgentSpec",
	"AgentSpecError",
	"run_agent_spec",
]
