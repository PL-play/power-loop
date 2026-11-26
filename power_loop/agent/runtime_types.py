"""Session-native identities and public storage contracts (design 215)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

from power_loop.runtime.message_ref import MessageRef

PROTOCOL_VERSION = 43
Priority = Literal["normal", "urgent"]
SessionKind = Literal["agent", "workflow_driver", "background"]
ExecutionState = Literal["idle", "queued", "running"]
InputState = Literal["pending", "delivered", "void", "waiting_continuation"]
TerminalStatus = Literal["completed", "failed", "cancelled", "interrupted"]
ResultKind = Literal[
    "agent_result", "workflow_result", "workflow_node_result",
    "background_result", "managed_effect_result",
]
CancelPropagation = Literal["linked", "independent"]
Lifecycle = Literal["open", "suspended", "closing", "closed"]
RESERVED_INPUT_PREFIXES = ("res_", "wfc_", "sys_")


class RuntimeRequestError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(code + (": " + detail if detail else ""))


class CapacityError(RuntimeRequestError):
    def __init__(self):
        super().__init__("capacity_exceeded")


class WaitTimeout(TimeoutError):
    pass


class StoreError(RuntimeError):
    """A commit outcome is unknown; retry or inspect the original InputRef."""

    def __init__(self, input_ref):
        self.code = "store_commit_unknown"
        self.input_ref = input_ref
        super().__init__("store_commit_unknown: inspect or retry the original InputRef")


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def canonical(value) -> str:
    return json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def sha256_canonical(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def freeze(value):
    """Validate strict JSON and detach nested caller-owned data."""

    def immutable(v):
        if isinstance(v, dict):
            return MappingProxyType({k: immutable(x) for k, x in v.items()})
        if isinstance(v, list):
            return tuple(immutable(x) for x in v)
        return v

    try:
        return immutable(json.loads(canonical(value)))
    except (TypeError, ValueError) as exc:
        raise RuntimeRequestError("invalid_options", "strict JSON required") from exc


def identifier(value, limit=64):
    if (
        not isinstance(value, str) or not 0 < len(value) <= limit
        or value != value.strip() or not value.isprintable()
    ):
        raise RuntimeRequestError("invalid_options", "invalid identifier")


@dataclass(frozen=True)
class TargetRef:
    store_id: str
    session_id: str

    def __post_init__(self):
        identifier(self.store_id, 255)
        identifier(self.session_id, 255)


@dataclass(frozen=True)
class InputRef:
    store_id: str
    session_id: str
    message_id: str

    def __post_init__(self):
        identifier(self.store_id, 255)
        identifier(self.session_id, 255)
        identifier(self.message_id)


@dataclass(frozen=True)
class TrustedActor:
    tenant_id: str
    principal_id: str
    session_id: str | None
    call_ref: MessageRef | None = None
    tool_call_id: str | None = None

    def __post_init__(self):
        identifier(self.tenant_id, 255)
        identifier(self.principal_id, 255)
        if self.session_id is not None:
            identifier(self.session_id, 255)


@dataclass(frozen=True)
class RuntimeDescriptor:
    schema_version: Literal[2]
    kind: SessionKind
    definition_ref: str
    model_profile_ref: str | None
    tool_profile_ref: str
    hook_profile_ref: str
    resource_profile_ref: str
    config: dict
    host_context: dict

    def __post_init__(self):
        if self.schema_version != 2 or self.kind not in {
            "agent", "workflow_driver", "background"
        }:
            raise RuntimeRequestError("definition_unavailable")
        for name in (
            self.definition_ref, self.tool_profile_ref,
            self.hook_profile_ref, self.resource_profile_ref,
        ):
            identifier(name, 255)
            if not name.endswith("@215"):
                raise RuntimeRequestError("definition_unavailable", name)
        if self.model_profile_ref is not None:
            identifier(self.model_profile_ref, 255)
            if not self.model_profile_ref.endswith("@215"):
                raise RuntimeRequestError("definition_unavailable", self.model_profile_ref)
        object.__setattr__(self, "config", freeze(self.config))
        object.__setattr__(self, "host_context", freeze(self.host_context))


@dataclass(frozen=True)
class Receipt:
    input_ref: InputRef | None
    target: TargetRef
    message_id: str | None
    newly_accepted: bool
    state: str
    send_index: int | None
    notification: str
    reason: str | None = None


@dataclass(frozen=True)
class Result:
    input_ref: InputRef
    target: TargetRef
    status: TerminalStatus
    send_index: int | None
    result_seq: int | None
    final_text: str = ""
    usage: dict = None
    structured_output: Any = None
    error: str | None = None
    error_code: str | None = None
    result_delivery_error: str | None = None
    cancel_by: str | None = None
    rounds: int = 0

    def __post_init__(self):
        object.__setattr__(self, "usage", freeze(self.usage or {}))
        object.__setattr__(self, "structured_output", freeze(self.structured_output))


@dataclass(frozen=True)
class Status:
    input_ref: InputRef | None
    target: TargetRef
    execution_state: ExecutionState
    input_state: InputState | None
    terminal_status: TerminalStatus | None
    send_index: int | None
    result_available: bool
    result_seq: int | None
    cancel_requested: bool
    reason: str | None
    error_code: str | None
    result_delivery_error: str | None
    continuation_of: str | None
    continuation_index: int | None
    lifecycle: Lifecycle = "open"
    need_wake: bool = False
    stopped_descendant_ids: tuple[str, ...] = ()
    voided_pending_inputs: int = 0


@dataclass(frozen=True)
class MessageRow:
    id: int
    session_id: str
    seq: int
    role: str
    name: str | None
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    round_index: int | None
    state: str
    meta: dict[str, Any]
    send_index: int | None
    created_at: int


@dataclass(frozen=True)
class MessagePage:
    items: tuple[MessageRow, ...]
    next_after_seq: int | None


@dataclass(frozen=True)
class SessionPage:
    items: tuple[TargetRef, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class RequestOptions:
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_rounds: int | None = None
    max_tokens_per_run: int | None = None
    tools: tuple[str, ...] | None = None
    system_prompt: str | None = None
    response_format: dict | None = None
    history: bool = True
    representation_ref: str | None = None
    metadata: dict | None = None


def result_message_id(
    child_target: TargetRef,
    start_input: InputRef,
    send_index: int | None,
    result_kind: ResultKind,
) -> str:
    if (child_target.store_id, child_target.session_id) != (
        start_input.store_id, start_input.session_id
    ):
        raise RuntimeRequestError("invalid_options", "result target/start mismatch")
    if result_kind not in {
        "agent_result", "workflow_result", "workflow_node_result",
        "background_result", "managed_effect_result",
    }:
        raise RuntimeRequestError("invalid_options", "result_kind")
    execution_key = (
        f"send:{send_index}" if send_index is not None
        else f"input:{start_input.message_id}"
    )
    return "res_" + sha256_canonical([
        child_target.store_id, child_target.session_id, execution_key, result_kind,
    ])[:56]








@dataclass(frozen=True)
class ShutdownResult:
    drained: bool
    interrupted_inputs: tuple[InputRef, ...] = ()


@dataclass(frozen=True)
class OutputItem:
    input_ref: InputRef
    sequence: int
    kind: str
    payload: dict


@dataclass(frozen=True)
class ContextBudget:
    """Context window budget & current inline estimate for a target session or descriptor."""

    model: str | None
    context_budget_tokens: int
    max_tokens: int | None
    max_context_rows: int | None
    used_tokens: int
    remaining_tokens: int
    sends: int = 0
    compact_to_send: int | None = None


@dataclass(frozen=True)
class RepairResult:
    """Result of an explicit ``RuntimeClient.repair`` / ``RuntimeInputStore.repair`` call."""

    target: TargetRef
    mode: Literal["heal_broken", "fold_tools_to_text", "compact"]
    repaired_messages: int
    repaired_projections: int
    send_index: int
    next_seq: int
    summary: str | None = None


@dataclass(frozen=True)
class RebindResult:
    """Result of an atomic ``RuntimeClient.rebind_children`` / ``RuntimeInputStore.rebind_children`` call."""

    from_parent: TargetRef
    to_parent: TargetRef
    rebound_children: tuple[str, ...]
    updated_descendants: tuple[str, ...]
    replayed_results: int
    new_root_session_id: str
    depth_delta: int
    need_wake: bool = False
