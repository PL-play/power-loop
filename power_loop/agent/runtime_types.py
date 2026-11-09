"""Serializable identities and contracts for the message runtime (design 203)."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any

from power_loop.runtime.message_ref import MessageRef


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
    """The submit transaction failed; its commit outcome may be unknown."""

    def __init__(self, input_ref):
        self.input_ref = input_ref
        super().__init__("input commit not confirmed; inspect or retry the original InputRef")


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def canonical(value) -> str:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def freeze(value):
    """Validate at the public boundary and detach all nested caller-owned data."""
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
    if not isinstance(value, str) or not 0 < len(value) <= limit:
        raise RuntimeRequestError("invalid_options", "invalid identifier")


@dataclass(frozen=True)
class TargetRef:
    store_id: str
    session_id: str
    job_id: str | None = None

    def __post_init__(self):
        identifier(self.store_id, 255)
        identifier(self.session_id, 255)
        if self.job_id is not None:
            identifier(self.job_id)


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
class JobRef:
    store_id: str
    job_id: str
    generation: int | None = None

    def __post_init__(self):
        identifier(self.store_id, 255)
        identifier(self.job_id)
        if self.generation is not None and (type(self.generation) is not int or self.generation < 0):
            raise RuntimeRequestError("stale_generation")


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
    schema_version: int
    kind: str
    definition_ref: str
    model_profile_ref: str | None
    tool_profile_ref: str
    hook_profile_ref: str
    resource_profile_ref: str
    config: dict
    host_context: dict

    def __post_init__(self):
        if self.schema_version != 1 or self.kind not in {
            "agent", "workflow_driver", "workflow_node", "background_tool", "background_command"
        }:
            raise RuntimeRequestError("definition_unavailable")
        for name in (self.definition_ref, self.tool_profile_ref, self.hook_profile_ref,
                     self.resource_profile_ref):
            identifier(name, 255)
        if self.model_profile_ref is not None:
            identifier(self.model_profile_ref, 255)
        object.__setattr__(self, "config", freeze(self.config))
        object.__setattr__(self, "host_context", freeze(self.host_context))


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

    def __post_init__(self):
        for name in ("tools", "response_format", "metadata"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        if self.response_format is not None:
            if not isinstance(self.response_format, Mapping):
                raise RuntimeRequestError("invalid_options", "response_format must be an object")
            if self.response_format.get("type") == "json_schema":
                schema = self.response_format.get("json_schema")
                if not isinstance(schema, Mapping) or not isinstance(schema.get("schema"), Mapping):
                    raise RuntimeRequestError("invalid_options", "json_schema.schema must be an object")
        if type(self.history) is not bool:
            raise RuntimeRequestError("invalid_options", "history must be boolean")
        for name in ("max_tokens", "max_rounds", "max_tokens_per_run"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise RuntimeRequestError("invalid_options", name)
        if self.temperature is not None and (type(self.temperature) not in (int, float) or self.temperature < 0):
            raise RuntimeRequestError("invalid_options", "temperature")
        freeze(self.temperature)
        for name in ("model", "system_prompt", "representation_ref"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise RuntimeRequestError("invalid_options", name)
        if self.tools is not None and (not isinstance(self.tools, tuple) or any(not isinstance(t, str) or not t for t in self.tools)):
            raise RuntimeRequestError("invalid_options", "tools")


@dataclass(frozen=True)
class Command:
    message_id: str
    target: TargetRef
    operation: str
    payload: dict
    urgent: bool = False
    descriptor_ref: str | None = None
    options: dict | None = None
    expires_at_ms: int | None = None
    expected_job_generation: int | None = None
    scope: str = "self"

    def __post_init__(self):
        identifier(self.message_id)
        if (self.operation not in {"send", "resume", "cancel"} or self.scope not in {"self", "subtree"}
                or type(self.urgent) is not bool or self.urgent and self.operation != "send"):
            raise RuntimeRequestError("invalid_options", "operation/urgent/scope")
        object.__setattr__(self, "payload", freeze(self.payload))
        object.__setattr__(self, "options", freeze(self.options or {}))
        for name in ("expires_at_ms", "expected_job_generation"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise RuntimeRequestError("invalid_options", name)


@dataclass(frozen=True)
class Receipt:
    input_ref: InputRef | None
    target: TargetRef
    message_id: str
    newly_accepted: bool
    job_ref: JobRef | None
    state: str
    notification: str
    reason: str | None = None


@dataclass(frozen=True)
class Status:
    input_ref: InputRef | None
    job_ref: JobRef | None
    state: str
    reason: str | None
    result_available: bool
    cancel_requested: bool
    resource_state: str


@dataclass(frozen=True)
class Result:
    job_ref: JobRef | None
    input_ref: InputRef | None
    status: str
    final_text: str = ""
    usage: dict | None = None
    structured_output: Any = None
    error: str | None = None
    error_code: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "usage", freeze(self.usage or {}))
        object.__setattr__(self, "structured_output", freeze(self.structured_output))


@dataclass(frozen=True)
class JobPage:
    items: tuple[Status, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class ShutdownResult:
    drained: bool
    interrupted_jobs: tuple[JobRef, ...] = ()


@dataclass(frozen=True)
class OutputItem:
    input_ref: InputRef
    job_ref: JobRef | None
    sequence: int
    kind: str
    payload: dict
