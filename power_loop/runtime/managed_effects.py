"""Finite managed effects with durable preparation and query-only recovery."""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeRequestError,
    TargetRef,
    canonical,
    plain,
    sha256_canonical,
)

from .tool_contracts import RuntimeTool, ToolDispatchPolicy, ToolRuntimeContract, resolve

FIELD = "managed_effect"
_LIMIT = 65536


def _document(value, limit=16384):
    def invalid_constant(_value):
        raise ValueError("non-finite JSON constant")

    try:
        if not isinstance(value, Mapping) or len(canonical(value).encode()) > limit:
            raise ValueError()
        return json.loads(canonical(value), parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        raise RuntimeRequestError("invalid_options", "managed_effect_document_invalid") from None


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ManagedToolResult:
    display: str
    reference: dict
    resolution: str = "committed"

    def __post_init__(self):
        if type(self.display) is not str or not self.display or "\x00" in self.display or len(self.display) > 4096:
            raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
        if self.resolution not in ("committed", "rejected", "not_dispatched"):
            raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
        object.__setattr__(self, "reference", _document(self.reference))

    def to_dict(self):
        return {"display": self.display, "reference": _document(self.reference), "resolution": self.resolution}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"display", "reference", "resolution"}:
            raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
        return cls(**value)


@dataclass(frozen=True)
class ManagedToolPreparation:
    reference: dict
    local_result: ManagedToolResult | None = None

    def __post_init__(self):
        object.__setattr__(self, "reference", _document(self.reference))
        if self.local_result is not None and type(self.local_result) is not ManagedToolResult:
            raise RuntimeRequestError("invalid_options", "managed_effect_preparation_invalid")

    def to_dict(self):
        return {
            "reference": _document(self.reference),
            "local_result": self.local_result.to_dict() if self.local_result is not None else None,
        }

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"reference", "local_result"}:
            raise RuntimeRequestError("invalid_options", "managed_effect_preparation_invalid")
        return cls(
            value["reference"],
            ManagedToolResult.from_dict(value["local_result"]) if value["local_result"] is not None else None,
        )


@dataclass(frozen=True)
class ManagedToolContract:
    key: str
    implementation_hash: str
    max_result_chars: int = 4096
    resource_model: str = "finite_effect"
    paid_requests: int = 0

    def __post_init__(self):
        if (
            type(self.key) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.key) is None
            or not _digest(self.implementation_hash)
            or type(self.max_result_chars) is not int
            or not 1 <= self.max_result_chars <= 4096
            or self.resource_model != "finite_effect"
            or type(self.paid_requests) is not int
            or self.paid_requests != 0
        ):
            raise RuntimeRequestError("invalid_options", "managed_effect_adapter_invalid")

    def to_dict(self):
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if (
            type(value) is not dict
            or set(value) != {"version", *cls.__dataclass_fields__}
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise RuntimeRequestError("invalid_options", "managed_effect_adapter_invalid")
        return cls(**{k: value[k] for k in cls.__dataclass_fields__})


class ManagedTool(ABC):
    @property
    @abstractmethod
    def contract(self) -> ManagedToolContract:
        raise NotImplementedError()

    @abstractmethod
    async def prepare(self, context, arguments) -> ManagedToolPreparation:
        raise NotImplementedError()

    @abstractmethod
    async def execute(self, context, original_reference) -> ManagedToolResult:
        raise NotImplementedError()

    @abstractmethod
    async def recover(self, context, original_reference) -> ManagedToolResult | None:
        raise NotImplementedError()

    def bind(self, runtime, definition):
        return bind(runtime, definition, self)


@dataclass(frozen=True)
class FunctionManagedTool(ManagedTool):
    descriptor: ManagedToolContract
    on_prepare: object = field(repr=False)
    on_execute: object = field(repr=False)
    on_recover: object = field(repr=False)

    def __post_init__(self):
        if type(self.descriptor) is not ManagedToolContract or any(
            not callable(f) for f in (self.on_prepare, self.on_execute, self.on_recover)
        ):
            raise RuntimeRequestError("invalid_options", "managed_effect_adapter_invalid")

    @property
    def contract(self):
        return self.descriptor

    async def prepare(self, context, arguments):
        return await self.on_prepare(context, arguments)

    async def execute(self, context, original_reference):
        return await self.on_execute(context, original_reference)

    async def recover(self, context, original_reference):
        return await self.on_recover(context, original_reference)


@dataclass(frozen=True)
class ManagedToolContext:
    runtime: object = field(repr=False)
    source_actor: object
    call_ref: object
    tool_call_id: str
    session_id: str
    send_index: int | None


def bind(runtime, definition, adapter):
    from power_loop.contracts.tools import ToolDefinition

    if (
        not isinstance(adapter, ManagedTool)
        or type(adapter.contract) is not ManagedToolContract
        or type(definition) is not ToolDefinition
    ):
        raise RuntimeRequestError("invalid_options", "managed_effect_adapter_invalid")
    if definition.async_capable or definition.interrupt != "finish":
        raise RuntimeRequestError("invalid_options", "managed_effect_tool_policy_invalid")
    return _Handler(
        definition,
        adapter,
        sha256_canonical(definition.to_openai_tool()),
        canonical(adapter.contract.to_dict()),
    )


@dataclass(frozen=True)
class _Handler(RuntimeTool):
    definition: object = field(repr=False)
    adapter: ManagedTool = field(repr=False)
    schema_hash: str
    contract_json: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def runtime_contract(self):
        return ToolRuntimeContract("managed-effect:" + self.adapter.contract.key, background_allowed=False)

    def dispatch_policy(self, arguments, *, runtime=None, definition=None):
        self.check(definition or self.definition)
        return ToolDispatchPolicy(native_admitted=True)

    def format_result(self, result, max_chars):
        prefix = "Error: "
        if type(result) is str and result.startswith(prefix):
            code = result[len(prefix):]
            if (0 < len(code) <= 128 and code[0].isascii() and code[0].islower()
                    and all(c.isascii() and (c.islower() or c.isdigit() or c == "_") for c in code)):
                return result
        return None

    def pending_call_spec(self, arguments, *, runtime=None, definition=None):
        self.check(definition or self.definition)
        return {"kind": FIELD, "request": {FIELD: {
            "version": 1,
            "adapter_key": self.adapter.contract.key,
            "contract_hash": self.adapter.contract.implementation_hash,
            "tool_name": self.definition.name,
            "schema_hash": self.schema_hash,
            "arguments_hash": sha256_canonical(_document(arguments, 131072)),
            "contract": self.adapter.contract.to_dict(),
        }}}

    def check(self, definition):
        if (
            definition.name != self.definition.name
            or sha256_canonical(definition.to_openai_tool()) != self.schema_hash
            or type(self.adapter.contract) is not ManagedToolContract
            or canonical(self.adapter.contract.to_dict()) != self.contract_json
        ):
            raise RuntimeRequestError("invalid_options", "managed_effect_binding_changed")

    async def __call__(self, **arguments):
        from power_loop.core.agent_context import get_current_loop

        from .child_tasks import tool_actor
        from .task_runtime import current_session_send

        loop = get_current_loop()
        binding = current_session_send(loop.store, loop._runtime_worker_context.session_id)
        if binding is None:
            raise RuntimeRequestError("invalid_options", "managed_effect_original_send_required")
        capacity = getattr(getattr(loop, "config", None), "tool_result_max_chars", None)
        if capacity is not None and capacity < self.adapter.contract.max_result_chars:
            raise RuntimeRequestError("invalid_options", "managed_effect_result_capacity_unavailable")
        arguments = _document(arguments, limit=131072)
        async with self.lock:
            _, actor = await tool_actor(loop)
            descriptor = _managed_descriptor(loop, self, actor, arguments)
            parent = TargetRef(loop.store.store_id, actor.session_id)
            target, start, receipt = await loop._runtime_worker.inputs.create_child(
                descriptor,
                parent=parent,
                actor=actor,
                call_ref=actor.call_ref,
                tool_call_id=actor.tool_call_id,
                body={FIELD: descriptor.config[FIELD]},
                result_kind="managed_effect_result",
            )
            await loop._runtime_worker.runtime.client._notify(
                target, start.message_id, start, ["work", "activation"],
            )
            return canonical({
                "target_ref": plain(target),
                "input_ref": plain(start),
                "status": "accepted",
                "newly_accepted": receipt.newly_accepted,
                "managed_effect_pending": True,
            })


def _managed_descriptor(loop, handler, actor, arguments):
    from dataclasses import replace

    parent = loop._runtime_worker_context.descriptor
    frozen = {
        "tool": handler.definition.name,
        "arguments": arguments,
        "contract": handler.adapter.contract.to_dict(),
        "call_ref": actor.call_ref.to_dict(),
        "call_id": actor.tool_call_id,
    }
    return replace(
        parent,
        kind="background",
        config={**plain(parent.config), FIELD: frozen},
    )


def _state_key(frozen):
    return "managed_effect:" + sha256_canonical([frozen["call_ref"], frozen["call_id"]])[:80]


def _context(store, actor, frozen, send_index):
    from power_loop.runtime.message_ref import MessageRef
    raw = frozen["call_ref"]
    return ManagedToolContext(
        store,
        actor,
        MessageRef(
            raw["store_id"], raw["session_id"],
            int(raw["id"]) if raw.get("id") is not None else None,
            int(raw["seq"]), raw["role"], raw.get("tool_call_id"),
            int(raw["send_index"]) if raw.get("send_index") is not None else None,
            raw.get("round_index"),
        ),
        frozen["call_id"],
        raw["session_id"],
        send_index,
    )


async def _write_state(store, session_id, key, value):
    async with store._db.transaction() as tx:
        sql = store._db.dialect.upsert(
            store.t.session_runtime_state,
            ("session_id", "state_key"),
            ("value_json", "updated_at"),
        )
        await tx.execute(sql, (session_id, key, canonical(value), now_ms()))


async def run_session(worker, ctx, env, config):
    from .background_execution import SessionToolResult

    frozen = ctx.descriptor.config[FIELD]
    registered = env.tool_registry.get(frozen["tool"]) if env.tool_registry is not None else None
    handler = resolve(registered)
    if not isinstance(handler, _Handler) or canonical(
        handler.adapter.contract.to_dict()
    ) != canonical(frozen["contract"]):
        return SessionToolResult(
            "failed", error="definition_unavailable: managed adapter changed",
            error_code="definition_unavailable",
        )
    adapter = handler.adapter
    key = _state_key(frozen)
    existing = await worker.runtime.store.get_runtime_state(ctx.session_id, key)
    if existing and existing.get("result") is not None:
        result = ManagedToolResult.from_dict(existing["result"])
        return SessionToolResult(
            "completed", result.display, usage={}, structured_output=result.to_dict(),
        )
    actor = ctx.actor
    context = _context(worker.runtime.store, actor, frozen, ctx.send_index)
    try:
        prepared = await adapter.prepare(context, _document(frozen["arguments"], 131072))
        if type(prepared) is not ManagedToolPreparation:
            raise RuntimeRequestError("invalid_options", "managed_effect_preparation_invalid")
        await _write_state(worker.runtime.store, ctx.session_id, key, {
            "phase": "dispatched",
            "preparation": prepared.to_dict(),
            "contract": frozen["contract"],
            "result": None,
        })
    except Exception as exc:
        result = ManagedToolResult(
            "The managed effect was not dispatched because preparation did not complete.",
            {"decision": "not_dispatched", "error": str(exc)[:500]},
            "not_dispatched",
        )
        await _write_state(worker.runtime.store, ctx.session_id, key, {
            "phase": "completed", "preparation": None, "contract": frozen["contract"],
            "result": result.to_dict(),
        })
        return SessionToolResult(
            "completed", result.display, usage={}, structured_output=result.to_dict(),
        )
    if prepared.local_result is not None:
        result = prepared.local_result
        await _write_state(worker.runtime.store, ctx.session_id, key, {
            "phase": "completed", "preparation": prepared.to_dict(),
            "contract": frozen["contract"], "result": result.to_dict(),
        })
        return SessionToolResult(
            "completed", result.display, usage={}, structured_output=result.to_dict(),
        )
    try:
        result = await adapter.execute(context, _document(prepared.reference))
        if type(result) is not ManagedToolResult or len(result.display) > adapter.contract.max_result_chars:
            raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
    except Exception as exc:
        await _write_state(worker.runtime.store, ctx.session_id, key, {
            "phase": "unknown", "preparation": prepared.to_dict(),
            "contract": frozen["contract"], "result": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        })
        return SessionToolResult(
            "interrupted",
            "Managed effect outcome is unknown; query the original adapter to recover it.",
            error="managed effect outcome is unknown",
            error_code="interrupted",
            structured_output={"managed_effect_pending": True},
        )
    await _write_state(worker.runtime.store, ctx.session_id, key, {
        "phase": "completed", "preparation": prepared.to_dict(),
        "contract": frozen["contract"], "result": result.to_dict(),
    })
    return SessionToolResult(
        "completed", result.display, usage={}, structured_output=result.to_dict(),
    )


async def reconcile(store, actor, start: InputRef, adapter: ManagedTool, *, client=None):
    """Query the original adapter and route a late result as a new ordinary send."""
    port = RuntimeInputStore(store)
    target = TargetRef(start.store_id, start.session_id)
    session = await port.authorize(port.db, target, actor)
    frozen = json.loads(session["descriptor_json"])["config"].get(FIELD)
    if frozen is None or canonical(frozen["contract"]) != canonical(adapter.contract.to_dict()):
        raise RuntimeRequestError("forbidden", "managed adapter or original call mismatch")
    key = _state_key(frozen)
    state = await store.get_runtime_state(target.session_id, key)
    if state is None:
        raise RuntimeRequestError("not_found", "managed preparation")
    if state.get("result") is not None:
        return ManagedToolResult.from_dict(state["result"])
    if state.get("phase") != "unknown":
        return None
    original = await port.get_result(start, actor=actor)
    if original is None or original.status != "interrupted":
        return None
    context = _context(store, actor, frozen, original.send_index)
    result = await adapter.recover(context, _document(state["preparation"]["reference"]))
    if result is None:
        return None
    if type(result) is not ManagedToolResult:
        raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
    message_id = "inp_" + sha256_canonical([
        "managed_recover", target.store_id, target.session_id, start.message_id,
        original.send_index,
    ])[:56]
    receipt = await port.accept(
        target,
        {FIELD: "recovered", "result": result.to_dict()},
        message_id=message_id,
        actor=actor,
    )
    batch = await port.claim(target, node_id="managed-recovery", boot_id="managed-recovery")
    final_batch = None
    if batch is not None:
        final_batch = await port.final(
            batch,
            final_text=result.display,
            structured_output=result.to_dict(),
        )
    if client is not None and final_batch is not None and final_batch.result_activation is not None:
        parent = final_batch.result_activation
        result_ref = final_batch.result_ref
        await client._notify(parent, result_ref.message_id, result_ref, ["work", "activation"])
    recovered = await port.get_result(receipt.input_ref, actor=actor)
    if recovered is None or recovered.status != "completed":
        return None
    await _write_state(store, target.session_id, key, {
        **state,
        "phase": "completed",
        "result": result.to_dict(),
        "recovered_send_index": recovered.send_index,
    })
    return result


__all__ = [
    "FunctionManagedTool",
    "ManagedTool",
    "ManagedToolContract",
    "ManagedToolContext",
    "ManagedToolPreparation",
    "ManagedToolResult",
    "bind",
    "reconcile",
]
