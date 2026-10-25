"""Trusted finite managed effects as required SDK background-tool children.

Callbacks are trusted host code for one finite managed effect, not arbitrary paid
tools, subprocesses or detached work. No import path is taken from saved/model
data. External outcome resolution, local coroutine exit and fee settlement differ.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from .invocation import current_invocation
from .jobs import JobError, JobLease, _hash, _json
from .tool_contracts import RuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

FIELD = "managed_effect"
_LIMIT = 65536


def _document(value, limit=16384):
    def invalid_constant(value):
        raise ValueError("non-finite JSON constant")

    try:
        if type(value) is not dict or len(_json(value).encode()) > limit:
            raise ValueError()
        return json.loads(_json(value), parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        raise JobError("managed_effect_document_invalid") from None


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ManagedToolResult:
    display: str
    reference: dict
    resolution: str = "committed"

    def __post_init__(self):
        if type(self.display) is not str or not self.display or "\x00" in self.display or len(self.display) > 4096:
            raise JobError("managed_effect_result_invalid")
        if self.resolution not in ("committed", "rejected", "not_dispatched"):
            raise JobError("managed_effect_result_invalid")
        object.__setattr__(self, "reference", _document(self.reference))

    def to_dict(self):
        return {"display": self.display, "reference": _document(self.reference), "resolution": self.resolution}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"display", "reference", "resolution"}:
            raise JobError("managed_effect_result_invalid")
        return cls(**value)


@dataclass(frozen=True)
class ManagedToolPreparation:
    reference: dict
    local_result: ManagedToolResult | None = None

    def __post_init__(self):
        object.__setattr__(self, "reference", _document(self.reference))
        if self.local_result is not None and type(self.local_result) is not ManagedToolResult:
            raise JobError("managed_effect_preparation_invalid")

    def to_dict(self):
        return {
            "reference": _document(self.reference),
            "local_result": self.local_result.to_dict() if self.local_result is not None else None,
        }

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"reference", "local_result"}:
            raise JobError("managed_effect_preparation_invalid")
        return cls(
            value["reference"],
            ManagedToolResult.from_dict(value["local_result"]) if value["local_result"] is not None else None,
        )


@dataclass(frozen=True)
class ManagedToolContract:
    """Host-declared implementation identity and SDK resource capabilities.

    This first resource model is finite and creates no untracked paid request
    or detached work. Other resource models need their own SDK lifecycle and
    admission implementation; a declaration alone cannot opt into them.
    """

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
            raise JobError("managed_effect_adapter_invalid")

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
            raise JobError("managed_effect_adapter_invalid")
        return cls(**{k: value[k] for k in cls.__dataclass_fields__})


class ManagedTool(ABC):
    """Platform-neutral implementation contract for a managed tool.

    Implementations inherit this base and provide four things: a frozen
    contract, preparation, one original execution, and query-only recovery.
    Register ``tool.bind(runtime, definition)`` as the handler; the SDK then
    owns admission, task tracking, cancellation and receipt checks. Neither
    the implementation's class name nor a tool name grants permission.
    """

    @property
    @abstractmethod
    def contract(self) -> ManagedToolContract:
        raise NotImplementedError()

    @abstractmethod
    async def prepare(self, context, arguments) -> ManagedToolPreparation:
        """Freeze local metadata/references; do not dispatch the effect here."""
        raise NotImplementedError()

    @abstractmethod
    async def execute(self, context, original_reference) -> ManagedToolResult:
        """Execute the original finite action once, with cooperative cleanup."""
        raise NotImplementedError()

    @abstractmethod
    async def recover(self, context, original_reference) -> ManagedToolResult | None:
        """Query the original outcome; never prepare or execute again."""
        raise NotImplementedError()

    def bind(self, runtime, definition):
        return bind(runtime, definition, self)


@dataclass(frozen=True)
class FunctionManagedTool(ManagedTool):
    """Optional function-based implementation of the same public base."""

    descriptor: ManagedToolContract
    on_prepare: object = field(repr=False)
    on_execute: object = field(repr=False)
    on_recover: object = field(repr=False)

    def __post_init__(self):
        if type(self.descriptor) is not ManagedToolContract or any(
            not callable(f) for f in (self.on_prepare, self.on_execute, self.on_recover)
        ):
            raise JobError("managed_effect_adapter_invalid")

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
    source_lease: JobLease
    call_ref: object
    tool_call_id: str
    job_id: str


def _authority(runtime):
    from .root_context import _authority

    return _authority(runtime)


def bind(runtime, definition, adapter):
    from power_loop.contracts.tools import ToolDefinition

    from .task_runtime import _OWNER, _SEND

    if _SEND.get() is not None or _OWNER.get() is not None:
        raise JobError("managed_effect_binding_requires_host")
    if (
        not isinstance(adapter, ManagedTool)
        or type(adapter.contract) is not ManagedToolContract
        or type(definition) is not ToolDefinition
    ):
        raise JobError("managed_effect_adapter_invalid")
    if definition.async_capable or definition.interrupt != "finish":
        raise JobError("managed_effect_tool_policy_invalid")
    return _Handler(
        runtime,
        definition,
        adapter,
        _authority(runtime),
        _hash(definition.to_openai_tool()),
        _json(adapter.contract.to_dict()),
    )


@dataclass(frozen=True)
class _Handler(RuntimeTool):
    runtime: object = field(repr=False)
    definition: object = field(repr=False)
    adapter: ManagedTool = field(repr=False)
    authority: tuple = field(repr=False)
    schema_hash: str
    contract_json: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def runtime_contract(self):
        return ToolRuntimeContract("managed-effect:" + self.adapter.contract.key, background_allowed=False)

    def dispatch_policy(self, arguments, *, runtime=None):
        self.check(runtime, self.definition.name, self.definition)
        return ToolDispatchPolicy(native_admitted=True)

    def check(self, runtime, tool_name, definition):
        current = _authority(runtime)
        if (
            runtime is not self.runtime
            or any(a is not b for a, b in zip(current[:3], self.authority[:3], strict=True))
            or current[3:] != self.authority[3:]
            or tool_name != self.definition.name
            or _hash(definition.to_openai_tool()) != self.schema_hash
            or _hash(self.definition.to_openai_tool()) != self.schema_hash
            or type(self.adapter.contract) is not ManagedToolContract
            or _json(self.adapter.contract.to_dict()) != self.contract_json
        ):
            raise JobError("managed_effect_binding_changed")

    async def __call__(self, **arguments):
        from power_loop.core.agent_context import get_current_loop

        from .task_runtime import _SEND

        execution = _SEND.get()
        if execution is None:
            raise JobError("managed_effect_original_send_required")
        self.check(execution.runtime, self.definition.name, self.definition)
        loop = get_current_loop()
        capacity = getattr(getattr(loop, "config", None), "tool_result_max_chars", None)
        if capacity is not None and capacity < self.adapter.contract.max_result_chars:
            raise JobError("managed_effect_result_capacity_unavailable")
        arguments = _document(arguments, limit=131072)
        async with self.lock:
            return await _invoke(self, execution, arguments)


def registration(request, *, kind, parent, call_ref, tool_call_id, session_id, independent):
    if FIELD not in request:
        return None
    value = request[FIELD]
    keys = {"version", "adapter_key", "contract_hash", "tool_name", "schema_hash", "arguments_hash", "contract"}
    if (
        kind != "background_tool"
        or parent is None
        or call_ref is None
        or not tool_call_id
        or session_id is not None
        or independent
        or type(value) is not dict
        or set(value) != keys
        or type(value["version"]) is not int
        or value["version"] != 1
        or any(not _digest(value[k]) for k in ("contract_hash", "schema_hash", "arguments_hash"))
        or any(type(value[k]) is not str or not 0 < len(value[k]) <= 64 for k in ("adapter_key", "tool_name"))
    ):
        raise JobError("managed_effect_registration_invalid")
    contract = ManagedToolContract.from_dict(value["contract"])
    if (contract.key, contract.implementation_hash) != (value["adapter_key"], value["contract_hash"]):
        raise JobError("managed_effect_registration_invalid")
    return value


def job_id(manager, actor, call_ref, tool_call_id):
    return (
        "effect_"
        + _hash(
            [
                manager.store.store_id,
                actor.tenant_id,
                actor.principal_id,
                call_ref.session_id,
                call_ref.id,
                tool_call_id,
            ]
        )[:48]
    )


def _id(manager, kind, job, generation=1):
    return "mt_" + kind + "_" + _hash([manager.store.store_id, job, generation])[:48]


async def _read_event(manager, tx, event_id, *, required=True):
    length = (
        "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
    )
    size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?", (event_id,))
    if size is None:
        if required:
            raise JobError("managed_effect_evidence_missing")
        return None
    if size["n"] > _LIMIT:
        raise JobError("managed_effect_evidence_too_large")
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    try:
        value = _document(json.loads(row["payload_json"]), _LIMIT)
        if row["request_hash"] != _hash(value):
            raise ValueError()
        return row, value
    except (ValueError, TypeError):
        raise JobError("managed_effect_evidence_corrupt") from None


async def _event(manager, tx, row, kind, value):
    event_id = _id(manager, kind, row["job_id"], row["generation"])
    old = await _read_event(manager, tx, event_id, required=False)
    if old is not None:
        if (
            old[0]["job_id"] != row["job_id"]
            or old[0]["event_type"] != "managed_effect_" + kind
            or old[0]["delivery_status"] != "not_applicable"
            or _hash(old[1]) != _hash(value)
        ):
            raise JobError("managed_effect_evidence_conflict")
        return event_id
    payload = _json(_document(value, _LIMIT))
    await tx.execute(
        f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            event_id,
            row["job_id"],
            "managed_effect_" + kind,
            row["version"],
            _hash(value),
            payload,
            "not_applicable",
            manager.now(),
        ),
    )
    return event_id


async def validate_call(manager, tx, actor, lease, ref, call_id, name, arguments_hash):
    source = await manager._fence(tx, actor, lease)
    return await _verify_call(manager, tx, actor, source, ref, call_id, name, arguments_hash)


async def _verify_call(manager, tx, actor, source, ref, call_id, name, arguments_hash):
    seq = ref.seq
    if source["kind"] != "send" or source["session_id"] != actor.session_id:
        raise JobError("managed_effect_original_send_required")
    length = (
        "length(CAST(tool_calls_json AS BLOB))"
        if manager.db.dialect.name == "sqlite"
        else "octet_length(tool_calls_json)"
    )
    size = await tx.fetchone(
        f"SELECT {length} AS n FROM {manager.store.t.messages} WHERE session_id=? AND seq=?", (actor.session_id, seq)
    )
    if size is None or size["n"] is None or size["n"] > 131072:
        raise JobError("managed_effect_call_evidence_unavailable")
    await manager._validate_ref(tx, actor, ref, role="assistant", metadata_only=True)
    creates = await tx.fetchall(
        f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'",
        (source["job_id"],),
    )
    if len(creates) != 1:
        raise JobError("managed_effect_creation_missing")
    size = await tx.fetchone(
        f"SELECT {'length(CAST(payload_json AS BLOB))' if manager.db.dialect.name == 'sqlite' else 'octet_length(payload_json)'} AS n FROM {manager.events} WHERE event_id=?",
        (creates[0]["event_id"],),
    )
    if size["n"] > _LIMIT:
        raise JobError("managed_effect_evidence_too_large")
    original = await tx.fetchone(
        f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (creates[0]["event_id"],)
    )
    try:
        index = json.loads(original["payload_json"])["request"]["send_index"]
        if type(index) is not int or index != ref.send_index:
            raise ValueError()
    except (ValueError, TypeError, KeyError, RecursionError):
        raise JobError("managed_effect_call_evidence_mismatch") from None
    row = await tx.fetchone(
        f"SELECT tool_calls_json FROM {manager.store.t.messages} WHERE session_id=? AND seq=?", (actor.session_id, seq)
    )
    try:
        calls = json.loads(row["tool_calls_json"])
        matches = [c for c in calls if type(c) is dict and c.get("id") == call_id]
        if (
            len(matches) != 1
            or matches[0]["function"]["name"] != name
            or _hash(json.loads(matches[0]["function"]["arguments"])) != arguments_hash
        ):
            raise ValueError()
    except (ValueError, TypeError, KeyError, RecursionError):
        raise JobError("managed_effect_call_evidence_mismatch") from None
    return ref, call_id


def _ref(value):
    from .message_ref import MessageRef

    try:
        ref = MessageRef(
            value["store_id"],
            value["session_id"],
            int(value["id"]),
            int(value["seq"]),
            value["role"],
            value["tool_call_id"],
            int(value["send_index"]),
            value["round_index"],
        )
        if _json(ref.to_dict()) != _json(value):
            raise ValueError()
        return ref
    except (ValueError, TypeError, KeyError):
        raise JobError("managed_effect_evidence_corrupt") from None


async def _original(manager, tx, row, creation, binding):
    """Validate historical membership and raw coordinates without a live fence."""
    from . import execution_membership
    from .jobs import JobActor

    ref = _ref(json.loads(creation["call_ref_json"]))
    actor = JobActor(row["tenant_id"], row["principal_id"], ref.session_id)
    source = await manager._authorize(tx, actor, row["parent_job_id"])
    proof = JobLease(row["parent_job_id"], row["parent_executor_id"], row["parent_boot_id"], row["parent_generation"])
    if (
        row["created_by_session_id"] != ref.session_id
        or creation["caller_session_id"] != ref.session_id
        or source["root_job_id"] != row["root_job_id"]
        or row["job_id"] != job_id(manager, actor, ref, creation["tool_call_id"])
    ):
        raise JobError("managed_effect_evidence_corrupt")
    event_id = execution_membership.scope_id(manager, source["job_id"], proof.generation)
    length = (
        "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
    )
    size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?", (event_id,))
    if size is None or size["n"] is None or size["n"] > 32768:
        raise JobError("managed_effect_source_evidence_unavailable")
    scope = await execution_membership.scope(manager, tx, source["job_id"], proof.generation)
    if (
        scope["lease"] != asdict(proof)
        or scope["session_id"] != actor.session_id
        or (scope["tenant_id"], scope["principal_id"]) != (actor.tenant_id, actor.principal_id)
    ):
        raise JobError("managed_effect_source_evidence_mismatch")
    await _verify_call(
        manager, tx, actor, source, ref, creation["tool_call_id"], binding["tool_name"], binding["arguments_hash"]
    )
    return actor, proof, ref


def _proof(row):
    return {key: row[key] for key in ("job_id", "executor_id", "boot_id", "generation")}


async def _evidence(manager, tx, row, kind, *, required=True):
    found = await _read_event(manager, tx, _id(manager, kind, row["job_id"], row["generation"]), required=required)
    if found is None:
        return None
    event, value = found
    fields = {
        "prepared": {"adapter", "preparation"},
        "dispatch": {"preparation_hash"},
        "resource_exit": {"exit", "dispatch_started"},
        "observation": {"adapter", "result"},
        "tool_receipt": {"creation_id", "result_id", "result_hash", "receipt_ref", "display_hash"},
    }
    version = value.get("version")
    if kind == "resource_exit" and type(version) is int and version == 2:
        fields["resource_exit"] = fields["resource_exit"] | {"completion"}
    if (
        event["job_id"] != row["job_id"]
        or event["event_type"] != "managed_effect_" + kind
        or event["delivery_status"] != "not_applicable"
        or set(value) != {"version", "lease", *fields[kind]}
        or type(value["version"]) is not int
        or value["version"] not in ((1, 2) if kind == "resource_exit" else (1,))
        or _json(value["lease"]) != _json(_proof(row))
    ):
        raise JobError("managed_effect_evidence_corrupt")
    if kind == "prepared":
        ManagedToolPreparation.from_dict(value["preparation"])
    elif kind == "observation":
        ManagedToolResult.from_dict(value["result"])
    elif kind == "dispatch" and not _digest(value["preparation_hash"]):
        raise JobError("managed_effect_evidence_corrupt")
    return found


async def _observation(manager, tx, row, binding, *, required=True):
    found = await _evidence(manager, tx, row, "observation", required=required)
    if found is not None and _json(found[1]["adapter"]) != _json(binding):
        raise JobError("managed_effect_binding_changed")
    return found


async def descriptor(manager, tx, row):
    events = await tx.fetchall(
        f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'",
        (row["job_id"],),
    )
    if len(events) != 1:
        raise JobError("managed_effect_creation_missing")
    # Creation is an existing SDK operation, whose hash covers the full spec.
    length = (
        "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
    )
    size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?", (events[0]["event_id"],))
    if size["n"] > _LIMIT:
        raise JobError("managed_effect_evidence_too_large")
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (events[0]["event_id"],))
    try:
        payload = json.loads(event["payload_json"])
        request = payload.get("request", {})
        if FIELD not in request:
            return None
        call = json.loads(event["call_ref_json"])
        binding = request[FIELD]
        if (
            row["kind"] != "background_tool"
            or not row["blocks_completion"]
            or row["parent_job_id"] is None
            or payload["job_id"] != row["job_id"]
            or payload["status"] != "queued"
            or event["request_hash"]
            != _hash([row["kind"], row["title"], None, row["parent_job_id"], False, request, call])
            or type(binding["version"]) is not int
            or binding["version"] != 1
        ):
            raise ValueError()
        registration(
            request,
            kind=row["kind"],
            parent=row["parent_job_id"],
            call_ref=_ref(call),
            tool_call_id=event["tool_call_id"],
            session_id=row["session_id"],
            independent=False,
        )
        await _registered(manager, tx, row, event, binding)
        await _original(manager, tx, row, event, binding)
        return event, binding
    except (ValueError, TypeError, KeyError):
        raise JobError("managed_effect_evidence_corrupt") from None


async def require_source_resolved(manager, tx, source):
    rows = await tx.fetchall(
        f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? AND kind='background_tool' ORDER BY job_id LIMIT 1001",
        (source["job_id"],),
    )
    if len(rows) > 1000:
        raise JobError("managed_effect_history_requires_compaction")
    for row in rows:
        found = await descriptor(manager, tx, row)
        if found is None:
            continue
        if row["generation"] == 0:
            await unstarted_references(manager, tx, row)
            continue
        if row["resource_state"] != "stopped":
            raise JobError("managed_effect_pending")
        await _require_resolved(manager, tx, row, found[1])


async def _invoke(handler, execution, arguments):
    seq, call_id = current_invocation(execution.actor.session_id)
    slot = handler.runtime._managed_completions.reserve(execution.actor, seq, call_id)
    try:
        return await _invoke_original(handler, execution, arguments, slot)
    finally:
        handler.runtime._managed_completions.release_uncompleted(slot)


async def _invoke_original(handler, execution, arguments, slot):
    runtime, manager, actor = handler.runtime, handler.runtime.manager, execution.actor
    seq, call_id = current_invocation(actor.session_id)
    if seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    ref = await manager.store.message_ref(actor.session_id, seq)
    async with manager._write() as tx:
        await require_source_resolved(manager, tx, await manager._row(tx, execution.lease.job_id))
        await validate_call(
            manager, tx, actor, execution.lease, ref, call_id, handler.definition.name, _hash(arguments)
        )
    request = {
        FIELD: {
            "version": 1,
            "adapter_key": handler.adapter.contract.key,
            "contract_hash": handler.adapter.contract.implementation_hash,
            "tool_name": handler.definition.name,
            "schema_hash": handler.schema_hash,
            "arguments_hash": _hash(arguments),
            "contract": handler.adapter.contract.to_dict(),
        }
    }
    registration = await manager.register_job(
        actor,
        kind="background_tool",
        title=handler.definition.name,
        parent=execution.lease,
        call_ref=ref,
        tool_call_id=call_id,
        request=request,
    )
    if registration.replayed:
        raise JobError("managed_effect_original_requires_reconciliation")
    binding = await runtime.claim(actor, registration)
    context = ManagedToolContext(runtime, actor, execution.lease, ref, call_id, binding.lease.job_id)
    prepared, result, error, dispatch_started = None, None, None, False
    try:
        async with runtime.execution(binding, owner=True):
            prepared = await handler.adapter.prepare(context, json.loads(_json(arguments)))
            handler.check(runtime, handler.definition.name, handler.definition)
            if type(prepared) is not ManagedToolPreparation:
                raise JobError("managed_effect_preparation_invalid")
            prepared = ManagedToolPreparation(_document(prepared.reference), prepared.local_result)
            async with manager._write() as tx:
                row = await manager._fence(tx, actor, binding.lease)
                await _event(
                    manager,
                    tx,
                    row,
                    "prepared",
                    {
                        "version": 1,
                        "lease": asdict(binding.lease),
                        "adapter": request[FIELD],
                        "preparation": prepared.to_dict(),
                    },
                )
            if prepared.local_result is not None:
                result = prepared.local_result
            else:
                handler.check(runtime, handler.definition.name, handler.definition)
                async with manager._write() as tx:
                    row = await manager._fence(tx, actor, binding.lease)
                    await _event(
                        manager,
                        tx,
                        row,
                        "dispatch",
                        {"version": 1, "lease": asdict(binding.lease), "preparation_hash": _hash(prepared.to_dict())},
                    )
                dispatch_started = True
                result = await handler.adapter.execute(context, _document(prepared.reference))
            if type(result) is not ManagedToolResult or len(result.display) > handler.adapter.contract.max_result_chars:
                raise JobError("managed_effect_result_invalid")
            result = ManagedToolResult(result.display, result.reference, result.resolution)
    except BaseException as exc:
        error = exc
        if type(result) is not ManagedToolResult or len(result.display) > handler.adapter.contract.max_result_chars:
            result = None
    # The owned finite callback coroutine has actually exited. Its business
    # outcome may still be unknown; that separate obligation blocks closure.
    binding.stop_renewal.set()
    pending_display = (
        "managed effect outcome is unresolved; the host must reconcile the original task " + binding.lease.job_id + "."
    )
    if len(pending_display) > handler.adapter.contract.max_result_chars:
        pending_display = "Managed effect is pending."[: handler.adapter.contract.max_result_chars]
    if error is not None and not dispatch_started:
        result = ManagedToolResult(
            "The managed effect was not dispatched because its preparation did not complete."[:handler.adapter.contract.max_result_chars],
            {"decision": "not_dispatched", "job_id": binding.lease.job_id},
            "not_dispatched",
        )
    payload = {
        "status": "completed" if error is None else "interrupted",
        "managed_effect_pending": error is not None and dispatch_started,
        "managed_effect_result": result.to_dict() if result is not None else None,
        "display": result.display if result is not None else pending_display,
        "usage": {},
    }
    completion = {
        "version": 2,
        "lease": asdict(binding.lease),
        "exit": "returned" if error is None else "cancelled" if isinstance(error, asyncio.CancelledError) else "raised",
        "dispatch_started": dispatch_started,
        "completion": {
            "outcome": "succeeded" if error is None else "interrupted",
            "result": payload,
            "resource_state": "stopped",
        },
    }
    handler.runtime._managed_completions.complete(slot, binding, completion)
    from .managed_completion import _persist_local

    await _persist_local(handler.runtime, actor, slot)
    if isinstance(error, asyncio.CancelledError):
        raise error
    return payload["display"]


async def record_registration(manager, tx, row, operation, binding):
    event = await tx.fetchone(f"SELECT request_hash FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
    await _event(
        manager,
        tx,
        row,
        "registered",
        {
            "version": 1,
            "binding": binding,
            "creation_id": operation.operation_id,
            "creation_hash": event["request_hash"],
            "source_job_id": row["parent_job_id"],
        },
    )


async def _registered(manager, tx, row, creation, binding):
    found = await _read_event(manager, tx, _id(manager, "registered", row["job_id"], 0))
    expected = {
        "version": 1,
        "binding": binding,
        "creation_id": creation["event_id"],
        "creation_hash": creation["request_hash"],
        "source_job_id": row["parent_job_id"],
    }
    if (
        found[0]["event_type"] != "managed_effect_registered"
        or found[0]["job_id"] != row["job_id"]
        or found[0]["delivery_status"] != "not_applicable"
        or _hash(found[1]) != _hash(expected)
    ):
        raise JobError("managed_effect_evidence_corrupt")


async def require_owned_resolved(manager, tx, owner):
    rows = await tx.fetchall(
        f"SELECT * FROM {manager.jobs} WHERE control_owner_job_id=? AND kind='background_tool' ORDER BY job_id LIMIT 1001",
        (owner["job_id"],),
    )
    if len(rows) > 1000:
        raise JobError("managed_effect_history_requires_compaction")
    for row in rows:
        found = await descriptor(manager, tx, row)
        if found is None:
            continue
        if row["generation"] == 0:
            await unstarted_references(manager, tx, row)
            continue
        await _require_resolved(manager, tx, row, found[1])


async def _require_resolved(manager, tx, row, binding):
    if row["resource_state"] != "stopped" or row["result_status"] != "handled":
        raise JobError("managed_effect_pending")
    from .result_revisions import result_id

    identity = result_id(row["job_id"], row["generation"])
    length = (
        "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
    )
    size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?", (identity,))
    if size is None or size["n"] is None or size["n"] > 65536:
        raise JobError("managed_effect_result_missing")
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
    payload = _document(json.loads(event["payload_json"]), 65536)
    if (
        event["job_id"] != row["job_id"]
        or event["event_type"] != "result"
        or event["delivery_status"] != "handled"
        or payload.get("resource_state") != row["resource_state"]
        or payload.get("outcome") != row["outcome"]
        or type(payload.get("result")) is not dict
        or type(payload["result"].get("managed_effect_pending")) is not bool
        or _hash(payload.get("result")) != _hash(json.loads(row["result_json"] or "null"))
    ):
        raise JobError("managed_effect_result_mismatch")
    if (
        payload["result"].get("managed_effect_pending") is True
        and await _observation(manager, tx, row, binding, required=False) is None
    ):
        raise JobError("managed_effect_pending")
    await require_closed(manager, tx, row, event)


async def _never_claimed(manager, tx, row):
    if (
        row["generation"] != 0
        or row["executor_id"] is not None
        or row["boot_id"] is not None
        or row["lease_until_ms"] is not None
    ):
        raise JobError("managed_effect_unstarted_conflict")
    history = await tx.fetchone(
        f"SELECT event_id FROM {manager.events} WHERE job_id=? AND (event_type IN ('execution_scope','execution_member') OR event_type LIKE 'managed_effect_%' AND event_type NOT IN ('managed_effect_registered','managed_effect_unstarted','managed_effect_raw_receipt')) LIMIT 1",
        (row["job_id"],),
    )
    children = await tx.fetchone(
        f"SELECT job_id FROM {manager.jobs} WHERE parent_job_id=? OR control_owner_job_id=? LIMIT 1",
        (row["job_id"], row["job_id"]),
    )
    # An admitted member edge is stored on its owner, independently of the
    # child's mutable row and scope. Losing both cannot erase that first claim.
    from .execution_membership import member_id

    membership = await tx.fetchone(
        f"SELECT event_id FROM {manager.events} WHERE event_id=?", (member_id(manager, row["job_id"], 1),)
    )
    if history is not None or children is not None or membership is not None:
        raise JobError("managed_effect_unstarted_conflict")


def _unstarted_payload(row):
    result = ManagedToolResult(
        "The original managed effect was cancelled before execution.",
        {"job_id": row["job_id"], "decision": "not_dispatched"},
        "not_dispatched",
    )
    return {
        "outcome": "cancelled",
        "resource_state": "stopped",
        "unstarted": True,
        "result": {
            "status": "cancelled",
            "managed_effect_pending": False,
            "managed_effect_result": result.to_dict(),
            "display": result.display,
            "usage": {},
        },
    }


async def unstarted_references(manager, tx, row):
    """Original negative disposition, not a fabricated tool/model acknowledgement."""
    found = await descriptor(manager, tx, row)
    if found is None:
        raise JobError("finalization_adapter_required")
    await _never_claimed(manager, tx, row)
    creation, binding = found
    await raw_receipt(manager, tx, row, creation, binding, required=False)
    payload = _unstarted_payload(row)
    result_id = "result_" + row["job_id"]
    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (result_id,))
    terminal = await _read_event(manager, tx, _id(manager, "unstarted", row["job_id"], 0))
    expected = {
        "version": 1,
        "decision": "never_claimed",
        "creation_id": creation["event_id"],
        "creation_hash": creation["request_hash"],
        "binding": binding,
        "result_id": result_id,
        "result_hash": _hash(payload),
    }
    if (
        row["resource_state"] != "stopped"
        or row["intake"] != "closed"
        or row["outcome"] != "cancelled"
        or row["result_status"] != "suppressed"
        or _hash(json.loads(row["result_json"] or "null")) != _hash(payload["result"])
        or result is None
        or result["job_id"] != row["job_id"]
        or result["event_type"] != "result"
        or result["delivery_status"] != "suppressed"
        or _hash(json.loads(result["payload_json"])) != _hash(payload)
        or terminal[0]["job_id"] != row["job_id"]
        or terminal[0]["event_type"] != "managed_effect_unstarted"
        or terminal[0]["delivery_status"] != "not_applicable"
        or _hash(terminal[1]) != _hash(expected)
    ):
        raise JobError("managed_effect_unstarted_evidence_mismatch")
    from .finalization import _reference

    return [
        await _reference(manager, tx, key)
        for key in (
            creation["event_id"],
            _id(manager, "registered", row["job_id"], 0),
            result_id,
            terminal[0]["event_id"],
        )
    ]


async def terminate_unstarted(manager, actor, job):
    """Cancel a provably unclaimed effect atomically; never invoke host callbacks."""
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError("finalization_adapter_required")
        await _never_claimed(manager, tx, row)
        if row["resource_state"] == "stopped":
            await unstarted_references(manager, tx, row)
        else:
            if (
                row["resource_state"] != "not_started"
                or row["status"] not in ("queued", "cancelled")
                or row["intake"] not in ("open", "closed")
            ):
                raise JobError("managed_effect_unstarted_conflict")
            if (
                await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", ("result_" + job,))
                is not None
            ):
                raise JobError("managed_effect_unstarted_evidence_mismatch")
            creation, binding = found
            payload = _unstarted_payload(row)
            version = row["version"] + 1
            await tx.execute(
                f"UPDATE {manager.jobs} SET status='settling',outcome='cancelled',resource_state='stopped',intake='closed',result_status='suppressed',result_json=?,version=?,updated_at=? WHERE job_id=?",
                (_json(payload["result"]), version, manager.now(), job),
            )
            await tx.execute(
                f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                ("result_" + job, job, "result", version, _json(payload), "suppressed", manager.now()),
            )
            row = await manager._row(tx, job)
            await _event(
                manager,
                tx,
                row,
                "unstarted",
                {
                    "version": 1,
                    "decision": "never_claimed",
                    "creation_id": creation["event_id"],
                    "creation_hash": creation["request_hash"],
                    "binding": binding,
                    "result_id": "result_" + job,
                    "result_hash": _hash(payload),
                },
            )
        from .finalization import _prepare, _publish

        state = {"done": {}, "visiting": set(), "plans": []}
        reference = await _prepare(manager, tx, actor, row, 0, state)
        await _publish(manager, tx, state["plans"])
        return {
            "job_id": job,
            "generation": 0,
            "resolution": "not_dispatched",
            "closure": reference,
            "execution_authority": False,
            "billing_settled": False,
            "tool_receipt_written": False,
        }


async def _receipt_content(manager, tx, receipt_ref, job, tool_name):
    from .jobs import JobActor

    actor = JobActor(job["tenant_id"], job["principal_id"], receipt_ref.session_id)
    await manager._validate_ref(tx, actor, receipt_ref, role="tool", metadata_only=True)
    length = "length(CAST(content AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(content)"
    size = await tx.fetchone(
        f"SELECT {length} AS n FROM {manager.store.t.messages} WHERE session_id=? AND id=?",
        (receipt_ref.session_id, receipt_ref.id),
    )
    if size is None or size["n"] is None or size["n"] > 16384:
        raise JobError("managed_effect_receipt_too_large")
    raw = await tx.fetchone(
        f"SELECT content,name FROM {manager.store.t.messages} WHERE session_id=? AND id=?",
        (receipt_ref.session_id, receipt_ref.id),
    )
    if raw["name"] != tool_name:
        raise JobError("managed_effect_receipt_mismatch")
    return raw["content"]


async def raw_receipt(manager, tx, row, creation, binding, *, required=True):
    found = await _read_event(manager, tx, _id(manager, "raw_receipt", row["job_id"], 0), required=required)
    if found is None:
        return None
    event, value = found
    try:
        ref = _ref(value["receipt_ref"])
        call = _ref(json.loads(creation["call_ref_json"]))
        if (
            set(value)
            != {
                "version",
                "creation_id",
                "creation_hash",
                "binding",
                "receipt_ref",
                "content_hash",
                "acknowledges_result",
            }
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["acknowledges_result"] is not False
            or event["job_id"] != row["job_id"]
            or event["event_type"] != "managed_effect_raw_receipt"
            or event["delivery_status"] != "not_applicable"
            or value["binding"] != binding
            or value["creation_id"] != creation["event_id"]
            or value["creation_hash"] != creation["request_hash"]
            or (ref.session_id, ref.send_index, ref.tool_call_id)
            != (call.session_id, call.send_index, creation["tool_call_id"])
            or ref.seq <= call.seq
            or json.loads(creation["receipt_ref_json"] or "null") != ref.to_dict()
            or _hash(await _receipt_content(manager, tx, ref, row, binding["tool_name"])) != value["content_hash"]
        ):
            raise ValueError()
    except (KeyError, ValueError, TypeError):
        raise JobError("managed_effect_raw_receipt_mismatch") from None
    return found


async def read_raw_receipt(manager, actor, job):
    """Read original call/response coordinates; this acknowledges no outcome."""
    from .task_history import _snapshot

    async with _snapshot(manager) as tx:
        row = await manager._authorize(tx, actor, job)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError("managed_effect_evidence_missing")
        creation, binding = found
        event, value = await raw_receipt(manager, tx, row, creation, binding)
        return {
            "job_id": job,
            "call_ref": json.loads(creation["call_ref_json"]),
            "receipt_ref": value["receipt_ref"],
            "content_hash": value["content_hash"],
            "observation_id": event["event_id"],
            "observation_hash": event["request_hash"],
            "acknowledges_result": False,
            "execution_authority": False,
            "resource_stop_confirmed": False,
            "billing_settled": False,
        }


async def reconcile_raw_receipt(manager, actor, job):
    """Index one retained original tool row; never synthesize a row or execute."""
    from .jobs import JobActor
    from .task_history import _snapshot

    async with _snapshot(manager) as tx:
        row = await manager._authorize(tx, actor, job)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError("managed_effect_evidence_missing")
        creation, binding = found
        call = _ref(json.loads(creation["call_ref_json"]))
        if creation["receipt_ref_json"]:
            ref = _ref(json.loads(creation["receipt_ref_json"]))
        else:
            rows = await tx.fetchall(
                f"SELECT id,seq,role,tool_call_id,send_index,round_index FROM {manager.store.t.messages} WHERE session_id=? AND role='tool' AND tool_call_id=? AND send_index=? AND seq>? ORDER BY seq LIMIT 2",
                (call.session_id, creation["tool_call_id"], call.send_index, call.seq),
            )
            if not rows:
                return None
            if len(rows) != 1:
                raise JobError("managed_effect_receipt_ambiguous")
            from .message_ref import MessageRef

            item = rows[0]
            ref = MessageRef(
                manager.store.store_id,
                call.session_id,
                int(item["id"]),
                int(item["seq"]),
                item["role"],
                item["tool_call_id"],
                item["send_index"],
                item["round_index"],
            )
        original_actor = JobActor(row["tenant_id"], row["principal_id"], call.session_id)
        operation_id = creation["event_id"]
    # The binding transaction rechecks original lineage, call/row coordinates
    # and any concurrent binding. The historical source grants no execution.
    await manager.reconcile_receipt(original_actor, operation_id, ref)
    return await read_raw_receipt(manager, actor, job)


async def bind_receipt(manager, tx, row, creation, receipt_ref):
    found = await descriptor(manager, tx, row)
    if found is None:
        raise JobError("managed_effect_evidence_missing")
    _, binding = found
    content = await _receipt_content(manager, tx, receipt_ref, row, binding["tool_name"])
    call = _ref(json.loads(creation["call_ref_json"]))
    if (receipt_ref.session_id, receipt_ref.send_index, receipt_ref.tool_call_id) != (
        call.session_id,
        call.send_index,
        creation["tool_call_id"],
    ):
        raise JobError("managed_effect_receipt_mismatch")
    # Observation of a real tool row is separate from acknowledging a canonical
    # effect result. Error rows can be indexed without inventing either result.
    await _event(
        manager,
        tx,
        {**row, "generation": 0},
        "raw_receipt",
        {
            "version": 1,
            "creation_id": creation["event_id"],
            "creation_hash": creation["request_hash"],
            "binding": binding,
            "receipt_ref": receipt_ref.to_dict(),
            "content_hash": _hash(content),
            "acknowledges_result": False,
        },
    )
    result = json.loads(row["result_json"] or "null")
    if row["generation"] == 0 or type(result) is not dict or type(result.get("display")) is not str:
        return
    raw = {"content": content}
    if raw["content"] != result["display"]:
        return
    from .result_revisions import result_id

    result_event = await tx.fetchone(
        f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'",
        (result_id(row["job_id"], row["generation"]),),
    )
    if result_event is None or _hash(json.loads(result_event["payload_json"])["result"]) != _hash(result):
        raise JobError("managed_effect_result_missing")
    await _event(
        manager,
        tx,
        row,
        "tool_receipt",
        {
            "version": 1,
            "lease": {
                "job_id": row["job_id"],
                "executor_id": row["executor_id"],
                "boot_id": row["boot_id"],
                "generation": row["generation"],
            },
            "creation_id": creation["event_id"],
            "result_id": result_event["event_id"],
            "result_hash": _hash(json.loads(result_event["payload_json"])),
            "receipt_ref": receipt_ref.to_dict(),
            "display_hash": _hash(raw["content"]),
        },
    )
    await tx.execute(
        f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (result_event["event_id"],)
    )
    await tx.execute(f"UPDATE {manager.jobs} SET result_status='handled' WHERE job_id=?", (row["job_id"],))


async def require_closed(manager, tx, row, result_event):
    found = await descriptor(manager, tx, row)
    if found is None:
        raise JobError("finalization_adapter_required")
    creation, binding = found
    observed_raw = await raw_receipt(manager, tx, row, creation, binding)
    await _registered(manager, tx, row, creation, binding)
    from .finalization import _reference

    refs = [
        await _reference(manager, tx, creation["event_id"]),
        await _reference(manager, tx, _id(manager, "registered", row["job_id"], 0)),
        await _reference(manager, tx, observed_raw[0]["event_id"]),
    ]
    prepared = await _evidence(manager, tx, row, "prepared", required=False)
    exited = await _evidence(manager, tx, row, "resource_exit")
    proof = {
        "job_id": row["job_id"],
        "executor_id": row["executor_id"],
        "boot_id": row["boot_id"],
        "generation": row["generation"],
    }
    if (
        exited[0]["event_type"] != "managed_effect_resource_exit"
        or exited[1].get("lease") != proof
        or exited[1].get("exit") not in ("returned", "raised", "cancelled")
        or type(exited[1].get("dispatch_started")) is not bool
    ):
        raise JobError("managed_effect_resource_exit_unconfirmed")
    if exited[1]["version"] == 2:
        from .managed_completion import validate_exit
        from .result_payloads import require_replay

        checkpoint = await validate_exit(manager, tx, row, found)
        require_replay(json.loads(result_event["payload_json"]), checkpoint["completion"])
        refs.append(await _reference(manager, tx, _id(manager, "completion", row["job_id"], row["generation"])))
    result = json.loads(result_event["payload_json"])["result"]
    if prepared is not None:
        if (
            prepared[0]["event_type"] != "managed_effect_prepared"
            or prepared[1].get("lease") != proof
            or prepared[1].get("adapter") != binding
        ):
            raise JobError("managed_effect_evidence_corrupt")
        refs.append(await _reference(manager, tx, prepared[0]["event_id"]))
    if exited[1]["dispatch_started"]:
        dispatched = await _evidence(manager, tx, row, "dispatch")
        if (
            prepared is None
            or dispatched[0]["event_type"] != "managed_effect_dispatch"
            or dispatched[1].get("lease") != proof
            or dispatched[1].get("preparation_hash") != _hash(prepared[1]["preparation"])
        ):
            raise JobError("managed_effect_evidence_corrupt")
        refs.append(await _reference(manager, tx, dispatched[0]["event_id"]))
    if result.get("managed_effect_pending") is True:
        observation = await _observation(manager, tx, row, binding)
        if (
            observation[0]["event_type"] != "managed_effect_observation"
            or observation[1].get("lease") != proof
            or observation[1].get("adapter") != binding
        ):
            raise JobError("managed_effect_evidence_corrupt")
        observed_result = ManagedToolResult.from_dict(observation[1]["result"])
        if prepared is None and observed_result.resolution != "not_dispatched":
            raise JobError("managed_effect_evidence_missing")
        refs.append(await _reference(manager, tx, observation[0]["event_id"]))
    else:
        observed_result = ManagedToolResult.from_dict(result["managed_effect_result"])
        if prepared is None and (observed_result.resolution != "not_dispatched" or exited[1]["dispatch_started"]):
            raise JobError("managed_effect_evidence_missing")
    receipt = await _evidence(manager, tx, row, "tool_receipt")
    if (
        receipt[0]["event_type"] != "managed_effect_tool_receipt"
        or receipt[1].get("lease") != proof
        or receipt[1].get("result_id") != result_event["event_id"]
        or receipt[1].get("result_hash") != _hash(json.loads(result_event["payload_json"]))
        or receipt[1].get("display_hash") != _hash(result["display"])
        or json.loads(creation["receipt_ref_json"] or "null") != receipt[1].get("receipt_ref")
        or row["result_status"] != "handled"
        or result_event["delivery_status"] != "handled"
    ):
        raise JobError("managed_effect_receipt_mismatch")
    refs.extend(
        [await _reference(manager, tx, exited[0]["event_id"]), await _reference(manager, tx, receipt[0]["event_id"])]
    )
    return refs


async def reconcile(runtime, actor, job, adapter):
    """Original observation only; never prepare/dispatch or claim a new lease."""
    if not isinstance(adapter, ManagedTool) or type(adapter.contract) is not ManagedToolContract:
        raise JobError("managed_effect_adapter_invalid")
    manager = runtime.manager
    from .task_history import _snapshot

    contract_json = _json(adapter.contract.to_dict())

    async with _snapshot(manager) as tx:
        row = await manager._authorize(tx, actor, job)
        found = await descriptor(manager, tx, row)
        if found is None:
            raise JobError("managed_effect_evidence_missing")
        creation, descriptor_value = found
        await _registered(manager, tx, row, creation, descriptor_value)
        if _json(descriptor_value["contract"]) != contract_json:
            raise JobError("managed_effect_binding_changed")
        source_actor, source_proof, ref = await _original(manager, tx, row, creation, descriptor_value)
        context = ManagedToolContext(runtime, source_actor, source_proof, ref, creation["tool_call_id"], job)
        prepared = await _evidence(manager, tx, row, "prepared", required=False)
        if prepared is not None and _json(prepared[1]["adapter"]) != _json(descriptor_value):
            raise JobError("managed_effect_binding_changed")
        reference = prepared[1]["preparation"]["reference"] if prepared is not None else None
    result = await adapter.recover(context, reference)
    if result is None:
        return None
    if type(result) is not ManagedToolResult or len(result.display) > adapter.contract.max_result_chars:
        raise JobError("managed_effect_result_invalid")
    result = ManagedToolResult(result.display, result.reference, result.resolution)
    if type(adapter.contract) is not ManagedToolContract or _json(adapter.contract.to_dict()) != contract_json:
        raise JobError("managed_effect_binding_changed")
    async with manager._write() as tx:
        current = await manager._authorize(tx, actor, job)
        if current["generation"] != row["generation"]:
            raise JobError("managed_effect_generation_changed")
        if await descriptor(manager, tx, current) != found:
            raise JobError("managed_effect_binding_changed")
        if await _evidence(manager, tx, current, "prepared", required=False) != prepared:
            raise JobError("managed_effect_evidence_conflict")
        await _event(
            manager,
            tx,
            current,
            "observation",
            {
                "version": 1,
                "lease": {
                    "job_id": current["job_id"],
                    "executor_id": current["executor_id"],
                    "boot_id": current["boot_id"],
                    "generation": current["generation"],
                },
                "adapter": descriptor_value,
                "result": result.to_dict(),
            },
        )
    return result
