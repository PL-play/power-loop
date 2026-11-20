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
    StoreError,
    TargetRef,
    canonical,
    plain,
    sha256_canonical,
)
from power_loop.contracts.errors import CancellationRequested, SessionLeaseLost

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


async def _write_state(tx, port, session_id, key, value, timestamp):
    sql = port.db.dialect.upsert(
        port.store.t.session_runtime_state, ("session_id", "state_key"),
        ("value_json", "updated_at"),
    )
    await tx.execute(sql, (session_id, key, canonical(value), timestamp))


async def _write_owned_state(port, batch, key, value):
    sid = batch.target.session_id
    try:
        async with port.db.transaction() as tx:
            state = await port.db.dialect.lock_state(tx, port.store.t.sessions, sid)
            if (state["execution_state"], int(state["send_index"]),
                int(state["owner_epoch"])) != (
                    "running", batch.send_index, batch.owner_epoch):
                raise SessionLeaseLost(sid)
            if value["phase"] == "dispatched" and (
                state["lifecycle"] in ("closing", "closed")
                or int(state["cancel_seq"]) > batch.cancel_seq
            ):
                raise CancellationRequested(state["cancel_reason"] or "cancelled")
            await _write_state(tx, port, sid, key, value, now_ms())
    except (SessionLeaseLost, CancellationRequested):
        raise
    except Exception as exc:
        raise StoreError(batch.input_refs[0]) from exc


def _checked_result(adapter, result):
    if type(result) is not ManagedToolResult or len(result.display) > adapter.contract.max_result_chars:
        raise RuntimeRequestError("invalid_options", "managed_effect_result_invalid")
    return result


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
    batch = ctx.batch
    if not any(ref.message_id == batch.start_input_message_id for ref in batch.input_refs):
        return SessionToolResult(
            "failed", error="managed effect requires the original start input",
            error_code="managed_effect_original_send_required",
        )
    existing = await worker.runtime.store.get_runtime_state(ctx.session_id, key)
    if existing and existing.get("result") is not None:
        result = ManagedToolResult.from_dict(existing["result"])
        return SessionToolResult(
            "completed", result.display, usage={}, structured_output=result.to_dict(),
        )
    if existing or batch.send_index != 1:
        return _unknown_outcome()

    context = _context(worker.runtime.store, ctx.actor, frozen, batch.send_index)
    prepared = None
    try:
        prepared = await adapter.prepare(context, _document(frozen["arguments"], 131072))
        if type(prepared) is not ManagedToolPreparation:
            raise RuntimeRequestError("invalid_options", "managed_effect_preparation_invalid")
        result = prepared.local_result
        if result is not None:
            _checked_result(adapter, result)
    except Exception as exc:
        prepared = None
        result = ManagedToolResult(
            "The managed effect was not dispatched because preparation did not complete.",
            {"decision": "not_dispatched", "error": str(exc)[:500]},
            "not_dispatched",
        )

    state = {"preparation": prepared.to_dict() if prepared is not None else None,
             "contract": frozen["contract"]}
    if result is None:
        # An exception from this storage boundary is not an adapter failure. In
        # particular, a lost commit acknowledgement must not start the external action.
        await _write_owned_state(worker.inputs, batch, key, {
            **state, "phase": "dispatched", "result": None,
        })
        try:
            result = _checked_result(adapter, await adapter.execute(context, _document(prepared.reference)))
        except Exception as exc:
            await _write_owned_state(worker.inputs, batch, key, {
                **state, "phase": "unknown", "result": None,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
            return _unknown_outcome()
    await _write_owned_state(worker.inputs, batch, key, {
        **state, "phase": "completed", "result": result.to_dict(),
    })
    return SessionToolResult("completed", result.display, usage={}, structured_output=result.to_dict())


def _unknown_outcome():
    from .background_execution import SessionToolResult

    return SessionToolResult(
        "interrupted", "Managed effect outcome is unknown; query the original adapter to recover it.",
        error="managed effect outcome is unknown", error_code="interrupted",
        structured_output={"managed_effect_pending": True},
    )


_OBSERVATION = "managed_recovery_observation"


async def _read_original(port, reader, start, *, key=None, recovery=None):
    """Read only by the frozen start, original action key and deterministic result ID."""
    extra_columns = "s.execution_state AS execution_state"
    extra_joins = ""
    params = (start.message_id, start.session_id)
    if recovery is not None:
        # The session row has already been locked and is not read again here.
        extra_columns = (
            "a.value_json AS managed_value,r.item_id AS recovery_item,"
            f"r.result_delivery_error AS recovery_error,{port._now_sql()} AS db_now"
        )
        extra_joins = (
            f"LEFT JOIN {port.store.t.session_runtime_state} a ON a.session_id=s.session_id AND a.state_key=? "
            f"LEFT JOIN {port.q} r ON r.session_id=s.session_id AND r.item_id=?"
        )
        params = (start.message_id, key, recovery.message_id, start.session_id)
    row = await reader.fetchone(
        "SELECT s.session_id,s.tenant_id,s.principal_id,s.parent_session_id,s.lifecycle,"
        "s.descriptor_json,s.start_input_message_id,s.creation_call_ref_json,s.creation_tool_call_id,"
        "s.result_route_json,q.item_id,q.terminal_status,q.result_send_index,q.meta_json,"
        f"{extra_columns} FROM {port.store.t.sessions} s "
        f"LEFT JOIN {port.q} q ON q.session_id=s.session_id AND q.item_id=? "
        f"{extra_joins} WHERE s.session_id=?", params,
    )
    if row is None:
        raise RuntimeRequestError("not_found")
    return row


async def _authorize_original(port, reader, start, target, actor, adapter, row):
    await port._authorize_known(reader, target, actor, row)
    descriptor = json.loads(row["descriptor_json"])
    frozen = descriptor["config"].get(FIELD)
    route = json.loads(row["result_route_json"]) if row["result_route_json"] else None
    if (frozen is None or route is None or row["start_input_message_id"] != start.message_id
            or InputRef(**route["input_ref"]) != start or route["result_kind"] != "managed_effect_result"
            or canonical(frozen["contract"]) != canonical(adapter.contract.to_dict())
            or canonical(frozen["call_ref"]) != row["creation_call_ref_json"]
            or frozen["call_id"] != row["creation_tool_call_id"]):
        raise RuntimeRequestError("forbidden", "managed adapter or original call mismatch")
    if row["lifecycle"] != "open":
        raise RuntimeRequestError("session_closed")
    if row["item_id"] is None:
        raise RuntimeRequestError("not_found", "managed original input")
    return frozen, route


def _recovery_ref(start, original_send):
    return InputRef(start.store_id, start.session_id, "inp_" + sha256_canonical([
        "managed_recover", start.store_id, start.session_id, start.message_id, original_send,
    ])[:56])


def _check_equal(actual, expected):
    if canonical(actual) != canonical(expected):
        raise RuntimeRequestError("request_conflict", "managed recovery outcome changed")


def _recovery_source(row, state):
    if row["terminal_status"] != "interrupted":
        return None
    if state["phase"] == "completed" and state["result"] is not None:
        return "stored" if state.get("recovered_send_index") is None else "published"
    if state["phase"] in ("dispatched", "unknown") and state["preparation"] is not None:
        return "probe"
    return None


async def _commit_recovery(port, actor, start, target, adapter, frozen, route, key,
                           original_send, state, result_document):
    recovery = _recovery_ref(start, original_send)
    parent_sid = route["target"]["session_id"]
    sid = target.session_id
    try:
        async with port.db.transaction() as tx:
            try:
                # Lock deep-first (child before parent) consistent with Design 217 lock ordering.
                lock_order = [sid] if sid == parent_sid else [sid, parent_sid]
                states = {s: await port.db.dialect.lock_state(tx, port.store.t.sessions, s)
                          for s in lock_order}
            except ValueError as exc:
                raise RuntimeRequestError("not_found") from exc
            row = await _read_original(port, tx, start, key=key, recovery=recovery)
            current_frozen, current_route = await _authorize_original(
                port, tx, start, target, actor, adapter, row,
            )
            _check_equal(current_frozen, frozen)
            _check_equal(current_route, route)
            if row["terminal_status"] != "interrupted" or row["result_send_index"] != original_send:
                raise RuntimeRequestError("request_conflict", "managed original input changed")
            current = json.loads(row["managed_value"]) if row["managed_value"] else None
            if current is None:
                raise RuntimeRequestError("not_found", "managed preparation")
            _check_equal(current["preparation"], state["preparation"])
            _check_equal(current["contract"], frozen["contract"])
            db_now = int(row["db_now"])
            source = _recovery_source(row, current)
            if source in ("published", "stored") and result_document is not None:
                _check_equal(current["result"], result_document)
            if source == "published":
                if row["recovery_item"] is None:
                    raise RuntimeRequestError("not_found", "managed recovery result")
                if row["recovery_error"] is None:
                    return current["result"], None, None
                record = await port.input(tx, recovery)
                activations = []
                parent_ref, _ = await port._route_stored_result_tx(
                    tx, row, route, record, activations, parent_state=states[parent_sid],
                )
                return current["result"], None, parent_ref if activations else None
            if source is None or result_document is None:
                return None, None, None
            if row["recovery_item"] is not None:
                raise RuntimeRequestError("request_conflict", "managed recovery input identity occupied")
            child = states[sid]
            busy = child["execution_state"] == "running"
            if source == "probe":
                meta = json.loads(row["meta_json"])
                observation = {"action_key": key, "preparation": current["preparation"], "result": result_document}
                if _OBSERVATION in meta:
                    _check_equal(meta[_OBSERVATION], observation)
                elif busy:
                    meta[_OBSERVATION] = observation
                    changed = await tx.execute(
                        f"UPDATE {port.q} SET meta_json=? WHERE session_id=? AND item_id=? "
                        "AND terminal_status='interrupted' AND result_send_index=?",
                        (canonical(meta), sid, start.message_id, original_send),
                    )
                    if changed != 1:
                        raise RuntimeRequestError("request_conflict", "managed original input changed")
            if busy:
                return None, None, None
            final = await port._complete_managed_recovery_tx(
                tx, row, child, route, states[parent_sid], recovery, result_document, db_now,
            )
            recovered = final.results[0]
            await _write_state(tx, port, sid, key, {
                **current, "phase": "completed", "result": result_document,
                "recovered_send_index": recovered.send_index,
            }, db_now)
            return result_document, recovered, final.result_ref if final.result_activation else None
    except RuntimeRequestError:
        raise
    except (Exception, asyncio.CancelledError) as exc:
        # Resolve an uncertain commit through the deterministic terminal input. There
        # is no compensation write and no repeat of the external probe here.
        try:
            saved = await port.get_result(recovery, actor=actor)
        except Exception:
            saved = None
        if (saved is not None and saved.input_ref == recovery
                and saved.status == "completed" and result_document is not None):
            _check_equal(saved.structured_output, result_document)
            return result_document, None, None
        raise StoreError(recovery) from exc


async def reconcile(store, actor, start: InputRef, adapter: ManagedTool, *, client=None):
    """Publish a stored outcome or query the original action outside locks."""
    port = RuntimeInputStore(store)
    port.check_store(start)
    target = TargetRef(start.store_id, start.session_id)
    row = await _read_original(port, port.db, start)
    frozen, route = await _authorize_original(port, port.db, start, target, actor, adapter, row)
    key = _state_key(frozen)
    state = await store.get_runtime_state(target.session_id, key)
    if state is None:
        raise RuntimeRequestError("not_found", "managed preparation")
    _check_equal(state["contract"], frozen["contract"])
    result_document = None
    original_send = row["result_send_index"]
    source = _recovery_source(row, state)
    if source is None:
        if state["phase"] == "completed" and state.get("recovered_send_index") is None:
            return ManagedToolResult.from_dict(state["result"])
        return None
    if source != "published" and row["execution_state"] == "running":
        return None
    if source == "stored":
        result_document = ManagedToolResult.from_dict(state["result"]).to_dict()
    elif source == "probe":
        meta = json.loads(row["meta_json"])
        if observation := meta.get(_OBSERVATION):
            _check_equal(observation["action_key"], key)
            _check_equal(observation["preparation"], state["preparation"])
            found = ManagedToolResult.from_dict(observation["result"])
        else:
            context = _context(store, actor, frozen, original_send)
            found = await adapter.recover(context, _document(state["preparation"]["reference"]))
            if found is None:
                return None
        result_document = _checked_result(adapter, found).to_dict()

    result, recovered, parent_ref = await _commit_recovery(
        port, actor, start, target, adapter, frozen, route, key, original_send, state, result_document,
    )
    if client is not None and recovered is not None:
        await client._notify(
            target, recovered.input_ref.message_id, recovered.input_ref, ["output"],
            {"kind": "completed", "session_id": target.session_id, "send_index": recovered.send_index},
        )
    if client is not None and parent_ref is not None:
        parent = TargetRef(parent_ref.store_id, parent_ref.session_id)
        await client._notify(parent, parent_ref.message_id, parent_ref, ["work", "activation"])
    return ManagedToolResult.from_dict(result) if result is not None else None


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
