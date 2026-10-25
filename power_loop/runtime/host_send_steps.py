"""Trusted host processing steps bind atomically to their original main send.

A step is provenance and a duplicate-dispatch fence, not a lease or a budget.
Hosts must look it up before staging input or retrying a loop operation.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from . import root_context
from .jobs import JobError, _hash, _json
from .task_history import _snapshot

_CURRENT = ContextVar("power_loop_host_send_step", default=None)
FIELD = "host_send_step"


def descriptor(namespace, plan_hash, step_id):
    if (type(namespace) is not str or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", namespace) is None
            or type(plan_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_hash) is None
            or type(step_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", step_id) is None):
        raise JobError("invalid_host_send_step")
    return {"version": 1, "namespace": namespace, "plan_hash": plan_hash, "step_id": step_id}


def _document(value):
    if type(value) is not dict or set(value) != {"version", "namespace", "plan_hash", "step_id"} or type(value["version"]) is not int or value["version"] != 1:
        raise JobError("invalid_host_send_step")
    return descriptor(value["namespace"], value["plan_hash"], value["step_id"])


@dataclass(frozen=True)
class _Binding:
    runtime: object
    session_id: str
    document_json: str
    root_hash: str
    authority: tuple


def bind(runtime, session_id, *, namespace, plan_hash, step_id):
    runtime.actor(session_id)
    source = root_context.for_send(runtime, session_id)
    if source is None:
        raise JobError("host_send_step_root_required")
    binding = _Binding(runtime, session_id, _json(descriptor(namespace, plan_hash, step_id)),
        source["context_hash"], root_context._authority(runtime))
    return _activate(binding)


def _check(binding):
    if not root_context._same_authority(binding):
        raise JobError("host_send_step_binding_mismatch")
    source = root_context.for_send(binding.runtime, binding.session_id)
    if source is None or source["context_hash"] != binding.root_hash:
        raise JobError("host_send_step_binding_mismatch")


@contextmanager
def _activate(binding):
    from .task_runtime import _OWNER, _SEND
    if _OWNER.get() is not None or _SEND.get() is not None:
        raise JobError("host_send_step_requires_host")
    if _CURRENT.get() is not None:
        raise JobError("host_send_step_already_bound")
    _check(binding)
    token = _CURRENT.set(binding)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def for_send(runtime, session_id):
    binding = _CURRENT.get()
    if binding is None:
        return None
    if binding.runtime is not runtime or binding.session_id != session_id:
        raise JobError("host_send_step_binding_mismatch")
    _check(binding)
    return json.loads(binding.document_json)


def request_step(request, *, actor, kind, parent, session_id, system_key):
    if FIELD not in request:
        return None
    step = _document(request[FIELD])
    try:
        source = request["host_context"]
        original = root_context._document(source["namespace"], source["payload"])
        index = request["send_index"]
        if (kind != "send" or parent is not None or session_id != actor.session_id
                or type(index) is not int or not 1 <= index < 2**63
                or type(request["model_attempt_protocol"]) is not int or request["model_attempt_protocol"] != 1
                or source != original or system_key != f"send:{index}"):
            raise ValueError()
    except (TypeError, ValueError, KeyError):
        raise JobError("host_send_step_main_root_required") from None
    return step, source


def _digest(manager, actor, step, source):
    return _hash([manager.store.store_id, actor.tenant_id, actor.principal_id,
        actor.session_id, source["context_hash"], step])


def _identity(manager, actor, step, source):
    return "host_step_" + _digest(manager, actor, step, source)[:48]


def job_id(manager, actor, binding):
    # The job is an independent durable anchor. Losing the auxiliary binding
    # event must not turn an already dispatched stage into a new random job.
    return None if binding is None else "job_" + _digest(manager, actor, *binding)[:32]


async def _event(manager, tx, event_id):
    length = "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
    size = await tx.fetchone(f"SELECT {length} AS n FROM {manager.events} WHERE event_id=?", (event_id,))
    if size is None:
        return None
    if size["n"] > 32768:
        raise JobError("host_send_step_evidence_too_large")
    return await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))


async def _verify(manager, tx, actor, event, step, source):
    try:
        row = await manager._authorize(tx, actor, event["job_id"])
        if (row["job_id"] != job_id(manager, actor, (step, source))
                or row["kind"] != "send" or row["parent_job_id"] is not None or row["session_id"] != actor.session_id
                or row["execution_role"] != "main" or row["created_by_session_id"] != actor.session_id):
            raise ValueError()
        length = "length(CAST(payload_json AS BLOB))" if manager.db.dialect.name == "sqlite" else "octet_length(payload_json)"
        sizes = await tx.fetchall(f"SELECT {length} AS n FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (row["job_id"],))
        if len(sizes) != 1 or sizes[0]["n"] > 65536:
            raise ValueError()
        origin = await root_context._read(manager, tx, actor, row["job_id"], namespace=source["namespace"])
        if any(origin[k] != source[k] for k in source):
            raise ValueError()
        create = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (origin["creation_event_id"],))
        request = json.loads(create["payload_json"])["request"]
        if _document(request[FIELD]) != step:
            raise ValueError()
        expected = {"version": 1, "step": step, "root_context_hash": source["context_hash"],
            "session_id": actor.session_id, "job_id": row["job_id"],
            "creation_event_id": create["event_id"], "creation_request_hash": create["request_hash"]}
        if (event["event_id"] != _identity(manager, actor, step, source)
                or event["event_type"] != "host_send_step" or event["job_version"] != 1
                or event["delivery_status"] != "not_applicable"
                or _hash(json.loads(event["payload_json"])) != _hash(expected)
                or event["request_hash"] != _hash(expected)):
            raise ValueError()
        return {**expected, "store_id": manager.store.store_id, "binding_event_id": event["event_id"],
            "binding_hash": event["request_hash"], "execution_authority": False,
            "resource_stop_confirmed": False, "billing_settled": False}
    except (ValueError, TypeError, KeyError):
        raise JobError("host_send_step_evidence_corrupt") from None


async def require_available(manager, tx, actor, binding):
    if binding is None:
        return
    step, source = binding
    event = await _event(manager, tx, _identity(manager, actor, step, source))
    if event is not None:
        await _verify(manager, tx, actor, event, step, source)
        raise JobError("host_send_step_already_registered")
    if await manager._row(tx, job_id(manager, actor, binding)) is not None:
        raise JobError("host_send_step_evidence_missing")


async def record(manager, tx, actor, binding, operation):
    if binding is None:
        return
    step, source = binding
    payload = {"version": 1, "step": step, "root_context_hash": source["context_hash"],
        "session_id": actor.session_id, "job_id": operation.job_id,
        "creation_event_id": operation.operation_id}
    # The operation DTO contains its public payload, not its source hash.
    create = await tx.fetchone(f"SELECT request_hash FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
    payload["creation_request_hash"] = create["request_hash"]
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,request_hash,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (_identity(manager, actor, step, source), operation.job_id, "host_send_step", 1,
            _json(payload), _hash(payload), "not_applicable", manager.now()))


async def require_replay(manager, tx, actor, binding, job_id):
    if binding is None:
        return
    step, source = binding
    event = await _event(manager, tx, _identity(manager, actor, step, source))
    if event is None or event["job_id"] != job_id:
        raise JobError("host_send_step_evidence_missing")
    await _verify(manager, tx, actor, event, step, source)


async def lookup(manager, actor, *, namespace, plan_hash, step_id, expected_host_context):
    step = descriptor(namespace, plan_hash, step_id)
    if type(expected_host_context) is not dict or set(expected_host_context) != {"namespace", "payload"}:
        raise JobError("invalid_host_send_step_context")
    source = root_context._document(**expected_host_context)
    async with _snapshot(manager) as tx:
        event = await _event(manager, tx, _identity(manager, actor, step, source))
        if event is None:
            if await manager._row(tx, job_id(manager, actor, (step, source))) is not None:
                raise JobError("host_send_step_evidence_missing")
            return None
        return await _verify(manager, tx, actor, event, step, source)
