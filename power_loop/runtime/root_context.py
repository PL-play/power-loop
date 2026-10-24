"""Host-only immutable root provenance. Metadata never grants execution/budget."""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .jobs import JobError, _hash, _json

_CURRENT = ContextVar("power_loop_root_context", default=None)


def _document(namespace, payload):
    if type(namespace) is not str or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", namespace) is None:
        raise JobError("invalid_root_context_namespace")
    if type(payload) is not dict or not payload:
        raise JobError("invalid_root_context_payload")
    remaining = [1000]
    def validate(value, depth=0):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 16:
            raise JobError("invalid_root_context_payload")
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str or not key or not key.isprintable():
                    raise JobError("invalid_root_context_payload")
                validate(child, depth + 1)
        elif type(value) is list:
            for child in value:
                validate(child, depth + 1)
        elif type(value) is str:
            if "\x00" in value:
                raise JobError("invalid_root_context_payload")
        elif type(value) is int:
            if not -(2**63) <= value < 2**63:
                raise JobError("invalid_root_context_payload")
        elif value is not None and type(value) is not bool:
            raise JobError("invalid_root_context_payload")
    validate(payload)
    try:
        encoded = _json(payload)
        if len(encoded.encode("utf-8")) > 16384:
            raise JobError("invalid_root_context_payload")
    except (ValueError, TypeError, UnicodeError):
        raise JobError("invalid_root_context_payload") from None
    result = {"version": 1, "namespace": namespace, "payload": json.loads(encoded)}
    return {**result, "context_hash": _hash(result)}


@dataclass(frozen=True)
class _Binding:
    runtime: object
    session_id: str
    document_json: str = field(repr=False)
    authority: tuple = field(repr=False)


def _authority(runtime):
    try:
        manager = runtime.manager
        return (manager, manager.store, manager.db, manager.store.store_id, manager.prefix,
            manager.jobs, manager.events, runtime.tenant_id, runtime.principal_id,
            runtime.executor_id, runtime.boot_id)
    except AttributeError:
        raise JobError("invalid_root_context_runtime") from None


def _same_authority(binding):
    current = _authority(binding.runtime)
    return (all(a is b for a, b in zip(current[:3], binding.authority[:3], strict=True))
        and current[3:] == binding.authority[3:])


def bind(runtime, session_id, *, namespace, payload):
    """Snapshot eagerly; entering is legal only outside a model/owner scope.

    Hosts must use a typed namespace and keep credentials/leases out of payload.
    The payload is audit provenance, never instructions or an authorization port.
    """
    runtime.actor(session_id)
    if session_id != session_id.strip() or not session_id.isprintable():
        raise JobError("invalid_root_context_session")
    document_json = _json(_document(namespace, payload))
    binding = _Binding(runtime, session_id, document_json, _authority(runtime))
    return _activate(binding)


@contextmanager
def _activate(binding):
    from .task_runtime import _OWNER, _SEND
    if _OWNER.get() is not None or _SEND.get() is not None:
        raise JobError("root_context_requires_host")
    if _CURRENT.get() is not None:
        raise JobError("root_context_already_bound")
    if not _same_authority(binding):
        raise JobError("root_context_binding_mismatch")
    token = _CURRENT.set(binding)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def for_send(runtime, session_id):
    """Called only for an external root; child sends retain their original tree."""
    binding = _CURRENT.get()
    if binding is None:
        return None
    if binding.runtime is not runtime or binding.session_id != session_id or not _same_authority(binding):
        raise JobError("root_context_binding_mismatch")
    return json.loads(binding.document_json)


async def read(manager, actor, source_job_id, *, namespace=None, required=True):
    """Read saved provenance even after exit; never renew a lease or grant work."""
    async with manager._write() as tx:
        source = await manager._authorize(tx, actor, source_job_id)
        # The root pointer is a cache, not provenance. Follow the authorized
        # original ancestry so a damaged pointer cannot reattribute old work
        # to a different, otherwise valid root in the same main session.
        root = source
        for _ in range(64):
            if root is None or root["root_job_id"] != source["root_job_id"]:
                raise JobError("root_context_lineage_mismatch")
            if root["parent_job_id"] is None:
                if root["job_id"] != source["root_job_id"]:
                    raise JobError("root_context_lineage_mismatch")
                break
            root = await manager._row(tx, root["parent_job_id"])
        else:
            raise JobError("root_context_lineage_mismatch")
        if (root["kind"] != "send" or root["execution_role"] != "main"
                or root["root_job_id"] != root["job_id"] or root["parent_job_id"] is not None
                or root["control_owner_job_id"] is not None or root["session_id"] != actor.session_id):
            raise JobError("root_context_scope_mismatch")
        events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
            "AND event_type='operation' AND action='create'", (root["job_id"],))
        if len(events) != 1:
            raise JobError("root_context_evidence_corrupt")
        [event] = events
        try:
            creation = json.loads(event["payload_json"])
            request = creation["request"]
            index = request["send_index"]
            protocol = request["model_attempt_protocol"]
            if (type(index) is not int or index < 1 or type(protocol) is not int or protocol != 1
                    or creation["job_id"] != root["job_id"] or creation["status"] != "queued"
                    or type(event["job_version"]) is not int or event["job_version"] != 1
                    or event["caller_session_id"] != root["session_id"]
                    or event["call_ref_json"] is not None or event["tool_call_id"] is not None
                    or event["request_hash"] != _hash(["send", root["title"], root["session_id"], None, False, request, None])
                    or event["operation_key"] != manager._origin_key(actor, "create", None, None, f"send:{index}")):
                raise ValueError()
            saved = request.get("host_context")
            if saved is None:
                if required:
                    raise JobError("root_context_missing")
                return None
            if (type(saved) is not dict or set(saved) != {"version", "namespace", "payload", "context_hash"}
                    or type(saved["version"]) is not int or saved["version"] != 1
                    or _document(saved["namespace"], saved["payload"]) != saved):
                raise ValueError()
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
            if isinstance(exc, JobError) and exc.code == "root_context_missing":
                raise
            raise JobError("root_context_evidence_corrupt") from None
        if namespace is not None and saved["namespace"] != namespace:
            raise JobError("root_context_namespace_mismatch")
        return {**saved, "root_job_id": root["job_id"], "root_generation": root["generation"],
            "main_session_id": root["session_id"], "creation_event_id": event["event_id"],
            "creation_request_hash": event["request_hash"], "execution_authority": False}
