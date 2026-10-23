"""Trusted host boot admission and durable drain, not physical-stop evidence.

Ports are deliberately absent from model tools. Historical unregistered task
executions retain their old capabilities; a registered execution cannot downgrade.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .jobs import KINDS, JobActor, JobError, _hash, _json

_STATES = {"starting", "accepting", "draining", "unknown"}
_FIELDS = ("executor_key", "tenant_id", "principal_id", "executor_id", "boot_id", "contract_hash",
    "state", "version", "lease_until_ms", "created_at", "updated_at")
_LIMIT = 10000
FENCE_ERRORS = frozenset({"executor_not_accepting", "executor_lease_expired", "executor_registration_required",
    "executor_binding_missing", "executor_binding_mismatch", "executor_evidence_missing", "executor_evidence_corrupt"})


@dataclass(frozen=True)
class ExecutorIdentity:
    tenant_id: str
    principal_id: str
    executor_id: str
    boot_id: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            if (not isinstance(value, str) or not 0 < len(value) <= (255 if key in ("tenant_id", "principal_id") else 64) or
                    any(ord(c) < 32 for c in value)):
                raise JobError("invalid_executor_identity")


def identity_for(row):
    return ExecutorIdentity(*(row[k] for k in ("tenant_id", "principal_id", "executor_id", "boot_id")))


def _key(manager, identity):
    if type(identity) is not ExecutorIdentity:
        raise JobError("trusted_executor_identity_required")
    return "exec_" + _hash([manager.store.store_id, asdict(identity)])[:48]


def _event_id(manager, key, version):
    return "exe_" + _hash([manager.store.store_id, key, version])[:48]


def _tables(manager):
    return manager.prefix + "task_executors", manager.prefix + "task_executor_events"


def _positive(value, maximum=2**63-1):
    return type(value) is int and 1 <= value <= maximum


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _snapshot(row):
    return {k: row[k] for k in _FIELDS}


def _validate_snapshot(manager, identity, value):
    if (not isinstance(value, dict) or set(value) != set(_FIELDS) or value["executor_key"] != _key(manager, identity) or
            any(value[k] != v for k, v in asdict(identity).items()) or not _digest(value["contract_hash"]) or
            value["state"] not in _STATES or not _positive(value["version"]) or
            any(type(value[k]) is not int or not 0 <= value[k] < 2**63 for k in ("lease_until_ms", "created_at", "updated_at")) or
            value["updated_at"] < value["created_at"]):
        raise JobError("executor_evidence_corrupt")


def _decode(manager, identity, event):
    try:
        data = json.loads(event["payload_json"])
        if (not isinstance(data, dict) or set(data) != {"version", "identity", "request", "response", "effects", "evidence_hash"} or
                type(data["version"]) is not int or data["version"] != 1 or data["identity"] != asdict(identity) or
                not isinstance(data["request"], dict) or not isinstance(data["effects"], dict) or
                data["evidence_hash"] != _hash({k: v for k, v in data.items() if k != "evidence_hash"}) or
                event["executor_key"] != _key(manager, identity) or event["request_hash"] != _hash(data["request"]) or
                event["action"] != data["request"].get("action")):
            raise JobError("executor_evidence_corrupt")
        _validate_snapshot(manager, identity, data["response"])
        request, response = data["request"], data["response"]
        action = request["action"]
        if action == "register":
            valid = (set(request) == {"action", "contract_hash", "startup_lease_ms"} and
                request["contract_hash"] == response["contract_hash"] and _positive(request["startup_lease_ms"], 300000) and
                response["version"] == 1 and response["state"] == "starting" and not data["effects"])
        elif action in ("activate", "heartbeat"):
            valid = (set(request) == {"action", "expected_version", "lease_ms"} and _positive(request["expected_version"]) and
                _positive(request["lease_ms"], 300000) and response["version"] == request["expected_version"]+1 and
                response["state"] == "accepting" and not data["effects"])
        elif action == "drain":
            valid = (set(request) == {"action", "reason"} and isinstance(request["reason"], str) and
                0 < len(request["reason"].strip()) <= 1000 and "\x00" not in request["reason"] and response["state"] == "draining" and
                set(data["effects"]) == {"executions", "stop_operations", "scope_incomplete"} and
                isinstance(data["effects"]["executions"], list) and isinstance(data["effects"]["stop_operations"], list) and
                type(data["effects"]["scope_incomplete"]) is bool and
                len(data["effects"]["executions"]) == len(data["effects"]["stop_operations"]) <= _LIMIT)
            if valid:
                for job in data["effects"]["executions"]:
                    if (not isinstance(job, dict) or set(job) != {"job_id", "generation", "kind", "executor_id", "boot_id", "resource_state"} or
                            not isinstance(job["job_id"], str) or not 0 < len(job["job_id"]) <= 64 or
                            type(job["generation"]) is not int or not 0 <= job["generation"] < 2**63 or
                            job["kind"] not in KINDS or job["resource_state"] not in ("not_started", "running", "stopped", "unknown") or
                            any(job[k] is not None and (not isinstance(job[k], str) or not 0 < len(job[k]) <= 64)
                                for k in ("executor_id", "boot_id"))):
                        valid = False
                        break
                valid = valid and all(isinstance(v, str) and 0 < len(v) <= 64 for v in data["effects"]["stop_operations"])
        else:
            valid = False
        if not valid:
            raise JobError("executor_evidence_corrupt")
        if event["event_id"] != _event_id(manager, event["executor_key"], data["response"]["version"]):
            raise JobError("executor_evidence_corrupt")
        return data
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("executor_evidence_corrupt") from exc


async def _event(manager, tx, identity, version):
    _, events = _tables(manager)
    row = await tx.fetchone(f"SELECT * FROM {events} WHERE event_id=?", (_event_id(manager, _key(manager, identity), version),))
    if row is None:
        raise JobError("executor_evidence_missing")
    return row, _decode(manager, identity, row)


async def _row(manager, tx, identity, *, required=True):
    executors, _ = _tables(manager)
    row = await tx.fetchone(f"SELECT * FROM {executors} WHERE executor_key=?", (_key(manager, identity),))
    if row is None:
        _, events = _tables(manager)
        retained = await tx.fetchone(f"SELECT event_id FROM {events} WHERE executor_key=? LIMIT 1", (_key(manager, identity),))
        bound = await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE executor_key=? LIMIT 1", (_key(manager, identity),))
        if retained or bound:
            raise JobError("executor_registration_required")
        if required:
            raise JobError("executor_registration_required")
        return None
    _validate_snapshot(manager, identity, _snapshot(row))
    _, current = await _event(manager, tx, identity, row["version"])
    _, original = await _event(manager, tx, identity, 1)
    if (current["response"] != _snapshot(row) or original["request"].get("action") != "register" or
            original["response"]["state"] != "starting" or original["response"]["contract_hash"] != row["contract_hash"] or
            original["response"]["created_at"] != row["created_at"]):
        raise JobError("executor_evidence_corrupt")
    return row


def _operation_key(manager, identity, action, system_key):
    if not isinstance(system_key, str) or not 0 < len(system_key) <= 200 or any(ord(c) < 32 for c in system_key):
        raise JobError("invalid_executor_operation_key")
    return "exec_op_" + _hash([manager.store.store_id, _key(manager, identity), action, system_key])[:48]


def _receipt(event, data, replayed):
    return {"event_id": event["event_id"], "executor": data["response"], "effects": data["effects"],
        "replayed": replayed, "physical_stop_confirmed": False, "execution_authority": False}


async def _recall(manager, tx, identity, operation_key, request):
    _, events = _tables(manager)
    event = await tx.fetchone(f"SELECT * FROM {events} WHERE operation_key=?", (operation_key,))
    if event is None:
        return None
    data = _decode(manager, identity, event)
    if data["request"] != request:
        raise JobError("executor_operation_conflict")
    return _receipt(event, data, True)


async def _save(manager, tx, identity, request, operation_key, value, *, old=None, effects=None):
    executors, events = _tables(manager)
    _validate_snapshot(manager, identity, value)
    key = _key(manager, identity)
    if old is None:
        await tx.execute(f"INSERT INTO {executors} ({','.join(_FIELDS)}) VALUES ({','.join('?' for _ in _FIELDS)})",
            tuple(value[k] for k in _FIELDS))
    else:
        if await tx.execute(f"UPDATE {executors} SET state=?,version=?,lease_until_ms=?,updated_at=? WHERE executor_key=? AND version=?",
                (value["state"], value["version"], value["lease_until_ms"], value["updated_at"], key, old["version"])) != 1:
            raise JobError("executor_write_conflict")
    data = {"version": 1, "identity": asdict(identity), "request": request, "response": value, "effects": effects or {}}
    data["evidence_hash"] = _hash(data)
    event_id = _event_id(manager, key, value["version"])
    await tx.execute(f"INSERT INTO {events} (event_id,executor_key,action,operation_key,request_hash,payload_json,created_at) VALUES (?,?,?,?,?,?,?)",
        (event_id, key, request["action"], operation_key, _hash(request), _json(data), manager.now()))
    return _receipt({"event_id": event_id}, data, False)


async def register(manager, identity, *, contract_hash, startup_lease_ms=30000):
    key = _key(manager, identity)
    if not _digest(contract_hash) or not _positive(startup_lease_ms, 300000):
        raise JobError("invalid_executor_registration")
    request = {"action": "register", "contract_hash": contract_hash, "startup_lease_ms": startup_lease_ms}
    operation = _operation_key(manager, identity, "register", "registration")
    async with manager._write() as tx:
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay
        executors, _ = _tables(manager)
        if await tx.fetchone(f"SELECT executor_key FROM {executors} WHERE executor_key=?", (key,)):
            await _row(manager, tx, identity)  # missing original evidence is not a new boot
            raise JobError("executor_registration_conflict")
        if await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND executor_id=? AND boot_id=? LIMIT 1",
                tuple(asdict(identity).values())):
            raise JobError("executor_history_requires_reconciliation")
        now = manager.now()
        value = {"executor_key": key, **asdict(identity), "contract_hash": contract_hash, "state": "starting", "version": 1,
            "lease_until_ms": now+startup_lease_ms, "created_at": now, "updated_at": now}
        return await _save(manager, tx, identity, request, operation, value)


async def activate(manager, identity, *, expected_version, system_key, lease_ms=30000):
    operation = _operation_key(manager, identity, "activate", system_key)
    if not _positive(expected_version) or not _positive(lease_ms, 300000):
        raise JobError("invalid_executor_activation")
    request = {"action": "activate", "expected_version": expected_version, "lease_ms": lease_ms}
    async with manager._write() as tx:
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay
        row = await _row(manager, tx, identity)
        if row["state"] != "starting" or row["version"] != expected_version:
            raise JobError("executor_activation_conflict")
        now = manager.now()
        if row["lease_until_ms"] <= now:
            raise JobError("executor_lease_expired")
        executors, _ = _tables(manager)
        if await tx.fetchone(f"SELECT executor_key FROM {executors} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
                "AND boot_id<>? AND state IN ('starting','accepting') AND lease_until_ms>? LIMIT 1", (*asdict(identity).values(), now)):
            raise JobError("executor_prior_boot_open")
        if await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
                "AND boot_id<>? AND resource_state IN ('running','unknown') LIMIT 1", tuple(asdict(identity).values())):
            raise JobError("executor_prior_execution_unsettled")
        # Open responsibilities cannot use a stopped cache to hide an original
        # model dispatch that never acquired returned-response evidence.
        prior_sends = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
            "AND boot_id<>? AND executor_key IS NOT NULL AND kind='send' AND supervision_status<>'closed' "
            "AND resource_state='stopped' ORDER BY job_id LIMIT ?", (*asdict(identity).values(), _LIMIT+1))
        if len(prior_sends) > _LIMIT:
            raise JobError("executor_prior_execution_unsettled")
        from .execution_membership import read
        from .model_attempts import require_quiescent
        for send in prior_sends:
            try:
                saved = await read(manager, tx, send, send["generation"])
                if not saved["sealed"] or saved["model"] is None:
                    raise JobError("executor_prior_execution_unsettled")
                await require_quiescent(manager, tx, send)
            except JobError as exc:
                raise JobError("executor_prior_execution_unsettled") from exc
        value = {**_snapshot(row), "state": "accepting", "version": row["version"]+1, "lease_until_ms": now+lease_ms, "updated_at": now}
        return await _save(manager, tx, identity, request, operation, value, old=row)


async def heartbeat(manager, identity, *, expected_version, system_key, lease_ms=30000):
    operation = _operation_key(manager, identity, "heartbeat", system_key)
    if not _positive(expected_version) or not _positive(lease_ms, 300000):
        raise JobError("invalid_executor_heartbeat")
    request = {"action": "heartbeat", "expected_version": expected_version, "lease_ms": lease_ms}
    async with manager._write() as tx:
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay
        row = await require_accepting(manager, tx, identity, required=True)
        if row["version"] != expected_version:
            raise JobError("executor_heartbeat_conflict")
        value = {**_snapshot(row), "version": row["version"]+1,
            "lease_until_ms": max(row["lease_until_ms"], manager.now()+lease_ms), "updated_at": manager.now()}
        return await _save(manager, tx, identity, request, operation, value, old=row)


async def require_accepting(manager, tx, identity, *, required=False):
    row = await _row(manager, tx, identity, required=required)
    if row is not None:
        if row["state"] != "accepting":
            raise JobError("executor_not_accepting")
        if row["lease_until_ms"] <= manager.now():
            raise JobError("executor_lease_expired")
    return row


async def check_execution(manager, tx, row):
    identity = identity_for(row)
    registered = await require_accepting(manager, tx, identity, required=row["executor_key"] is not None)
    if registered is not None and row["executor_key"] != registered["executor_key"]:
        raise JobError("executor_binding_missing")
    if row["executor_key"] is not None:
        from .execution_membership import scope
        saved = await scope(manager, tx, row["job_id"], row["generation"])
        if saved["version"] != 2 or saved["executor_binding"]["executor_key"] != row["executor_key"]:
            raise JobError("executor_binding_missing")


async def registration_binding(manager, tx, row):
    identity = identity_for(row)
    if row["executor_key"] != _key(manager, identity):
        raise JobError("executor_binding_mismatch")
    event, data = await _event(manager, tx, identity, 1)
    if data["request"].get("action") != "register":
        raise JobError("executor_evidence_corrupt")
    return {"executor_key": row["executor_key"], "registration_event_id": event["event_id"],
        "registration_hash": _hash(data), "contract_hash": data["response"]["contract_hash"]}


async def validate_binding(manager, tx, data):
    identity = ExecutorIdentity(data["tenant_id"], data["principal_id"], data["lease"]["executor_id"], data["lease"]["boot_id"])
    expected = await registration_binding(manager, tx, {**asdict(identity), "executor_key": _key(manager, identity)})
    if data["executor_binding"] != expected:
        raise JobError("executor_binding_mismatch")


async def get(manager, identity):
    async with manager._write() as tx:
        row = await _row(manager, tx, identity)
        return {**_snapshot(row), "admission_open": row["state"] == "accepting" and row["lease_until_ms"] > manager.now(),
            "physical_stop_confirmed": False}


async def drain(manager, identity, *, system_key, reason):
    operation = _operation_key(manager, identity, "drain", system_key)
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000 or "\x00" in reason:
        raise JobError("invalid_executor_drain_reason")
    request = {"action": "drain", "reason": reason}
    async with manager._write() as tx:
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay
        row = await _row(manager, tx, identity)
        jobs = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND supervision_status<>'closed' "
            "AND ((executor_id=? AND boot_id=?) OR (resource_state='not_started' AND parent_executor_id=? AND parent_boot_id=?)) "
            "AND (intake='open' OR resource_state IN ('not_started','running','unknown')) ORDER BY job_id LIMIT ?",
            (*asdict(identity).values(), identity.executor_id, identity.boot_id, _LIMIT+1))
        if len(jobs) > _LIMIT:
            # Persist the admission barrier even when the bounded stop snapshot
            # cannot be completed. The caller must retain pending control work.
            value = {**_snapshot(row), "state": "draining", "version": row["version"]+1, "updated_at": manager.now()}
            return await _save(manager, tx, identity, request, operation, value, old=row,
                effects={"executions": [], "stop_operations": [], "scope_incomplete": True})
        stops, snapshot = [], []
        for job in jobs:
            snapshot.append({k: job[k] for k in ("job_id", "generation", "kind", "executor_id", "boot_id", "resource_state")})
            if job["cancel_requested_at"] is None:
                await tx.execute(f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,intake='closed',"
                    "version=version+1,updated_at=? WHERE job_id=?", ("cancelled" if job["resource_state"] == "not_started" else "cancelling",
                    manager.now(), reason, manager.now(), job["job_id"]))
            saved = await manager._row(tx, job["job_id"])
            actor = JobActor(identity.tenant_id, identity.principal_id, job["created_by_session_id"])
            key = manager._origin_key(actor, "stop", None, None, operation+":"+job["job_id"]+":"+str(job["generation"]))
            stop = await manager._insert_operation(tx, job_id=job["job_id"], version=saved["version"], actor=actor,
                action="stop", key=key, fingerprint=_hash([job["job_id"], reason]),
                payload={"job_id": job["job_id"], "status": saved["status"], "cancel_requested": True,
                    "resource_state": saved["resource_state"], "executor_drain_operation": operation})
            stops.append(stop.operation_id)
        value = {**_snapshot(row), "state": "draining", "version": row["version"]+1, "updated_at": manager.now()}
        return await _save(manager, tx, identity, request, operation, value, old=row,
            effects={"executions": snapshot, "stop_operations": stops, "scope_incomplete": False})
