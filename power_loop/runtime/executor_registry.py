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
        elif action == "activate":
            valid = (set(request) == {"action", "expected_version", "lease_ms"} and _positive(request["expected_version"]) and
                _positive(request["lease_ms"], 300000) and response["version"] == request["expected_version"]+1 and
                response["state"] == "accepting" and not data["effects"])
        elif action == "drain":
            valid = (set(request) == {"action", "reason"} and isinstance(request["reason"], str) and
                0 < len(request["reason"].strip()) <= 1000 and "\x00" not in request["reason"] and response["state"] == "draining" and
                set(data["effects"]) == {"executions", "stop_operations", "scope_incomplete"} and
                isinstance(data["effects"]["executions"], list) and isinstance(data["effects"]["stop_operations"], list) and
                type(data["effects"]["scope_incomplete"]) is bool and
                len(data["effects"]["stop_operations"]) <= len(data["effects"]["executions"]) <= _LIMIT and
                (data["effects"]["scope_incomplete"] or
                 len(data["effects"]["executions"]) == len(data["effects"]["stop_operations"])))
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


async def _row(manager, tx, identity, *, required=True, admission=False, exclusive=False):
    executors, _ = _tables(manager)
    writer = manager._control_intents.get()
    row = (await writer.executor_admission(_key(manager, identity), exclusive=exclusive) if admission and writer is not None else
           await tx.fetchone(f"SELECT * FROM {executors} WHERE executor_key=?", (_key(manager, identity),)))
    if row is None and admission and writer is not None:
        from .executor_admissions import lock
        registered = await lock(manager, writer, identity)
        if registered:
            # A first registration may have committed after the missing-boot
            # statement's snapshot. The held group gate now fences that race.
            row = await writer.executor_admission(_key(manager, identity), exclusive=exclusive, refresh=True)
            if row is None:
                raise JobError('executor_registration_required')
        elif required:
            raise JobError('executor_registration_required')
        else:
            return None
    if row is None:
        _, events = _tables(manager)
        # Once this logical executor is registered, another boot cannot fall
        # back to unregistered dispatch while activation proves old roots.
        retained = await tx.fetchone(f"SELECT event_id FROM {events} WHERE executor_key=? UNION ALL "
            f"SELECT executor_key AS event_id FROM {executors} WHERE tenant_id=? AND principal_id=? AND executor_id=? LIMIT 1",
            (_key(manager, identity), identity.tenant_id, identity.principal_id, identity.executor_id))
        bound = await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE executor_key=? LIMIT 1", (_key(manager, identity),))
        if retained or bound:
            raise JobError("executor_registration_required")
        if required:
            raise JobError("executor_registration_required")
        return None
    _validate_snapshot(manager, identity, _snapshot(row))
    if not manager._trust_own_writes.get():
        _, current = await _event(manager, tx, identity, row["version"])
        _, original = await _event(manager, tx, identity, 1)
        if (any(current["response"][key] != row[key] for key in _FIELDS
                if key not in ("lease_until_ms", "updated_at")) or original["request"].get("action") != "register" or
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
        await tx.write_executor(key, f"INSERT INTO {executors} ({','.join(_FIELDS)}) VALUES ({','.join('?' for _ in _FIELDS)})",
            tuple(value[k] for k in _FIELDS))
    else:
        if await tx.write_executor(key, f"UPDATE {executors} SET state=?,version=?,lease_until_ms=?,updated_at=? WHERE executor_key=? AND version=?",
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
        from .budget_admissions import initialize
        await initialize(manager, tx, identity)
        from .executor_admissions import lock
        await lock(manager, tx, identity, registration=True)
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


class _ActivationChanged(Exception):
    pass


@dataclass(frozen=True)
class _ActivationProof:
    boots: tuple[tuple[str, int], ...]


async def _activation_boots(manager, tx, identity):
    """Only boot metadata locks; never acquire an old root while holding them."""
    executors, _ = _tables(manager)
    suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR UPDATE'
    rows = await tx.fetchall(f"SELECT * FROM {executors} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
        "AND state IN ('starting','accepting') ORDER BY executor_key"+suffix, (identity.tenant_id, identity.principal_id, identity.executor_id))
    for row in rows:
        tx.remember_row(executors, 'executor_key', row['executor_key'], row)
    return rows


async def _activation_current(manager, tx, identity, request, boots):
    row = await _row(manager, tx, identity)
    if row['state'] != 'starting' or row['version'] != request['expected_version']:
        raise JobError('executor_activation_conflict')
    if row['lease_until_ms'] <= manager.now():
        raise JobError('executor_lease_expired')
    for other in boots:
        if (other['boot_id'] != identity.boot_id and other['state'] in ('starting', 'accepting')
                and other['lease_until_ms'] > manager.now()):
            raise JobError('executor_prior_boot_open')
    return row


async def _begin_activation(manager, identity, operation, request):
    """Fence expired prior boots before inspecting their original task trees."""
    async with manager._write() as tx:
        boots = await _activation_boots(manager, tx, identity)
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay, ()
        await _activation_current(manager, tx, identity, request, boots)
        reason = 'activation:'+_hash([operation, request])
        drain_request = {'action': 'drain', 'reason': reason}
        for offset, old in enumerate(boots):
            if old['boot_id'] == identity.boot_id:
                continue
            prior = identity_for(old)
            barrier = _operation_key(manager, prior, 'drain', 'activation:'+operation)
            await _recall(manager, tx, prior, barrier, drain_request)
            if old['state'] in ('starting', 'accepting'):
                old = await _row(manager, tx, prior)
                value = {**_snapshot(old), 'state': 'draining', 'version': old['version']+1,
                         'updated_at': manager.now()}
                await _save(manager, tx, prior, drain_request, barrier, value, old=old,
                    effects={'executions': [], 'stop_operations': [], 'scope_incomplete': True})
                boots[offset] = value
        return None, tuple((row['executor_key'], row['version']) for row in boots
                           if row['state'] in ('starting', 'accepting'))


async def _inspect_prior_executions(manager, identity):
    """Cold transitive model/membership proof uses one coherent read snapshot."""
    async with manager._read() as tx:
        if await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
                "AND boot_id<>? AND resource_state IN ('running','unknown') LIMIT 1", tuple(asdict(identity).values())):
            raise JobError("executor_prior_execution_unsettled")
        # Open responsibilities cannot use a stopped cache to hide an original
        # model dispatch that never acquired returned-response evidence.
        prior_sends = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND executor_id=? "
            "AND boot_id<>? AND executor_key IS NOT NULL AND kind='send' AND supervision_status='open' "
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


async def _publish_activation(manager, identity, operation, request, prepared):
    async with manager._write() as tx:
        boots = await _activation_boots(manager, tx, identity)
        replay = await _recall(manager, tx, identity, operation, request)
        if replay is not None:
            return replay
        row = await _activation_current(manager, tx, identity, request, boots)
        if tuple((boot['executor_key'], boot['version']) for boot in boots) != prepared.boots:
            raise _ActivationChanged()
        now = manager.now()
        value = {**_snapshot(row), "state": "accepting", "version": row["version"]+1,
                 "lease_until_ms": now+request['lease_ms'], "updated_at": now}
        return await _save(manager, tx, identity, request, operation, value, old=row)


async def activate(manager, identity, *, expected_version, system_key, lease_ms=30000):
    operation = _operation_key(manager, identity, "activate", system_key)
    if not _positive(expected_version) or not _positive(lease_ms, 300000):
        raise JobError("invalid_executor_activation")
    request = {"action": "activate", "expected_version": expected_version, "lease_ms": lease_ms}
    replay = await _recall(manager, manager.db, identity, operation, request)
    if replay is not None:
        return replay
    while True:
        replay, boots = await _begin_activation(manager, identity, operation, request)
        if replay is not None:
            return replay
        await _inspect_prior_executions(manager, identity)
        try:
            return await _publish_activation(manager, identity, operation, request, _ActivationProof(boots))
        except _ActivationChanged:
            continue


async def heartbeat(manager, identity, *, expected_version, lease_ms=30000):
    """Extend the current admission lease without adding a control event."""
    if not _positive(expected_version) or not _positive(lease_ms, 300000):
        raise JobError("invalid_executor_heartbeat")
    async with manager._write() as tx:
        row = await require_accepting(manager, tx, identity, required=True, exclusive=True)
        if row["version"] != expected_version:
            raise JobError("executor_heartbeat_conflict")
        now = manager.now()
        value = {**_snapshot(row), "lease_until_ms": max(row["lease_until_ms"], now+lease_ms), "updated_at": now}
        executors, _ = _tables(manager)
        changed = await tx.write_executor(row["executor_key"],
            f"UPDATE {executors} SET lease_until_ms=?,updated_at=? "
            "WHERE executor_key=? AND version=? AND state='accepting' AND lease_until_ms>?",
            (value["lease_until_ms"], now, row["executor_key"], expected_version, now))
        if changed != 1:
            raise JobError("executor_heartbeat_conflict")
        return {"executor": value, "physical_stop_confirmed": False, "execution_authority": False}


async def require_accepting(manager, tx, identity, *, required=False, exclusive=False):
    row = await _row(manager, tx, identity, required=required, admission=True, exclusive=exclusive)
    if row is not None:
        if row["state"] != "accepting":
            raise JobError("executor_not_accepting")
    return row


async def admit_unbudgeted(manager, tx, identity, *, required):
    """Before tree publication,join mandatory-policy and boot admission."""
    from . import budget_admissions, executor_admissions

    policy_key = budget_admissions.key(manager, identity.tenant_id, identity.principal_id)
    row = await tx.unbudgeted_executor_admission(_key(manager, identity), policy_key)
    if row is None:
        # Optional unregistered runtimes instead share the logical-executor
        # absence gate. Row-presence caching never caches the policy value.
        await budget_admissions.initialize(manager, tx, identity)
        await executor_admissions.lock(manager, tx, identity, principal_key=policy_key)
    await require_accepting(manager, tx, identity, required=required)
    if tx._principal_admission_hashes[policy_key] is not None:
        raise JobError('budget_provider_required')
    tx._budget_policy_admissions[policy_key] = None


async def check_execution(manager, tx, row):
    from .session_ownership import current_ownership
    from .task_runtime import _TASK_OWNERS
    original = identity_for(row)
    ownership = current_ownership(manager.store, row["session_id"])
    runtime = next((runtime for _, runtime in reversed(_TASK_OWNERS.get()) if runtime.manager is manager), None)
    identity = (ExecutorIdentity(row["tenant_id"], row["principal_id"], runtime.executor_id, runtime.boot_id)
                if ownership is not None and runtime is not None else original)
    registered = await require_accepting(manager, tx, identity, required=row["executor_key"] is not None)
    if identity == original and registered is not None and row["executor_key"] != registered["executor_key"]:
        raise JobError("executor_binding_missing")
    if row["executor_key"] is not None and not manager._trust_own_writes.get():
        from .execution_membership import scope
        saved = await scope(manager, tx, row["job_id"], row["generation"])
        if saved["version"] != 2 or saved["executor_binding"]["executor_key"] != row["executor_key"]:
            raise JobError("executor_binding_missing")


async def registration_binding(manager, tx, row):
    identity = identity_for(row)
    if row["executor_key"] != _key(manager, identity):
        raise JobError("executor_binding_mismatch")
    # The registration event is immutable; read it once per process.
    cache = manager.__dict__.setdefault("_executor_bindings", {})
    found = cache.get(row["executor_key"]) if manager._trust_own_writes.get() else None
    if found is None:
        event, data = await _event(manager, tx, identity, 1)
        if data["request"].get("action") != "register":
            raise JobError("executor_evidence_corrupt")
        found = cache[row["executor_key"]] = {"executor_key": row["executor_key"], "registration_event_id": event["event_id"],
            "registration_hash": _hash(data), "contract_hash": data["response"]["contract_hash"]}
    return dict(found)


async def validate_binding(manager, tx, data):
    identity = ExecutorIdentity(data["tenant_id"], data["principal_id"], data["lease"]["executor_id"], data["lease"]["boot_id"])
    expected = await registration_binding(manager, tx, {**asdict(identity), "executor_key": _key(manager, identity)})
    if data["executor_binding"] != expected:
        raise JobError("executor_binding_mismatch")


async def get(manager, identity):
    async with manager._read() as tx:
        row = await _row(manager, tx, identity)
        return {**_snapshot(row), "admission_open": row["state"] == "accepting" and row["lease_until_ms"] > manager.now(),
            "physical_stop_confirmed": False}


async def _drain_barrier(manager, identity, operation, request):
    """Commit admission closure without acquiring or changing any task tree."""
    key = "edb_" + _hash([operation])[:48]
    async with manager._write() as tx:
        row = await _row(manager, tx, identity, admission=True, exclusive=True)
        final = await _recall(manager, tx, identity, operation, request)
        if final is not None:
            return final, True
        old = await _recall(manager, tx, identity, key, request)
        if old is not None:
            return old, False
        value = {**_snapshot(row), "state": "draining", "version": row["version"]+1,
                 "updated_at": manager.now()}
        return await _save(manager, tx, identity, request, key, value, old=row,
            effects={"executions": [], "stop_operations": [], "scope_incomplete": True}), False


async def _drain_scope(manager, identity, operation, request):
    """Freeze the retryable scope after admission closure, outside tree locks."""
    key = "eds_" + _hash([operation])[:48]
    original = await _recall(manager, manager.db, identity, key, request)
    if original is not None:
        return original
    jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? AND supervision_status<>'closed' "
        "AND executor_id=? AND boot_id=? AND status<>'waiting' "
        "AND (intake='open' OR resource_state IN ('not_started','running','unknown')) ORDER BY job_id LIMIT ?",
        (*asdict(identity).values(), _LIMIT+1))
    if len(jobs) > _LIMIT:
        return None
    snapshot = [{k: job[k] for k in ("job_id", "generation", "kind", "executor_id", "boot_id", "resource_state")}
                for job in jobs]
    async with manager._write() as tx:
        row = await _row(manager, tx, identity, admission=True, exclusive=True)
        original = await _recall(manager, tx, identity, key, request)
        if original is not None:
            return original
        value = {**_snapshot(row), "state": "draining", "version": row["version"]+1,
                 "updated_at": manager.now()}
        return await _save(manager, tx, identity, request, key, value, old=row,
            effects={"executions": snapshot, "stop_operations": [], "scope_incomplete": True})


async def _drain_tree(manager, identity, operation, reason, targets, creators):
    """One tree's original stop intents commit together; retries reuse them."""
    stops, incomplete = [], False
    async with manager._write() as tx:
        for target in targets:
            job_id = target["job_id"]
            actor = JobActor(identity.tenant_id, identity.principal_id, creators[job_id])
            key = manager._origin_key(actor, "stop", None, None,
                operation+":"+job_id+":"+str(target["generation"]))
            fingerprint = _hash([target, reason])
            job = await manager._row(tx, job_id)
            old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
            if old is not None:
                if old["request_hash"] != fingerprint:
                    raise JobError("executor_drain_stop_conflict")
                try:
                    saved = json.loads(old["payload_json"])
                    if (old["job_id"] != job_id or old["action"] != "stop"
                            or saved["requested_execution"] != target
                            or saved["executor_drain_operation"] != operation
                            or type(saved["scope_changed"]) is not bool or type(saved["scope_closed"]) is not bool):
                        raise ValueError()
                except (KeyError, TypeError, ValueError):
                    raise JobError("executor_drain_stop_evidence_corrupt") from None
                stops.append(old["event_id"])
                incomplete |= saved["scope_changed"]
                if not saved["scope_changed"] and not saved["scope_closed"]:
                    # An unknown commit may have skipped the original local
                    # notification. Replay the signal, never the database stop.
                    manager._control_notifications.get().add(job_id)
                continue
            changed = any(job[k] != target[k] for k in ("generation", "executor_id", "boot_id"))
            closed = not changed and job["resource_state"] == "stopped" and (
                job["intake"] == "closed" or job["status"] == "waiting")
            if not changed and not closed and job["cancel_requested_at"] is None:
                status = job["status"] if job["supervision_status"] == "closed" or job["status"] == "model_closed" else (
                    "cancelled" if job["resource_state"] == "not_started" else "cancelling")
                now = manager.now()
                count = await tx.write_job(job_id, f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,intake='closed',"
                    "version=version+1,updated_at=? WHERE job_id=? AND generation=?",
                    (status, now, reason, now, job_id, target["generation"]))
                if count != 1:
                    raise JobError("stale_execution")
                job = {**job, "status": status, "cancel_requested_at": now, "cancel_reason": reason,
                       "intake": "closed", "version": job["version"]+1, "updated_at": now}
                tx.remember_row(manager.jobs, "job_id", job_id, job)
            # A different generation/boot is never stopped by this old drain.
            # Its durable intent remains incomplete, with no physical-exit claim.
            stop = await manager._insert_operation(tx, job_id=job_id, version=job["version"], actor=actor,
                action="stop", key=key, fingerprint=fingerprint, notify_stop=not changed and not closed,
                payload={"job_id": job_id, "status": job["status"],
                    "cancel_requested": not changed and job["cancel_requested_at"] is not None,
                    "resource_state": target["resource_state"] if changed else job["resource_state"],
                    "executor_drain_operation": operation, "requested_execution": target,
                    "scope_changed": changed, "scope_closed": closed})
            stops.append(stop.operation_id)
            incomplete |= changed
    return stops, incomplete


async def drain(manager, identity, *, system_key, reason):
    operation = _operation_key(manager, identity, "drain", system_key)
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000 or "\x00" in reason:
        raise JobError("invalid_executor_drain_reason")
    request = {"action": "drain", "reason": reason}
    barrier, final = await _drain_barrier(manager, identity, operation, request)
    if final:
        return barrier
    scope = await _drain_scope(manager, identity, operation, request)
    if scope is None:
        return barrier
    targets = scope["effects"]["executions"]
    jobs, creators, trees = {}, {}, {}
    ids = [target["job_id"] for target in targets]
    for offset in range(0, len(ids), 100):
        part = ids[offset:offset+100]
        rows = await manager.db.fetchall(f"SELECT job_id,root_job_id,created_by_session_id FROM {manager.jobs} WHERE job_id IN (" +
            ",".join("?" for _ in part) + ")", tuple(part))
        jobs.update((row["job_id"], row) for row in rows)
    if set(ids) != jobs.keys():
        raise JobError("executor_drain_original_scope_missing")
    for target in targets:
        row = jobs[target["job_id"]]
        creators[row["job_id"]] = row["created_by_session_id"]
        trees.setdefault(row["root_job_id"], []).append(target)
    stops, incomplete = {}, False
    for root_id in sorted(trees):
        group = trees[root_id]
        done, partial = await _drain_tree(manager, identity, operation, reason, group, creators)
        stops.update(zip((target["job_id"] for target in group), done, strict=True))
        incomplete |= partial
    async with manager._write() as tx:
        row = await _row(manager, tx, identity, admission=True, exclusive=True)
        original = await _recall(manager, tx, identity, operation, request)
        if original is not None:
            return original
        value = {**_snapshot(row), "state": "draining", "version": row["version"]+1,
                 "updated_at": manager.now()}
        return await _save(manager, tx, identity, request, operation, value, old=row,
            effects={"executions": targets, "stop_operations": [stops[target["job_id"]] for target in targets],
                     "scope_incomplete": incomplete})
