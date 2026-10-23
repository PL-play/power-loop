"""Immutable control-owner edges for historical per-generation execution usage."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import KINDS, JobError, JobLease, _hash, _json


def _generation(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value < 2**63:
        raise JobError("invalid_result_generation")
    return value


def _id(manager, kind, job_id, generation):
    return kind + "_" + _hash([manager.store.store_id, job_id, _generation(generation)])[:48]


def scope_id(manager, job_id, generation):
    return _id(manager, "scope", job_id, generation)


def member_id(manager, job_id, generation):
    return _id(manager, "member", job_id, generation)


def seal_id(manager, job_id, generation):
    return _id(manager, "members_sealed", job_id, generation)


def _object(raw):
    def invalid_constant(value):
        raise ValueError("non-finite JSON constant")
    try:
        data = json.loads(raw, parse_constant=invalid_constant)
    except (TypeError, ValueError) as exc:
        raise JobError("execution_membership_corrupt") from exc
    if not isinstance(data, dict):
        raise JobError("execution_membership_corrupt")
    return data


def _lease(raw):
    if (not isinstance(raw, dict) or set(raw) != {"job_id", "executor_id", "boot_id", "generation"} or
            any(not isinstance(raw.get(k), str) or not 0 < len(raw[k]) <= 64 for k in ("job_id", "executor_id", "boot_id"))):
        raise JobError("execution_membership_corrupt")
    try:
        _generation(raw["generation"])
    except JobError as exc:
        raise JobError("execution_membership_corrupt") from exc
    return raw


def _snapshot(row):
    return {"version": 1,
        "lease": asdict(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"])),
        "owner": ({"job_id": row["control_owner_job_id"], "generation": row["control_owner_generation"],
                   "executor_id": row["control_owner_executor_id"], "boot_id": row["control_owner_boot_id"]}
                  if row["control_owner_job_id"] is not None else None),
        **{k: row[k] for k in ("tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id", "session_id")}}


def _event(row, job_id, kind):
    if (row["job_id"] != job_id or row["event_type"] != kind or row["delivery_status"] != "not_applicable"):
        raise JobError("execution_membership_corrupt")
    return _object(row["payload_json"])


async def scope(manager, reader, job_id, generation):
    row = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (scope_id(manager, job_id, generation),))
    if row is None:
        raise JobError("execution_membership_missing")
    data = _event(row, job_id, "execution_scope")
    proof = _lease(data.get("lease"))
    if (set(data) != {"version", "lease", "owner", "tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id", "session_id"} or
            type(data.get("version")) is not int or data["version"] != 1 or
            proof["job_id"] != job_id or proof["generation"] != generation or data.get("kind") not in KINDS or
            data.get("execution_role") not in ("main", "workflow", "workflow_node", "subagent") or
            any(not isinstance(data.get(k), str) or not data[k] for k in ("tenant_id", "principal_id", "created_by_session_id")) or
            any(data.get(k) is not None and (not isinstance(data[k], str) or not 0 < len(data[k]) <= 64) for k in ("parent_job_id", "session_id"))):
        raise JobError("execution_membership_corrupt")
    if data["owner"] is not None:
        _lease(data["owner"])
    return data


async def require_current(manager, tx, row):
    saved = await read(manager, tx, row, row["generation"])
    if not saved["sealed"]:
        raise JobError("execution_membership_unsealed")
    # Resuming cannot make incomplete/corrupt historical descendants disappear.
    await usage(manager, tx, row, row["generation"])
    data = saved["scope"]
    current = _snapshot(row)
    # A workflow driver is bound after claim; the execution lease/ownership is
    # immutable independently of that deferred physical-session binding.
    if _hash({k: v for k, v in data.items() if k != "session_id"}) != _hash({k: v for k, v in current.items() if k != "session_id"}):
        raise JobError("execution_membership_mismatch")
    return data


async def _insert(manager, tx, event_id, job_id, kind, version, payload):
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if old is not None:
        if _hash(_event(old, job_id, kind)) != _hash(payload):
            raise JobError("execution_membership_conflict")
        return
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
        (event_id, job_id, kind, version, _json(payload), "not_applicable", manager.now()))


async def start(manager, tx, row):
    """Atomic with claim/admission; no execution capability is granted here."""
    data = _snapshot(row)
    _lease(data["lease"])
    if data["owner"] is not None:
        owner = _lease(data["owner"])
        parent = await scope(manager, tx, owner["job_id"], owner["generation"])
        if (_hash(parent["lease"]) != _hash(owner) or
                (parent["tenant_id"], parent["principal_id"]) != (data["tenant_id"], data["principal_id"])):
            raise JobError("execution_membership_mismatch")
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (seal_id(manager, owner["job_id"], owner["generation"]),)):
            raise JobError("execution_membership_closed")
        owner_row = await manager._row(tx, owner["job_id"])
        await _insert(manager, tx, member_id(manager, row["job_id"], row["generation"]), owner["job_id"], "execution_member", owner_row["version"],
            {"version": 1, "owner": owner, "member": data["lease"]})
    await _insert(manager, tx, scope_id(manager, row["job_id"], row["generation"]), row["job_id"], "execution_scope", row["version"], data)


async def members(manager, reader, data):
    proof = data["lease"]
    rows = await reader.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='execution_member' ORDER BY event_id", (proof["job_id"],))
    result = []
    for row in rows:
        payload = _event(row, proof["job_id"], "execution_member")
        owner, member = _lease(payload.get("owner")), _lease(payload.get("member"))
        if owner["job_id"] != proof["job_id"] or row["event_id"] != member_id(manager, member["job_id"], member["generation"]):
            raise JobError("execution_membership_corrupt")
        if owner["generation"] != proof["generation"]:
            continue
        if _hash(owner) != _hash(proof) or _hash(payload) != _hash({"version": 1, "owner": owner, "member": member}):
            raise JobError("execution_membership_corrupt")
        result.append({"event_id": row["event_id"], "payload": payload})
    return result


async def closed_model(manager, reader, data):
    proof = data["lease"]
    row = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", ("model_closed_"+proof["job_id"],))
    if row is None:
        return None
    payload = _event(row, proof["job_id"], "model_closed")
    if (data["kind"] != "send" or type(payload.get("generation")) is not int or payload["generation"] != proof["generation"] or
            not isinstance(payload.get("evidence"), dict) or not isinstance(payload["evidence"].get("usage", {}), dict)):
        raise JobError("execution_membership_corrupt")
    return {"event_id": row["event_id"], "hash": _hash(payload), "payload": payload}


async def seal(manager, tx, row):
    """Freeze admitted direct members, not final supervision or provider costs."""
    anchor = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (scope_id(manager, row["job_id"], row["generation"]),))
    if anchor is None:
        return  # Preserve exit evidence for uninstrumented legacy executions.
    data = await scope(manager, tx, row["job_id"], row["generation"])
    edges, model = await members(manager, tx, data), await closed_model(manager, tx, data)
    await _insert(manager, tx, seal_id(manager, row["job_id"], row["generation"]), row["job_id"], "execution_membership_seal", row["version"]+1,
        {"version": 1, "lease": data["lease"], "members_hash": _hash(edges),
         "model": {k: model[k] for k in ("event_id", "hash")} if model else None})


async def read(manager, reader, row, generation):
    data = await scope(manager, reader, row["job_id"], generation)
    if (data["tenant_id"], data["principal_id"]) != (row["tenant_id"], row["principal_id"]):
        raise JobError("execution_membership_mismatch")
    edges, model = await members(manager, reader, data), await closed_model(manager, reader, data)
    sealed = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (seal_id(manager, row["job_id"], generation),))
    if sealed:
        expected = {"version": 1, "lease": data["lease"], "members_hash": _hash(edges),
            "model": {k: model[k] for k in ("event_id", "hash")} if model else None}
        if _hash(_event(sealed, row["job_id"], "execution_membership_seal")) != _hash(expected):
            raise JobError("execution_membership_corrupt")
    elif model is not None:
        raise JobError("execution_membership_missing")
    else:
        from .result_revisions import result_id
        if await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (result_id(row["job_id"], generation),)):
            raise JobError("execution_membership_missing")
    return {"scope": data, "members": edges, "sealed": sealed is not None, "model": model}


async def get(manager, actor, job_id, generation):
    _generation(generation)
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job_id)
        return await read(manager, tx, row, generation)


async def usage(manager, tx, row, generation):
    root = await read(manager, tx, row, generation)
    pending, seen, totals = [(root, None)], set(), {}
    for _ in range(64):
        if not pending:
            return totals
        following = []
        for item, owner in pending:
            data = item["scope"]
            key = (data["lease"]["job_id"], data["lease"]["generation"])
            if key in seen or (owner is not None and _hash(data["owner"]) != _hash(owner)):
                raise JobError("invalid_execution_membership")
            seen.add(key)
            if item["model"] is not None:
                for name, value in item["model"]["payload"].get("evidence", {}).get("usage", {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[name] = totals.get(name, 0) + value
            for edge in item["members"]:
                member = edge["payload"]["member"]
                child = await read(manager, tx, {**row, "job_id": member["job_id"]}, member["generation"])
                if _hash(child["scope"]["lease"]) != _hash(member):
                    raise JobError("execution_membership_corrupt")
                following.append((child, data["lease"]))
        pending = following
    raise JobError("job_depth_exceeded")
