"""Immutable control-owner edges for historical per-generation execution usage."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import KINDS, JobError, JobLease, _hash, _json
from .result_revisions import result_id


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


def proof(row):
    return asdict(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"]))


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


def _snapshot(row, owner=None):
    return {"version": 1,
        "lease": asdict(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"])),
        "owner": owner,
        **{k: row[k] for k in ("tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id", "session_id")}}


def _event(row, job_id, kind):
    if (row["job_id"] != job_id or row["event_type"] != kind or row["delivery_status"] != "not_applicable"):
        raise JobError("execution_membership_corrupt")
    return _object(row["payload_json"])


def _cache(manager, name):
    from collections import OrderedDict
    return manager.__dict__.setdefault(name, OrderedDict())


def _trusted(manager, name, key):
    if not manager._trust_own_writes.get():
        return None
    overlay = manager._control_overlay.get()
    if overlay and (name, key) in overlay:
        return overlay[name, key]  # Written earlier in this still-open transaction.
    return _cache(manager, name).get(key)


def _remember(manager, name, key, value):
    cache = _cache(manager, name)
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > 4096:
        cache.popitem(last=False)


async def scope(manager, reader, job_id, generation):
    # Scope events are immutable; one written or validated here is not reread.
    known = _trusted(manager, "_membership_scopes", (job_id, generation))
    if known is not None:
        return json.loads(known)
    row = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (scope_id(manager, job_id, generation),))
    data = _event(row, job_id, "execution_scope") if row else None
    if data is None:
        raise JobError("execution_membership_missing")
    proof = _lease(data.get("lease"))
    keys = {"version", "lease", "owner", "tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id", "session_id"}
    if data.get("version") == 2:
        keys.add("executor_binding")
    if (set(data) != keys or
            type(data.get("version")) is not int or data["version"] not in (1, 2) or
            proof["job_id"] != job_id or proof["generation"] != generation or data.get("kind") not in KINDS or
            data.get("execution_role") not in ("main", "workflow", "workflow_node", "subagent") or
            any(not isinstance(data.get(k), str) or not data[k] for k in ("tenant_id", "principal_id", "created_by_session_id")) or
            any(data.get(k) is not None and (not isinstance(data[k], str) or not 0 < len(data[k]) <= 64) for k in ("parent_job_id", "session_id"))):
        raise JobError("execution_membership_corrupt")
    if data["owner"] is not None:
        _generation(data["owner"]["generation"])
    if data["version"] == 2 and row is not None:
        from .executor_registry import validate_binding
        await validate_binding(manager, reader, data)
    if manager._control_commit_hooks.get() is None:
        _remember(manager, "_membership_scopes", (job_id, generation), _json(data))
    else:
        _remember_after_commit(manager, reader, "_membership_scopes", (job_id, generation), _json(data))
    return data


async def require_current(manager, tx, row):
    data = await scope(manager, tx, row["job_id"], row["generation"])
    if data["lease"]["generation"] != row["generation"] or data["session_id"] != row["session_id"]:
        raise JobError("execution_membership_mismatch")
    return data


async def _insert(manager, tx, event_id, job_id, kind, version, payload):
    columns = ("event_id", "job_id", "event_type", "job_version", "payload_json", "delivery_status", "created_at")
    inserted = await tx.append_event(columns, (event_id, job_id, kind, version, _json(payload), "not_applicable", manager.now()), ignore=True)
    if inserted:
        remember_ref(manager, event_id, job_id, kind, version, payload)
        return
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if old is None or _hash(_event(old, job_id, kind)) != _hash(payload):
        raise JobError("execution_membership_conflict")


async def start(manager, tx, row):
    """Atomic with claim/admission; no execution capability is granted here."""
    owner_row = await manager._row(tx, row["control_owner_job_id"]) if row["control_owner_job_id"] else None
    owner = ({"job_id": owner_row["job_id"], "generation": owner_row["generation"]} if owner_row else None)
    data = _snapshot(row, owner)
    if row["executor_key"] is not None:
        from .executor_registry import registration_binding
        data["version"] = 2
        data["executor_binding"] = await registration_binding(manager, tx, row)
    _lease(data["lease"])
    if owner is not None:
        await _insert(manager, tx, member_id(manager, row["job_id"], row["generation"]), owner["job_id"],
            "execution_member", owner_row["version"], {"version": 1, "owner": owner, "member": data["lease"]})
    await _insert(manager, tx, scope_id(manager, row["job_id"], row["generation"]), row["job_id"], "execution_scope", row["version"], data)
    _remember_after_commit(manager, tx, "_membership_scopes", (row["job_id"], row["generation"]), _json(data))


def note_child(manager, parent_job_id):
    """Only the executing parent can register children, so this process sees all of them."""
    _remember_after_commit(manager, None, "_parents_with_children", parent_job_id, "1")


def childless(manager, job_id):
    """True when this live process ran ``job_id`` and registered no child under it."""
    return bool(_trusted(manager, "_first_claims", job_id) is not None
                and _trusted(manager, "_parents_with_children", job_id) is None)


def note_first_claim(manager, job_id):
    """This process claimed generation 1, so no earlier generation registered children."""
    _remember_after_commit(manager, None, "_first_claims", job_id, "1")


def resolved(manager, job_id, version):
    """A review source this process already proved handled at this row version."""
    return _trusted(manager, "_resolved_reviews", (job_id, int(version))) is not None


def mark_resolved(manager, job_id, version):
    _remember_after_commit(manager, None, "_resolved_reviews", (job_id, int(version)), "1")


def remember_ref(manager, event_id, job_id, kind, version, payload, request_hash=None):
    """Remember the finalization reference of an immutable event this process inserted."""
    _remember_after_commit(manager, None, "_event_refs", event_id, _json({
        "event_id": event_id, "job_id": job_id, "event_type": kind, "revision": version,
        "request_hash": request_hash,
        "payload_hash": _hash(json.loads(_json(payload)))}))


def trusted_ref(manager, event_id):
    known = _trusted(manager, "_event_refs", event_id)
    return None if known is None else json.loads(known)


def remember_merge(manager, name, key, items):
    """Accumulate rows this process wrote under ``key`` (visible in-transaction, cached on commit)."""
    overlay = manager._control_overlay.get()
    base = overlay.get((name, key)) if overlay and (name, key) in overlay else _cache(manager, name).get(key)
    _remember_after_commit(manager, None, name, key, {**(base or {}), **items})


def _remember_after_commit(manager, tx, name, key, value):
    # A rolled-back admission must not leave a cached scope behind.
    pending = manager._control_commit_hooks.get()
    if pending is None:
        _remember(manager, name, key, value)  # Autocommit reader outside a control transaction.
    else:
        manager._control_overlay.get()[name, key] = value
        pending.append(lambda: _remember(manager, name, key, value))


async def members(manager, reader, data):
    proof = data["lease"]
    rows = await reader.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='execution_member' ORDER BY event_id", (proof["job_id"],))
    result = []
    for row in rows:
        payload = _event(row, proof["job_id"], "execution_member")
        owner, member = payload["owner"], _lease(payload["member"])
        if owner["job_id"] != proof["job_id"] or row["event_id"] != member_id(manager, member["job_id"], member["generation"]):
            raise JobError("execution_membership_corrupt")
        if owner["generation"] != proof["generation"]:
            continue

        result.append({"event_id": row["event_id"], "payload": payload})
    return result


async def closed_model(manager, reader, data):
    proof = data["lease"]
    row = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (result_id(proof["job_id"], proof["generation"]),))
    if row is None:
        return None
    payload = _object(row["payload_json"])
    if (data["kind"] != "send" or type(payload.get("generation")) is not int or payload["generation"] != proof["generation"] or
            not isinstance(payload.get("evidence"), dict) or not isinstance(payload["evidence"].get("usage", {}), dict)):
        raise JobError("execution_membership_corrupt")
    return {"event_id": row["event_id"], "hash": _hash(payload), "payload": payload}


async def read(manager, reader, row, generation):
    data = await scope(manager, reader, row["job_id"], generation)
    if (data["tenant_id"], data["principal_id"]) != (row["tenant_id"], row["principal_id"]):
        raise JobError("execution_membership_mismatch")
    model = await closed_model(manager, reader, data) if data["kind"] == "send" else None
    result = model or await reader.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?",
                                           (result_id(row["job_id"], generation),))
    return {"scope": data, "members": await members(manager, reader, data), "sealed": result is not None, "model": model}


async def get(manager, actor, job_id, generation):
    _generation(generation)
    async with manager._read() as tx:
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
            if key in seen or (owner is not None and data["owner"]["job_id"] != owner["job_id"]):
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
