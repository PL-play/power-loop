"""Trusted budget templates resolved against original job and control-owner IDs.

The same task-store control transaction owns scope creation and reservations.
A scope instance's creation marker prevents lazy recreation of a deleted balance.
Settlement uses the original attempt's frozen set, never its current driver/SID.
"""

from __future__ import annotations

from .jobs import KINDS, JobError, _hash, _json

COUNTERS = ("used_calls", "held_calls", "used_tokens", "held_tokens", "inflight")
_CAP_FIELDS = ("key", "max_calls", "max_tokens", "max_inflight")
MAX_SCOPES = 256


def scope_key(policy_key, limits):
    parts = [policy_key, limits["key"]]
    if "binding" in limits:
        parts.extend((limits["binding"], limits["subject_job_id"]))
    return "bs_" + _hash(parts)[:48]


def _instance(template, subject):
    return {
        **{k: template[k] for k in _CAP_FIELDS},
        "binding": template["binding"],
        "subject_job_id": subject["job_id"],
        "subject_kind": subject["kind"],
        "template_hash": _hash(template),
    }


def _identity(job, reference):
    if (
        job is None
        or job["kind"] not in KINDS
        or any(job[k] != reference[k] for k in ("tenant_id", "principal_id", "root_job_id"))
    ):
        raise JobError("budget_scope_lineage_changed")


async def _lineage(manager, tx, job):
    current, result = job, {}
    for _ in range(64):
        _identity(current, job)
        if current["job_id"] in result:
            raise JobError("budget_scope_lineage_changed")
        result[current["job_id"]] = current
        if current["parent_job_id"] is None:
            if current["job_id"] != job["root_job_id"]:
                raise JobError("budget_scope_lineage_changed")
            return result, current
        current = await manager._row(tx, current["parent_job_id"])
    raise JobError("budget_scope_depth_exceeded")


def _owners(job, lineage):
    current, result, seen = job, [], {job["job_id"]}
    # Main sends are their own budget owners. A non-send direct host attempt
    # also counts its own scope before its controlling ancestors.
    if current["control_owner_job_id"] is None or current["kind"] != "send":
        result.append(current)
    for _ in range(64):
        key = current["control_owner_job_id"]
        if key is None:
            return result
        owner = lineage.get(key)
        if (
            key in seen
            or owner is None
            or (
                current["control_owner_generation"],
                current["control_owner_executor_id"],
                current["control_owner_boot_id"],
            )
            != (owner["generation"], owner["executor_id"], owner["boot_id"])
        ):
            raise JobError("budget_scope_lineage_changed")
        seen.add(key)
        result.append(owner)
        current = owner
    raise JobError("budget_scope_depth_exceeded")


def _marker(policy_key, limits):
    key = scope_key(policy_key, limits)
    return "mb_scope_" + key[3:], {"version": 1, "policy_key": policy_key, "scope_key": key, "limits": limits}


async def check_marker(manager, tx, policy_key, limits):
    if "binding" not in limits:
        return
    event_id, body = _marker(policy_key, limits)
    event = await tx.fetchone(
        f"SELECT job_id,event_type,request_hash,payload_json,delivery_status FROM {manager.events} WHERE event_id=?",
        (event_id,),
    )
    if event is None or (
        event["job_id"],
        event["event_type"],
        event["request_hash"],
        event["payload_json"],
        event["delivery_status"],
    ) != (limits["subject_job_id"], "model_budget_scope", _hash(body), _json(body), "not_applicable"):
        raise JobError("budget_scope_evidence_corrupt")


async def _ensure_instance(manager, tx, policy, limits, subject):
    key = scope_key(policy["policy_key"], limits)
    row = await tx.fetchone(f"SELECT scope_key FROM {manager.prefix}task_budget_scopes WHERE scope_key=?", (key,))
    event_id, body = _marker(policy["policy_key"], limits)
    marker = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (event_id,))
    if row is None:
        if marker is not None:
            raise JobError("budget_scope_missing_or_changed")
        counters = dict.fromkeys(COUNTERS, 0)
        await tx.execute(
            f"INSERT INTO {manager.prefix}task_budget_scopes "
            "(scope_key,policy_key,limits_hash,limits_json,counters_hash,counters_json) VALUES (?,?,?,?,?,?)",
            (key, policy["policy_key"], _hash(limits), _json(limits), _hash(counters), _json(counters)),
        )
        await tx.execute(
            f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                event_id,
                subject["job_id"],
                "model_budget_scope",
                subject["version"],
                _hash(body),
                _json(body),
                "not_applicable",
                manager.now(),
            ),
        )
    await check_marker(manager, tx, policy["policy_key"], limits)


async def resolve(runtime, tx, policy, job, *, create=False):
    templates = policy["manifest"]["scopes"]
    if policy["manifest"]["version"] == 1:
        return templates
    lineage, root = await _lineage(runtime.manager, tx, job)
    owners = _owners(job, lineage)
    selected, subjects = [], {}
    for template in templates:
        binding = template.get("binding", "authority")
        if binding == "authority":
            selected.append(template)
            continue
        matches = (
            [root]
            if binding == "root"
            else [o for o in owners if not template.get("owner_kinds") or o["kind"] in template["owner_kinds"]]
        )
        for subject in matches:
            limits = _instance(template, subject)
            selected.append(limits)
            subjects[scope_key(policy["policy_key"], limits)] = subject
    if not selected:
        raise JobError("budget_scope_not_applicable")
    if len(selected) > MAX_SCOPES:
        raise JobError("budget_scope_count_exceeded")
    result = sorted(selected, key=lambda s: scope_key(policy["policy_key"], s))
    if len({scope_key(policy["policy_key"], s) for s in result}) != len(result):
        raise JobError("budget_scope_duplicate")
    if create:
        for limits in result:
            if "binding" in limits:
                await _ensure_instance(
                    runtime.manager, tx, policy, limits, subjects[scope_key(policy["policy_key"], limits)]
                )
    return result


def frozen(policy, state):
    if policy["manifest"]["version"] == 1:
        return policy["manifest"]["scopes"]
    scopes = state.get("scopes")
    if type(scopes) is not list or not 1 <= len(scopes) <= MAX_SCOPES:
        raise JobError("budget_scope_evidence_corrupt")
    templates = {t["key"]: t for t in policy["manifest"]["scopes"]}
    keys, selected_templates = set(), set()
    for limits in scopes:
        if type(limits) is not dict or type(limits.get("key")) is not str or limits["key"] not in templates:
            raise JobError("budget_scope_evidence_corrupt")
        template = templates[limits["key"]]
        if template.get("binding", "authority") == "authority":
            expected = template
        else:
            kind, subject_id = limits.get("subject_kind"), limits.get("subject_job_id")
            if type(kind) is not str or kind not in KINDS or type(subject_id) is not str or not subject_id:
                raise JobError("budget_scope_evidence_corrupt")
            if template["binding"] == "root" and subject_id != state["context"]["root_job_id"]:
                raise JobError("budget_scope_evidence_corrupt")
            if template.get("owner_kinds") and kind not in template["owner_kinds"]:
                raise JobError("budget_scope_evidence_corrupt")
            expected = _instance(template, {"job_id": subject_id, "kind": kind})
        key = scope_key(policy["policy_key"], limits)
        if limits != expected or key in keys:
            raise JobError("budget_scope_evidence_corrupt")
        keys.add(key)
        selected_templates.add(limits["key"])
    if any(t["key"] not in selected_templates for t in templates.values() if t.get("binding") != "owner"):
        raise JobError("budget_scope_evidence_corrupt")
    return scopes
