"""Owned legacy workflow history; metadata matches alone never grant reads."""
from __future__ import annotations

import base64
import copy
import json
import re
from contextlib import asynccontextmanager

from power_loop.runtime.jobs import JobError, _hash

from . import journal
from .spec import AgentNode, ForeachNode, LoopNode, WorkflowSpec, WorkflowSpecError

_LIMIT = 5000


def history_key(run_id):
    # Preserve the conventional key for generated IDs. Long host SDK IDs must
    # not overflow state_key merely because this prefix is longer than journal's.
    return "workflow:history:"+(run_id if len(run_id) <= 128 else "sdk_"+_hash(run_id))


def _object(raw):
    try:
        value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
        if not isinstance(value, dict):
            raise ValueError()
        _hash(value)  # Reject non-JSON/nonfinite contract data as well.
        return value
    except (ValueError, TypeError) as exc:
        raise JobError("legacy_workflow_history_corrupt") from exc


def _run(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{8,64}", value):
        raise JobError("workflow_run_id_required")
    return value


def _saved_identifier(value, name):
    # The host SDK predates model tools and accepts caller-defined opaque IDs,
    # including external executor session IDs. These are only parameterized
    # journal references, never authority to read a physical transcript.
    if not isinstance(value, str) or not value:
        raise JobError("legacy_workflow_history_invalid_"+name)
    return value


def _sid(value):
    if not isinstance(value, str) or not re.fullmatch(r"sess_[0-9a-f]{8,64}", value):
        raise JobError("invalid_workflow_history_session")
    return value


def _nodes(spec):
    from .engine import WorkflowEngine
    found = {}
    def visit(node):
        if isinstance(node, AgentNode):
            found[node.id] = node
        for child in WorkflowEngine._children(node):
            visit(child)
        if getattr(node, "fallback", None) is not None:
            visit(node.fallback)
    visit(spec.root)
    return found


async def _state(store, tx, sid, key):
    row = await tx.fetchone(f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?", (sid, key))
    return _object(row["value_json"]) if row is not None else None


async def _owned(store, tx, owner, run_id):
    _saved_identifier(run_id, "run_id")
    saved = await _state(store, tx, owner, journal.run_key(run_id))
    if saved is None:
        raise JobError("workflow_run_not_found")
    if saved.get("run_id") != run_id:
        raise JobError("legacy_workflow_history_owner_mismatch")
    steps, live, clamp = saved.get("steps", []), saved.get("live_sessions", {}), saved.get("allowed_tools")
    if (not isinstance(steps, list) or not isinstance(live, dict) or
            any(not isinstance(step, dict) for step in steps) or
            any(not isinstance(node, str) or not isinstance(sid, str) for node, sid in live.items()) or
            clamp is not None and (not isinstance(clamp, list) or any(not isinstance(tool, str) for tool in clamp))):
        raise JobError("legacy_workflow_history_journal_corrupt")
    if len(steps) > _LIMIT or len(live) > _LIMIT:
        raise JobError("legacy_workflow_history_scope_too_large")
    for step in steps:
        if step.get("node_id") is not None and not isinstance(step["node_id"], str):
            raise JobError("legacy_workflow_history_journal_corrupt")
        if step.get("session_id") is not None:
            _saved_identifier(step["session_id"], "session_id")
    for sid in live.values():
        _saved_identifier(sid, "session_id")
    try:
        spec = WorkflowSpec.from_json(saved.get("spec"))
    except WorkflowSpecError as exc:
        raise JobError("legacy_workflow_history_spec_unavailable") from exc
    return saved, spec, _nodes(spec)


def _registry(value, owner, run_id, spec):
    expected = {"version", "owner_session_id", "run_id", "spec_hash", "complete", "drivers", "sessions"}
    value = _object(value)
    if (set(value) != expected or type(value["version"]) is not int or value["version"] != 1 or
            value["owner_session_id"] != owner or value["run_id"] != run_id or
            value["spec_hash"] != _hash(spec.to_dict()) or type(value["complete"]) is not bool or
            not isinstance(value["drivers"], dict) or not isinstance(value["sessions"], dict) or
            len(value["drivers"]) > _LIMIT or len(value["sessions"]) > _LIMIT):
        raise JobError("legacy_workflow_history_binding_mismatch")
    for sid, driver in value["drivers"].items():
        _sid(sid)
        if driver != {"session_id": sid, "owner_session_id": owner, "run_id": run_id}:
            raise JobError("legacy_workflow_history_driver_mismatch")
    return value


def _route(route, iteration, spec, node_id):
    if not isinstance(route, list) or len(route) > 64:
        raise JobError("legacy_workflow_history_route_mismatch")
    last = None
    for part in route:
        if not isinstance(part, dict):
            raise JobError("legacy_workflow_history_route_mismatch")
        if set(part) == {"foreach", "index"}:
            value, minimum, label = part["index"], 0, part["foreach"]
            last = value
        elif set(part) == {"loop", "iteration"}:
            value, minimum, label = part["iteration"], 1, part["loop"]
        else:
            raise JobError("legacy_workflow_history_route_mismatch")
        if type(value) is not int or not minimum <= value < 2**63 or label is not None and (not isinstance(label, str) or not label):
            raise JobError("legacy_workflow_history_route_mismatch")
    if iteration != last or iteration is not None and type(iteration) is not int:
        raise JobError("legacy_workflow_history_route_mismatch")
    from .engine import WorkflowEngine
    paths = {}
    def visit(node, path):
        if isinstance(node, AgentNode):
            paths[node.id] = path
        nested = [*path, node] if isinstance(node, (ForeachNode, LoopNode)) else path
        for child in WorkflowEngine._children(node):
            visit(child, nested)
        if getattr(node, "fallback", None) is not None:
            visit(node.fallback, path)
    visit(spec.root, [])
    path = paths.get(node_id)
    if path is None or len(path) != len(route):
        raise JobError("legacy_workflow_history_route_mismatch")
    for node, part in zip(path, route, strict=True):
        kind, coordinate = ("foreach", "index") if isinstance(node, ForeachNode) else ("loop", "iteration")
        if set(part) != {kind, coordinate} or part[kind] != node.id:
            raise JobError("legacy_workflow_history_route_mismatch")
        if isinstance(node, ForeachNode) and node.items is not None and part[coordinate] >= len(node.items) or isinstance(node, LoopNode) and part[coordinate] > node.max_iterations:
            raise JobError("legacy_workflow_history_route_mismatch")


async def register_driver(store, owner, run_id, driver_sid, spec):
    async with store._db.transaction() as tx:
        saved, frozen, _ = await _owned(store, tx, owner, run_id)
    if saved.get("status") != "running" or _hash(spec.to_dict()) != _hash(frozen.to_dict()):
        raise JobError("legacy_workflow_history_admission_closed")
    driver = await store.get_session(driver_sid)
    if (driver is None or driver.metadata.get("kind") != "wf_driver" or
            driver.metadata.get("workflow_run_id") != run_id or driver.metadata.get("legacy_workflow_owner_sid") != owner):
        raise JobError("legacy_workflow_history_driver_mismatch")
    def bind(current):
        value = _registry(current, owner, run_id, spec) if current is not None else {
            "version": 1, "owner_session_id": owner, "run_id": run_id, "spec_hash": _hash(spec.to_dict()),
            "complete": not bool(saved.get("steps") or saved.get("live_sessions")), "drivers": {}, "sessions": {}}
        if driver_sid not in value["drivers"] and len(value["drivers"]) >= _LIMIT:
            raise JobError("legacy_workflow_history_scope_too_large")
        value["drivers"][driver_sid] = {"session_id": driver_sid, "owner_session_id": owner, "run_id": run_id}
        return value
    await store.mutate_runtime_state(owner, history_key(run_id), bind)


def _contract(value, node, run_id, clamp, *, owner=None, require_owner=False):
    value = _object(value)
    spec = value.get("spec")
    metadata = spec.get("metadata") if isinstance(spec, dict) else None
    tools = value.get("tools")
    if (type(value.get("version")) is not int or value["version"] != 1 or not isinstance(metadata, dict) or
            spec.get("name") != node.spec.name or spec.get("system_prompt") != node.spec.system_prompt or
            spec.get("lifecycle") != "linked" or metadata.get("workflow_run_id") != run_id or
            metadata.get("workflow_node_id") != node.id or not isinstance(tools, list) or
            any(not isinstance(tool, str) for tool in tools) or
            node.spec.tools is not None and not set(tools) <= set(node.spec.tools) or
            clamp is not None and not set(tools) <= set(clamp) or
            owner is not None and (metadata.get("legacy_workflow_owner_sid") not in (None, owner) or
                require_owner and metadata.get("legacy_workflow_owner_sid") != owner)):
        raise JobError("legacy_workflow_history_contract_mismatch")
    attempt = metadata.get("attempt")
    if type(attempt) is not int or not 1 <= attempt <= (node.retry.max_attempts if node.retry else 1):
        raise JobError("legacy_workflow_history_attempt_mismatch")
    return value


async def register_session(store, owner, run_id, driver_sid, node, sid, route, attempt):
    async with store._db.transaction() as tx:
        saved, spec, nodes = await _owned(store, tx, owner, run_id)
    if saved.get("status") != "running" or node.id not in nodes or nodes[node.id] != node:
        raise JobError("legacy_workflow_history_admission_closed")
    session = await store.get_session(sid)
    contract = _contract(await store.get_runtime_state(sid, "workflow_execution_contract"), node, run_id,
        saved.get("allowed_tools"), owner=owner, require_owner=True)
    if (session is None or session.parent_session_id != driver_sid or session.metadata.get("execution_role") != "workflow_node" or
            session.metadata.get("workflow_run_id") != run_id or session.metadata.get("workflow_node_id") != node.id or
            session.metadata.get("legacy_workflow_owner_sid") != owner or type(session.metadata.get("attempt")) is not int or session.metadata.get("attempt") != attempt):
        raise JobError("legacy_workflow_history_session_mismatch")
    binding = {"version": 1, "session_id": sid, "node_id": node.id, "driver_sid": driver_sid,
        "route": copy.deepcopy(route), "iteration": session.metadata.get("workflow_iteration"),
        "attempt": attempt, "contract_hash": _hash(contract)}
    _route(binding["route"], binding["iteration"], spec, node.id)
    def bind(current):
        value = _registry(current, owner, run_id, spec)
        if driver_sid not in value["drivers"]:
            raise JobError("legacy_workflow_history_driver_mismatch")
        existing = value["sessions"].get(sid)
        if existing is not None and existing != binding:
            raise JobError("legacy_workflow_history_binding_mismatch")
        if existing is None and len(value["sessions"]) >= _LIMIT:
            raise JobError("legacy_workflow_history_scope_too_large")
        value["sessions"][sid] = binding
        return value
    await store.mutate_runtime_state(owner, history_key(run_id), bind)


async def _entry(store, tx, owner, run_id, saved, node, sid, binding=None):
    row = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {store.t.sessions} WHERE session_id=?", (sid,))
    source = "owned_history_registry" if binding else "owned_journal_and_contract"
    base = {"session_id": sid, "node_id": node.id, "binding_source": source, "execution_authority": False,
        "history_scope": "physical_session", "route": binding["route"] if binding else None,
        "attempt": binding["attempt"] if binding else None, "iteration": binding["iteration"] if binding else None}
    if row is None:
        return {**base, "available": False, "reason": "session_deleted" if binding else "session_unavailable"}
    raw = await _state(store, tx, sid, "workflow_execution_contract")
    if raw is None:
        if binding:
            raise JobError("legacy_workflow_history_contract_missing")
        return {**base, "available": False, "reason": "execution_contract_missing"}
    contract = _contract(raw, node, run_id, saved.get("allowed_tools"), owner=owner, require_owner=binding is not None)
    metadata = _object(row["metadata_json"] or "{}")
    cm = contract["spec"]["metadata"]
    if (metadata.get("workflow_run_id") != run_id or metadata.get("workflow_node_id") != node.id or
            type(metadata.get("attempt")) is not int or metadata.get("attempt") != cm["attempt"] or
            metadata.get("workflow_iteration") is not None and type(metadata["workflow_iteration"]) is not int or
            metadata.get("legacy_workflow_owner_sid") not in (None, owner) or
            metadata.get("execution_role") not in (None, "workflow_node") or
            binding and (row["parent_session_id"] not in (None, binding["driver_sid"]) or
                metadata.get("legacy_workflow_owner_sid") != owner or metadata.get("execution_role") != "workflow_node" or
                binding["attempt"] != cm["attempt"] or binding["iteration"] != metadata.get("workflow_iteration") or
                _hash(contract) != binding["contract_hash"])):
        raise JobError("legacy_workflow_history_binding_mismatch")
    return {**base, "attempt": cm["attempt"], "iteration": metadata.get("workflow_iteration"), "available": True, "reason": None}


async def collect(store, tx, owner, run_id):
    saved, spec, nodes = await _owned(store, tx, owner, run_id)
    raw = await _state(store, tx, owner, history_key(run_id))
    registry = _registry(raw, owner, run_id, spec) if raw is not None else None
    entries = {}
    if registry is not None:
        keys = {"version", "session_id", "node_id", "driver_sid", "route", "iteration", "attempt", "contract_hash"}
        for sid, binding in registry["sessions"].items():
            _sid(sid)
            if (not isinstance(binding, dict) or set(binding) != keys or binding["session_id"] != sid or
                    type(binding["version"]) is not int or binding["version"] != 1 or not isinstance(binding["node_id"], str) or binding["node_id"] not in nodes or
                    not isinstance(binding["driver_sid"], str) or binding["driver_sid"] not in registry["drivers"] or not isinstance(binding["route"], list) or
                    type(binding["attempt"]) is not int or not 1 <= binding["attempt"] <= (nodes[binding["node_id"]].retry.max_attempts if nodes[binding["node_id"]].retry else 1)):
                raise JobError("legacy_workflow_history_binding_mismatch")
            _route(binding["route"], binding["iteration"], spec, binding["node_id"])
            entries[sid] = await _entry(store, tx, owner, run_id, saved, nodes[binding["node_id"]], sid, binding)
    referenced = {step.get("session_id") for step in saved.get("steps") or [] if step.get("node_id") in nodes and step.get("session_id")}
    referenced.update(sid for node_id, sid in (saved.get("live_sessions") or {}).items() if node_id in nodes and sid)
    complete = bool(registry is not None and registry["complete"] and referenced <= set(registry["sessions"]))
    if not complete:
        witnesses = {}
        for step in saved.get("steps") or []:
            if step.get("node_id") in nodes and step.get("session_id"):
                witnesses[step["session_id"]] = step["node_id"]
        for node_id, sid in (saved.get("live_sessions") or {}).items():
            if node_id in nodes and sid:
                witnesses[sid] = node_id
        candidates = await tx.fetchall(f"SELECT session_id,value_json FROM {store.t.session_runtime_state} "
            "WHERE state_key=? AND value_json LIKE ? ORDER BY session_id LIMIT ?",
            ("workflow_execution_contract", "%"+run_id+"%", _LIMIT+1))
        if len(candidates) > _LIMIT or len(witnesses)+len(entries) > _LIMIT:
            raise JobError("legacy_workflow_history_scope_too_large")
        for candidate in candidates:
            sid = candidate["session_id"]
            if sid in entries:
                continue
            try:
                contract = _object(candidate["value_json"])
                cs = contract.get("spec")
                metadata = cs.get("metadata") if isinstance(cs, dict) else None
                if not isinstance(metadata, dict):
                    continue
                node_id = metadata.get("workflow_node_id")
                if metadata.get("workflow_run_id") != run_id or node_id not in nodes:
                    continue
                entry = await _entry(store, tx, owner, run_id, saved, nodes[node_id], sid)
                if not entry["available"]:
                    continue
            except JobError:
                continue  # Unwitnessed discovery is never authority by itself.
            entries[sid] = entry
        for sid, node_id in witnesses.items():
            if sid not in entries:
                try:
                    entries[sid] = await _entry(store, tx, owner, run_id, saved, nodes[node_id], sid)
                except JobError:
                    entries[sid] = {"session_id": sid, "node_id": node_id, "available": False,
                        "reason": "execution_contract_mismatch", "binding_source": "owned_journal_reference", "execution_authority": False}
    if len(entries) > _LIMIT:
        raise JobError("legacy_workflow_history_scope_too_large")
    return {"run_id": run_id, "owner_session_id": owner, "history_complete": complete,
        "raw_message_id_status": "available" if store.has_message_ids else "unavailable",
        "spec_hash": _hash(spec.to_dict()), "nodes": [entries[sid] for sid in sorted(entries)]}


@asynccontextmanager
async def snapshot(store):
    async with store._db.transaction() as tx:
        if store._db.dialect.name == "postgres":
            await tx.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        yield tx


def _num(value, name, low=1, high=2**63-1):
    if isinstance(value, str) and len(value) <= 19 and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not low <= value <= high:
        raise JobError("invalid_legacy_workflow_history_"+name)
    return value


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def _locate(store, tx, owner, run_id, sid):
    if run_id is not None:
        return await collect(store, tx, owner, _run(run_id))
    _sid(sid)
    row = await tx.fetchone(f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (owner, journal.INDEX_KEY))
    ids = json.loads(row["value_json"]) if row else []
    if not isinstance(ids, list) or len(ids) > 100 or any(not isinstance(v, str) for v in ids):
        raise JobError("legacy_workflow_history_run_required")
    matches = []
    for value in ids:
        data = await collect(store, tx, owner, _saved_identifier(value, "run_id"))
        if any(entry["session_id"] == sid for entry in data["nodes"]):
            matches.append(data)
    if len(matches) != 1:
        raise JobError("legacy_workflow_history_session_not_owned")
    return matches[0]


def _base(data):
    return {"version": 1, "backend": "legacy", "run_id": data["run_id"],
        "history_complete": data["history_complete"], "physical_state_confirmed": False,
        "execution_authority": False, "raw_message_id_status": data["raw_message_id_status"]}


def _fingerprint(data):
    return _hash(data)


def _bounded(value, budget):
    if len(_dump(value)) > budget:
        raise JobError("legacy_workflow_history_budget_too_small")
    return value


async def leaves(loop, owner, options):
    store = await loop._ensure_store()
    limit = _num(options.get("limit", 20), "limit", high=100)
    budget = _num(options.get("max_chars", 8000), "budget", low=1024, high=32000)
    async with snapshot(store) as tx:
        data = await _locate(store, tx, owner, options.get("run_id"), None)
    scope, after = _fingerprint(data), ""
    if options.get("cursor") is not None:
        try:
            raw = options["cursor"]
            if not isinstance(raw, str) or not 0 < len(raw) <= 2048:
                raise ValueError()
            state = json.loads(base64.urlsafe_b64decode(raw+"="*(-len(raw)%4)))
            if not isinstance(state, dict) or set(state) != {"scope", "after"} or state["scope"] != scope:
                raise ValueError()
            after = _sid(state["after"])
        except (ValueError, TypeError, UnicodeError) as exc:
            raise JobError("invalid_legacy_workflow_history_cursor") from exc
    remaining = [entry for entry in data["nodes"] if entry["session_id"] > after]
    shown = remaining[:limit]
    while True:
        more = len(shown) < len(remaining)
        cursor = base64.urlsafe_b64encode(_dump({"scope": scope, "after": shown[-1]["session_id"]}).encode()).decode().rstrip("=") if more and shown else None
        result = {**_base(data), "nodes": shown, "matched_count": len(data["nodes"]), "truncated": more, "next_cursor": cursor}
        if len(_dump(result)) <= budget:
            return result
        if len(shown) <= 1:
            raise JobError("legacy_workflow_history_budget_too_small")
        shown.pop()


class _SnapshotStore:
    """Reuse recall formatting/export against the already authorized read TX."""
    def __init__(self, store, tx):
        self.store, self.tx, self._db = store, tx, store._db

    async def history_bounds(self, sid, **kwargs):
        return await self.store.history_bounds(sid, _fetchone=self.tx.fetchone, **kwargs)

    async def history_records(self, sid, **kwargs):
        kwargs["_fetchall"] = self.tx.fetchall
        return await self.store.history_records(sid, **kwargs)


@asynccontextmanager
async def _same_transaction(tx):
    yield tx


def _clip(value, size):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= size else text[:size]+"…"


async def _timeline(store, tx, sid, base, options, budget):
    tail = _num(options.get("tail", 40), "tail", high=200)
    verbose = options.get("verbose", False)
    if type(verbose) is not bool:
        raise JobError("invalid_legacy_workflow_history_verbose")
    assistants = await tx.fetchall(f"SELECT seq,send_index,round_index,SUBSTR(content,1,2000) AS content,"
        f"SUBSTR(tool_calls_json,1,32769) AS tool_calls_json FROM {store.t.messages} WHERE session_id=? AND role='assistant' ORDER BY seq DESC LIMIT ?", (sid, tail))
    if not assistants:
        return _bounded({**base, "status": "unavailable", "reason": "records_not_found"}, budget)
    raw = await tx.fetchall(f"SELECT tool_call_id,SUBSTR(content,1,2000) AS content FROM {store.t.messages} "
        "WHERE session_id=? AND role='tool' AND seq>=? ORDER BY seq LIMIT ?", (sid, assistants[-1]["seq"], _LIMIT+1))
    if len(raw) > _LIMIT:
        raise JobError("legacy_workflow_history_tail_too_large_use_coordinates")
    results = {r["tool_call_id"]: r["content"] for r in raw}
    rounds = []
    for row in reversed(assistants):
        calls = []
        encoded = row["tool_calls_json"]
        truncated = bool(encoded and len(encoded) > 32768)
        if encoded and not truncated:
            try:
                for call in json.loads(encoded):
                    fn = call.get("function") or {}
                    calls.append({"name": fn.get("name"), "arguments": _clip(fn.get("arguments"), 200 if verbose else 60),
                        "result": _clip(results.get(call.get("id")), 400 if verbose else 90)})
                    truncated = truncated or len(calls[-1]["arguments"]) > (200 if verbose else 60) or len(calls[-1]["result"]) > (400 if verbose else 90)
            except (ValueError, TypeError, AttributeError):
                truncated = True
        text = _clip(row["content"], 400 if verbose else 90)
        rounds.append({"seq": str(row["seq"]), "send_index": str(row["send_index"]) if row["send_index"] is not None else None,
            "round_index": row["round_index"], "text": text, "calls": calls,
            "truncated": truncated or len(text) > (400 if verbose else 90)})
    wanted, shortened = len(rounds), False
    while True:
        result = {**base, "status": "ok", "format": "round_preview", "rounds": rounds, "requested_tail": tail,
            "shown_rounds": len(rounds), "preview_truncated": shortened or len(rounds) < wanted or any(row["truncated"] for row in rounds)}
        if len(_dump(result)) <= budget:
            return result
        shortened = True
        if len(rounds) > 1:
            rounds.pop(0)
        elif rounds[0]["calls"]:
            rounds[0]["calls"].pop(0)
            rounds[0]["truncated"] = True
        elif rounds[0]["text"]:
            rounds[0]["text"] = ""
            rounds[0]["truncated"] = True
        else:
            raise JobError("legacy_workflow_history_budget_too_small")


async def logs(loop, owner, options):
    from power_loop.runtime.recall import RecallInputError, recall
    from power_loop.runtime.store.schema import _column_exists

    if "node_job_id" in options or "message_id" in options or "message_ids" in options:
        raise JobError("legacy_workflow_history_native_ids_unavailable")
    sid, node = options.get("sid"), options.get("node")
    if sid is not None:
        _sid(sid)
    if node is not None and (not isinstance(node, str) or not node.strip() or len(node) > 1000):
        raise JobError("invalid_workflow_node")
    if sid is None and node is None:
        raise JobError("workflow_history_node_required")
    source = options.get("source", "messages")
    if source not in ("messages", "projection"):
        raise JobError("invalid_legacy_workflow_history_source")
    budget = _num(options.get("max_chars", 8000), "budget", low=1024, high=32000)
    store = await loop._ensure_store()
    async with snapshot(store) as tx:
        data = await _locate(store, tx, owner, options.get("run_id"), sid)
        matches = [entry for entry in data["nodes"] if (sid is None or entry["session_id"] == sid) and (node is None or entry["node_id"] == node)]
        if not matches:
            raise JobError("legacy_workflow_history_session_not_owned")
        if len(matches) != 1:
            raise JobError("legacy_workflow_history_session_required")
        selected = matches[0]
        sid = selected["session_id"]
        base = {**_base(data), **selected, "source": source}
        if not selected["available"]:
            return _bounded({**base, "status": "unavailable"}, budget)
        coordinate_keys = {"ids", "seq", "id", "send_index", "cursor", "output_file", "char_offset", "limit"}
        timeline = source == "messages" and not any(key in options for key in coordinate_keys)
        if timeline:
            return await _timeline(store, tx, sid, base, options, budget)
        if "tail" in options or options.get("verbose", False):
            raise JobError("ambiguous_legacy_workflow_history_view")
        if source == "projection" and not await _column_exists(tx, store._db.dialect.name, store.t.project_messages, "id"):
            return _bounded({**base, "status": "unavailable", "reason": "projection_ids_unavailable"}, budget)
        query = {key: options[key] for key in ("ids", "seq", "id", "send_index", "limit", "cursor", "char_offset", "output_file") if key in options}
        for key in ("seq", "id", "send_index"):
            if key in query:
                query[key] = _num(query[key], key)
        if "ids" in query:
            if not isinstance(query["ids"], list) or len(query["ids"]) != 2:
                raise JobError("invalid_legacy_workflow_history_ids")
            query["ids"] = [_num(value, "ids") for value in query["ids"]]
        output = query.get("output_file")
        if output is not None:
            if not isinstance(output, str) or not output or len(output) > 4096:
                raise JobError("invalid_legacy_workflow_history_output_file")
            from power_loop.runtime.env import RuntimeEnvError, get_runtime_env
            try:
                get_runtime_env().require_workspace_dir()
            except RuntimeEnvError as exc:
                raise JobError("legacy_workflow_history_workspace_required") from exc
            _bounded({**base, "status": "exported", "path": output, "records": 10000, "bytes": 16*1024*1024}, budget)
        overhead = len(_dump(base))+32
        body_budget = budget-overhead
        if body_budget < 256:
            raise JobError("legacy_workflow_history_budget_too_small")
        if output is None:
            query["char_limit"] = body_budget
        try:
            answer = json.loads(await recall(_SnapshotStore(store, tx), sid, source=source, **query,
                _scope_id=_fingerprint(data), _strict_errors=True, _all_records=True, _export_options={"transaction": _same_transaction(tx),
                    "configure_snapshot": False, "max_records": 10000, "max_bytes": 16*1024*1024, "workspace_only": True}))
        except RecallInputError as exc:
            raise JobError("invalid_legacy_workflow_history_selection") from exc
        except OSError as exc:
            raise JobError("legacy_workflow_history_export_failed") from exc
        return _bounded({**base, **answer}, budget)
