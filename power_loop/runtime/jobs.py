"""Experimental B1 task registry, enabled only on an explicit v10 store.

This owns registration/operation identities and execution evidence, not the B2
success gate or B3 external cancellation adapters. A recorded result stays in
settling with supervision open until those protocols are implemented.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .message_ref import MessageRef
from .store.task_schema import TASK_PROTOCOL_VERSION

KINDS = frozenset({"send", "workflow", "workflow_node", "subagent", "background_tool", "background_command", "timer"})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class JobError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class JobActor:
    """Trusted host attribution; never populated directly from model arguments."""
    tenant_id: str
    principal_id: str
    session_id: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v or len(v) > 255
               for v in (self.tenant_id, self.principal_id, self.session_id)):
            raise JobError("invalid_actor")


@dataclass(frozen=True)
class JobLease:
    job_id: str
    executor_id: str
    boot_id: str
    generation: int


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    parent_job_id: str | None
    root_job_id: str
    session_id: str | None
    created_by_session_id: str
    kind: str
    execution_role: str
    control_owner_job_id: str | None
    title: str
    completion_scope_job_id: str
    blocks_completion: bool
    status: str
    version: int
    generation: int
    resource_state: str
    supervision_status: str
    result_status: str
    outcome: str | None
    result: dict | None
    cancel_requested_at: int | None


@dataclass(frozen=True)
class JobOperation:
    operation_id: str
    job_id: str
    action: str
    call_ref: dict | None
    receipt_ref: dict | None
    result: dict


@dataclass(frozen=True)
class JobRegistration:
    job: JobRecord
    operation: JobOperation
    replayed: bool


def _record(row) -> JobRecord:
    fields = JobRecord.__dataclass_fields__
    data = {key: row[key] for key in fields if key != "result"}
    data["blocks_completion"] = bool(data["blocks_completion"])
    data["result"] = json.loads(row["result_json"]) if row["result_json"] else None
    return JobRecord(**data)


def _operation(row) -> JobOperation:
    return JobOperation(row["event_id"], row["job_id"], row["action"],
        json.loads(row["call_ref_json"]) if row["call_ref_json"] else None,
        json.loads(row["receipt_ref_json"]) if row["receipt_ref_json"] else None,
        json.loads(row["payload_json"]))


class JobManager:
    def __init__(self, store, *, now_ms: Callable[[], int] | None = None):
        if store.schema_version != 10 or not store.store_id:
            raise JobError("task_registry_unsupported")
        self.store = store
        self.db = store._db
        self.prefix = store.table_prefix
        self.jobs = self.prefix + "jobs"
        self.events = self.prefix + "job_events"
        self.now = now_ms or (lambda: time.time_ns() // 1_000_000)

    @asynccontextmanager
    async def _write(self):
        # One short store control lock avoids parent/child lock inversion. No
        # transaction spans a model/provider request. B1 favours correctness over
        # multi-tenant throughput; replace only with equivalent race tests.
        async with self.db.transaction() as tx:
            suffix = "" if self.db.dialect.name == "sqlite" else " FOR UPDATE"
            row = await tx.fetchone(f"SELECT store_id,protocol_version FROM {self.prefix}task_store WHERE id=1" + suffix)
            if row is None or row["store_id"] != self.store.store_id or row["protocol_version"] != TASK_PROTOCOL_VERSION:
                raise JobError("store_identity_changed")
            yield tx

    async def _row(self, reader, job_id):
        return await reader.fetchone(f"SELECT * FROM {self.jobs} WHERE job_id=?", (job_id,))

    async def _authorize(self, reader, actor: JobActor, job_id: str):
        first = current = await self._row(reader, job_id)
        if first is None:
            raise JobError("access_denied")
        allowed, seen = False, set()
        for _ in range(64):
            if current is None or current["job_id"] in seen:
                raise JobError("invalid_job_lineage")
            seen.add(current["job_id"])
            if (current["tenant_id"], current["principal_id"]) != (actor.tenant_id, actor.principal_id):
                raise JobError("access_denied")
            allowed |= actor.session_id in (current["session_id"], current["created_by_session_id"])
            parent = current["parent_job_id"]
            if parent is None:
                if not allowed:
                    raise JobError("access_denied")
                return first
            current = await self._row(reader, parent)
        raise JobError("job_depth_exceeded")

    async def _supervision_barriers(self, tx, row):
        """Historical ancestors retain stop/unknown barriers, not model leases."""
        current, seen, independent_creator = row, set(), False
        for _ in range(64):
            if current is None or current["job_id"] in seen:
                raise JobError("invalid_job_lineage")
            seen.add(current["job_id"])
            if current["cancel_requested_at"] is not None:
                raise JobError("cancel_requested")
            if current["resource_state"] == "unknown" and not independent_creator:
                raise JobError("unresolved_prior_attempt")
            if not current["parent_job_id"]:
                return
            child = current
            current = await self._row(tx, current["parent_job_id"])
            # Previously admitted independent work has its own execution
            # authority. An uncertain creator's *model* request is a separate
            # action, not authority to replay it or to cancel that child. A
            # persistent subtree stop still applies at every ancestor above.
            independent_creator = bool(current is not None and current["kind"] == "send" and
                current["execution_role"] == "main" and current["status"] == "model_closed" and
                child["kind"] in ("workflow", "subagent") and child["control_owner_job_id"] is None and
                not child["blocks_completion"] and child["completion_scope_job_id"] == child["job_id"])
        raise JobError("job_depth_exceeded")

    async def _control_owners(self, tx, row):
        current, seen = row, {row["job_id"]}
        for _ in range(64):
            owner_id = current["control_owner_job_id"]
            if owner_id is None:
                return
            if owner_id in seen:
                raise JobError("invalid_control_owner")
            seen.add(owner_id)
            owner = await self._row(tx, owner_id)
            if owner is None:
                raise JobError("invalid_control_owner")
            if (current["control_owner_generation"], current["control_owner_executor_id"],
                    current["control_owner_boot_id"]) != (owner["generation"], owner["executor_id"], owner["boot_id"]):
                raise JobError("stale_parent_execution")
            self._execution_open(owner)
            from .completion_windows import require_time
            await require_time(self, tx, owner)
            from .executor_registry import check_execution
            await check_execution(self, tx, owner)
            current = owner
        raise JobError("job_depth_exceeded")

    def _execution_open(self, row):
        if row["intake"] != "open":
            raise JobError("job_intake_closed")
        if row["lease_until_ms"] is None or row["lease_until_ms"] <= self.now():
            raise JobError("execution_lease_expired")
        if row["status"] not in ("running", "waiting", "settling"):
            raise JobError("execution_not_running")

    @staticmethod
    def _check_proof(row, lease):
        if (row["executor_id"], row["boot_id"], row["generation"]) != (
                lease.executor_id, lease.boot_id, lease.generation):
            raise JobError("stale_execution")

    async def _fence(self, tx, actor, lease):
        if not isinstance(lease, JobLease):
            raise JobError("execution_lease_required")
        row = await self._authorize(tx, actor, lease.job_id)
        self._check_proof(row, lease)
        await self._supervision_barriers(tx, row)
        self._execution_open(row)
        from .completion_windows import require_time
        await require_time(self, tx, row)
        from .executor_registry import check_execution
        await check_execution(self, tx, row)
        await self._control_owners(tx, row)
        return row

    async def close_model_execution(self, actor: JobActor, lease: JobLease, *, evidence: Mapping[str, Any] | None = None,
                                    resource_state: str = "stopped") -> JobRecord:
        """Close a send's model phase, retaining its supervision/stop obligations.

        This neither publishes final task success nor releases descendant budget.
        Required children remain in their completion scopes for the B2 gate.
        """
        if resource_state not in ("stopped", "unknown"):
            raise JobError("invalid_resource_state")
        evidence_data = json.loads(_json(dict(evidence or {})))
        async with self._write() as tx:
            row = await self._authorize(tx, actor, lease.job_id)
            if row["kind"] != "send":
                raise JobError("not_model_send")
            self._check_proof(row, lease)
            if row["status"] == "model_closed":
                return _record(row)
            from .model_exit import SNAPSHOT, capture
            if SNAPSHOT in evidence_data or "model_attempt_snapshot_error" in evidence_data:
                raise JobError("reserved_model_exit_evidence")
            try:
                snapshot = await capture(self, tx, row)
                if snapshot is not None:
                    evidence_data[SNAPSHOT] = snapshot
            except JobError as exc:
                evidence_data["model_attempt_snapshot_error"] = exc.code
                resource_state = "unknown"
            # Closing an already executed attempt is evidence, not permission to
            # dispatch more work. Cancellation/lease expiry must not prevent it.
            if row["resource_state"] not in ("running", "unknown"):
                raise JobError("execution_not_running")
            await tx.execute(f"UPDATE {self.jobs} SET status='model_closed',intake='closed',"
                "resource_state=?,lease_until_ms=NULL,version=version+1,updated_at=? WHERE job_id=?",
                (resource_state, self.now(), lease.job_id))
            await tx.execute(f"INSERT INTO {self.events} "
                "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                ("model_closed_" + lease.job_id, lease.job_id, "model_closed", row["version"]+1,
                 _json({"generation": lease.generation, "evidence": evidence_data,
                        "resource_state": resource_state}), "not_applicable", self.now()))
            from .execution_membership import seal
            await seal(self, tx, row)
            return _record(await self._row(tx, lease.job_id))

    async def check_execution(self, actor: JobActor, lease: JobLease) -> None:
        async with self._write() as tx:
            row = await self._fence(tx, actor, lease)
            from .workflow_controller import validate_dispatch
            await validate_dispatch(self, tx, row)

    async def renew_execution(self, actor: JobActor, lease: JobLease, *, lease_ms: int = 30000) -> None:
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 1 <= lease_ms <= 300000:
            raise JobError("invalid_lease_duration")
        async with self._write() as tx:
            await self._fence(tx, actor, lease)
            await tx.execute(f"UPDATE {self.jobs} SET lease_until_ms=?,updated_at=? WHERE job_id=?",
                             (self.now() + lease_ms, self.now(), lease.job_id))

    async def bind_session(self, actor: JobActor, job_id: str, session_id: str) -> JobRecord:
        """Bind an admitted subagent to its actual child before any model call."""
        async with self._write() as tx:
            row = await self._authorize(tx, actor, job_id)
            if row["kind"] != "subagent" or row["created_by_session_id"] != actor.session_id:
                raise JobError("session_ownership_mismatch")
            if row["session_id"] is not None:
                if row["session_id"] != session_id:
                    raise JobError("session_already_bound")
                return _record(row)
            await self._supervision_barriers(tx, row)
            await self._control_owners(tx, row)
            if row["status"] != "queued":
                raise JobError("job_claim_conflict")
            session = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {self.store.t.sessions} WHERE session_id=?", (session_id,))
            if (session is None or session["parent_session_id"] != actor.session_id or
                    json.loads(session["metadata_json"] or "{}").get("execution_role") != "subagent"):
                raise JobError("session_ownership_mismatch")
            await tx.execute(f"UPDATE {self.jobs} SET session_id=?,version=version+1,updated_at=? WHERE job_id=?",
                             (session_id, self.now(), job_id))
            return _record(await self._row(tx, job_id))

    async def bind_workflow_driver(self, actor: JobActor, lease: JobLease, session_id: str) -> JobRecord:
        """Bind the single synthetic driver to an admitted workflow execution."""
        async with self._write() as tx:
            row = await self._fence(tx, actor, lease)
            if row["kind"] != "workflow" or row["created_by_session_id"] != actor.session_id:
                raise JobError("session_ownership_mismatch")
            from .workflow_controller import bind
            await bind(self, tx, row, session_id)
            return _record(await self._row(tx, lease.job_id))

    async def get_workflow_controller(self, actor, job_id, generation):
        from .workflow_controller import get
        return await get(self, actor, job_id, generation)

    def _node_session_event_id(self, job_id, session_id, generation=1):
        from .node_inputs import event_id
        return event_id(self, "session_", job_id, session_id, generation)

    async def bind_node_session(self, actor: JobActor, lease: JobLease, session_id: str) -> None:
        """A node scope may own several fresh retry sessions; each is durable evidence.

        The ephemeral driver is the node's controller, never its model identity.
        Rebinding an old session to another node execution is deliberately refused.
        """
        async with self._write() as tx:
            try:
                row = await self._fence(tx, actor, lease)
            except JobError as exc:
                if exc.code not in ("completion_window_evidence_corrupt", "completion_window_creation_missing"):
                    raise
                # A new derived deadline check must not obscure an original
                # malformed node contract's precise route/binding diagnostic.
                # This read-only probe cannot grant admission; if the original
                # contract is valid, retain the window error unchanged.
                original = await self._authorize(tx, actor, lease.job_id)
                self._check_proof(original, lease)
                from .node_inputs import session_data
                await session_data(self, tx, original, session_id)
                raise
            from .workflow_restore_execution import controls_node, has_authority
            restored = await has_authority(self, tx, row)
            if row["kind"] != "workflow_node" or (restored and not await controls_node(self, tx, row, actor, session_id)) or (
                    not restored and row["created_by_session_id"] != actor.session_id):
                raise JobError("session_ownership_mismatch")
            # A generation bump is not node-resume admission. Keep this closed
            # until the owning engine/dispatch protocol can issue that proof.
            if row["generation"] > 1:
                if restored:
                    from .node_inputs import descriptor
                    await descriptor(self, tx, row, session_id)
                    return
                from .node_resume import validate_execution
                await validate_execution(self, tx, row, session_id, dispatch=True)
                from .workflow_controller import for_node
                await for_node(self, tx, row)
                return
            from .workflow_controller import for_node
            await for_node(self, tx, row)
            from .node_inputs import descriptor, session_data
            data = await session_data(self, tx, row, session_id)
            event_id = self._node_session_event_id(lease.job_id, session_id, row["generation"])
            if await tx.fetchone(f"SELECT event_id FROM {self.events} WHERE event_id=?", (event_id,)):
                await descriptor(self, tx, row, session_id)
                return
            await tx.execute(f"UPDATE {self.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (self.now(), lease.job_id))
            await tx.execute(f"INSERT INTO {self.events} "
                "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                (event_id, lease.job_id, "node_session", row["version"]+1,
                 _json(data), "not_applicable", self.now()))

    async def operation_for_call(self, actor: JobActor, call_ref: MessageRef, tool_call_id: str) -> JobOperation | None:
        async with self._write() as tx:
            await self._validate_call(tx, actor, call_ref, tool_call_id)
            rows = []
            for action in ("create", "nudge", "steer", "stop", "wait_decision"):
                key = self._origin_key(actor, action, call_ref, tool_call_id, None)
                row = await tx.fetchone(f"SELECT * FROM {self.events} WHERE operation_key=?", (key,))
                if row is not None:
                    rows.append(row)
            if not rows:
                return None
            if len(rows) > 1:
                raise JobError("ambiguous_tool_operation")
            [row] = rows
            await self._authorize(tx, actor, row["job_id"])
            return _operation(row)

    async def accept_nudge(self, actor: JobActor, target: JobLease, *, caller: JobLease, call_ref: MessageRef,
                           tool_call_id: str, session_id: str, message: str, node: dict | None = None) -> tuple[JobOperation, bool]:
        from .nudges import accept
        return await accept(self, actor, target, caller=caller, call_ref=call_ref,
                            tool_call_id=tool_call_id, session_id=session_id, message=message, node=node)

    async def resume_subagent(self, actor, job_id, **kwargs):
        from .subagent_resume import admit
        return await admit(self, actor, job_id, **kwargs)

    async def resume_workflow_node(self, actor, job_id, **kwargs):
        from .node_resume import admit
        return await admit(self, actor, job_id, **kwargs)

    async def get_workflow_profile(self, actor, job_id):
        from .workflow_profile import get
        return await get(self, actor, job_id)

    async def get_workflow_restore_plan(self, actor, job_id, **kwargs):
        from .workflow_restore import preview
        return await preview(self, actor, job_id, **kwargs)

    async def claim_resume_dispatch(self, actor, operation_id, **kwargs):
        from .subagent_resume import claim_dispatch
        return await claim_dispatch(self, actor, operation_id, **kwargs)

    async def get_result_revision(self, actor, job_id, generation):
        from .result_revisions import get
        return await get(self, actor, job_id, generation)

    async def get_execution_membership(self, actor, job_id, generation):
        from .execution_membership import get
        return await get(self, actor, job_id, generation)

    async def _validate_ref(self, tx, actor, ref: MessageRef, *, role: str):
        if not isinstance(ref, MessageRef) or ref.store_id != self.store.store_id or ref.id is None:
            raise JobError("invalid_message_ref")
        if ref.session_id != actor.session_id:
            raise JobError("access_denied")
        row = await tx.fetchone(f"SELECT * FROM {self.store.t.messages} WHERE session_id=? AND id=?",
                               (ref.session_id, ref.id))
        if (row is None or int(row["seq"]) != ref.seq or row["role"] != role or ref.role != role or
                row["tool_call_id"] != ref.tool_call_id or row["send_index"] != ref.send_index or
                row["round_index"] != ref.round_index):
            raise JobError("message_ref_mismatch")
        return row

    def _origin_key(self, actor, action, call_ref, tool_call_id, system_key):
        if call_ref is not None:
            if system_key is not None or not tool_call_id or not isinstance(call_ref, MessageRef):
                raise JobError("invalid_operation_origin")
            origin = ["model", call_ref.store_id, actor.session_id, call_ref.seq, tool_call_id]
        else:
            if tool_call_id is not None or not isinstance(system_key, str) or not system_key or len(system_key) > 255:
                raise JobError("persistent_system_key_required")
            origin = ["system", actor.session_id, system_key]
        return _hash([self.store.store_id, actor.tenant_id, actor.principal_id, action, origin])

    async def _validate_call(self, tx, actor, ref, tool_call_id):
        row = await self._validate_ref(tx, actor, ref, role="assistant")
        calls = json.loads(row["tool_calls_json"] or "[]")
        if not any(call.get("id") == tool_call_id for call in calls):
            raise JobError("tool_call_not_found")

    async def _insert_operation(self, tx, *, job_id, version, actor, action, key, fingerprint,
                                call_ref=None, tool_call_id=None, payload=None):
        operation_id = "op_" + secrets.token_hex(16)
        await tx.execute(f"INSERT INTO {self.events} (event_id,job_id,event_type,job_version,operation_key,"
            "request_hash,action,caller_session_id,tool_call_id,call_ref_json,payload_json,delivery_status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (operation_id, job_id, "operation", version, key,
            fingerprint, action, actor.session_id, tool_call_id,
            _json(call_ref.to_dict()) if call_ref else None, _json(payload or {}), "not_applicable", self.now()))
        return _operation(await tx.fetchone(f"SELECT * FROM {self.events} WHERE event_id=?", (operation_id,)))

    async def get_job(self, actor: JobActor, job_id: str, *, include_result: bool = True) -> JobRecord:
        """Read authorized metadata, optionally omitting the mutable result cache.

        Canonical-event readers must remain usable when that projection is
        damaged, and alias resolution does not need to parse its full output.
        """
        if type(include_result) is not bool:
            raise JobError("invalid_job_read_options")
        row = await self._authorize(self.db, actor, job_id)
        return _record(row if include_result else {**row, "result_json": None})

    async def finalize_supervision(self, actor: JobActor, job_id: str, *, generation: int) -> dict:
        from .finalization import finalize
        try:
            return await finalize(self, actor, job_id, generation=generation)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            if isinstance(exc, JobError):
                raise
            raise JobError("finalization_evidence_corrupt") from exc

    async def execution_usage(self, actor: JobActor, job_id: str, *, generation: int | None = None) -> dict:
        """Sum each persisted model-send exit once, never recursive scope rollups.

        This is execution evidence, not billing or a transactional completion gate.
        Unknown/in-flight provider charges remain a separate recovery obligation.
        """
        await self._authorize(self.db, actor, job_id)
        if generation is not None and (isinstance(generation, bool) or not isinstance(generation, int) or generation < 1):
            raise JobError("invalid_result_generation")
        if generation is not None:
            from .execution_membership import usage
            async with self._write() as tx:
                row = await self._authorize(tx, actor, job_id)
                return await usage(self, tx, row, generation)
        frontier, seen, totals = [job_id], set(), {}
        for _ in range(64):
            if not frontier:
                return totals
            following = []
            for offset in range(0, len(frontier), 256):
                batch = frontier[offset:offset+256]
                if seen.intersection(batch):
                    raise JobError("invalid_job_lineage")
                seen.update(batch)
                placeholders = ",".join("?" for _ in batch)
                events = await self.db.fetchall(f"SELECT payload_json FROM {self.events} "
                    f"WHERE job_id IN ({placeholders}) AND event_type='model_closed'", tuple(batch))
                for event in events:
                    usage = json.loads(event["payload_json"]).get("evidence", {}).get("usage", {})
                    for name, value in usage.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            totals[name] = totals.get(name, 0) + value
                rows = await self.db.fetchall(f"SELECT job_id FROM {self.jobs} WHERE parent_job_id IN ({placeholders})", tuple(batch))
                following.extend(row["job_id"] for row in rows)
            frontier = following
        raise JobError("job_depth_exceeded")

    async def get_operation(self, actor: JobActor, operation_id: str) -> JobOperation:
        row = await self.db.fetchone(f"SELECT * FROM {self.events} WHERE event_id=? AND event_type='operation'", (operation_id,))
        if row is None:
            raise JobError("operation_not_found")
        await self._authorize(self.db, actor, row["job_id"])
        return _operation(row)

    async def register_job(self, actor: JobActor, *, kind: str, title: str = "",
            session_id: str | None = None, parent: JobLease | None = None,
            call_ref: MessageRef | None = None, tool_call_id: str | None = None,
            system_key: str | None = None, independent: bool = False,
            request: Mapping[str, Any] | None = None) -> JobRegistration:
        if kind not in KINDS or not isinstance(title, str) or len(title) > 200 or not isinstance(independent, bool):
            raise JobError("invalid_job_spec")
        if parent is None and call_ref is not None:
            raise JobError("model_operation_requires_parent_execution")
        request_data = json.loads(_json(dict(request or {})))
        key = self._origin_key(actor, "create", call_ref, tool_call_id, system_key)
        fingerprint = _hash([kind, title, session_id, parent.job_id if parent else None, independent,
                             request_data, call_ref.to_dict() if call_ref else None])
        async with self._write() as tx:
            old = await tx.fetchone(f"SELECT * FROM {self.events} WHERE operation_key=?", (key,))
            if old:
                if old["request_hash"] != fingerprint:
                    raise JobError("operation_conflict")
                job = await self._authorize(tx, actor, old["job_id"])
                return JobRegistration(_record(job), _operation(old), True)
            parent_row = await self._fence(tx, actor, parent) if parent else None
            restored_parent = False
            if parent_row:
                from .workflow_restore_execution import controls_node, has_authority
                restored_parent = parent_row["kind"] == "workflow_node" and await has_authority(self, tx, parent_row)
                if ((restored_parent and not await controls_node(self, tx, parent_row, actor)) or
                        (not restored_parent and (parent_row["session_id"] or parent_row["created_by_session_id"]) != actor.session_id)):
                    raise JobError("caller_is_not_executing_parent")
            role = parent_row["execution_role"] if parent_row else "main"
            if (kind == "workflow" and role != "main") or (kind == "subagent" and role not in ("main", "workflow_node")):
                raise JobError("role_forbids_delegation")
            if kind == "workflow_node" and (not parent_row or parent_row["kind"] != "workflow"):
                raise JobError("role_forbids_workflow_node")
            if kind == "workflow_node":
                from .workflow_controller import current, has_binding
                if await has_binding(self, tx, parent_row):
                    await current(self, tx, parent_row)
            if independent and kind == "subagent" and role == "workflow_node":
                raise JobError("required_child_cannot_detach")
            role = {"workflow": "workflow", "workflow_node": "workflow_node", "subagent": "subagent"}.get(kind, role)
            if call_ref:
                await self._validate_call(tx, actor, call_ref, tool_call_id)
            if session_id is not None:
                session = await tx.fetchone(f"SELECT * FROM {self.store.t.sessions} WHERE session_id=?", (session_id,))
                granted = restored_parent and await controls_node(self, tx, parent_row, actor, session_id)
                if session is None or (session_id != actor.session_id and session["parent_session_id"] != actor.session_id and not granted):
                    raise JobError("session_ownership_mismatch")
                if kind == "send" and parent_row is not None and parent_row["kind"] != "workflow_node":
                    from .workflow_controller import validate_dispatch
                    await validate_dispatch(self, tx, parent_row)
                if parent_row and parent_row["kind"] == "workflow_node" and kind == "send":
                    bound = await tx.fetchone(f"SELECT job_id FROM {self.events} WHERE event_id=? AND event_type='node_session'",
                        (self._node_session_event_id(parent.job_id, session_id, parent_row["generation"]),))
                    if bound is None or bound["job_id"] != parent.job_id:
                        raise JobError("node_session_admission_required")
                    from .node_inputs import descriptor
                    await descriptor(self, tx, parent_row, session_id)
                    from .workflow_controller import for_node
                    await for_node(self, tx, parent_row)
            if parent is None and session_id != actor.session_id:
                raise JobError("root_session_required")
            job_id = "job_" + secrets.token_hex(16)
            root = parent_row["root_job_id"] if parent_row else job_id
            scope = job_id
            if not independent and parent_row:
                scope = (parent_row["job_id"] if parent_row["kind"] in ("workflow", "workflow_node", "subagent")
                         else parent_row["completion_scope_job_id"])
            owner = None
            if parent_row and not independent:
                # A node model send is only provenance. Its durable node scope
                # remains the owner when that model phase has already closed.
                owner = (await self._row(tx, parent_row["control_owner_job_id"])
                         if parent_row["kind"] == "send" and parent_row["control_owner_job_id"] else parent_row)
            now = self.now()
            wait_binding = None
            if kind == "timer" and "managed_wait" in request_data:
                from .timer_waits import prepare_binding

                wait_binding = await prepare_binding(self, tx, actor, parent_row, request_data,
                    caller=parent, session_id=session_id, independent=independent, call_ref=call_ref,
                    tool_call_id=tool_call_id)
            await tx.execute(f"INSERT INTO {self.jobs} (job_id,parent_job_id,root_job_id,tenant_id,principal_id,"
                "session_id,created_by_session_id,kind,title,completion_scope_job_id,blocks_completion,"
                "status,version,generation,intake,resource_state,supervision_status,result_status,created_at,updated_at,"
                "parent_generation,parent_executor_id,parent_boot_id,execution_role,control_owner_job_id,"
                "control_owner_generation,control_owner_executor_id,control_owner_boot_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (job_id, parent.job_id if parent else None,
                root, actor.tenant_id, actor.principal_id, session_id, actor.session_id, kind, title,
                scope, 0 if independent or parent is None else 1, "queued", 1, 0, "open", "not_started", "open", "none", now, now,
                parent.generation if parent else None, parent.executor_id if parent else None, parent.boot_id if parent else None,
                role, owner["job_id"] if owner else None, owner["generation"] if owner else None,
                owner["executor_id"] if owner else None, owner["boot_id"] if owner else None))
            payload = {"job_id": job_id, "status": "queued"}
            if kind in ("send", "subagent", "workflow", "workflow_node", "timer") and request is not None:
                # Recovery must not reread a potentially edited YAML/file or
                # reconstruct a full execution contract from a request hash.
                payload["request"] = request_data
            if wait_binding is not None:
                payload["wait_binding"] = wait_binding
            operation = await self._insert_operation(tx, job_id=job_id, version=1, actor=actor, action="create",
                key=key, fingerprint=fingerprint, call_ref=call_ref, tool_call_id=tool_call_id,
                payload=payload)
            return JobRegistration(_record(await self._row(tx, job_id)), operation, False)

    async def claim_job(self, actor: JobActor, job_id: str, *, executor_id: str, boot_id: str,
                        expected_version: int, lease_ms: int = 30000) -> JobLease:
        if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
            raise JobError("invalid_executor")
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 1 <= lease_ms <= 300000:
            raise JobError("invalid_lease_duration")
        async with self._write() as tx:
            row = await self._authorize(tx, actor, job_id)
            await self._supervision_barriers(tx, row)
            if row["status"] != "queued" or row["version"] != expected_version:
                raise JobError("job_claim_conflict")
            if row["kind"] == "timer":
                creation = await tx.fetchone(f"SELECT payload_json FROM {self.events} WHERE job_id=? AND event_type='operation' AND action='create'", (job_id,))
                data = json.loads(creation["payload_json"]) if creation else {}
                if "wait_binding" in data or "managed_wait" in data.get("request", {}):
                    from .timer_waits import _read
                    await _read(self, tx, actor, job_id)
                    raise JobError("managed_wait_has_no_executor")
            await self._control_owners(tx, row)
            from .executor_registry import ExecutorIdentity, require_accepting
            inherited = await self._row(tx, row["control_owner_job_id"] or row["parent_job_id"]) if row["parent_job_id"] else None
            executor = await require_accepting(self, tx, ExecutorIdentity(actor.tenant_id, actor.principal_id, executor_id, boot_id),
                required=row["executor_key"] is not None or inherited is not None and inherited["executor_key"] is not None)
            if row["kind"] == "workflow_node":
                workflow = await self._row(tx, row["parent_job_id"])
                from .workflow_controller import for_node, has_binding
                if workflow is not None and (await has_binding(self, tx, workflow) or row["created_by_session_id"] != workflow["created_by_session_id"]):
                    controller = await for_node(self, tx, row)
                    if actor.session_id != controller["session_id"]:
                        raise JobError("session_ownership_mismatch")
            elif row["kind"] in ("send", "subagent") and row["session_id"] is not None:
                from .workflow_controller import validate_dispatch
                await validate_dispatch(self, tx, row)
            generation = int(row["generation"]) + 1
            await tx.execute(f"UPDATE {self.jobs} SET status='running',version=version+1,generation=?,"
                "executor_id=?,boot_id=?,executor_key=?,lease_until_ms=?,resource_state='running',updated_at=? WHERE job_id=?",
                (generation, executor_id, boot_id, executor["executor_key"] if executor else None, self.now() + lease_ms, self.now(), job_id))
            from .execution_membership import start
            await start(self, tx, await self._row(tx, job_id))
            if row["kind"] in ("subagent", "workflow_node"):
                from .completion_windows import admit
                await admit(self, tx, await self._row(tx, job_id))
            return JobLease(job_id, executor_id, boot_id, generation)

    async def register_operation(self, actor: JobActor, lease: JobLease, *, action: str,
            caller: JobLease, call_ref: MessageRef, tool_call_id: str,
            request: Mapping[str, Any], result: Mapping[str, Any]) -> JobOperation:
        if action not in ("nudge", "steer"):
            raise JobError("unsupported_operation")
        request_data, result_data = json.loads(_json(dict(request))), json.loads(_json(dict(result)))
        key = self._origin_key(actor, action, call_ref, tool_call_id, None)
        fingerprint = _hash([lease.job_id, request_data, call_ref.to_dict()])
        async with self._write() as tx:
            old = await tx.fetchone(f"SELECT * FROM {self.events} WHERE operation_key=?", (key,))
            if old:
                await self._authorize(tx, actor, old["job_id"])
                if old["request_hash"] != fingerprint:
                    raise JobError("operation_conflict")
                return _operation(old)
            row = await self._fence(tx, actor, lease)
            caller_row = await self._fence(tx, actor, caller)
            if (caller_row["session_id"] or caller_row["created_by_session_id"]) != actor.session_id:
                raise JobError("caller_is_not_executing_parent")
            if (row["kind"] == "workflow" and caller_row["execution_role"] != "main") or (
                    row["kind"] == "subagent" and caller_row["execution_role"] not in ("main", "workflow_node")):
                raise JobError("role_forbids_delegation")
            await self._validate_call(tx, actor, call_ref, tool_call_id)
            await tx.execute(f"UPDATE {self.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (self.now(), lease.job_id))
            return await self._insert_operation(tx, job_id=lease.job_id, version=int(row["version"]) + 1,
                actor=actor, action=action, key=key, fingerprint=fingerprint,
                call_ref=call_ref, tool_call_id=tool_call_id, payload=result_data)

    async def bind_receipt(self, actor: JobActor, operation_id: str, receipt_ref: MessageRef, *, caller: JobLease) -> JobOperation:
        return await self._bind_receipt(actor, operation_id, receipt_ref, caller=caller)

    async def reconcile_receipt(self, actor: JobActor, operation_id: str, receipt_ref: MessageRef) -> JobOperation:
        """Attach existing raw evidence after caller closure; grants no execution."""
        return await self._bind_receipt(actor, operation_id, receipt_ref)

    async def _bind_receipt(self, actor, operation_id, receipt_ref, *, caller=None):
        async with self._write() as tx:
            op = await tx.fetchone(f"SELECT * FROM {self.events} WHERE event_id=? AND event_type='operation'", (operation_id,))
            if op is None:
                raise JobError("operation_not_found")
            await self._authorize(tx, actor, op["job_id"])
            if caller is not None:
                caller_row = await self._fence(tx, actor, caller)
                if (caller_row["session_id"] or caller_row["created_by_session_id"]) != actor.session_id:
                    raise JobError("caller_is_not_executing_parent")
            row = await self._validate_ref(tx, actor, receipt_ref, role="tool")
            call = json.loads(op["call_ref_json"]) if op["call_ref_json"] else None
            if (not call or op["caller_session_id"] != actor.session_id or
                    row["tool_call_id"] != op["tool_call_id"] or receipt_ref.seq <= int(call["seq"])):
                raise JobError("receipt_call_mismatch")
            later = await tx.fetchall(f"SELECT tool_calls_json FROM {self.store.t.messages} "
                "WHERE session_id=? AND seq>? AND seq<? AND role='assistant'", (actor.session_id, int(call["seq"]), receipt_ref.seq))
            if any(c.get("id") == op["tool_call_id"] for r in later for c in json.loads(r["tool_calls_json"] or "[]")):
                raise JobError("receipt_call_mismatch")
            encoded = _json(receipt_ref.to_dict())
            if op["receipt_ref_json"] and op["receipt_ref_json"] != encoded:
                raise JobError("receipt_already_bound")
            await tx.execute(f"UPDATE {self.events} SET receipt_ref_json=? WHERE event_id=?", (encoded, operation_id))
            return _operation(await tx.fetchone(f"SELECT * FROM {self.events} WHERE event_id=?", (operation_id,)))

    async def record_result(self, actor: JobActor, lease: JobLease, *, outcome: str, result: Mapping[str, Any],
                            resource_state: str = "unknown") -> str:
        return await self._record_result(actor, lease, outcome=outcome, result=result, resource_state=resource_state)

    async def record_execution_result(self, actor: JobActor, lease: JobLease, *, outcome: str,
                                      result: Mapping[str, Any], resource_state: str, require_children_stopped: bool = False) -> str:
        """Record a known attempt even after cancellation/expiry, never restart it.

        Only the current generation's proof is accepted. Reclaiming an unknown
        old generation remains unsupported; stale workers cannot overwrite it.
        """
        return await self._record_result(actor, lease, outcome=outcome, result=result,
                                         resource_state=resource_state, evidence_only=True,
                                         require_children_stopped=require_children_stopped)

    async def prepare_processing(self, actor: JobActor, caller: JobLease, *, round_index: int,
                                 send_index: int, messages: list[dict]) -> str | None:
        from .processing import prepare
        return await prepare(self, actor, caller, round_index=round_index, send_index=send_index, messages=messages)

    async def commit_processing_decision(self, actor: JobActor, caller: JobLease, *, snapshot_id: str,
                                          response_ref: MessageRef, decision: str = "consumed", reason: str) -> list[str]:
        from .processing import commit
        return await commit(self, actor, caller, snapshot_id=snapshot_id, response_ref=response_ref,
                            decision=decision, reason=reason)

    async def recall_processing_decision(self, actor: JobActor, caller: JobLease, *, snapshot_id: str,
                                          response_ref: MessageRef, decision: str = "consumed", reason: str) -> list[str] | None:
        from .processing import recall
        return await recall(self, actor, caller, snapshot_id=snapshot_id, response_ref=response_ref,
                            decision=decision, reason=reason)

    async def _require_scope_ready(self, tx, row):
        from .processing import require_inputs_handled
        await require_inputs_handled(self, tx, row)
        if row["kind"] == "workflow" and row["generation"] > 1:
            from .workflow_restore_execution import require_settled
            from .workflow_resume import execution
            op, _ = await execution(self, tx, row)
            await require_settled(self, tx, row, op, success=True)
        children = await tx.fetchall(f"SELECT * FROM {self.jobs} WHERE control_owner_job_id=?", (row["job_id"],))
        for child in children:
            if child["resource_state"] in ("running", "unknown") or (
                    child["resource_state"] == "not_started" and child["status"] != "cancelled"):
                raise JobError("required_execution_unsettled")
            if row["kind"] != "workflow_node" or child["kind"] != "subagent" or child["status"] == "cancelled":
                continue
            from .processing import require_handled
            await require_handled(self, tx, child)

    async def record_unstarted_result(self, actor: JobActor, job_id: str, *, outcome: str, result: Mapping[str, Any]) -> dict:
        """Close a provably unclaimed admission; an ambiguous claim never qualifies."""
        if outcome not in ("failed", "cancelled", "timed_out"):
            raise JobError("invalid_unstarted_outcome")
        result_data = json.loads(_json(dict(result)))
        event_id = "result_" + job_id
        payload = {"outcome": outcome, "result": result_data, "resource_state": "stopped", "unstarted": True}
        async with self._write() as tx:
            row = await self._authorize(tx, actor, job_id)
            if row["created_by_session_id"] != actor.session_id:
                raise JobError("access_denied")
            old = await tx.fetchone(f"SELECT payload_json FROM {self.events} WHERE event_id=?", (event_id,))
            if old:
                from .result_payloads import require_replay
                require_replay(json.loads(old["payload_json"]), payload)
                return json.loads(row["result_json"])
            if row["generation"] != 0 or row["resource_state"] != "not_started" or row["status"] not in ("queued", "cancelled"):
                raise JobError("unstarted_result_conflict")
            effective = outcome
            ancestor = row
            while ancestor is not None:
                if ancestor["cancel_requested_at"] is not None:
                    effective = "cancelled"
                    break
                ancestor = await self._row(tx, ancestor["parent_job_id"]) if ancestor["parent_job_id"] else None
            saved = {**result_data, "status": effective}
            from .result_payloads import canonical
            payload = canonical(payload, effective, saved)
            await tx.execute(f"UPDATE {self.jobs} SET status='settling',outcome=?,result_json=?,resource_state='stopped',"
                "intake='closed',lease_until_ms=NULL,result_status='pending_delivery',version=version+1,updated_at=? WHERE job_id=?",
                (effective, _json(saved), self.now(), job_id))
            await tx.execute(f"INSERT INTO {self.events} "
                "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                (event_id, job_id, "result", row["version"]+1, _json(payload), "pending_delivery", self.now()))
            return saved

    async def _record_result(self, actor, lease, *, outcome, result, resource_state, evidence_only=False, require_children_stopped=False):
        result = json.loads(_json(dict(result)))
        if outcome not in ("succeeded", "failed", "cancelled", "timed_out", "budget_exceeded", "interrupted"):
            raise JobError("invalid_outcome")
        if resource_state not in ("stopped", "unknown"):
            raise JobError("invalid_resource_state")
        from .result_revisions import result_id
        event_id = result_id(lease.job_id, lease.generation)
        payload = {"outcome": outcome, "result": dict(result), "resource_state": resource_state}
        if lease.generation > 1:
            payload["generation"] = lease.generation
        async with self._write() as tx:
            row = await self._authorize(tx, actor, lease.job_id)
            self._check_proof(row, lease)
            old = await tx.fetchone(f"SELECT * FROM {self.events} WHERE event_id=?", (event_id,))
            if old:
                from .result_payloads import require_replay
                require_replay(json.loads(old["payload_json"]), payload)
                from .execution_membership import seal
                await seal(self, tx, row)
                return event_id
            if evidence_only:
                if row["resource_state"] not in ("running", "unknown"):
                    raise JobError("execution_not_running")
            else:
                await self._fence(tx, actor, lease)
            effective_outcome, effective_result = outcome, dict(result)
            if evidence_only:
                ancestor = row
                while ancestor is not None:
                    if ancestor["cancel_requested_at"] is not None:
                        effective_outcome = "cancelled"
                        if "status" in effective_result and effective_result["status"] != "cancelled":
                            effective_result["execution_status"] = effective_result["status"]
                            effective_result["status"] = "cancelled"
                        break
                    ancestor = await self._row(tx, ancestor["parent_job_id"]) if ancestor["parent_job_id"] else None
            if effective_outcome == "succeeded" and row["executor_key"] is not None:
                from .executor_registry import check_execution
                await check_execution(self, tx, row)
            if effective_outcome == "succeeded":
                from .timer_waits import has_required_wait

                if require_children_stopped or await has_required_wait(self, tx, actor, row):
                    from .wait_effects import settle_owner_tx
                    await settle_owner_tx(self, tx, actor, row)
                    await self._require_scope_ready(tx, row)
            from .result_payloads import canonical
            payload = canonical(payload, effective_outcome, effective_result)
            await tx.execute(f"UPDATE {self.jobs} SET status='settling',outcome=?,result_json=?,"
                "result_status='pending_delivery',resource_state=?,intake='closed',lease_until_ms=NULL,"
                "version=version+1,updated_at=? WHERE job_id=?",
                (effective_outcome, _json(effective_result), resource_state, self.now(), lease.job_id))
            await tx.execute(f"INSERT INTO {self.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) "
                "VALUES (?,?,?,?,?,?,?)", (event_id, lease.job_id, "result", int(row["version"]) + 1, _json(payload), "pending_delivery", self.now()))
            from .execution_membership import seal
            await seal(self, tx, row)
            return event_id

    async def request_cancel(self, actor: JobActor, job_id: str, *, system_key: str, reason: str) -> JobOperation:
        if not isinstance(reason, str) or not reason or len(reason) > 1000:
            raise JobError("invalid_cancel_reason")
        key = self._origin_key(actor, "stop", None, None, system_key)
        fingerprint = _hash([job_id, reason])
        async with self._write() as tx:
            old = await tx.fetchone(f"SELECT * FROM {self.events} WHERE operation_key=?", (key,))
            if old:
                await self._authorize(tx, actor, old["job_id"])
                if old["request_hash"] != fingerprint:
                    raise JobError("operation_conflict")
                return _operation(old)
            row = await self._authorize(tx, actor, job_id)
            if row["cancel_requested_at"] is None:
                # Host scope-stop remains a future admission barrier even on
                # a settled scope. Preserve its verified terminal view; no
                # physical execution is restarted by recording that barrier.
                status = row["status"] if row["supervision_status"] == "closed" else (
                    "cancelled" if row["resource_state"] == "not_started" else "cancelling")
                await tx.execute(f"UPDATE {self.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,"
                    "intake='closed',version=version+1,updated_at=? WHERE job_id=?", (status, self.now(), reason, self.now(), job_id))
                row = await self._row(tx, job_id)
            return await self._insert_operation(tx, job_id=job_id, version=row["version"], actor=actor,
                action="stop", key=key, fingerprint=fingerprint, payload={"job_id": job_id,
                "status": row["status"], "cancel_requested": True, "resource_state": row["resource_state"]})
