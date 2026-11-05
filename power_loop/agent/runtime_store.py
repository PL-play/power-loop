"""Request storage ports. No publisher needs an execution loop or its lease."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

from power_loop.runtime.jobs import JobActor, JobManager

from .runtime_types import (
    InputRef,
    JobRef,
    Receipt,
    RequestOptions,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    Status,
    TargetRef,
    canonical,
    plain,
)


def now_ms():
    return int(time.time() * 1000)


@dataclass(frozen=True)
class InputRecord:
    ref: InputRef
    row: dict
    meta: dict


@dataclass(frozen=True)
class Page:
    items: tuple
    next_cursor: str | None


class RuntimeInputStore:
    def __init__(self, store):
        self.store = store
        self.q = store.t.follow_up_queue
        self.state = store.t.session_state
        self.jobs = store.table_prefix + "jobs"
        self.events = store.table_prefix + "job_events"

    @property
    def db(self):
        return self.store._db

    def check_store(self, ref):
        if ref.store_id != self.store.store_id:
            raise RuntimeRequestError("wrong_store")

    async def authorize(self, reader, target, actor):
        self.check_store(target)
        session = await reader.fetchone(
            f"SELECT metadata_json,parent_session_id FROM {self.store.t.sessions} WHERE session_id=?",
            (target.session_id,))
        if session is None:
            raise RuntimeRequestError("not_found")
        metadata = json.loads(session["metadata_json"])
        if (metadata.get("tenant_id"), metadata.get("principal_id")) != (actor.tenant_id, actor.principal_id):
            raise RuntimeRequestError("forbidden")
        if actor.session_id is not None:
            sid = target.session_id
            parent = session["parent_session_id"]
            while sid != actor.session_id:
                if parent is None:
                    raise RuntimeRequestError("forbidden")
                sid = parent
                ancestor = await reader.fetchone(
                    f"SELECT parent_session_id FROM {self.store.t.sessions} WHERE session_id=?", (sid,))
                if ancestor is None:
                    raise RuntimeRequestError("forbidden")
                parent = ancestor["parent_session_id"]
            if actor.call_ref is None or actor.tool_call_id is None:
                raise RuntimeRequestError("forbidden", "persisted tool call required")
            manager = JobManager(self.store)
            await manager._validate_call(reader, JobActor(actor.tenant_id, actor.principal_id, actor.session_id),
                                         actor.call_ref, actor.tool_call_id)
        if target.job_id is not None:
            job = await reader.fetchone(f"SELECT session_id,tenant_id,principal_id FROM {self.jobs} WHERE job_id=?",
                                        (target.job_id,))
            if job is None:
                raise RuntimeRequestError("not_found")
            if (job["session_id"], job["tenant_id"], job["principal_id"]) != (
                target.session_id, actor.tenant_id, actor.principal_id
            ):
                raise RuntimeRequestError("forbidden")
        return metadata

    async def input(self, reader, ref):
        self.check_store(ref)
        row = await reader.fetchone(f"SELECT * FROM {self.q} WHERE session_id=? AND item_id=?",
                                   (ref.session_id, ref.message_id))
        return self.record(row) if row else None

    def record(self, row):
        return InputRecord(InputRef(self.store.store_id, row["session_id"], row["item_id"]),
                           dict(row), json.loads(row["meta_json"]))

    def receipt(self, record, *, new=False, reason=None):
        row = record.row
        job = (JobRef(self.store.store_id, row["delivered_job_id"], row["delivered_job_generation"])
               if row["delivered_job_id"] else None)
        stored_reason = record.meta.get("reason")
        if isinstance(stored_reason, dict):
            stored_reason = stored_reason["detail"]
        return Receipt(record.ref, TargetRef(self.store.store_id, record.ref.session_id, row["target_job_id"]),
                       record.ref.message_id, new, job,
                       "accepted" if row["status"] in {"pending", "claimed"} else row["status"],
                       "not_needed", reason or stored_reason)

    async def accept_input(self, tx, command, actor, *, descriptor=None):
        """A: conditionally gate intake, freeze once, insert/dedupe the original row."""
        defaults = await self.authorize(tx, command.target, actor)
        changed = await tx.execute(
            f"UPDATE {self.state} SET inbox_version=inbox_version+1 WHERE session_id=? AND lifecycle='open'",
            (command.target.session_id,))
        if not changed:
            raise RuntimeRequestError("session_closed")
        ref = InputRef(command.target.store_id, command.target.session_id, command.message_id)
        old = await self.input(tx, ref)
        frozen_descriptor = (plain(descriptor) if descriptor is not None else
                             old.meta["descriptor"] if old else defaults["runtime_descriptor"])
        definition = RuntimeDescriptor(**frozen_descriptor)
        if definition.kind != "agent":
            raise RuntimeRequestError("feature_unavailable", definition.kind + " is not enabled")
        from power_loop.agent.types import AgentLoopConfig
        config_defaults = AgentLoopConfig()
        base = old.meta["options"] if old else {
            name: definition.config.get(name, getattr(config_defaults, name, value))
            for name, value in plain(RequestOptions()).items()
        }
        options = dict(base)
        for name, value in command.options.items():
            if name not in base:
                raise RuntimeRequestError("invalid_options", name)
            if value is not None:
                options[name] = plain(value)
        options = plain(RequestOptions(**options))
        allowed_models = definition.config.get("allowed_models", ())
        if options["model"] is not None and options["model"] not in allowed_models:
            raise RuntimeRequestError("invalid_options", "model outside frozen profile")
        digest = hashlib.sha256(canonical({
            "operation": command.operation, "target": command.target, "body": command.payload,
            "descriptor": definition, "options": options, "deadline": command.expires_at_ms,
            "generation": command.expected_job_generation, "scope": command.scope,
            "source": actor,
        }).encode()).hexdigest()
        if old:
            if old.meta["request_hash"] != digest:
                raise RuntimeRequestError("duplicate_input_conflict")
            return self.receipt(old)
        if command.expires_at_ms is not None and command.expires_at_ms <= now_ms():
            raise RuntimeRequestError("deadline_expired")
        if command.target.job_id:
            job = await tx.fetchone(f"SELECT generation,status FROM {self.jobs} WHERE job_id=?", (command.target.job_id,))
            if command.expected_job_generation != job["generation"]:
                raise RuntimeRequestError("stale_generation")
            if job["status"] != "running":
                raise RuntimeRequestError("target_not_running" if job["generation"] == 0 else "target_terminal")
        meta = dict(protocol_version=42, operation=command.operation, execution_mode=command.execution_mode,
                    request_id=command.message_id, request_hash=digest, descriptor=plain(definition),
                    options=options, source_job_id=None, scope=command.scope, result_ref=None,
                    interaction=None, reason=None)
        columns = ("session_id", "item_id", "kind", "mode", "status", "content", "meta_json", "created_at",
                   "target_job_id", "expected_job_generation", "expires_at")
        body = plain(command.payload["body"])
        values = (ref.session_id, ref.message_id, "user", "steer" if command.operation == "steer" else "queue",
                  "pending", canonical(body), canonical(meta), now_ms(), command.target.job_id,
                  command.expected_job_generation, command.expires_at_ms)
        inserted = await tx.execute(self.db.dialect.insert_ignore(self.q, columns), values)
        if not inserted:
            existing = await self.input(tx, ref)
            if existing.meta["request_hash"] != digest:
                raise RuntimeRequestError("duplicate_input_conflict")
            return self.receipt(existing)
        row = dict(zip(columns, values, strict=True), delivered_job_id=None, delivered_job_generation=None)
        return self.receipt(InputRecord(ref, row, meta), new=True)

    async def lock_owned(self, tx, target, ownership):
        changed = await tx.execute(
            f"UPDATE {self.state} SET owner_epoch=owner_epoch WHERE session_id=? AND owner_epoch=? AND owner_token=?",
            (target.session_id, ownership.epoch, ownership.lease.token))
        ownership.lease.require_owned()
        if not changed:
            raise RuntimeRequestError("ownership_lost")

    async def claim_inputs(self, tx, target, ownership, limit=100, input_refs=None):
        if not 1 <= limit <= 100 or input_refs is not None and len(input_refs) > 100:
            raise ValueError("claim limit must be in 1..100")
        await self.lock_owned(tx, target, ownership)
        params = [target.session_id]
        condition = ""
        if input_refs is not None:
            if not input_refs:
                return ()
            for ref in input_refs:
                if (ref.store_id, ref.session_id) != (target.store_id, target.session_id):
                    raise RuntimeRequestError("wrong_store")
            condition = f" AND item_id IN ({','.join('?' for _ in input_refs)})"
            params.extend(ref.message_id for ref in input_refs)
        rows = await tx.fetchall(
            f"SELECT * FROM {self.q} WHERE session_id=? AND status='pending'{condition} ORDER BY id LIMIT ?",
            (*params, limit))
        if not rows:
            return ()
        token, timestamp = uuid.uuid4().hex, now_ms()
        ids = [row["id"] for row in rows]
        await tx.execute(
            f"UPDATE {self.q} SET status='claimed',claim_token=?,claimed_at=?,claim_owner_epoch=? "
            f"WHERE status='pending' AND id IN ({','.join('?' for _ in ids)})",
            (token, timestamp, ownership.epoch, *ids))
        return tuple(self.record(dict(row, status="claimed", claim_token=token, claimed_at=timestamp,
                                      claim_owner_epoch=ownership.epoch)) for row in rows)

    async def bind_input_job(self, tx, input_ref, job_ref, ownership, *, claim_token):
        """C: called inside the same transaction that creates/claims the job."""
        await self.lock_owned(tx, TargetRef(input_ref.store_id, input_ref.session_id), ownership)
        changed = await tx.execute(
            f"UPDATE {self.q} SET delivered_job_id=?,delivered_job_generation=? "
            "WHERE session_id=? AND item_id=? AND status='claimed' AND claim_owner_epoch=? AND claim_token=? "
            "AND (expires_at IS NULL OR expires_at>?) "
            "AND (delivered_job_id IS NULL OR (delivered_job_id=? AND delivered_job_generation=0))",
            (job_ref.job_id, job_ref.generation, input_ref.session_id, input_ref.message_id,
             ownership.epoch, claim_token, now_ms(), job_ref.job_id))
        if not changed:
            raise RuntimeRequestError("input_not_startable")

    async def void_unbound(self, tx, ref, *, reason, status=None):
        record = await self.input(tx, ref)
        if record is None:
            raise RuntimeRequestError("not_found")
        meta = dict(record.meta, reason={"status": status or ("cancelled" if reason == "cancelled" else "failed"), "detail": reason})
        changed = await tx.execute(
            f"UPDATE {self.q} SET status='void',meta_json=?,claim_token=NULL,claimed_at=NULL,claim_owner_epoch=NULL "
            "WHERE session_id=? AND item_id=? AND status IN ('pending','claimed') AND delivered_job_id IS NULL",
            (canonical(meta), ref.session_id, ref.message_id))
        if changed:
            record = InputRecord(ref, dict(record.row, status="void"), meta)
        return self.receipt(record)

    async def list_unfinished(self, cursor=None, limit=100):
        if not 1 <= limit <= 100:
            raise ValueError("page limit must be in 1..100")
        rows = await self.db.fetchall(
            f"SELECT id,session_id,item_id FROM {self.q} WHERE status='pending' AND id>? ORDER BY id LIMIT ?",
            (int(cursor or 0), limit))
        return Page(tuple(InputRef(self.store.store_id, r["session_id"], r["item_id"]) for r in rows),
                    str(rows[-1]["id"]) if len(rows) == limit else None)

    async def has_pending(self, sid):
        return await self.db.fetchone(f"SELECT id FROM {self.q} WHERE session_id=? AND status='pending' LIMIT 1", (sid,)) is not None

    async def resolve_job(self, reader, ref, actor):
        self.check_store(ref)
        row = await reader.fetchone(f"SELECT * FROM {self.jobs} WHERE job_id=?", (ref.job_id,))
        if row is None:
            raise RuntimeRequestError("not_found")
        await self.authorize(reader, TargetRef(ref.store_id, row["session_id"], ref.job_id), actor)
        return row

    async def read(self, ref, actor):
        """Resolve once, then read the selected immutable result, including 0/1."""
        input_ref, job_ref = None, ref
        async with self.db.read_snapshot() as reader:
            if isinstance(ref, InputRef):
                input_ref = ref
                await self.authorize(reader, TargetRef(ref.store_id, ref.session_id), actor)
                record = await self.input(reader, ref)
                if record is None:
                    raise RuntimeRequestError("not_found")
                row = record.row
                if row["delivered_job_id"] is None:
                    reason = record.meta.get("reason")
                    terminal = row["status"] == "void"
                    state = reason["status"] if terminal else "accepted"
                    detail = reason["detail"] if terminal else None
                    return (Status(ref, None, state, detail, terminal, state == "cancelled", "not_started"),
                            Result(None, ref, state, error=detail) if terminal else None, ref)
                job_ref = JobRef(ref.store_id, row["delivered_job_id"], row["delivered_job_generation"])
            job = await self.resolve_job(reader, job_ref, actor)
            choice = job["generation"] if job_ref.generation is None else job_ref.generation
            if choice > job["generation"]:
                raise RuntimeRequestError("stale_generation")
            selected = 1 if choice == 0 and job["generation"] > 0 else choice
            fixed = JobRef(ref.store_id, job["job_id"], choice)
            actual = JobRef(ref.store_id, job["job_id"], selected)
            from power_loop.runtime.result_revisions import result_id
            event = await reader.fetchone(f"SELECT payload_json FROM {self.events} WHERE event_id=? AND event_type='result'",
                                          (result_id(job["job_id"], selected),))
            if event:
                payload = json.loads(event["payload_json"])
                if payload["generation"] != selected:
                    raise RuntimeRequestError("not_found")
                value = payload["result"]
                result = Result(actual, input_ref, value["status"], value["final_text"], value["usage"],
                                value["structured_output"], value["error"])
                return (Status(input_ref, actual, result.status, result.error, True,
                               result.status == "cancelled", payload["resource_state"]), result, fixed)
            if selected < job["generation"] or job["supervision_status"] == "closed":
                raise RuntimeRequestError("not_found", "result no longer available")
            state = "accepted" if selected == 0 else "running"
            return (Status(input_ref, actual, state, None, False, job["cancel_requested_at"] is not None,
                           "not_started" if selected == 0 else job["resource_state"]), None, fixed)

    async def get_result(self, ref, actor):
        return (await self.read(ref, actor))[1]

    async def commit_result_and_parent_ref(self, tx, job_ref, result, ownership):
        """F: one original result; a queued failure never invents an execution."""
        route = await self._lock_result_sessions(tx, job_ref)
        return await self._commit_result(tx, job_ref, result, ownership, route)

    async def _lock_result_sessions(self, tx, job_ref):
        """Lock destination states before callers modify the job or its input."""
        self.check_store(job_ref)
        suffix = "" if self.db.dialect.name == "sqlite" else " FOR UPDATE OF state"
        rows = await tx.fetchall(
            f"SELECT state.session_id,job.session_id AS source_session_id,parent.session_id AS parent_session_id "
            f"FROM {self.jobs} job LEFT JOIN {self.jobs} parent ON parent.job_id=job.parent_job_id "
            f"JOIN {self.state} state ON state.session_id IN (job.session_id,parent.session_id) "
            "WHERE job.job_id=? ORDER BY state.session_id" + suffix, (job_ref.job_id,))
        if not rows:
            raise RuntimeRequestError("not_found")
        return rows[0]

    async def _commit_result(self, tx, job_ref, result, ownership, route):
        from power_loop.runtime.result_revisions import result_id
        job = await tx.job_row(job_ref.job_id)
        if job["generation"] != job_ref.generation:
            raise RuntimeRequestError("stale_generation")
        if job_ref.generation > 0:
            await self.lock_owned(tx, TargetRef(job_ref.store_id, job["session_id"]), ownership)
        value = plain(result)
        result_key = result_id(job_ref.job_id, job_ref.generation)
        payload = dict(generation=job_ref.generation,
                       outcome=job["outcome"],
                       resource_state=job["resource_state"] if job_ref.generation else "stopped",
                       result=value)
        if job_ref.generation == 0:
            payload["unstarted"] = True
        inserted = await tx.append_event(("event_id", "job_id", "event_type", "job_version", "payload_json", "delivery_status", "created_at"),
            (result_key, job_ref.job_id, "result", job["version"] + 1,
             canonical(payload), "not_applicable", now_ms()), ignore=True)
        if inserted:
            changed = await tx.write_job(job_ref.job_id,
                f"UPDATE {self.jobs} SET result_json=?,result_status='available',version=version+1 "
                "WHERE job_id=? AND generation=? AND version=?",
                (canonical(value), job_ref.job_id, job_ref.generation, job["version"]))
            if not changed:
                raise RuntimeRequestError("stale_generation")
        parent_sid = route["parent_session_id"]
        if parent_sid is None:
            return None
        item_id = hashlib.sha256(("result:" + result_key).encode()).hexdigest()
        ref = InputRef(job_ref.store_id, parent_sid, item_id)
        result_ref = dict(result_id=result_key, job_id=job_ref.job_id, generation=job_ref.generation)
        meta = dict(protocol_version=42, operation="result_ready", execution_mode="result",
                    request_id=item_id, request_hash=hashlib.sha256(canonical(result_ref).encode()).hexdigest(),
                    descriptor=None, options=None, source_job_id=job_ref.job_id, scope="self",
                    result_ref=result_ref, interaction=None, reason=None)
        columns = ("session_id", "item_id", "kind", "mode", "status", "content", "meta_json", "created_at")
        new_ref = await tx.execute(self.db.dialect.insert_ignore(self.q, columns),
            (parent_sid, item_id, "task_done", "queue", "pending", "", canonical(meta), now_ms()))
        if new_ref:
            await tx.execute(f"UPDATE {self.state} SET inbox_version=inbox_version+1 WHERE session_id=?", (parent_sid,))
        return ref

    async def poll_notifications(self, binding, *, cursor, limit):
        from .runtime_transport import TransportPacket
        if binding.store_id != self.store.store_id:
            raise RuntimeRequestError("wrong_store")
        if binding.channel == "output":
            # Authorization was checked by RuntimeClient before creating this binding.
            mid = binding.message_id or "result-check"
            return (TransportPacket(42, binding.namespace, mid,
                    TargetRef(binding.store_id, binding.session_id), "output",
                    payload={"kind": "check"}),), None
        if binding.channel == "control":
            kind, after = json.loads(cursor) if cursor else ("jobs", "")
            if kind == "jobs":
                rows = await self.db.fetchall(
                    f"SELECT job_id,session_id,generation FROM {self.jobs} WHERE cancel_requested_at IS NOT NULL "
                    "AND resource_state='running' AND job_id>? ORDER BY job_id LIMIT ?", (after, limit))
                packets = tuple(TransportPacket(42, binding.namespace, "cancel:" + r["job_id"],
                    TargetRef(binding.store_id, r["session_id"], r["job_id"]), "control", payload={"generation": r["generation"]}) for r in rows)
                next_cursor = ("jobs", rows[-1]["job_id"]) if len(rows) == limit else ("steer", 0)
            else:
                rows = await self.db.fetchall(f"SELECT id,session_id,item_id,target_job_id FROM {self.q} "
                    "WHERE status IN ('pending','claimed') AND mode='steer' AND id>? ORDER BY id LIMIT ?", (after, limit))
                packets = tuple(TransportPacket(42, binding.namespace, r["item_id"],
                    TargetRef(binding.store_id, r["session_id"], r["target_job_id"]), "control",
                    InputRef(binding.store_id, r["session_id"], r["item_id"])) for r in rows)
                next_cursor = ("steer", rows[-1]["id"]) if len(rows) == limit else ("jobs", "")
            return packets, canonical(next_cursor)
        page = await self.list_unfinished(cursor, limit)
        return tuple(TransportPacket(42, binding.namespace, r.message_id,
                     TargetRef(r.store_id, r.session_id), binding.channel, r) for r in page.items), page.next_cursor
