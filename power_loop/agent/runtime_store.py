"""Session-native accept/claim/final storage. No job or generation reads."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

from power_loop.contracts.errors import SessionLeaseLost

from .runtime_types import (
    PROTOCOL_VERSION,
    CapacityError,
    InputRef,
    MessagePage,
    MessageRow,
    Receipt,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    SessionPage,
    Status,
    StoreError,
    TargetRef,
    canonical,
    plain,
    result_message_id,
    sha256_canonical,
)


def now_ms():
    import time
    return int(time.time() * 1000)


@dataclass(frozen=True)
class InputRecord:
    ref: InputRef
    row: dict
    meta: dict


@dataclass(frozen=True)
class ClaimedBatch:
    target: TargetRef
    send_index: int
    message_seq: int
    input_refs: tuple[InputRef, ...]
    owner_epoch: int
    owner_token: str
    owner_node_id: str
    owner_boot_id: str
    inbox_version: int = 0


@dataclass(frozen=True)
class FinalBatch:
    execution_state: str
    results: tuple[Result, ...]
    result_ref: InputRef | None = None
    result_activation: TargetRef | None = None


class RuntimeInputStore:
    """The only schema-17 execution storage port used by P1."""

    def __init__(self, store, *, inbox_item_limit=100, inbox_byte_limit=1_048_576):
        self.store = store
        self.q = store.t.follow_up_queue
        self.state = store.t.session_state
        self.inbox_item_limit = inbox_item_limit
        self.inbox_byte_limit = inbox_byte_limit

    @property
    def db(self):
        return self.store._db

    def check_store(self, ref):
        if ref.store_id != self.store.store_id:
            raise RuntimeRequestError("wrong_store")

    async def _db_now(self, reader):
        name = self.db.dialect.name
        if name == "sqlite":
            sql = "SELECT CAST(strftime('%s','now') AS INTEGER)*1000 AS now"
        elif name == "postgres":
            sql = "SELECT floor(extract(epoch from clock_timestamp())*1000)::bigint AS now"
        else:
            sql = "SELECT CAST(UNIX_TIMESTAMP(CURRENT_TIMESTAMP(3))*1000 AS SIGNED) AS now"
        return int((await reader.fetchone(sql))["now"])

    async def create_session(
        self, descriptor: RuntimeDescriptor, *, actor, parent=None, result_route=None,
        cancel_propagation="linked", options=None,
    ) -> TargetRef:
        if cancel_propagation not in {"linked", "independent"}:
            raise RuntimeRequestError("invalid_options", "cancel_propagation")
        opts = dict(options or {})
        allowed = {
            "session_id", "creation_call_ref", "creation_tool_call_id", "start_input",
        }
        if set(opts) - allowed:
            raise RuntimeRequestError("invalid_options", "create options")
        parent_id = None
        if parent is not None:
            self.check_store(parent)
            parent_id = parent.session_id
        session_id = opts.get("session_id")
        if session_id is None:
            import secrets
            session_id = "sess_" + secrets.token_hex(12)
        route = self._result_route(parent, result_route, session_id)
        sid = await self.store.create_session(
            descriptor,
            actor=actor,
            parent_session_id=parent_id,
            created_by_session_id=actor.session_id,
            result_route=route,
            cancel_propagation=cancel_propagation,
            session_id=session_id,
            creation_call_ref=opts.get("creation_call_ref"),
            creation_tool_call_id=opts.get("creation_tool_call_id"),
            start_input_message_id=(
                opts["start_input"].message_id if opts.get("start_input") is not None else None
            ),
        )
        return TargetRef(self.store.store_id, sid)

    async def create_child(
        self, descriptor: RuntimeDescriptor, *, parent: TargetRef, actor, call_ref,
        tool_call_id: str, body, cancel_propagation="linked", result_kind="agent_result",
        require_assistant_call=True,
    ):
        """Atomically create the deterministic child and accept its original start input."""
        from power_loop.contracts.errors import CancellationRequested

        self.check_store(parent)
        if actor.session_id != parent.session_id:
            raise RuntimeRequestError("forbidden", "child creation requires the direct parent actor")
        if cancel_propagation not in {"linked", "independent"}:
            raise RuntimeRequestError("invalid_options", "cancel_propagation")
        if result_kind not in {
            "agent_result", "background_result", "managed_effect_result",
            "workflow_result", "workflow_node_result",
        }:
            raise RuntimeRequestError("invalid_options", "result_kind")
        session_id = "ses_" + sha256_canonical([
            "child", parent.store_id, parent.session_id, call_ref.to_dict(), tool_call_id,
        ])[:56]
        start = InputRef(
            parent.store_id, session_id,
            "inp_" + sha256_canonical(["child_start", parent.store_id, session_id])[:56],
        )
        target = TargetRef(parent.store_id, session_id)
        route = self._result_route(
            parent,
            {"target": parent, "input_ref": start, "result_kind": result_kind},
            session_id,
        )
        identity = sha256_canonical([
            parent.store_id, parent.session_id, call_ref.to_dict(), tool_call_id,
            descriptor, start,
        ])
        async with self.db.transaction() as tx:
            parent_session = await self.authorize(tx, parent, actor)
            existing = await tx.fetchone(
                f"SELECT * FROM {self.store.t.sessions} WHERE session_id=?", (session_id,)
            )
            lock_ids = {parent.session_id}
            if existing is not None:
                lock_ids.add(session_id)
            states = {}
            for sid in sorted(lock_ids):
                states[sid] = await self.db.dialect.lock_state(tx, self.state, sid)
            parent_state = states[parent.session_id]
            if parent_state["cancel_send_index"] is not None:
                raise CancellationRequested(parent_state["cancel_reason"] or "cancelled")
            if existing is None:
                if require_assistant_call:
                    call = await tx.fetchone(
                        f"SELECT tool_calls_json FROM {self.store.t.messages} "
                        "WHERE session_id=? AND seq=? AND role='assistant'",
                        (parent.session_id, call_ref.seq),
                    )
                    calls = json.loads(call["tool_calls_json"] or "[]") if call is not None else []
                    if not any(item.get("id") == tool_call_id for item in calls):
                        raise RuntimeRequestError("forbidden", "original assistant tool call not found")
                database_now = await self._db_now(tx)
                await tx.execute(
                    f"INSERT INTO {self.store.t.sessions} ("
                    "session_id,parent_session_id,created_by_session_id,session_kind,descriptor_json,"
                    "descriptor_hash,creation_identity_hash,creation_call_ref_json,creation_tool_call_id,"
                    "start_input_message_id,tenant_id,principal_id,result_route_json,cancel_propagation,"
                    "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session_id, parent.session_id, actor.session_id, descriptor.kind,
                        canonical(descriptor), sha256_canonical(descriptor), identity,
                        canonical(call_ref.to_dict()), tool_call_id, start.message_id,
                        parent_session["tenant_id"], parent_session["principal_id"],
                        canonical(route), cancel_propagation, database_now, database_now,
                    ),
                )
                await tx.execute(
                    f"INSERT INTO {self.state} (session_id) VALUES (?)", (session_id,)
                )
            else:
                expected = {
                    "parent_session_id": parent.session_id,
                    "descriptor_hash": sha256_canonical(descriptor),
                    "creation_identity_hash": identity,
                    "creation_call_ref_json": canonical(call_ref.to_dict()),
                    "creation_tool_call_id": tool_call_id,
                    "start_input_message_id": start.message_id,
                    "result_route_json": canonical(route),
                    "cancel_propagation": cancel_propagation,
                }
                if any(existing[key] != value for key, value in expected.items()):
                    raise RuntimeRequestError("creation_conflict")
            receipt = await self._accept_tx(
                tx, target, body, start, self.request_hash(target, body, "normal"),
                canonical(body), actor=actor, priority="normal", deadline_ms=None,
                kind="user", result_kind=None,
            )
        return target, start, receipt

    async def create_workflow_session(
        self, descriptor: RuntimeDescriptor, *, parent: TargetRef, actor,
        target: TargetRef, start: InputRef, body, result_kind: str,
        cancel_propagation="linked", run_journal=None,
    ):
        """Atomically align/create a deterministic workflow session and its start.

        Node journals are deliberately written by the caller in a second transaction.
        A driver run journal may be included here because its worker must never discover
        a runnable driver without its frozen run coordinates.
        """
        from power_loop.contracts.errors import CancellationRequested

        self.check_store(parent)
        if (target.store_id, start.store_id, start.session_id) != (
            self.store.store_id, self.store.store_id, target.session_id
        ):
            raise RuntimeRequestError("invalid_options", "workflow target/start mismatch")
        if result_kind not in {"workflow_result", "workflow_node_result"}:
            raise RuntimeRequestError("invalid_options", "result_kind")
        route = self._result_route(
            parent, {"target": parent, "input_ref": start, "result_kind": result_kind},
            target.session_id,
        )
        identity = sha256_canonical([
            "workflow_session", parent.store_id, parent.session_id,
            target.session_id, descriptor, start,
        ])
        async with self.db.transaction() as tx:
            await self.authorize(tx, parent, actor)
            parent_state = await self.db.dialect.lock_state(tx, self.state, parent.session_id)
            if parent_state["cancel_send_index"] is not None:
                raise CancellationRequested(parent_state["cancel_reason"] or "cancelled")
            existing = await tx.fetchone(
                f"SELECT * FROM {self.store.t.sessions} WHERE session_id=?",
                (target.session_id,),
            )
            if existing is not None:
                await self.db.dialect.lock_state(tx, self.state, target.session_id)
            if existing is None:
                database_now = await self._db_now(tx)
                await tx.execute(
                    f"INSERT INTO {self.store.t.sessions} ("
                    "session_id,parent_session_id,created_by_session_id,session_kind,descriptor_json,"
                    "descriptor_hash,creation_identity_hash,creation_call_ref_json,creation_tool_call_id,"
                    "start_input_message_id,tenant_id,principal_id,result_route_json,cancel_propagation,"
                    "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        target.session_id, parent.session_id, actor.session_id, descriptor.kind,
                        canonical(descriptor), sha256_canonical(descriptor), identity,
                        canonical(actor.call_ref.to_dict()) if actor.call_ref is not None else None,
                        actor.tool_call_id, start.message_id,
                        actor.tenant_id, actor.principal_id, canonical(route), cancel_propagation,
                        database_now, database_now,
                    ),
                )
                await tx.execute(f"INSERT INTO {self.state} (session_id) VALUES (?)", (target.session_id,))
            else:
                expected = {
                    "parent_session_id": parent.session_id,
                    "descriptor_hash": sha256_canonical(descriptor),
                    "creation_identity_hash": identity,
                    "start_input_message_id": start.message_id,
                    "result_route_json": canonical(route),
                    "cancel_propagation": cancel_propagation,
                }
                if any(existing[key] != value for key, value in expected.items()):
                    raise RuntimeRequestError("creation_conflict")
            receipt = await self._accept_tx(
                tx, target, body, start, self.request_hash(target, body, "normal"),
                canonical(body), actor=actor, priority="normal", deadline_ms=None,
                kind="user", result_kind=None,
            )
            if run_journal is not None and existing is None:
                from power_loop.workflow import journal

                await journal.put(
                    self.store, tx, target.session_id, journal.run_key(start.message_id), run_journal,
                )
        return target, start, receipt

    def _result_route(self, parent, route, child_id):
        if route is None:
            return None
        if parent is None:
            raise RuntimeRequestError("invalid_options", "result route requires a direct parent")
        if set(route) != {"target", "input_ref", "result_kind"}:
            raise RuntimeRequestError("invalid_options", "result route")
        target = route["target"]
        start = route["input_ref"]
        if target != parent or start.store_id != self.store.store_id:
            raise RuntimeRequestError("forbidden", "result route")
        # Validate and freeze the deterministic address before persisting it.
        result_message_id(TargetRef(self.store.store_id, child_id), start, None, route["result_kind"])
        return {
            "target": plain(target), "input_ref": plain(start),
            "result_kind": route["result_kind"],
        }

    async def _session(self, reader, session_id):
        row = await reader.fetchone(
            f"SELECT * FROM {self.store.t.sessions} WHERE session_id=?", (session_id,)
        )
        if row is None:
            raise RuntimeRequestError("not_found")
        return row

    async def authorize(self, reader, target, actor):
        self.check_store(target)
        session = await self._session(reader, target.session_id)
        if (session["tenant_id"], session["principal_id"]) != (
            actor.tenant_id, actor.principal_id
        ):
            raise RuntimeRequestError("forbidden")
        if actor.session_id is not None:
            sid, expected = target.session_id, actor.session_id
            ancestor = session
            while sid != expected:
                if ancestor["parent_session_id"] is None:
                    raise RuntimeRequestError("forbidden")
                sid = ancestor["parent_session_id"]
                ancestor = await self._session(reader, sid)
        return session

    def record(self, row):
        return InputRecord(
            InputRef(self.store.store_id, row["session_id"], row["item_id"]),
            dict(row), json.loads(row["meta_json"]),
        )

    async def input(self, reader, ref):
        self.check_store(ref)
        row = await reader.fetchone(
            f"SELECT * FROM {self.q} WHERE session_id=? AND item_id=?",
            (ref.session_id, ref.message_id),
        )
        return self.record(row) if row else None

    def receipt(self, record, *, new=False, state=None, reason=None):
        row = record.row
        if state is None:
            state = "accepted"
        return Receipt(
            record.ref, TargetRef(self.store.store_id, record.ref.session_id),
            record.ref.message_id, new, state,
            row["result_send_index"] or row["delivered_send_index"],
            "not_needed", reason if reason is not None else row["error_detail"],
        )

    @staticmethod
    def request_hash(target, body, priority):
        return sha256_canonical([
            PROTOCOL_VERSION, target.store_id, target.session_id, body, priority,
        ])

    async def accept(
        self, target, body, *, message_id, actor, priority="normal", deadline_ms=None,
        kind="user", result_kind=None, _with_activation=False,
    ) -> Receipt:
        if priority not in {"normal", "urgent"}:
            raise RuntimeRequestError("invalid_options", "priority")
        self.check_store(target)
        ref = InputRef(target.store_id, target.session_id, message_id)
        digest = self.request_hash(target, body, priority)
        # Validate JSON before opening the write transaction.
        content = canonical(body)
        activations = []
        try:
            async with self.db.transaction() as tx:
                receipt = await self._accept_tx(
                    tx, target, body, ref, digest, content, actor=actor, priority=priority,
                    deadline_ms=deadline_ms, kind=kind, result_kind=result_kind,
                    activations=activations,
                )
            return (receipt, bool(activations)) if _with_activation else receipt
        except RuntimeRequestError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            raise StoreError(ref) from exc

    async def _accept_tx(
        self, tx, target, body, ref, digest, content, *, actor, priority, deadline_ms,
        kind, result_kind, activations=None,
    ):
        try:
            state = await self.db.dialect.lock_state(tx, self.state, target.session_id)
        except ValueError as exc:
            raise RuntimeRequestError("not_found") from exc
        session = await self.authorize(tx, target, actor)
        if session["lifecycle"] != "open":
            raise RuntimeRequestError("session_closed")
        old = await self.input(tx, ref)
        if old is not None:
            if old.row["request_hash"] != digest:
                raise RuntimeRequestError("request_conflict")
            return self.receipt(old)
        count = await tx.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.q} "
            "WHERE session_id=? AND status IN ('pending','waiting_continuation')",
            (target.session_id,),
        )
        if int(count["n"]) >= self.inbox_item_limit:
            raise CapacityError()
        if self.db.dialect.name == "sqlite":
            size_sql = (
                "SELECT COALESCE(SUM(LENGTH(CAST(content_json AS BLOB))"
                "+LENGTH(CAST(meta_json AS BLOB))),0) AS n FROM " + self.q + " "
                "WHERE session_id=? AND status IN ('pending','waiting_continuation')"
            )
        else:
            size_sql = (
                "SELECT COALESCE(SUM(OCTET_LENGTH(content_json)+OCTET_LENGTH(meta_json)),0) AS n FROM "
                + self.q + " WHERE session_id=? AND status IN ('pending','waiting_continuation')"
            )
        size = int((await tx.fetchone(size_sql, (target.session_id,)))["n"])
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        if size + len(content.encode()) + len(meta.encode()) > self.inbox_byte_limit:
            raise CapacityError()
        database_now = await self._db_now(tx)
        if deadline_ms is not None and deadline_ms <= database_now:
            raise RuntimeRequestError("deadline_expired")
        execution_state = state["execution_state"]
        if execution_state == "idle":
            next_state = "queued"
        elif execution_state == "running" and int(state["owner_lease_until_ms"]) <= database_now:
            next_state = "recovering"
        else:
            next_state = execution_state
        await tx.execute(
            f"INSERT INTO {self.q} (session_id,item_id,kind,priority,status,content_json,"
            "meta_json,request_hash,result_kind,created_at,expires_at) "
            "VALUES (?,?,?,?,'pending',?,?,?,?,?,?)",
            (
                target.session_id, ref.message_id, kind, priority, content, meta, digest,
                result_kind, database_now, deadline_ms,
            ),
        )
        changed = await tx.execute(
            f"UPDATE {self.state} SET execution_state=?,inbox_version=inbox_version+1 "
            "WHERE session_id=? AND execution_state=?",
            (next_state, target.session_id, execution_state),
        )
        if changed != 1:
            raise SessionLeaseLost(target.session_id)
        row = {
            "session_id": target.session_id, "item_id": ref.message_id, "status": "pending",
            "delivered_send_index": None, "result_send_index": None, "error_detail": None,
        }
        if activations is not None and next_state != "running":
            activations.append(target)
        return self.receipt(InputRecord(ref, row, {}), new=True)

    async def accept_result(self, child_target, result: Result, *, result_kind):
        """Trusted delivery using only the child's frozen direct-parent route."""
        self.check_store(child_target)
        async with self.db.transaction() as tx:
            session = await self._session(tx, child_target.session_id)
            if not session["result_route_json"]:
                return None
            route = json.loads(session["result_route_json"])
            target = TargetRef(**route["target"])
            for sid in sorted({child_target.session_id, target.session_id}):
                await self.db.dialect.lock_state(tx, self.state, sid)
            activations = []
            ref, error = await self._route_result_tx(
                tx, session, route, result, result_kind, await self._db_now(tx), activations,
            )
            return ref, error, activations[0] if activations else None

    async def claim(self, target, *, node_id, boot_id, ttl_s=30.0, _owner_token=None) -> ClaimedBatch | None:
        self.check_store(target)
        if ttl_s <= 0:
            raise RuntimeRequestError("invalid_options", "ttl_s")
        context = {}
        try:
            async with self.db.transaction() as tx:
                state = await self.db.dialect.lock_state(tx, self.state, target.session_id)
                if state["execution_state"] != "queued":
                    return None
                # Repair any unresolved tool calls from an interrupted prior run BEFORE
                # the new merged user message, stamped with that run's send_index.
                repaired = await self._repair_pending_tx(
                    tx, target.session_id, int(state["send_index"]), state)
                snapshot = await tx.fetchone(
                    f"SELECT COALESCE(MAX(id),0) AS id FROM {self.q} "
                    "WHERE session_id=? AND status='pending'", (target.session_id,),
                )
                if not int(snapshot["id"]):
                    return None
                rows = await tx.fetchall(
                    f"SELECT * FROM {self.q} WHERE session_id=? AND status='pending' AND id<=? "
                    "ORDER BY CASE priority WHEN 'urgent' THEN 0 ELSE 1 END,id",
                    (target.session_id, snapshot["id"]),
                )
                refs = tuple(InputRef(target.store_id, target.session_id, row["item_id"]) for row in rows)
                context["refs"] = refs
                token = _owner_token or uuid.uuid4().hex
                send_index = int(state["send_index"]) + 1
                seq = int(state["next_seq"]) + repaired
                epoch = int(state["owner_epoch"]) + 1
                database_now = await self._db_now(tx)
                deadline = database_now + int(ttl_s * 1000)
                changed = await tx.execute(
                    f"UPDATE {self.state} SET execution_state='running',send_index=?,"
                    "active_send_index=?,next_seq=next_seq+1,owner_epoch=?,owner_token=?,"
                    "owner_node_id=?,owner_boot_id=?,owner_lease_until_ms=?,"
                    "cancel_send_index=NULL,cancel_reason=NULL "
                    "WHERE session_id=? AND execution_state='queued' AND send_index=? AND next_seq=? "
                    "AND active_send_index IS NULL AND owner_token IS NULL",
                    (
                        send_index, send_index, epoch, token, node_id, boot_id, deadline,
                        target.session_id, state["send_index"], seq,
                    ),
                )
                if changed != 1:
                    return None
                merged = self.merge_batch(rows)
                message_id = await self._insert_message(
                    tx, target.session_id, seq, "user", canonical(merged),
                    {"content_encoding": "json", "input_refs": [plain(ref) for ref in refs]},
                    send_index, database_now,
                )
                ids = [int(row["id"]) for row in rows]
                delivered = await tx.execute(
                    f"UPDATE {self.q} SET status='delivered',delivered_at=?,delivered_seq=?,"
                    f"delivered_send_index=? WHERE session_id=? AND status='pending' "
                    f"AND id IN ({','.join('?' for _ in ids)})",
                    (database_now, seq, send_index, target.session_id, *ids),
                )
                if delivered != len(ids):
                    raise SessionLeaseLost(target.session_id)
                batch = ClaimedBatch(
                    target, send_index, seq, refs, epoch, token, node_id, boot_id,
                    int(state["inbox_version"]),
                )
                context.update(batch=batch, message_id=message_id)
                return batch
        except (Exception, asyncio.CancelledError) as exc:
            recovered = await self._claim_after_unknown(target, context)
            if recovered is not None:
                return recovered
            ref = context.get("refs", (None,))[0]
            raise StoreError(ref) from exc

    async def release_claim(self, batch: ClaimedBatch):
        """Atomically release an unexecuted claim, honouring active cancellation."""
        self.check_store(batch.target)
        ids = [ref.message_id for ref in batch.input_refs]
        async with self.db.transaction() as tx:
            session = await self._session(tx, batch.target.session_id)
            route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
            lock_ids = {batch.target.session_id}
            if route is not None:
                lock_ids.add(route["target"]["session_id"])
            states = {}
            for sid in sorted(lock_ids):
                states[sid] = await self.db.dialect.lock_state(tx, self.state, sid)
            state = states[batch.target.session_id]
            if (
                state["execution_state"] != "running"
                or state["active_send_index"] != batch.send_index
                or state["owner_epoch"] != batch.owner_epoch
                or state["owner_token"] != batch.owner_token
            ):
                raise SessionLeaseLost(batch.target.session_id)
            if state["cancel_send_index"] == batch.send_index:
                # Reuse final's batch cancellation and trusted result routing. The merged
                # user row is removed because no model execution consumed this batch.
                await self._final_tx(
                    tx, batch, batch.input_refs, "cancelled", "", {}, None,
                    state["cancel_reason"] or "cancelled", "cancelled",
                )
            else:
                restored = await tx.execute(
                    f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                    "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                    f"AND delivered_send_index=? AND item_id IN ({','.join('?' for _ in ids)})",
                    (batch.target.session_id, batch.send_index, *ids),
                )
                if restored != len(ids):
                    raise SessionLeaseLost(batch.target.session_id)
                changed = await tx.execute(
                    f"UPDATE {self.state} SET execution_state='queued',active_send_index=NULL,"
                    "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                    "cancel_send_index=NULL,cancel_reason=NULL "
                    "WHERE session_id=? AND execution_state='running' AND active_send_index=? "
                    "AND owner_epoch=? AND owner_token=?",
                    (batch.target.session_id, batch.send_index, batch.owner_epoch, batch.owner_token),
                )
                if changed != 1:
                    raise SessionLeaseLost(batch.target.session_id)
            await tx.execute(
                f"DELETE FROM {self.store.t.messages} WHERE session_id=? AND seq=? "
                "AND role='user' AND send_index=?",
                (batch.target.session_id, batch.message_seq, batch.send_index),
            )

    @staticmethod
    def merge_batch(rows):
        from .follow_up import session_batch_content
        return session_batch_content(rows)

    async def boundary(self, target):
        """One current-version read at a loop boundary (design 215 §7.4).

        Returns the live ``session_state`` control fields so the caller can detect new
        inbox input, a matching cancellation, or a lost owner fence. This is the fixed
        per-boundary control cost; the inbox batch is only read when the version moved.
        """
        self.check_store(target)
        return await self.db.fetchone(
            f"SELECT execution_state,active_send_index,owner_epoch,owner_token,"
            f"inbox_version,cancel_send_index,cancel_reason FROM {self.state} WHERE session_id=?",
            (target.session_id,),
        )

    async def claim_follow_up(self, batch: ClaimedBatch):
        """Atomically claim the pending inbox snapshot into the RUNNING send.

        Same atomic boundary as :meth:`claim` but fenced on the live owner: lock the
        state, verify running + active_send_index + epoch/token, snapshot MAX(id), write
        one merged user message and bind every input to this send_index. Returns
        ``(refs, seq, merged)`` or ``None`` when nothing is pending or the fence moved.
        """
        self.check_store(batch.target)
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.state, batch.target.session_id)
            if (
                state["execution_state"] != "running"
                or state["active_send_index"] != batch.send_index
                or state["owner_epoch"] != batch.owner_epoch
                or state["owner_token"] != batch.owner_token
            ):
                raise SessionLeaseLost(batch.target.session_id)
            snapshot = await tx.fetchone(
                f"SELECT COALESCE(MAX(id),0) AS id FROM {self.q} "
                "WHERE session_id=? AND status='pending'", (batch.target.session_id,),
            )
            if not int(snapshot["id"]):
                return None
            rows = await tx.fetchall(
                f"SELECT * FROM {self.q} WHERE session_id=? AND status='pending' AND id<=? "
                "ORDER BY CASE priority WHEN 'urgent' THEN 0 ELSE 1 END,id",
                (batch.target.session_id, snapshot["id"]),
            )
            refs = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in rows
            )
            seq = int(state["next_seq"])
            database_now = await self._db_now(tx)
            merged = self.merge_batch(rows)
            changed = await tx.execute(
                f"UPDATE {self.state} SET next_seq=next_seq+1,inbox_version=inbox_version "
                "WHERE session_id=? AND execution_state='running' AND active_send_index=? "
                "AND owner_epoch=? AND owner_token=? AND next_seq=?",
                (batch.target.session_id, batch.send_index, batch.owner_epoch,
                 batch.owner_token, seq),
            )
            if changed != 1:
                raise SessionLeaseLost(batch.target.session_id)
            await self._insert_message(
                tx, batch.target.session_id, seq, "user", canonical(merged),
                {"content_encoding": "json", "input_refs": [plain(ref) for ref in refs]},
                batch.send_index, database_now,
            )
            ids = [int(row["id"]) for row in rows]
            delivered = await tx.execute(
                f"UPDATE {self.q} SET status='delivered',delivered_at=?,delivered_seq=?,"
                f"delivered_send_index=? WHERE session_id=? AND status='pending' "
                f"AND id IN ({','.join('?' for _ in ids)})",
                (database_now, seq, batch.send_index, batch.target.session_id, *ids),
            )
            if delivered != len(ids):
                raise SessionLeaseLost(batch.target.session_id)
            version = int(state["inbox_version"])
        # The merged follow-up row consumed a seq outside SessionStore.append_message;
        # advance the live send's in-memory cursor so its next conditional append matches
        # (otherwise the stale next_seq fence would reject the loop's own next write).
        from power_loop.runtime.store import send_reads
        holding = send_reads.current(self.store, batch.target.session_id)
        if holding is not None:
            holding.next_seq = seq + 1
            holding.change_snapshot = None
        return refs, seq, merged, version

    async def _repair_pending_tx(self, tx, session_id, send_index, state):
        """Synthesize unknown tool results for unresolved calls (design 215 §8.3).

        Runs inside the caller's fenced transaction. Only ids without a stored result
        get a fixed interrupted marker; existing results are never duplicated and no
        unknown tool is replayed. The synthetic rows carry the interrupted run's
        send_index. Returns the number of rows appended (so the caller can advance its
        seq cursor). pending_json is cleared in the same transaction.
        """
        pending = json.loads(state["pending_json"]) if state["pending_json"] else None
        if not pending:
            return 0
        tool_calls = pending.get("tool_calls") or [
            {"id": cid} for cid in pending.get("tool_call_ids", [])
        ]
        if not tool_calls:
            await tx.execute(
                f"UPDATE {self.state} SET pending_json=NULL WHERE session_id=?", (session_id,)
            )
            return 0
        assistant_seq = pending.get("assistant_seq")
        round_index = pending.get("round_index")
        replied = set()
        if assistant_seq is not None:
            rows = await tx.fetchall(
                f"SELECT tool_call_id FROM {self.store.t.messages} WHERE session_id=? "
                "AND role='tool' AND send_index=? AND seq>? AND tool_call_id IS NOT NULL",
                (session_id, send_index, assistant_seq),
            )
            replied = {row["tool_call_id"] for row in rows}
        names = {str(tc.get("id") or ""): (tc.get("function", {}) or {}).get("name")
                 if "function" in tc else tc.get("name") for tc in tool_calls}
        call_ref = None
        if assistant_seq is not None:
            from power_loop.runtime.message_ref import MessageRef
            assistant = await tx.fetchone(
                f"SELECT id,seq,role,tool_call_id,send_index,round_index FROM {self.store.t.messages} "
                "WHERE session_id=? AND seq=? AND role='assistant'",
                (session_id, assistant_seq),
            )
            if assistant is not None:
                call_ref = MessageRef(
                    self.store.store_id, session_id, assistant["id"], assistant["seq"],
                    assistant["role"], assistant["tool_call_id"], assistant["send_index"],
                    assistant["round_index"],
                )
        seq = int(state["next_seq"])
        database_now = await self._db_now(tx)
        appended = 0
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            if not cid or cid in replied:
                continue
            content = "<aborted: 执行中断，结果未知>"
            meta = {"interrupted": True}
            if call_ref is not None:
                child = await tx.fetchone(
                    f"SELECT session_id,start_input_message_id FROM {self.store.t.sessions} "
                    "WHERE parent_session_id=? AND creation_call_ref_json=? AND creation_tool_call_id=?",
                    (session_id, canonical(call_ref.to_dict()), cid),
                )
                if child is not None and child["start_input_message_id"] is not None:
                    child_target = TargetRef(self.store.store_id, child["session_id"])
                    start = InputRef(self.store.store_id, child["session_id"], child["start_input_message_id"])
                    content = canonical({
                        "creation_result_uncertain": True,
                        "target_ref": plain(child_target),
                        "input_ref": plain(start),
                        "status": "accepted",
                    })
                    meta["creation_result_uncertain"] = True
            await tx.execute(
                f"INSERT INTO {self.store.t.messages} (session_id,seq,role,name,content,"
                "tool_call_id,round_index,state,meta_json,send_index,created_at) "
                "VALUES (?,?, 'tool', ?, ?, ?, ?, 'active', ?, ?, ?)",
                (session_id, seq, names.get(cid), content, cid,
                 round_index, canonical(meta), send_index, database_now),
            )
            seq += 1
            appended += 1
        await tx.execute(
            f"UPDATE {self.state} SET pending_json=NULL,next_seq=? WHERE session_id=?",
            (seq, session_id),
        )
        return appended

    async def _insert_message(self, tx, session_id, seq, role, content, meta, send_index, created_at):
        sql = (
            f"INSERT INTO {self.store.t.messages} (session_id,seq,role,content,meta_json,"
            "send_index,created_at) VALUES (?,?,?,?,?,?,?)"
        )
        params = (session_id, seq, role, content, canonical(meta), send_index, created_at)
        if self.db.dialect.name == "mysql":
            await tx.execute(sql, params)
            return int((await tx.fetchone("SELECT LAST_INSERT_ID() AS id"))["id"])
        row = await tx.fetchone(sql + " RETURNING id", params)
        return int(row["id"])

    async def _claim_after_unknown(self, target, context):
        if "batch" not in context:
            return None
        batch = context["batch"]
        state = await self.db.fetchone(
            f"SELECT * FROM {self.state} WHERE session_id=?", (target.session_id,)
        )
        if not state or state["active_send_index"] != batch.send_index:
            return None
        if (
            state["owner_epoch"], state["owner_token"], state["execution_state"]
        ) != (batch.owner_epoch, batch.owner_token, "running"):
            return None
        ids = [ref.message_id for ref in batch.input_refs]
        rows = await self.db.fetchall(
            f"SELECT * FROM {self.q} WHERE session_id=? "
            f"AND item_id IN ({','.join('?' for _ in ids)})",
            (target.session_id, *ids),
        )
        if len(rows) != len(ids) or any(
            row["status"] != "delivered" or row["delivered_send_index"] != batch.send_index
            or row["delivered_seq"] != batch.message_seq for row in rows
        ):
            return None
        return batch

    async def final(
        self, batch: ClaimedBatch, *, consumed=None, status="completed", final_text="",
        usage=None, structured_output=None, error=None, error_code=None, result_seq=None,
    ) -> FinalBatch:
        # consumed=None means "every input delivered to this send" — including follow-ups
        # claimed at loop boundaries, which share send_index but are not in the original
        # ClaimedBatch.input_refs (design 215 §3.4/§7.4). An explicit consumed narrows it.
        if consumed is not None:
            consumed = tuple(consumed)
            if any(ref.session_id != batch.target.session_id for ref in consumed):
                raise RuntimeRequestError("invalid_options", "consumed input outside this session")
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise RuntimeRequestError("invalid_options", "terminal status")
        try:
            async with self.db.transaction() as tx:
                return await self._final_tx(
                    tx, batch, consumed, status, final_text, usage or {},
                    structured_output, error, error_code, result_seq,
                )
        except (RuntimeRequestError, SessionLeaseLost):
            raise
        except (Exception, asyncio.CancelledError) as exc:
            recovered = await self._final_after_unknown(batch, consumed)
            if recovered is not None:
                return recovered
            raise StoreError(consumed[0] if consumed else batch.input_refs[0]) from exc

    async def _final_tx(
        self, tx, batch, consumed, status, final_text, usage, structured_output,
        error, error_code, result_seq=None,
    ):
        session = await self._session(tx, batch.target.session_id)
        route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
        lock_ids = [batch.target.session_id]
        if route is not None:
            lock_ids.append(route["target"]["session_id"])
        states = {}
        for sid in sorted(set(lock_ids)):
            states[sid] = await self.db.dialect.lock_state(tx, self.state, sid)
        state = states[batch.target.session_id]
        if state["cancel_send_index"] == batch.send_index:
            status, error = "cancelled", state["cancel_reason"] or "cancelled"
        if consumed is None:
            delivered = await tx.fetchall(
                f"SELECT item_id FROM {self.q} WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL",
                (batch.target.session_id, batch.send_index),
            )
            consumed = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in delivered
            )
        consumed_ids = [ref.message_id for ref in consumed]
        unconsumed_ids = [ref.message_id for ref in batch.input_refs if ref not in consumed]
        result_ref, delivery_error = None, None
        activations = []
        database_now = await self._db_now(tx)
        # The running loop appends assistant turns in real time and passes the seq of its
        # LAST assistant row as ``result_seq``: that row IS the result message (design 215
        # §5.3), so final stamps its meta instead of writing a duplicate. A storage-only
        # final (no loop, result_seq=None) inserts the result message itself; a failure with
        # no assistant row leaves result_seq NULL and is expressed by the input terminal.
        write_message = result_seq is None and status == "completed"
        if write_message:
            result_seq = int(state["next_seq"])
        reuse_message = result_seq is not None and not write_message
        if consumed and route is not None:
            route_result = Result(
                consumed[0], batch.target, status, batch.send_index, result_seq,
                final_text, usage, structured_output, error, error_code,
            )
            result_ref, delivery_error = await self._route_result_tx(
                tx, session, route, route_result, route["result_kind"], database_now,
                activations,
            )
        next_seq = int(state["next_seq"]) + int(write_message)
        next_state = "idle"
        pending = await tx.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? AND status='pending'",
            (batch.target.session_id,),
        )
        has_pending = bool(int(pending["n"])) or bool(unconsumed_ids)
        next_state = "queued" if has_pending else "idle"
        changed = await tx.execute(
            f"UPDATE {self.state} SET execution_state=?,active_send_index=NULL,"
            "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
            "cancel_send_index=NULL,cancel_reason=NULL,next_seq=? "
            "WHERE session_id=? AND execution_state='running' AND active_send_index=? "
            "AND owner_epoch=? AND owner_token=?",
            (
                next_state, next_seq, batch.target.session_id, batch.send_index,
                batch.owner_epoch, batch.owner_token,
            ),
        )
        if changed != 1:
            raise SessionLeaseLost(batch.target.session_id)
        result_meta = {
            "usage": plain(usage), "structured_output": plain(structured_output),
            "error": error, "error_code": error_code,
            "result_delivery_error": delivery_error,
        }
        if write_message:
            await self._insert_message(
                tx, batch.target.session_id, result_seq, "assistant", final_text,
                result_meta, batch.send_index, database_now,
            )
        elif reuse_message:
            # Merge into the loop's assistant row, preserving reasoning/round-usage meta.
            row = await tx.fetchone(
                f"SELECT meta_json FROM {self.store.t.messages} WHERE session_id=? AND seq=?",
                (batch.target.session_id, result_seq),
            )
            meta = json.loads(row["meta_json"] or "{}") if row is not None else {}
            meta.update(result_meta)
            await tx.execute(
                f"UPDATE {self.store.t.messages} SET meta_json=? WHERE session_id=? AND seq=?",
                (canonical(meta), batch.target.session_id, result_seq),
            )
        if unconsumed_ids:
            await tx.execute(
                f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                f"AND delivered_send_index=? AND item_id IN ({','.join('?' for _ in unconsumed_ids)})",
                (batch.target.session_id, batch.send_index, *unconsumed_ids),
            )
        if consumed_ids:
            changed = await tx.execute(
                f"UPDATE {self.q} SET terminal_status=?,result_seq=?,result_send_index=?,"
                "error_code=?,error_detail=?,result_delivery_error=?,completed_at=? "
                "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                f"AND item_id IN ({','.join('?' for _ in consumed_ids)})",
                (
                    status, result_seq, batch.send_index, error_code, error, delivery_error,
                    database_now, batch.target.session_id, batch.send_index, *consumed_ids,
                ),
            )
            if changed != len(consumed_ids):
                raise SessionLeaseLost(batch.target.session_id)
        results = tuple(
            Result(
                ref, batch.target, status, batch.send_index, result_seq, final_text,
                usage, structured_output, error, error_code, delivery_error,
            )
            for ref in consumed
        )
        return FinalBatch(
            next_state, results, result_ref,
            activations[0] if activations else None,
        )

    async def _route_result_tx(
        self, tx, child_session, route, result, result_kind, database_now, activations=None,
    ):
        child_target = TargetRef(self.store.store_id, child_session["session_id"])
        route_start = InputRef(**route["input_ref"])
        start = result.input_ref if result.send_index is None else route_start
        target = TargetRef(**route["target"])
        message_id = result_message_id(child_target, start, result.send_index, result_kind)
        body = {
            "child_target": plain(child_target), "input_ref": plain(result.input_ref),
            "status": result.status, "final_text": result.final_text,
            "usage": plain(result.usage), "structured_output": plain(result.structured_output),
            "error": result.error, "error_code": result.error_code,
            "result_seq": result.result_seq,
        }
        ref = InputRef(target.store_id, target.session_id, message_id)
        try:
            parent_session = await self._session(tx, target.session_id)
            if (
                parent_session["lifecycle"] != "open"
                or parent_session["tenant_id"] != child_session["tenant_id"]
                or parent_session["principal_id"] != child_session["principal_id"]
                or child_session["parent_session_id"] != target.session_id
                or target.store_id != self.store.store_id
            ):
                raise RuntimeRequestError("session_closed" if parent_session["lifecycle"] != "open" else "forbidden")
            digest = self.request_hash(target, body, "normal")
            old = await self.input(tx, ref)
            parent_state = await tx.fetchone(
                f"SELECT * FROM {self.state} WHERE session_id=?", (target.session_id,)
            )
            execution_state = parent_state["execution_state"]
            if old is None:
                count = await tx.fetchone(
                    f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? "
                    "AND status IN ('pending','waiting_continuation')", (target.session_id,),
                )
                if int(count["n"]) >= self.inbox_item_limit:
                    raise CapacityError()
                if self.db.dialect.name == "sqlite":
                    size_sql = (
                        "SELECT COALESCE(SUM(LENGTH(CAST(content_json AS BLOB))"
                        "+LENGTH(CAST(meta_json AS BLOB))),0) AS n FROM " + self.q + " "
                        "WHERE session_id=? AND status IN ('pending','waiting_continuation')"
                    )
                else:
                    size_sql = (
                        "SELECT COALESCE(SUM(OCTET_LENGTH(content_json)+OCTET_LENGTH(meta_json)),0) AS n FROM "
                        + self.q + " WHERE session_id=? AND status IN ('pending','waiting_continuation')"
                    )
                content = canonical(body)
                meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
                size = int((await tx.fetchone(size_sql, (target.session_id,)))["n"])
                if size + len(content.encode()) + len(meta.encode()) > self.inbox_byte_limit:
                    raise CapacityError()
                next_state = "queued" if execution_state == "idle" else execution_state
                if (
                    execution_state == "running"
                    and int(parent_state["owner_lease_until_ms"]) <= database_now
                ):
                    next_state = "recovering"
                await tx.execute(
                    f"INSERT INTO {self.q} (session_id,item_id,kind,priority,status,content_json,"
                    "meta_json,request_hash,result_kind,created_at) VALUES (?,?,?,'normal','pending',?,?,?,?,?)",
                    (
                        target.session_id, message_id, result_kind, content, meta, digest,
                        result_kind, database_now,
                    ),
                )
                await tx.execute(
                    f"UPDATE {self.state} SET execution_state=?,inbox_version=inbox_version+1 "
                    "WHERE session_id=? AND execution_state=?",
                    (next_state, target.session_id, execution_state),
                )
            elif old.row["request_hash"] != digest:
                raise RuntimeRequestError("request_conflict")
            if activations is not None and execution_state == "idle":
                activations.append(target)
            return ref, None
        except (RuntimeRequestError, CapacityError) as exc:
            code = exc.code if isinstance(exc, RuntimeRequestError) else "capacity_exceeded"
            return ref, code

    async def _final_after_unknown(self, batch, consumed):
        state = await self.db.fetchone(
            f"SELECT * FROM {self.state} WHERE session_id=?", (batch.target.session_id,)
        )
        if not state or state["execution_state"] not in {"idle", "queued"}:
            return None
        if consumed is None:
            rows = await self.db.fetchall(
                f"SELECT item_id FROM {self.q} WHERE session_id=? AND result_send_index=? "
                "AND terminal_status IS NOT NULL",
                (batch.target.session_id, batch.send_index),
            )
            consumed = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in rows
            )
        results = []
        for ref in consumed:
            record = await self.input(self.db, ref)
            if (
                record is None or record.row["terminal_status"] is None
                or record.row["result_send_index"] != batch.send_index
            ):
                return None
            row = record.row
            final_text, usage, structured = "", {}, None
            if row["result_seq"] is not None:
                message = await self.db.fetchone(
                    f"SELECT content,meta_json FROM {self.store.t.messages} "
                    "WHERE session_id=? AND seq=? AND send_index=?",
                    (ref.session_id, row["result_seq"], row["result_send_index"]),
                )
                if message is not None:
                    final_text = message["content"] or ""
                    meta = json.loads(message["meta_json"] or "{}")
                    usage = meta.get("usage") or {}
                    structured = meta.get("structured_output")
            results.append(Result(
                ref, batch.target, row["terminal_status"], row["result_send_index"],
                row["result_seq"], final_text, usage, structured, row["error_detail"],
                row["error_code"], row["result_delivery_error"],
            ))
        return FinalBatch(state["execution_state"], tuple(results), None)

    async def recover(self, target, *, node_id, boot_id, ttl_s=30.0, ownership=None) -> FinalBatch | None:
        self.check_store(target)
        token = ownership.token if ownership is not None else uuid.uuid4().hex
        async with self.db.transaction() as tx:
            session = await self._session(tx, target.session_id)
            route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
            for sid in sorted({target.session_id, *( [route["target"]["session_id"]] if route else [] )}):
                state = await self.db.dialect.lock_state(tx, self.state, sid)
                if sid == target.session_id:
                    own_state = state
            database_now = await self._db_now(tx)
            execution_state = own_state["execution_state"]
            expired = (
                own_state["owner_lease_until_ms"] is not None
                and int(own_state["owner_lease_until_ms"]) <= database_now
            )
            if own_state["active_send_index"] is None:
                return None
            active = int(own_state["active_send_index"])
            if ownership is not None:
                ownership.require_owned()
                if not (
                    execution_state == "recovering"
                    and own_state["owner_token"] == ownership.token
                    and own_state["owner_epoch"] == ownership.epoch
                    and own_state["owner_node_id"] == node_id
                    and own_state["owner_boot_id"] == boot_id
                ):
                    raise SessionLeaseLost(target.session_id)
                epoch = int(own_state["owner_epoch"])
                changed = await tx.execute(
                    f"UPDATE {self.state} SET owner_lease_until_ms=? "
                    "WHERE session_id=? AND execution_state='recovering' AND active_send_index=? "
                    "AND owner_epoch=? AND owner_token=?",
                    (
                        database_now + int(ttl_s * 1000), target.session_id, active,
                        epoch, token,
                    ),
                )
            else:
                if execution_state not in {"running", "recovering"} or (
                    execution_state == "running" and not expired
                ) or (execution_state == "recovering" and own_state["owner_token"] and not expired):
                    return None
                epoch = int(own_state["owner_epoch"]) + 1
                changed = await tx.execute(
                    f"UPDATE {self.state} SET execution_state='recovering',owner_epoch=?,owner_token=?,"
                    "owner_node_id=?,owner_boot_id=?,owner_lease_until_ms=? "
                    "WHERE session_id=? AND execution_state=? AND active_send_index=?",
                    (
                        epoch, token, node_id, boot_id, database_now + int(ttl_s * 1000),
                        target.session_id, execution_state, active,
                    ),
                )
            if changed != 1:
                return None
            # Repair unresolved tool calls of the interrupted run before its terminal
            # state, stamped with the interrupted active_send_index (design 215 §9.3).
            await self._repair_pending_tx(tx, target.session_id, active, own_state)
            if session["session_kind"] == "workflow_driver":
                return await self._recover_workflow_tx(
                    tx, session, own_state, active, epoch, token, node_id, boot_id,
                    ttl_s, database_now,
                )
            rows = await tx.fetchall(
                f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL",
                (target.session_id, active),
            )
            cancelled = own_state["cancel_send_index"] == active
            terminal = "cancelled" if cancelled else "interrupted"
            detail = own_state["cancel_reason"] if cancelled else "execution owner lost"
            delivery_error = None
            result_ref = None
            activations = []
            if rows and route is not None:
                first = self.record(rows[0])
                route_result = Result(
                    first.ref, target, terminal, active, None,
                    error=detail, error_code=terminal,
                )
                result_ref, delivery_error = await self._route_result_tx(
                    tx, session, route, route_result, route["result_kind"], database_now,
                    activations,
                )
            if rows:
                await tx.execute(
                    f"UPDATE {self.q} SET terminal_status=?,result_send_index=?,error_code=?,"
                    "error_detail=?,result_delivery_error=?,completed_at=? "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    "AND terminal_status IS NULL",
                    (
                        terminal, active, terminal, detail, delivery_error, database_now,
                        target.session_id, active,
                    ),
                )
            pending = await tx.fetchone(
                f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? AND status='pending'",
                (target.session_id,),
            )
            next_state = "queued" if int(pending["n"]) else "idle"
            changed = await tx.execute(
                f"UPDATE {self.state} SET execution_state=?,active_send_index=NULL,"
                "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                "cancel_send_index=NULL,cancel_reason=NULL "
                "WHERE session_id=? AND execution_state='recovering' AND active_send_index=? "
                "AND owner_epoch=? AND owner_token=?",
                (next_state, target.session_id, active, epoch, token),
            )
            if changed != 1:
                raise SessionLeaseLost(target.session_id)
            results = tuple(
                Result(
                    self.record(row).ref, target, terminal, active, None,
                    error=detail, error_code=terminal, result_delivery_error=delivery_error,
                )
                for row in rows
            )
            return FinalBatch(
                next_state, results, result_ref,
                activations[0] if activations else None,
            )

    async def _continuation_child(self, reader, record):
        return await reader.fetchone(
            f"SELECT * FROM {self.q} WHERE session_id=? AND continuation_of=? "
            "ORDER BY continuation_index DESC LIMIT 1",
            (record.ref.session_id, record.ref.message_id),
        )

    async def _continuation_root(self, reader, record):
        current = record
        for _ in range(10000):
            parent_id = current.row["continuation_of"]
            if parent_id is None:
                return current
            parent = await self.input(
                reader, InputRef(current.ref.store_id, current.ref.session_id, parent_id),
            )
            if parent is None:
                raise RuntimeRequestError("workflow_continuation_broken")
            current = parent
        raise CapacityError("workflow continuation chain limit exceeded")

    async def _latest_continuation(self, reader, record):
        current = record
        for _ in range(10000):
            if current.row["status"] != "waiting_continuation":
                return current
            child = await self._continuation_child(reader, current)
            if child is None:
                return current
            current = self.record(child)
        raise CapacityError("workflow continuation chain limit exceeded")

    async def _recover_workflow_tx(
        self, tx, session, own_state, active, epoch, token, node_id, boot_id,
        ttl_s, database_now,
    ):
        """Continue a non-LLM driver instead of interrupting its original start."""
        rows = await tx.fetchall(
            f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
            "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id",
            (session["session_id"], active),
        )
        target = TargetRef(self.store.store_id, session["session_id"])
        if own_state["cancel_send_index"] == active:
            origins = [self.record(row) for row in rows if row["kind"] in {"user", "workflow_continue"}]
            if origins:
                origin = origins[-1]
                root = await self._continuation_root(tx, origin)
                route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
                delivery_error = None
                activations = []
                if route is not None:
                    cancelled = Result(
                        root.ref, target, "cancelled", active, None,
                        error=own_state["cancel_reason"] or "cancelled", error_code="cancelled",
                    )
                    _ref, delivery_error = await self._route_result_tx(
                        tx, session, route, cancelled, route["result_kind"], database_now, activations,
                    )
                reason = own_state["cancel_reason"] or "cancelled"
                await tx.execute(
                    f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',"
                    "result_send_index=?,error_code='cancelled',error_detail=?,"
                    "result_delivery_error=?,completed_at=? WHERE session_id=? "
                    "AND delivered_send_index=? AND terminal_status IS NULL",
                    (active, reason, delivery_error, database_now, session["session_id"], active),
                )
                # Ancestors were bound to earlier sends. They still share the final
                # continuation's terminal result, just like wait(original) does.
                current = origin
                ancestors = []
                while current.row["continuation_of"] is not None:
                    current = await self.input(tx, InputRef(
                        current.ref.store_id, current.ref.session_id, current.row["continuation_of"],
                    ))
                    ancestors.append(current.ref.message_id)
                if ancestors:
                    await tx.execute(
                        f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',"
                        "result_send_index=?,error_code='cancelled',error_detail=?,"
                        "result_delivery_error=?,completed_at=? WHERE session_id=? "
                        f"AND item_id IN ({','.join('?' for _ in ancestors)})",
                        (active, reason, delivery_error, database_now, session["session_id"], *ancestors),
                    )
                await tx.execute(
                    f"UPDATE {self.state} SET execution_state='idle',active_send_index=NULL,"
                    "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                    "cancel_send_index=NULL,cancel_reason=NULL WHERE session_id=? "
                    "AND execution_state='recovering' AND active_send_index=? AND owner_epoch=? AND owner_token=?",
                    (session["session_id"], active, epoch, token),
                )
                return FinalBatch(
                    "idle", (), result_activation=activations[0] if activations else None,
                )

        # Result notifications from the interrupted send are reconsidered by the
        # continuation. A committed journal/result consumption can never reach here.
        await tx.execute(
            f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
            "delivered_send_index=NULL WHERE session_id=? AND kind='workflow_node_result' "
            "AND status='delivered' AND delivered_send_index=? AND terminal_status IS NULL",
            (session["session_id"], active),
        )
        origins = [self.record(row) for row in rows if row["kind"] in {"user", "workflow_continue"}]
        if not origins:
            next_state = "queued" if rows else "idle"
        else:
            origin = origins[-1]
            root = await self._continuation_root(tx, origin)
            index = int(origin.row["continuation_index"] or 0) + 1
            message_id = "wfc_" + sha256_canonical([
                "workflow_continue", self.store.store_id, session["session_id"],
                root.ref.message_id, index,
            ])[:56]
            continuation = InputRef(self.store.store_id, session["session_id"], message_id)
            body = {
                "workflow_continue": {
                    "origin_input_ref": plain(root.ref),
                    "previous_input_ref": plain(origin.ref),
                    "continuation_index": index,
                },
            }
            existing = await self.input(tx, continuation)
            if existing is None:
                await tx.execute(
                    f"INSERT INTO {self.q} (session_id,item_id,kind,priority,status,content_json,"
                    "meta_json,request_hash,created_at,continuation_of,continuation_index) "
                    "VALUES (?,?,'workflow_continue','normal','pending',?,?,?,?,?,?)",
                    (
                        session["session_id"], message_id, canonical(body),
                        canonical({"accepted_protocol": PROTOCOL_VERSION}),
                        self.request_hash(target, body, "normal"), database_now,
                        origin.ref.message_id, index,
                    ),
                )
            elif existing.row["request_hash"] != self.request_hash(target, body, "normal"):
                raise RuntimeRequestError("request_conflict")
            await tx.execute(
                f"UPDATE {self.q} SET status='waiting_continuation' WHERE session_id=? "
                "AND item_id=? AND status='delivered' AND delivered_send_index=? "
                "AND terminal_status IS NULL",
                (session["session_id"], origin.ref.message_id, active),
            )
            next_state = "queued"
        changed = await tx.execute(
            f"UPDATE {self.state} SET execution_state=?,active_send_index=NULL,"
            "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
            "cancel_send_index=NULL,cancel_reason=NULL WHERE session_id=? "
            "AND execution_state='recovering' AND active_send_index=? AND owner_epoch=? AND owner_token=?",
            (next_state, session["session_id"], active, epoch, token),
        )
        if changed != 1:
            raise SessionLeaseLost(session["session_id"])
        return FinalBatch(next_state, ())

    async def discover_unfinished(self, *, limit=100):
        if not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options", "limit")
        database_now = await self._db_now(self.db)
        rows = await self.db.fetchall(
            f"SELECT session_id FROM {self.state} WHERE execution_state IN ('queued','recovering') "
            "OR (execution_state='running' AND owner_lease_until_ms<=?) "
            "ORDER BY execution_state,owner_lease_until_ms,session_id LIMIT ?",
            (database_now, limit),
        )
        return tuple(TargetRef(self.store.store_id, row["session_id"]) for row in rows)

    async def cancel(self, ref, *, actor, scope="self", reason="cancelled", _with_activation=False):
        if scope not in {"self", "subtree"}:
            raise RuntimeRequestError("invalid_options", "scope")
        if isinstance(ref, InputRef) and scope == "subtree":
            raise RuntimeRequestError("invalid_options", "InputRef subtree")
        target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
        activations = []
        async with self.db.transaction() as tx:
            session = await self.authorize(tx, target, actor)
            route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
            lock_ids = {target.session_id}
            if route is not None:
                lock_ids.add(route["target"]["session_id"])
            for sid in sorted(lock_ids):
                locked = await self.db.dialect.lock_state(tx, self.state, sid)
                if sid == target.session_id:
                    state = locked
            database_now = await self._db_now(tx)
            if isinstance(ref, InputRef):
                record = await self.input(tx, ref)
                if record is None:
                    raise RuntimeRequestError("not_found")
                if record.row["kind"] == "workflow_continue" or record.row["status"] == "waiting_continuation":
                    receipt = await self._cancel_workflow_input_tx(
                        tx, session, state, record, reason, database_now, activations,
                    )
                else:
                    receipt = await self._cancel_input_tx(
                        tx, session, state, record, reason, database_now, activations,
                    )
            elif state["execution_state"] in {"running", "recovering"}:
                active = state["active_send_index"]
                await tx.execute(
                    f"UPDATE {self.state} SET cancel_send_index=?,cancel_reason=? "
                    "WHERE session_id=? AND active_send_index=?",
                    (active, reason, target.session_id, active),
                )
                receipt = Receipt(None, target, None, False, "cancel_requested", active, "not_needed", reason)
            else:
                pending_rows = await tx.fetchall(
                    f"SELECT * FROM {self.q} WHERE session_id=? AND status='pending' ORDER BY id",
                    (target.session_id,),
                )
                changed = 0
                for row in pending_rows:
                    record = self.record(row)
                    delivery_error = None
                    if route is not None:
                        cancelled = Result(
                            record.ref, target, "cancelled", None, None,
                            error=reason, error_code="cancelled",
                        )
                        _result_ref, delivery_error = await self._route_result_tx(
                            tx, session, route, cancelled, route["result_kind"], database_now,
                            activations,
                        )
                    changed += await tx.execute(
                        f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                        "error_code='cancelled',error_detail=?,result_delivery_error=?,completed_at=? "
                        "WHERE session_id=? AND item_id=? AND status='pending'",
                        (reason, delivery_error, database_now, target.session_id, record.ref.message_id),
                    )
                if not pending_rows and route is not None:
                    start = await self.input(tx, InputRef(**route["input_ref"]))
                    if start is not None and start.row["terminal_status"] is not None:
                        await self._route_stored_result_tx(
                            tx, session, route, start, database_now, activations,
                        )
                if state["execution_state"] == "queued":
                    await tx.execute(
                        f"UPDATE {self.state} SET execution_state='idle' WHERE session_id=? "
                        "AND execution_state='queued' AND owner_token IS NULL", (target.session_id,)
                    )
                receipt = Receipt(
                    None, target, None, False, "voided" if changed else "already_finished",
                    None, "not_needed", reason if changed else None,
                )
        if scope == "subtree":
            rows = await self.db.fetchall(
                f"SELECT session_id FROM {self.store.t.sessions} "
                "WHERE parent_session_id=? AND cancel_propagation='linked' ORDER BY session_id",
                (target.session_id,),
            )
            for row in rows:
                child = TargetRef(self.store.store_id, row["session_id"])
                _receipt, child_activations = await self.cancel(
                    child, actor=actor, scope="subtree", reason=reason, _with_activation=True,
                )
                activations.extend(child_activations)
        if _with_activation:
            return receipt, activations
        return receipt

    async def _cancel_workflow_input_tx(
        self, tx, session, state, record, reason, database_now, activations,
    ):
        """Resolve cancel(original) to the newest continuation, then close its chain."""
        root = await self._continuation_root(tx, record)
        latest = await self._latest_continuation(tx, record)
        target = TargetRef(self.store.store_id, record.ref.session_id)
        if latest.row["terminal_status"] is not None:
            return self.receipt(record, state="already_finished")
        active = state["active_send_index"]
        if (
            latest.row["status"] == "delivered" and active is not None
            and latest.row["delivered_send_index"] == active
            and state["execution_state"] in {"running", "recovering"}
        ):
            await tx.execute(
                f"UPDATE {self.state} SET cancel_send_index=?,cancel_reason=? "
                "WHERE session_id=? AND active_send_index=? AND execution_state IN ('running','recovering')",
                (active, reason, target.session_id, active),
            )
            return Receipt(
                record.ref, target, record.ref.message_id, False, "cancel_requested",
                active, "not_needed", reason,
            )
        if latest.row["status"] != "pending":
            return self.receipt(record, state="already_finished")

        route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
        delivery_error = None
        if route is not None:
            cancelled = Result(
                root.ref, target, "cancelled", None, None,
                error=reason, error_code="cancelled",
            )
            _ref, delivery_error = await self._route_result_tx(
                tx, session, route, cancelled, route["result_kind"], database_now, activations,
            )
        changed = await tx.execute(
            f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
            "error_code='cancelled',error_detail=?,result_delivery_error=?,completed_at=? "
            "WHERE session_id=? AND item_id=? AND status='pending'",
            (reason, delivery_error, database_now, target.session_id, latest.ref.message_id),
        )
        if changed != 1:
            raise SessionLeaseLost(target.session_id)
        ancestors = []
        current = latest
        while current.row["continuation_of"] is not None:
            current = await self.input(tx, InputRef(
                current.ref.store_id, current.ref.session_id, current.row["continuation_of"],
            ))
            ancestors.append(current.ref.message_id)
        if record.ref.message_id not in ancestors:
            ancestors.append(record.ref.message_id)
        if ancestors:
            await tx.execute(
                f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',"
                "result_send_index=delivered_send_index,error_code='cancelled',error_detail=?,"
                "result_delivery_error=?,completed_at=? WHERE session_id=? "
                f"AND item_id IN ({','.join('?' for _ in ancestors)}) "
                "AND status='waiting_continuation' AND terminal_status IS NULL",
                (reason, delivery_error, database_now, target.session_id, *ancestors),
            )
        remaining = await tx.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? "
            "AND status IN ('pending','waiting_continuation')",
            (target.session_id,),
        )
        if not int(remaining["n"]) and state["execution_state"] == "queued":
            await tx.execute(
                f"UPDATE {self.state} SET execution_state='idle' WHERE session_id=? "
                "AND execution_state='queued' AND owner_token IS NULL",
                (target.session_id,),
            )
        return self.receipt(record, state="voided", reason=reason)

    async def _stored_result_tx(self, tx, record):
        row = record.row
        final_text, usage, structured = "", {}, None
        if row["result_seq"] is not None:
            message = await tx.fetchone(
                f"SELECT content,meta_json FROM {self.store.t.messages} "
                "WHERE session_id=? AND seq=? AND send_index=?",
                (record.ref.session_id, row["result_seq"], row["result_send_index"]),
            )
            if message is not None:
                final_text = message["content"] or ""
                meta = json.loads(message["meta_json"] or "{}")
                usage = meta.get("usage") or {}
                structured = meta.get("structured_output")
        return Result(
            record.ref, TargetRef(self.store.store_id, record.ref.session_id),
            row["terminal_status"], row["result_send_index"], row["result_seq"],
            final_text, usage, structured, row["error_detail"], row["error_code"],
            row["result_delivery_error"],
        )

    async def _route_stored_result_tx(
        self, tx, session, route, record, database_now, activations=None,
    ):
        existing = await self._stored_result_tx(tx, record)
        result_ref, delivery_error = await self._route_result_tx(
            tx, session, route, existing, route["result_kind"], database_now, activations,
        )
        row = record.row
        if delivery_error != row["result_delivery_error"]:
            await tx.execute(
                f"UPDATE {self.q} SET result_delivery_error=? WHERE session_id=? AND item_id=?",
                (delivery_error, record.ref.session_id, record.ref.message_id),
            )
            if row["result_seq"] is not None:
                message = await tx.fetchone(
                    f"SELECT meta_json FROM {self.store.t.messages} WHERE session_id=? AND seq=?",
                    (record.ref.session_id, row["result_seq"]),
                )
                meta = json.loads(message["meta_json"] or "{}") if message is not None else {}
                meta["result_delivery_error"] = delivery_error
                await tx.execute(
                    f"UPDATE {self.store.t.messages} SET meta_json=? WHERE session_id=? AND seq=?",
                    (canonical(meta), record.ref.session_id, row["result_seq"]),
                )
        return result_ref, delivery_error

    async def _cancel_input_tx(
        self, tx, session, state, record, reason, database_now, activations=None,
    ):
        row = record.row
        ref = record.ref
        route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
        if row["terminal_status"] is not None:
            if route is not None:
                await self._route_stored_result_tx(
                    tx, session, route, record, database_now, activations,
                )
            return self.receipt(record, state="already_finished")
        if row["status"] == "pending":
            delivery_error = None
            if route is not None:
                cancelled = Result(
                    ref, TargetRef(self.store.store_id, ref.session_id), "cancelled", None, None,
                    error=reason, error_code="cancelled",
                )
                _result_ref, delivery_error = await self._route_result_tx(
                    tx, session, route, cancelled, route["result_kind"], database_now,
                    activations,
                )
            changed = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                "error_code='cancelled',error_detail=?,result_delivery_error=?,completed_at=? "
                "WHERE session_id=? AND item_id=? AND status='pending'",
                (reason, delivery_error, database_now, ref.session_id, ref.message_id),
            )
            if changed:
                remaining = await tx.fetchone(
                    f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? "
                    "AND status IN ('pending','waiting_continuation')", (ref.session_id,),
                )
                if not int(remaining["n"]) and state["execution_state"] == "queued":
                    await tx.execute(
                        f"UPDATE {self.state} SET execution_state='idle' WHERE session_id=? "
                        "AND execution_state='queued' AND owner_token IS NULL", (ref.session_id,)
                    )
                row = dict(
                    row, status="void", terminal_status="cancelled", error_detail=reason,
                    result_delivery_error=delivery_error,
                )
                return self.receipt(InputRecord(ref, row, record.meta), state="voided", reason=reason)
        active = state["active_send_index"]
        if (
            row["status"] == "delivered" and active is not None
            and row["delivered_send_index"] == active
            and state["execution_state"] in {"running", "recovering"}
        ):
            await tx.execute(
                f"UPDATE {self.state} SET cancel_send_index=?,cancel_reason=? "
                "WHERE session_id=? AND active_send_index=? AND execution_state IN ('running','recovering')",
                (active, reason, ref.session_id, active),
            )
            return self.receipt(record, state="cancel_requested", reason=reason)
        return self.receipt(record, state="already_finished")

    async def cancel_active(self, batch: ClaimedBatch, *, reason="cancelled"):
        changed = await self.db.execute(
            f"UPDATE {self.state} SET cancel_send_index=?,cancel_reason=? "
            "WHERE session_id=? AND active_send_index=? AND owner_epoch=? AND owner_token=? "
            "AND execution_state='running'",
            (
                batch.send_index, reason, batch.target.session_id, batch.send_index,
                batch.owner_epoch, batch.owner_token,
            ),
        )
        if changed != 1:
            raise SessionLeaseLost(batch.target.session_id)
        return Receipt(
            None, batch.target, None, False, "cancel_requested", batch.send_index,
            "not_needed", reason,
        )

    async def release_owner(self, batch: ClaimedBatch):
        return bool(await self.db.execute(
            f"UPDATE {self.state} SET owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,"
            "owner_lease_until_ms=NULL WHERE session_id=? AND owner_epoch=? AND owner_token=? "
            "AND execution_state IN ('idle','queued')",
            (batch.target.session_id, batch.owner_epoch, batch.owner_token),
        ))

    async def close(self, target, *, actor):
        from power_loop.contracts.errors import SessionBusy

        self.check_store(target)
        async with self.db.transaction() as tx:
            await self.authorize(tx, target, actor)
            state = await self.db.dialect.lock_state(tx, self.state, target.session_id)
            if state["execution_state"] in {"running", "recovering"}:
                raise SessionBusy(target.session_id)
            pending = await tx.fetchone(
                f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? "
                "AND status IN ('pending','waiting_continuation')", (target.session_id,),
            )
            if int(pending["n"]):
                raise SessionBusy(target.session_id)
            linked_child = await tx.fetchone(
                f"SELECT session_id FROM {self.store.t.sessions} "
                "WHERE parent_session_id=? AND cancel_propagation='linked' "
                "AND lifecycle='open' LIMIT 1",
                (target.session_id,),
            )
            if linked_child is not None:
                raise SessionBusy(target.session_id)
            changed = await tx.execute(
                f"UPDATE {self.store.t.sessions} SET lifecycle='closed',updated_at=? "
                "WHERE session_id=? AND lifecycle='open'",
                (now_ms(), target.session_id),
            )
            if changed != 1:
                raise RuntimeRequestError("session_closed")

    async def inspect(self, ref, *, actor):
        target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
        async with self.db.read_snapshot() as reader:
            await self.authorize(reader, target, actor)
            state = await reader.fetchone(
                f"SELECT * FROM {self.state} WHERE session_id=?", (target.session_id,)
            )
            if isinstance(ref, TargetRef):
                send_index = state["active_send_index"] or state["send_index"] or None
                return Status(
                    None, target, state["execution_state"], None, None, send_index, False,
                    None, bool(
                        state["active_send_index"] is not None
                        and state["cancel_send_index"] == state["active_send_index"]
                    ),
                    state["cancel_reason"], None, None, None, None,
                )
            record = await self.input(reader, ref)
            if record is None:
                raise RuntimeRequestError("not_found")
            row = record.row
            return Status(
                ref, target, state["execution_state"], row["status"], row["terminal_status"],
                row["result_send_index"] or row["delivered_send_index"],
                row["terminal_status"] is not None, row["result_seq"],
                bool(
                    state["active_send_index"] is not None
                    and state["cancel_send_index"] == state["active_send_index"]
                    and row["delivered_send_index"] == state["active_send_index"]
                ),
                row["error_detail"] or (
                    state["cancel_reason"] if state["cancel_send_index"] == row["delivered_send_index"] else None
                ),
                row["error_code"], row["result_delivery_error"], row["continuation_of"],
                row["continuation_index"],
            )

    async def get_result(self, ref, *, actor):
        target = TargetRef(ref.store_id, ref.session_id)
        async with self.db.read_snapshot() as reader:
            await self.authorize(reader, target, actor)
            record = await self.input(reader, ref)
            if record is None:
                raise RuntimeRequestError("not_found")
            if record.row["status"] == "waiting_continuation" or record.row["kind"] == "workflow_continue":
                record = await self._latest_continuation(reader, record)
            if record.row["terminal_status"] is None:
                return None
            row = record.row
            final_text, usage, structured = "", {}, None
            if row["result_seq"] is not None:
                message = await reader.fetchone(
                    f"SELECT content,meta_json FROM {self.store.t.messages} "
                    "WHERE session_id=? AND seq=? AND send_index=?",
                    (ref.session_id, row["result_seq"], row["result_send_index"]),
                )
                if message is not None:
                    final_text = message["content"] or ""
                    meta = json.loads(message["meta_json"] or "{}")
                    usage = meta.get("usage") or {}
                    structured = meta.get("structured_output")
            return Result(
                ref, target, row["terminal_status"], row["result_send_index"], row["result_seq"],
                final_text, usage, structured, row["error_detail"], row["error_code"],
                row["result_delivery_error"],
            )

    async def messages(self, target, *, actor, after_seq=None, limit=20):
        if not 1 <= limit <= 100:
            raise RuntimeRequestError("invalid_options", "limit")
        async with self.db.read_snapshot() as reader:
            await self.authorize(reader, target, actor)
            rows = await reader.fetchall(
                f"SELECT * FROM {self.store.t.messages} WHERE session_id=? AND seq>? "
                "ORDER BY seq LIMIT ?", (target.session_id, after_seq or 0, limit),
            )
        items = tuple(
            MessageRow(
                int(row["id"]), row["session_id"], int(row["seq"]), row["role"], row["name"],
                row["content"], json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else None,
                row["tool_call_id"], row["round_index"], row["state"],
                json.loads(row["meta_json"] or "{}"), row["send_index"], int(row["created_at"]),
            )
            for row in rows
        )
        return MessagePage(items, int(rows[-1]["seq"]) if len(rows) == limit else None)

    async def list_children(self, target, *, actor, cursor=None, limit=100):
        if not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options", "limit")
        async with self.db.read_snapshot() as reader:
            await self.authorize(reader, target, actor)
            rows = await reader.fetchall(
                f"SELECT session_id FROM {self.store.t.sessions} WHERE parent_session_id=? "
                "AND session_id>? ORDER BY session_id LIMIT ?",
                (target.session_id, cursor or "", limit),
            )
        items = tuple(TargetRef(self.store.store_id, row["session_id"]) for row in rows)
        return SessionPage(items, items[-1].session_id if len(items) == limit else None)

    async def poll_notifications(self, binding, *, cursor=None, limit=100):
        """Bounded shared-store discovery; notifications never become a second ledger."""
        from .runtime_transport import PROTOCOL_VERSION, TransportPacket

        if binding.channel in {"work", "activation"}:
            refs = await self.discover_unfinished(limit=limit)
            packets = [
                TransportPacket(
                    PROTOCOL_VERSION, binding.namespace, ref.session_id, ref,
                    binding.channel, None, {"kind": "unfinished"},
                )
                for ref in refs
                if binding.session_id is None or ref.session_id == binding.session_id
            ]
            return packets, None
        if binding.channel != "output":
            return [], cursor
        rows = await self.db.fetchall(
            f"SELECT session_id,item_id,terminal_status,result_send_index,completed_at "
            f"FROM {self.q} WHERE terminal_status IS NOT NULL AND completed_at>? "
            "ORDER BY completed_at,id LIMIT ?",
            (cursor or 0, limit),
        )
        packets = []
        next_cursor = cursor
        for row in rows:
            if binding.session_id is not None and row["session_id"] != binding.session_id:
                continue
            ref = InputRef(self.store.store_id, row["session_id"], row["item_id"])
            packets.append(TransportPacket(
                PROTOCOL_VERSION, binding.namespace, row["item_id"],
                TargetRef(self.store.store_id, row["session_id"]), "output", ref,
                {"kind": "completed", "session_id": row["session_id"],
                 "send_index": row["result_send_index"], "status": row["terminal_status"]},
            ))
            next_cursor = int(row["completed_at"])
        return packets, next_cursor
