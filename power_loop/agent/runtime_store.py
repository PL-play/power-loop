"""Session-Core accept / claim / final / stop / recovery storage (Design 217 schema 18)."""
from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, replace
from typing import Any

from power_loop.contracts.errors import SessionLeaseLost
from power_loop.runtime.store.db import is_duplicate_key_error
from power_loop.runtime.store.types import (
    MessageRow as StoredMessageRow,
)
from power_loop.runtime.store.types import MessageState

from .runtime_types import (
    PROTOCOL_VERSION,
    RESERVED_INPUT_PREFIXES,
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

SOFT_PENDING_LIMIT = 100
HARD_PENDING_LIMIT = 1024
DEFAULT_MAX_PENDING_BYTES = 1_048_576
_VALID_INPUT_KINDS = frozenset({
    "user",
    "system_notice",
    "agent_result",
    "workflow_result",
    "workflow_node_result",
    "background_result",
    "managed_effect_result",
    "workflow_continue",
})
_PUBLIC_INPUT_KINDS = frozenset({"user", "system_notice"})
_VALID_RESULT_KINDS = frozenset({
    "agent_result",
    "workflow_result",
    "workflow_node_result",
    "background_result",
    "managed_effect_result",
})


def now_ms() -> int:
    import time
    return int(time.time() * 1000)


@dataclass(frozen=True)
class InputRecord:
    ref: InputRef
    row: dict
    meta: dict


_RESULT_INPUT_COLUMNS = (
    "session_id", "item_id", "kind", "status", "terminal_status",
    "result_seq", "result_send_index", "error_code", "error_detail",
    "result_delivery_error", "delivered_send_index", "continuation_of",
    "result_json", "cancel_by",
)


@dataclass(frozen=True)
class ResultInput:
    """Narrow input view needed to resolve and report a durable result."""
    ref: InputRef
    row: dict


@dataclass(frozen=True)
class ResultRead:
    """One authorized read of a target input and its durable result."""
    ref: InputRef
    target: TargetRef
    session: dict
    record: ResultInput | None
    resolved: ResultInput | None
    result: Result | None


@dataclass(frozen=True)
class ChildSpec:
    """Specification for one child session in ``create_many``."""
    session_id: str
    message_id: str
    body: Any
    descriptor: RuntimeDescriptor
    result_kind: str = "agent_result"
    reply_to_session_id: str | None = None
    cancel_propagation: str = "linked"
    creation_tool_call_id: str | None = None
    creation_call_ref: dict | None = None
    creation_identity_hash: str | None = None


@dataclass(frozen=True)
class ClaimedBatch:
    """One atomically claimed input snapshot on schema 18 (no owner_token)."""
    target: TargetRef
    send_index: int
    message_seq: int
    input_refs: tuple[InputRef, ...]
    owner_epoch: int
    owner_node_id: str
    owner_boot_id: str
    descriptor: RuntimeDescriptor
    tenant_id: str
    principal_id: str
    parent_session_id: str | None
    user_row: StoredMessageRow
    result_route: dict | None
    session_updated_at: int
    inbox_version: int = 0
    background_version: int = 0
    runtime_version: int = 0
    wake_inbox_seq: int = 0
    cancel_seq: int = 0
    cancel_by: str | None = None
    start_input_message_id: str | None = None
    root_session_id: str = ""
    depth: int = 0
    reply_to_session_id: str | None = None
    reply_result_kind: str | None = None
    is_reclaimed: bool = False
    reclaimed_has_assistant: bool = False
    reclaimed_cancelled: bool = False


@dataclass(frozen=True)
class CreateManyResult:
    preclaimed: tuple[ClaimedBatch, ...]
    queued_targets: tuple[TargetRef, ...]
    already_created: bool = False


@dataclass(frozen=True)
class FinalBatch:
    execution_state: str
    results: tuple[Result, ...]
    result_ref: InputRef | None = None
    result_activation: TargetRef | None = None
    result_delivery_error: str | None = None


class _TargetAlreadyCreated(Exception):
    pass


class _AcceptSlowPath(Exception):
    def __init__(self, reason: str = "slow"):
        self.reason = reason
        super().__init__(reason)


class _ClaimRollback(Exception):
    pass


class RuntimeInputStore:
    """Session-Core schema-18 storage port (SQLite / PostgreSQL 16 / MySQL 8.4)."""

    def __init__(
        self,
        store,
        *,
        inbox_item_limit: int = SOFT_PENDING_LIMIT,
        hard_pending_limit: int | None = None,
        inbox_byte_limit: int = DEFAULT_MAX_PENDING_BYTES,
    ):
        self.store = store
        self.s = store.t.sessions
        self.q = store.t.follow_up_queue
        self.inbox_item_limit = inbox_item_limit
        if hard_pending_limit is None:
            self.hard_pending_limit = (
                HARD_PENDING_LIMIT if inbox_item_limit == SOFT_PENDING_LIMIT else inbox_item_limit
            )
        else:
            self.hard_pending_limit = hard_pending_limit
        self.inbox_byte_limit = inbox_byte_limit

    @property
    def db(self):
        return self.store._db

    @property
    def _dialect(self) -> str:
        return self.db.dialect.name

    def check_store(self, ref) -> None:
        if ref.store_id != self.store.store_id:
            raise RuntimeRequestError("wrong_store")

    def _now_sql(self) -> str:
        name = self._dialect
        if name == "sqlite":
            return "CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)"
        if name == "postgres":
            return "CAST(EXTRACT(EPOCH FROM clock_timestamp()) * 1000 AS BIGINT)"
        return "FLOOR(UNIX_TIMESTAMP(CURRENT_TIMESTAMP(3)) * 1000)"

    def _p_text(self) -> str:
        return "CAST(? AS TEXT)" if self._dialect == "postgres" else "?"

    def _p_int(self) -> str:
        return "CAST(? AS BIGINT)" if self._dialect == "postgres" else "?"

    async def _db_now(self, reader) -> int:
        return int((await reader.fetchone(f"SELECT {self._now_sql()} AS now"))["now"])

    # ── Session creation & configuration ─────────────────────────────────────

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

    async def configure_session(
        self,
        target: TargetRef,
        *,
        actor,
        descriptor: RuntimeDescriptor | None = None,
        cancel_propagation: str | None = None,
    ) -> dict[str, Any]:
        """Hot-update session descriptor (immediate if not running; staged in next_descriptor_* if running).

        Staged descriptors are promoted atomically across all three claim entry points:
        fused ``accept_and_claim``, standard ``claim``, and 1-SQL pre-staged child ``claim``.
        """
        self.check_store(target)
        if cancel_propagation is not None and cancel_propagation not in {"linked", "independent"}:
            raise RuntimeRequestError("invalid_options", "cancel_propagation")
        desc_json = canonical(descriptor) if descriptor is not None else None
        desc_hash = sha256_canonical(descriptor) if descriptor is not None else None
        now_sql = self._now_sql()
        suffix = "" if self._dialect == "sqlite" else " FOR UPDATE"
        async with self.db.transaction() as tx:
            row = await tx.fetchone(
                f"SELECT session_id,tenant_id,principal_id,parent_session_id,lifecycle,execution_state "
                f"FROM {self.s} WHERE session_id=?" + suffix,
                (target.session_id,),
            )
            if row is None:
                raise RuntimeRequestError("not_found")
            await self._authorize_known(tx, target, actor, row)
            if row["lifecycle"] not in ("open", "suspended"):
                raise RuntimeRequestError("session_closed")
            sets = [f"updated_at={now_sql}"]
            params: list[Any] = []
            applied_immediately = row["execution_state"] != "running"
            if desc_json is not None:
                if applied_immediately:
                    sets.extend([
                        "descriptor_json=?",
                        "descriptor_hash=?",
                        "next_descriptor_json=NULL",
                        "next_descriptor_hash=NULL",
                    ])
                    params.extend([desc_json, desc_hash])
                else:
                    sets.extend(["next_descriptor_json=?", "next_descriptor_hash=?"])
                    params.extend([desc_json, desc_hash])
            if cancel_propagation is not None:
                sets.append("cancel_propagation=?")
                params.append(cancel_propagation)
            params.append(target.session_id)
            await tx.execute(
                f"UPDATE {self.s} SET {','.join(sets)} WHERE session_id=?",
                tuple(params),
            )
        return {
            "session_id": target.session_id,
            "applied_immediately": applied_immediately,
            "staged": desc_json is not None and not applied_immediately,
        }

    async def _lock_creation_parent_tx(self, tx, parent, actor, *, existing_id=None):
        """Lock deep-first (child before parent) consistent with Design 217 lock ordering."""
        from power_loop.contracts.errors import CancellationRequested

        lock_ids = []
        if existing_id is not None and existing_id != parent.session_id:
            lock_ids.append(existing_id)
        lock_ids.append(parent.session_id)
        states = {}
        for sid in lock_ids:
            try:
                states[sid] = await self.db.dialect.lock_state(tx, self.s, sid)
            except ValueError as exc:
                if sid == parent.session_id:
                    raise RuntimeRequestError("not_found") from exc
                raise RuntimeRequestError("creation_conflict") from exc
        parent_session = states[parent.session_id]
        await self._authorize_known(tx, parent, actor, parent_session)
        if parent_session["lifecycle"] != "open":
            raise RuntimeRequestError("session_closed")
        if parent_session["execution_state"] == "running" and parent_session["cancel_by"] is not None:
            raise CancellationRequested(parent_session["cancel_reason"] or "cancelled")
        return parent_session

    async def create_child(
        self, descriptor: RuntimeDescriptor, *, parent: TargetRef, actor, call_ref,
        tool_call_id: str, body, cancel_propagation="linked", result_kind="agent_result",
        require_assistant_call=True,
    ):
        """Atomically create the deterministic child and accept its original start input."""
        self.check_store(parent)
        if actor.session_id != parent.session_id:
            raise RuntimeRequestError("forbidden", "child creation requires the direct parent actor")
        if cancel_propagation not in {"linked", "independent"}:
            raise RuntimeRequestError("invalid_options", "cancel_propagation")
        if result_kind not in _VALID_RESULT_KINDS:
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
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        if self.inbox_item_limit <= 0 or content_bytes > self.inbox_byte_limit:
            raise CapacityError()
        req_hash = self.request_hash(target, body, "normal")
        for attempt in range(2):
            try:
                async with self.db.transaction() as tx:
                    existing = await tx.fetchone(
                        f"SELECT * FROM {self.s} WHERE session_id=?", (session_id,),
                    )
                    parent_session = await self._lock_creation_parent_tx(
                        tx, parent, actor,
                        existing_id=(session_id if existing is not None else None),
                    )
                    if existing is None:
                        if attempt:
                            raise RuntimeRequestError("creation_conflict")
                        if require_assistant_call:
                            call = await tx.fetchone(
                                f"SELECT tool_calls_json FROM {self.store.t.messages} "
                                "WHERE session_id=? AND seq=? AND role='assistant'",
                                (parent.session_id, call_ref.seq),
                            )
                            calls = json.loads(call["tool_calls_json"] or "[]") if call is not None else []
                            if not any(item.get("id") == tool_call_id for item in calls):
                                raise RuntimeRequestError(
                                    "forbidden", "original assistant tool call not found")
                        database_now = await self._db_now(tx)
                        root_id = str(parent_session["root_session_id"])
                        child_depth = int(parent_session["depth"]) + 1
                        if child_depth > self.store.max_spawn_depth:
                            raise RuntimeRequestError(
                                "invalid_options",
                                f"child depth exceeds max {self.store.max_spawn_depth}",
                            )
                        changed = await tx.execute(
                            f"INSERT INTO {self.s} ("
                            "session_id,parent_session_id,created_by_session_id,root_session_id,depth,"
                            "session_kind,descriptor_json,descriptor_hash,creation_identity_hash,"
                            "creation_call_ref_json,creation_tool_call_id,start_input_message_id,"
                            "tenant_id,principal_id,result_route_json,cancel_propagation,"
                            "execution_state,inbox_version,wake_inbox_seq,wake_pending_count,"
                            "pending_count,pending_bytes,created_at,updated_at) "
                            "SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',1,1,1,1,?,?,? "
                            f"WHERE NOT EXISTS (SELECT 1 FROM {self.s} WHERE session_id=?)",
                            (
                                session_id, parent.session_id, actor.session_id, root_id, child_depth,
                                descriptor.kind, canonical(descriptor), sha256_canonical(descriptor),
                                identity, canonical(call_ref.to_dict()), tool_call_id, start.message_id,
                                parent_session["tenant_id"], parent_session["principal_id"],
                                canonical(route), cancel_propagation,
                                content_bytes, database_now, database_now, session_id,
                            ),
                        )
                        if changed != 1:
                            if attempt == 0:
                                raise _TargetAlreadyCreated
                            raise RuntimeRequestError("creation_conflict")
                        await tx.execute(
                            f"INSERT INTO {self.q} ("
                            "session_id,item_id,inbox_version,kind,priority,wake,status,"
                            "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                            "meta_json,request_hash,result_kind,created_at,expires_at) "
                            "VALUES (?,?,1,'user','normal',1,'pending',?,?,?,?,?,?,NULL,?,NULL)",
                            (
                                session_id, start.message_id, parent.session_id, result_kind,
                                content, content_bytes, meta, req_hash, database_now,
                            ),
                        )
                        receipt = Receipt(start, target, start.message_id, True, "accepted", None, "not_needed")
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
                        receipt = await self._accept_existing_tx(
                            tx, target, start, req_hash, content, content_bytes,
                            actor=actor, priority="normal", deadline_ms=None,
                            kind="user", result_kind=None, wake=True,
                            reply_to_session_id=parent.session_id, reply_result_kind=result_kind,
                        )
                return target, start, receipt
            except _TargetAlreadyCreated:
                if attempt:
                    raise
                continue
        raise RuntimeRequestError("creation_conflict")

    async def create_many(
        self,
        *,
        parent: TargetRef,
        actor,
        specs: list[ChildSpec],
        node_id: str,
        boot_id: str,
        ttl_s: float = 30.0,
        can_execute: bool = True,
        free_slots: int = 0,
        parent_owner_epoch: int | None = None,
        parent_cancel_seq: int | None = None,
    ) -> CreateManyResult:
        """Batch-create N child sessions in 1 atomic transaction (Design 217 §4.2).

        - Pre-claims the first ``k = min(N, free_slots)`` when ``can_execute=True`` (0 Claim SQL).
        - Pre-stages the remaining ``N - k`` queued children with ``follow_up_queue(status='delivered')``,
          initial ``messages(role='user', seq=1)``, and ``ready_item_json`` so cross-container
          ``claim()`` completes in 1 SQL (SQLite/PG) or 2 SQL (MySQL) with 0 message inserts!
        """
        self.check_store(parent)
        if not specs:
            return CreateManyResult(preclaimed=(), queued_targets=())
        if ttl_s <= 0:
            raise RuntimeRequestError("invalid_options", "ttl_s")
        k = min(len(specs), max(0, int(free_slots))) if can_execute else 0
        ttl_ms = int(ttl_s * 1000)

        spec_hashes = {}
        for sp in specs:
            if sp.cancel_propagation not in {"linked", "independent"}:
                raise RuntimeRequestError("invalid_options", "cancel_propagation")
            if sp.result_kind not in _VALID_RESULT_KINDS:
                raise RuntimeRequestError("invalid_options", "result_kind")
            spec_hashes[sp.session_id] = sp.creation_identity_hash or sha256_canonical([
                parent.store_id, parent.session_id, sp.session_id,
                sp.creation_tool_call_id, sp.message_id, sp.descriptor,
            ])

        try:
            async with self.db.transaction() as tx:
                parent_row = await self.db.dialect.lock_state(tx, self.s, parent.session_id)
                await self._authorize_known(tx, parent, actor, parent_row)
                if parent_owner_epoch is not None and (
                    int(parent_row["owner_epoch"]) != int(parent_owner_epoch)
                    or parent_row["execution_state"] != "running"
                ):
                    raise SessionLeaseLost(parent.session_id)
                if parent_row["lifecycle"] != "open":
                    raise RuntimeRequestError("session_closed")
                if parent_cancel_seq is not None and int(parent_row["cancel_seq"]) > int(parent_cancel_seq):
                    raise RuntimeRequestError("cancelled", "parent_cancelled")

                database_now = await self._db_now(tx)
                root_id = str(parent_row["root_session_id"])
                child_depth = int(parent_row["depth"]) + 1
                if child_depth > self.store.max_spawn_depth:
                    raise RuntimeRequestError(
                        "invalid_options",
                        f"child depth exceeds max {self.store.max_spawn_depth}",
                    )
                lease_until = database_now + ttl_ms

                preclaimed: list[ClaimedBatch] = []
                queued: list[TargetRef] = []

                for idx, sp in enumerate(specs):
                    is_preclaimed = idx < k
                    child_target = TargetRef(parent.store_id, sp.session_id)
                    start_ref = InputRef(parent.store_id, sp.session_id, sp.message_id)
                    reply_to = sp.reply_to_session_id or parent.session_id
                    route = self._result_route(
                        parent,
                        {"target": parent, "input_ref": start_ref, "result_kind": sp.result_kind},
                        sp.session_id,
                    )
                    content = canonical(sp.body)
                    meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
                    content_bytes = len(content.encode()) + len(meta.encode())
                    if content_bytes > self.inbox_byte_limit:
                        raise CapacityError()
                    req_hash = self.request_hash(child_target, sp.body, "normal")
                    is_driver = sp.descriptor.kind == "workflow_driver"
                    merged = self.merge_batch([{
                        "item_id": sp.message_id, "kind": "user", "priority": "normal",
                        "content_json": content,
                    }])
                    merged_json = canonical(merged)
                    msg_meta = {"content_encoding": "json", "input_refs": [plain(start_ref)]}

                    ready_item = None if is_preclaimed else canonical({
                        "item_id": sp.message_id,
                        "kind": "user",
                        "priority": "normal",
                        "wake": 1,
                        "content_json": content,
                        "content_bytes": content_bytes,
                        "meta_json": meta,
                        "reply_to_session_id": reply_to,
                        "reply_result_kind": sp.result_kind,
                        "user_content": merged_json,
                        "user_seq": None if is_driver else 1,
                    })

                    execution_state_val = "running" if is_preclaimed else "queued"
                    send_idx_col = 1 if is_preclaimed else 0
                    next_seq_col = 1 if is_driver else 2
                    owner_epoch_col = 1 if is_preclaimed else 0
                    owner_node_col = node_id if is_preclaimed else None
                    owner_boot_col = boot_id if is_preclaimed else None
                    lease_col = lease_until if is_preclaimed else None
                    wake_pending_col = 0 if is_preclaimed else 1

                    await tx.execute(
                        f"INSERT INTO {self.s} ("
                        "session_id,parent_session_id,created_by_session_id,root_session_id,depth,"
                        "session_kind,descriptor_json,descriptor_hash,creation_identity_hash,"
                        "creation_call_ref_json,creation_tool_call_id,start_input_message_id,"
                        "tenant_id,principal_id,result_route_json,cancel_propagation,"
                        "next_seq,send_index,execution_state,owner_epoch,owner_node_id,"
                        "owner_boot_id,owner_lease_until_ms,inbox_version,wake_inbox_seq,"
                        "wake_pending_count,pending_count,pending_bytes,ready_item_json,"
                        "created_at,updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            sp.session_id, parent.session_id, actor.session_id, root_id, child_depth,
                            sp.descriptor.kind, canonical(sp.descriptor), sha256_canonical(sp.descriptor),
                            spec_hashes[sp.session_id],
                            canonical(sp.creation_call_ref) if sp.creation_call_ref else None,
                            sp.creation_tool_call_id, sp.message_id,
                            parent_row["tenant_id"], parent_row["principal_id"],
                            canonical(route), sp.cancel_propagation,
                            next_seq_col, send_idx_col, execution_state_val, owner_epoch_col,
                            owner_node_col, owner_boot_col, lease_col,
                            1, 1, wake_pending_col, 0, 0, ready_item,
                            database_now, database_now,
                        ),
                    )
                    deliv_seq = None if is_driver else 1
                    await tx.execute(
                        f"INSERT INTO {self.q} ("
                        "session_id,item_id,inbox_version,kind,priority,wake,status,"
                        "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                        "meta_json,request_hash,result_kind,created_at,delivered_at,"
                        "delivered_seq,delivered_send_index) "
                        "VALUES (?,?,1,'user','normal',1,'delivered',?,?,?,?,?,?,NULL,?,?,?,1)",
                        (
                            sp.session_id, sp.message_id, reply_to, sp.result_kind,
                            content, content_bytes, meta, req_hash, database_now,
                            database_now, deliv_seq,
                        ),
                    )
                    if not is_driver:
                        await self._insert_message(
                            tx, sp.session_id, 1, "user", merged_json, msg_meta, 1, database_now,
                        )
                    if is_preclaimed:
                        user_row = StoredMessageRow(
                            session_id=sp.session_id, seq=1, role="user", name=None,
                            content=merged_json, tool_calls=None, tool_call_id=None,
                            round_index=0, state=MessageState.ACTIVE, meta=msg_meta,
                            created_at=database_now, send_index=1,
                        )
                        preclaimed.append(ClaimedBatch(
                            target=child_target,
                            send_index=1,
                            message_seq=1 if not is_driver else 0,
                            input_refs=(start_ref,),
                            owner_epoch=1,
                            owner_node_id=node_id,
                            owner_boot_id=boot_id,
                            descriptor=sp.descriptor,
                            tenant_id=parent_row["tenant_id"],
                            principal_id=parent_row["principal_id"],
                            parent_session_id=parent.session_id,
                            user_row=user_row,
                            result_route=route,
                            session_updated_at=database_now,
                            inbox_version=1,
                            wake_inbox_seq=1,
                            cancel_seq=0,
                            start_input_message_id=sp.message_id,
                            root_session_id=root_id,
                            depth=child_depth,
                            reply_to_session_id=reply_to,
                            reply_result_kind=sp.result_kind,
                        ))
                    else:
                        queued.append(child_target)
                return CreateManyResult(preclaimed=tuple(preclaimed), queued_targets=tuple(queued))
        except Exception as exc:
            if not is_duplicate_key_error(exc):
                raise
            # Verify idempotent retry via creation_identity_hash
            ids = [sp.session_id for sp in specs]
            rows = await self.db.fetchall(
                f"SELECT session_id,creation_identity_hash FROM {self.s} "
                f"WHERE session_id IN ({','.join('?' for _ in ids)})",
                tuple(ids),
            )
            by_id = {r["session_id"]: r["creation_identity_hash"] for r in rows}
            if len(by_id) != len(ids) or any(by_id.get(sid) != h for sid, h in spec_hashes.items()):
                raise RuntimeRequestError("creation_conflict") from exc
            return CreateManyResult(preclaimed=(), queued_targets=(), already_created=True)

    async def create_workflow_session(
        self, descriptor: RuntimeDescriptor, *, parent: TargetRef, actor,
        target: TargetRef, start: InputRef, body, result_kind: str,
        cancel_propagation="linked", run_journal=None,
    ):
        """Atomically align/create a deterministic workflow session and its start."""
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
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        req_hash = self.request_hash(target, body, "normal")
        for attempt in range(2):
            try:
                async with self.db.transaction() as tx:
                    existing = await tx.fetchone(
                        f"SELECT * FROM {self.s} WHERE session_id=?", (target.session_id,),
                    )
                    parent_session = await self._lock_creation_parent_tx(
                        tx, parent, actor,
                        existing_id=(target.session_id if existing is not None else None),
                    )
                    if existing is None:
                        if attempt:
                            raise RuntimeRequestError("creation_conflict")
                        database_now = await self._db_now(tx)
                        root_id = str(parent_session["root_session_id"])
                        child_depth = int(parent_session["depth"]) + 1
                        changed = await tx.execute(
                            f"INSERT INTO {self.s} ("
                            "session_id,parent_session_id,created_by_session_id,root_session_id,depth,"
                            "session_kind,descriptor_json,descriptor_hash,creation_identity_hash,"
                            "creation_call_ref_json,creation_tool_call_id,start_input_message_id,"
                            "tenant_id,principal_id,result_route_json,cancel_propagation,"
                            "execution_state,inbox_version,wake_inbox_seq,wake_pending_count,"
                            "pending_count,pending_bytes,created_at,updated_at) "
                            "SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',1,1,1,1,?,?,? "
                            f"WHERE NOT EXISTS (SELECT 1 FROM {self.s} WHERE session_id=?)",
                            (
                                target.session_id, parent.session_id, actor.session_id,
                                root_id, child_depth, descriptor.kind, canonical(descriptor),
                                sha256_canonical(descriptor), identity,
                                canonical(actor.call_ref.to_dict()) if actor.call_ref is not None else None,
                                actor.tool_call_id, start.message_id,
                                actor.tenant_id, actor.principal_id, canonical(route),
                                cancel_propagation, content_bytes, database_now, database_now,
                                target.session_id,
                            ),
                        )
                        if changed != 1:
                            if attempt == 0:
                                raise _TargetAlreadyCreated
                            raise RuntimeRequestError("creation_conflict")
                        await tx.execute(
                            f"INSERT INTO {self.q} ("
                            "session_id,item_id,inbox_version,kind,priority,wake,status,"
                            "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                            "meta_json,request_hash,result_kind,created_at,expires_at) "
                            "VALUES (?,?,1,'user','normal',1,'pending',?,?,?,?,?,?,NULL,?,NULL)",
                            (
                                target.session_id, start.message_id, parent.session_id, result_kind,
                                content, content_bytes, meta, req_hash, database_now,
                            ),
                        )
                        receipt = Receipt(start, target, start.message_id, True, "accepted", None, "not_needed")
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
                        receipt = await self._accept_existing_tx(
                            tx, target, start, req_hash, content, content_bytes,
                            actor=actor, priority="normal", deadline_ms=None,
                            kind="user", result_kind=None, wake=True,
                            reply_to_session_id=parent.session_id, reply_result_kind=result_kind,
                        )
                    if run_journal is not None and existing is None:
                        from power_loop.workflow import journal
                        await journal.put(
                            self.store, tx, target.session_id,
                            journal.run_key(start.message_id), run_journal,
                        )
                return target, start, receipt
            except _TargetAlreadyCreated:
                if attempt:
                    raise
                continue
        raise RuntimeRequestError("creation_conflict")

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
        result_message_id(TargetRef(self.store.store_id, child_id), start, None, route["result_kind"])
        return {
            "target": plain(target), "input_ref": plain(start),
            "result_kind": route["result_kind"],
        }

    @staticmethod
    def _parse_reply_route(reply_route) -> tuple[str | None, str | None]:
        if reply_route is None:
            return None, None
        if not isinstance(reply_route, dict):
            raise RuntimeRequestError("invalid_options", "reply_route must be a mapping")
        target = reply_route.get("target") or reply_route.get("reply_to_session_id")
        if isinstance(target, TargetRef):
            reply_sid = target.session_id
        elif isinstance(target, dict) and "session_id" in target:
            reply_sid = str(target["session_id"])
        elif isinstance(target, str) and target:
            reply_sid = target
        else:
            raise RuntimeRequestError("invalid_options", "reply_route target required")
        rk = reply_route.get("result_kind") or reply_route.get("reply_result_kind") or "agent_result"
        if rk not in _VALID_RESULT_KINDS:
            raise RuntimeRequestError("invalid_options", "reply_route result_kind")
        return reply_sid, rk

    # ── Authorization & Lookup Helpers ───────────────────────────────────────

    async def _session(self, reader, session_id):
        row = await reader.fetchone(
            f"SELECT * FROM {self.s} WHERE session_id=?", (session_id,)
        )
        if row is None:
            raise RuntimeRequestError("not_found")
        return row

    async def _is_ancestor(self, reader, parent_session_id: str | None, actor_session_id: str) -> bool:
        cur = parent_session_id
        visited: set[str] = set()
        while cur is not None and cur not in visited:
            if cur == actor_session_id:
                return True
            visited.add(cur)
            row = await reader.fetchone(
                f"SELECT parent_session_id FROM {self.s} WHERE session_id=?", (cur,)
            )
            if row is None:
                return False
            cur = row["parent_session_id"]
        return False

    async def _authorize_known(self, reader, target, actor, session):
        if (session["tenant_id"], session["principal_id"]) != (
            actor.tenant_id, actor.principal_id
        ):
            raise RuntimeRequestError("forbidden")
        if actor.session_id is not None:
            if (
                actor.session_id != target.session_id
                and actor.session_id != session["parent_session_id"]
                and not await self._is_ancestor(reader, session["parent_session_id"], actor.session_id)
            ):
                raise RuntimeRequestError("forbidden")

    async def authorize(self, reader, target, actor):
        self.check_store(target)
        session = await self._session(reader, target.session_id)
        await self._authorize_known(reader, target, actor, session)
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
        return self._receipt(record.ref, record.row, new=new, state=state, reason=reason)

    def _receipt(self, ref, row, *, new=False, state=None, reason=None):
        if state is None:
            state = "accepted"
        return Receipt(
            ref, TargetRef(self.store.store_id, ref.session_id),
            ref.message_id, new, state,
            row.get("result_send_index") or row.get("delivered_send_index"),
            "not_needed", reason if reason is not None else row.get("error_detail"),
        )

    @staticmethod
    def request_hash(target, body, priority):
        return sha256_canonical([
            PROTOCOL_VERSION, target.store_id, target.session_id, body, priority,
        ])

    # ── Accept (Standard 2-SQL Fast Path + Slow Diagnostic Path) ─────────────

    def _validate_accept_args(
        self, target, body, *, message_id, priority, kind, wake, reply_route, public
    ):
        if priority not in {"normal", "urgent"}:
            raise RuntimeRequestError("invalid_options", "priority")
        if public and isinstance(message_id, str) and message_id.startswith(RESERVED_INPUT_PREFIXES):
            raise RuntimeRequestError("invalid_options", "reserved message_id prefix")
        if kind not in _VALID_INPUT_KINDS or (public and kind not in _PUBLIC_INPUT_KINDS):
            raise RuntimeRequestError("invalid_options", "kind")
        self.check_store(target)
        ref = InputRef(target.store_id, target.session_id, message_id)
        reply_to_sid, reply_rk = self._parse_reply_route(reply_route)
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        digest = self.request_hash(target, body, priority)
        wake_int = 1 if wake else 0
        return ref, digest, content, meta, content_bytes, wake_int, reply_to_sid, reply_rk

    async def accept(
        self, target, body, *, message_id, actor, priority="normal", deadline_ms=None,
        kind="user", result_kind=None, wake: bool = True, reply_route: dict | None = None,
        public: bool = True, _with_activation: bool = False,
    ) -> Receipt:
        """Accept an input into ``follow_up_queue`` (2 control SQL on the fast path).

        - ``wake=False`` ONLY increments ``inbox_version``, ``pending_count``, ``pending_bytes``
          (never increments ``wake_inbox_seq`` or ``wake_pending_count``, never flips ``queued``).
        - Public calls reject reserved prefixes ``res_`` / ``wfc_`` / ``sys_`` with 0 SQL.
        """
        ref, digest, content, meta, content_bytes, wake_int, reply_to_sid, reply_rk = (
            self._validate_accept_args(
                target, body, message_id=message_id, priority=priority,
                kind=kind, wake=wake, reply_route=reply_route, public=public,
            )
        )
        now_sql = self._now_sql()
        pt, pi = self._p_text(), self._p_int()
        limit = self.inbox_item_limit if public else self.hard_pending_limit
        try:
            try:
                async with self.db.transaction() as tx:
                    if self._dialect == "mysql":
                        rc, packed = await tx.execute_packed(
                            f"UPDATE {self.s} SET "
                            "inbox_version=(LAST_INSERT_ID("
                            "((inbox_version+1)<<1)|"
                            "(CASE WHEN execution_state='idle' AND ?=1 THEN 1 ELSE 0 END)"
                            ")>>1),"
                            "wake_inbox_seq=wake_inbox_seq+?,"
                            "cancel_by=CASE WHEN execution_state='idle' AND ?=1 THEN NULL ELSE cancel_by END,"
                            "wake_pending_count=wake_pending_count+?,"
                            "pending_count=pending_count+1,"
                            "pending_bytes=pending_bytes+?,"
                            "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' "
                            "AND tenant_id=? AND principal_id=? "
                            "AND (? IS NULL OR session_id=? OR parent_session_id=?) "
                            f"AND (? IS NULL OR ?>{now_sql}) "
                            "AND pending_count<? AND (pending_bytes+?)<=?",
                            (
                                wake_int, wake_int, wake_int, wake_int, content_bytes, wake_int,
                                target.session_id, actor.tenant_id, actor.principal_id,
                                actor.session_id, actor.session_id, actor.session_id,
                                deadline_ms, deadline_ms,
                                limit, content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if rc == 0:
                            raise _AcceptSlowPath("update_zero")
                        new_inbox_ver = packed >> 1
                        need_wake = bool(packed & 1)
                    else:
                        row = await tx.fetchone(
                            f"UPDATE {self.s} SET "
                            "inbox_version=inbox_version+1,"
                            "wake_inbox_seq=wake_inbox_seq+?,"
                            "cancel_by=CASE WHEN execution_state='idle' AND ?=1 THEN NULL ELSE cancel_by END,"
                            "wake_pending_count=wake_pending_count+?,"
                            "pending_count=pending_count+1,"
                            "pending_bytes=pending_bytes+?,"
                            "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' "
                            "AND tenant_id=? AND principal_id=? "
                            f"AND ({pt} IS NULL OR session_id={pt} OR parent_session_id={pt}) "
                            f"AND ({pi} IS NULL OR {pi}>{now_sql}) "
                            "AND pending_count<? AND (pending_bytes+?)<=? "
                            "RETURNING inbox_version,execution_state,wake_pending_count",
                            (
                                wake_int, wake_int, wake_int, content_bytes, wake_int,
                                target.session_id, actor.tenant_id, actor.principal_id,
                                actor.session_id, actor.session_id, actor.session_id,
                                deadline_ms, deadline_ms,
                                limit, content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if row is None:
                            raise _AcceptSlowPath("update_zero")
                        new_inbox_ver = int(row["inbox_version"])
                        need_wake = (
                            wake_int == 1
                            and row["execution_state"] == "queued"
                            and int(row["wake_pending_count"]) == 1
                        )
                    try:
                        await tx.execute(
                            f"INSERT INTO {self.q} ("
                            "session_id,item_id,inbox_version,kind,priority,wake,status,"
                            "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                            "meta_json,request_hash,result_kind,created_at,expires_at) "
                            f"VALUES (?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,{now_sql},?)",
                            (
                                target.session_id, ref.message_id, new_inbox_ver, kind, priority,
                                wake_int, reply_to_sid, reply_rk, content, content_bytes,
                                meta, digest, result_kind, deadline_ms,
                            ),
                        )
                    except Exception as exc:
                        if is_duplicate_key_error(exc):
                            raise _AcceptSlowPath("duplicate") from exc
                        raise
                receipt = Receipt(ref, target, ref.message_id, True, "accepted", None, "not_needed")
                return (receipt, need_wake) if _with_activation else receipt
            except _AcceptSlowPath:
                receipt, need_wake = await self._accept_slow_path(
                    target, ref, digest, content, meta, content_bytes,
                    actor=actor, priority=priority, deadline_ms=deadline_ms,
                    kind=kind, result_kind=result_kind, wake_int=wake_int,
                    limit=limit, reply_to_sid=reply_to_sid, reply_rk=reply_rk,
                )
                return (receipt, need_wake) if _with_activation else receipt
        except RuntimeRequestError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            raise StoreError(ref) from exc

    async def _accept_slow_path(
        self, target, ref, digest, content, meta, content_bytes, *,
        actor, priority, deadline_ms, kind, result_kind, wake_int, limit,
        reply_to_sid, reply_rk,
    ) -> tuple[Receipt, bool]:
        now_sql = self._now_sql()
        diag = await self.db.fetchone(
            "SELECT s.session_id,s.tenant_id,s.principal_id,s.parent_session_id,"
            f"s.lifecycle,s.pending_count,s.pending_bytes,{now_sql} AS db_now,"
            "q.id AS existing_id,q.request_hash AS existing_hash,"
            "q.result_send_index AS result_send_index,"
            "q.delivered_send_index AS delivered_send_index,q.error_detail AS error_detail "
            f"FROM {self.s} s LEFT JOIN {self.q} q "
            "ON q.session_id=s.session_id AND q.item_id=? WHERE s.session_id=?",
            (ref.message_id, target.session_id),
        )
        if diag is None:
            raise RuntimeRequestError("not_found")
        if (diag["tenant_id"], diag["principal_id"]) != (actor.tenant_id, actor.principal_id):
            raise RuntimeRequestError("forbidden")
        is_deep_ancestor = False
        if actor.session_id is not None:
            direct = actor.session_id in (diag["session_id"], diag["parent_session_id"])
            if not direct:
                if not await self._is_ancestor(self.db, diag["parent_session_id"], actor.session_id):
                    raise RuntimeRequestError("forbidden")
                is_deep_ancestor = True
        if diag["lifecycle"] != "open":
            if diag["lifecycle"] == "suspended":
                raise RuntimeRequestError("session_suspended")
            raise RuntimeRequestError("session_closed")
        if diag["existing_id"] is not None:
            if diag["existing_hash"] != digest:
                raise RuntimeRequestError("request_conflict")
            return self._receipt(ref, dict(diag)), False
        if deadline_ms is not None and deadline_ms <= int(diag["db_now"]):
            raise RuntimeRequestError("deadline_expired")
        if is_deep_ancestor:
            pi = self._p_int()
            async with self.db.transaction() as tx:
                try:
                    if self._dialect == "mysql":
                        rc, packed = await tx.execute_packed(
                            f"UPDATE {self.s} SET "
                            "inbox_version=(LAST_INSERT_ID("
                            "((inbox_version+1)<<1)|"
                            "(CASE WHEN execution_state='idle' AND ?=1 THEN 1 ELSE 0 END)"
                            ")>>1),"
                            "wake_inbox_seq=wake_inbox_seq+?,"
                            "wake_pending_count=wake_pending_count+?,"
                            "pending_count=pending_count+1,"
                            "pending_bytes=pending_bytes+?,"
                            "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' "
                            f"AND (? IS NULL OR ?>{now_sql}) "
                            "AND pending_count<? AND (pending_bytes+?)<=?",
                            (
                                wake_int, wake_int, wake_int, content_bytes, wake_int,
                                target.session_id, deadline_ms, deadline_ms,
                                limit, content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if rc == 0:
                            raise CapacityError()
                        new_ver, need_wake = packed >> 1, bool(packed & 1)
                    else:
                        row = await tx.fetchone(
                            f"UPDATE {self.s} SET "
                            "inbox_version=inbox_version+1,"
                            "wake_inbox_seq=wake_inbox_seq+?,"
                            "wake_pending_count=wake_pending_count+?,"
                            "pending_count=pending_count+1,"
                            "pending_bytes=pending_bytes+?,"
                            "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' "
                            f"AND ({pi} IS NULL OR {pi}>{now_sql}) "
                            "AND pending_count<? AND (pending_bytes+?)<=? "
                            "RETURNING inbox_version,execution_state,wake_pending_count",
                            (
                                wake_int, wake_int, content_bytes, wake_int,
                                target.session_id, deadline_ms, deadline_ms,
                                limit, content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if row is None:
                            raise CapacityError()
                        new_ver = int(row["inbox_version"])
                        need_wake = (
                            wake_int == 1
                            and row["execution_state"] == "queued"
                            and int(row["wake_pending_count"]) == 1
                        )
                    await tx.execute(
                        f"INSERT INTO {self.q} ("
                        "session_id,item_id,inbox_version,kind,priority,wake,status,"
                        "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                        "meta_json,request_hash,result_kind,created_at,expires_at) "
                        f"VALUES (?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,{now_sql},?)",
                        (
                            target.session_id, ref.message_id, new_ver, kind, priority,
                            wake_int, reply_to_sid, reply_rk, content, content_bytes,
                            meta, digest, result_kind, deadline_ms,
                        ),
                    )
                except Exception as exc:
                    if is_duplicate_key_error(exc):
                        ex = await self.db.fetchone(
                            f"SELECT request_hash,result_send_index,delivered_send_index,error_detail "
                            f"FROM {self.q} WHERE session_id=? AND item_id=?",
                            (target.session_id, ref.message_id),
                        )
                        if ex is not None:
                            if ex["request_hash"] != digest:
                                raise RuntimeRequestError("request_conflict") from None
                            return self._receipt(ref, dict(ex)), False
                    raise
            return Receipt(ref, target, ref.message_id, True, "accepted", None, "not_needed"), need_wake
        raise CapacityError()

    async def _accept_existing_tx(
        self, tx, target, ref, digest, content, content_bytes, *,
        actor, priority, deadline_ms, kind, result_kind, wake,
        reply_to_session_id=None, reply_result_kind=None,
    ) -> Receipt:
        """Accept into an already-locked session within an existing transaction (used by create retries)."""
        existing = await tx.fetchone(
            f"SELECT request_hash,result_send_index,delivered_send_index,error_detail "
            f"FROM {self.q} WHERE session_id=? AND item_id=?",
            (target.session_id, ref.message_id),
        )
        if existing is not None:
            if existing["request_hash"] != digest:
                raise RuntimeRequestError("request_conflict")
            return self._receipt(ref, dict(existing))
        now_sql = self._now_sql()
        wake_int = 1 if wake else 0
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        if self._dialect == "mysql":
            rc, packed = await tx.execute_packed(
                f"UPDATE {self.s} SET inbox_version=LAST_INSERT_ID(inbox_version+1),"
                "wake_inbox_seq=wake_inbox_seq+?,wake_pending_count=wake_pending_count+?,"
                "pending_count=pending_count+1,pending_bytes=pending_bytes+?,"
                "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                f"updated_at={now_sql} WHERE session_id=? AND lifecycle='open' "
                "AND pending_count<? AND (pending_bytes+?)<=?",
                (wake_int, wake_int, content_bytes, wake_int, target.session_id,
                 self.hard_pending_limit, content_bytes, self.inbox_byte_limit),
            )
            if rc == 0:
                raise CapacityError()
            new_ver = packed
        else:
            row = await tx.fetchone(
                f"UPDATE {self.s} SET inbox_version=inbox_version+1,"
                "wake_inbox_seq=wake_inbox_seq+?,wake_pending_count=wake_pending_count+?,"
                "pending_count=pending_count+1,pending_bytes=pending_bytes+?,"
                "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                f"updated_at={now_sql} WHERE session_id=? AND lifecycle='open' "
                "AND pending_count<? AND (pending_bytes+?)<=? RETURNING inbox_version",
                (wake_int, wake_int, content_bytes, wake_int, target.session_id,
                 self.hard_pending_limit, content_bytes, self.inbox_byte_limit),
            )
            if row is None:
                raise CapacityError()
            new_ver = int(row["inbox_version"])
        await tx.execute(
            f"INSERT INTO {self.q} ("
            "session_id,item_id,inbox_version,kind,priority,wake,status,"
            "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
            "meta_json,request_hash,result_kind,created_at,expires_at) "
            f"VALUES (?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,{now_sql},?)",
            (
                target.session_id, ref.message_id, new_ver, kind, priority, wake_int,
                reply_to_session_id, reply_result_kind, content, content_bytes,
                meta, digest, result_kind, deadline_ms,
            ),
        )
        return Receipt(ref, target, ref.message_id, True, "accepted", None, "not_needed")

    # ── Fused Local Accept + Claim (2-Tx Fast Path, Probe 8 Entry #1) ────────

    async def accept_and_claim(
        self, target, body, *, message_id, actor, node_id: str, boot_id: str,
        ttl_s: float = 30.0, priority: str = "normal", deadline_ms: int | None = None,
        kind: str = "user", result_kind: str | None = None, wake: bool = True,
        reply_route: dict | None = None, public: bool = True, can_execute=None,
    ) -> tuple[Receipt, ClaimedBatch | None, bool]:
        """Fused 1-tx accept + claim when idle & empty inbox; transparent fallback to ``accept()``.

        Promotes ``next_descriptor_json/hash`` atomically in the same ``UPDATE sessions`` statement.
        Returns ``(receipt, claimed_batch_or_None, need_wake)``.
        """
        if ttl_s <= 0:
            raise RuntimeRequestError("invalid_options", "ttl_s")
        ref, digest, content, meta, content_bytes, wake_int, reply_to_sid, reply_rk = (
            self._validate_accept_args(
                target, body, message_id=message_id, priority=priority,
                kind=kind, wake=wake, reply_route=reply_route, public=public,
            )
        )
        if not wake:
            receipt, need_wake = await self.accept(
                target, body, message_id=message_id, actor=actor, priority=priority,
                deadline_ms=deadline_ms, kind=kind, result_kind=result_kind,
                wake=False, reply_route=reply_route, public=public, _with_activation=True,
            )
            return receipt, None, need_wake

        now_sql = self._now_sql()
        pt, pi = self._p_text(), self._p_int()
        ttl_ms = int(ttl_s * 1000)
        try:
            try:
                async with self.db.transaction() as tx:
                    if self._dialect == "mysql":
                        rc = await tx.execute(
                            f"UPDATE {self.s} SET "
                            "execution_state='running',"
                            "owner_node_id=?,owner_boot_id=?,"
                            "owner_epoch=owner_epoch+1,"
                            "send_index=send_index+1,"
                            "inbox_version=inbox_version+1,"
                            "wake_inbox_seq=wake_inbox_seq+1,"
                            "next_seq=CASE WHEN session_kind='workflow_driver' THEN next_seq ELSE next_seq+1 END,"
                            f"owner_lease_until_ms={now_sql}+?,"
                            "wake_pending_count=0,pending_count=0,pending_bytes=0,"
                            "cancel_by=NULL,cancel_reason=NULL,ready_item_json=NULL,"
                            "descriptor_json=COALESCE(next_descriptor_json,descriptor_json),"
                            "descriptor_hash=COALESCE(next_descriptor_hash,descriptor_hash),"
                            "next_descriptor_json=NULL,next_descriptor_hash=NULL,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' AND execution_state='idle' "
                            "AND wake_pending_count=0 AND pending_count=0 AND pending_tool_seq IS NULL "
                            "AND tenant_id=? AND principal_id=? "
                            "AND (? IS NULL OR session_id=? OR parent_session_id=?) "
                            f"AND (? IS NULL OR ?>{now_sql}) "
                            "AND ?<=?",
                            (
                                node_id, boot_id, ttl_ms,
                                target.session_id, actor.tenant_id, actor.principal_id,
                                actor.session_id, actor.session_id, actor.session_id,
                                deadline_ms, deadline_ms,
                                content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if rc == 0:
                            raise _AcceptSlowPath("not_idle_fast")
                        srow = await tx.fetchone(
                            "SELECT owner_epoch,send_index,inbox_version,background_version,"
                            "runtime_version,wake_inbox_seq,next_seq,"
                            "cancel_seq,depth,session_kind,tenant_id,principal_id,parent_session_id,"
                            "root_session_id,result_route_json,start_input_message_id,"
                            f"descriptor_json,updated_at FROM {self.s} WHERE session_id=?",
                            (target.session_id,),
                        )
                        if srow is None:
                            raise _AcceptSlowPath("missing")
                    else:
                        srow = await tx.fetchone(
                            f"UPDATE {self.s} SET "
                            "execution_state='running',"
                            "owner_node_id=?,owner_boot_id=?,"
                            "owner_epoch=owner_epoch+1,"
                            "send_index=send_index+1,"
                            "inbox_version=inbox_version+1,"
                            "wake_inbox_seq=wake_inbox_seq+1,"
                            "next_seq=CASE WHEN session_kind='workflow_driver' THEN next_seq ELSE next_seq+1 END,"
                            f"owner_lease_until_ms={now_sql}+?,"
                            "wake_pending_count=0,pending_count=0,pending_bytes=0,"
                            "cancel_by=NULL,cancel_reason=NULL,ready_item_json=NULL,"
                            "descriptor_json=COALESCE(next_descriptor_json,descriptor_json),"
                            "descriptor_hash=COALESCE(next_descriptor_hash,descriptor_hash),"
                            "next_descriptor_json=NULL,next_descriptor_hash=NULL,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' AND execution_state='idle' "
                            "AND wake_pending_count=0 AND pending_count=0 AND pending_tool_seq IS NULL "
                            "AND tenant_id=? AND principal_id=? "
                            f"AND ({pt} IS NULL OR session_id={pt} OR parent_session_id={pt}) "
                            f"AND ({pi} IS NULL OR {pi}>{now_sql}) "
                            f"AND {pi}<={pi} "
                            "RETURNING owner_epoch,send_index,inbox_version,background_version,"
                            "runtime_version,wake_inbox_seq,next_seq,"
                            "cancel_seq,depth,session_kind,tenant_id,principal_id,parent_session_id,"
                            "root_session_id,result_route_json,start_input_message_id,"
                            "descriptor_json,updated_at",
                            (
                                node_id, boot_id, ttl_ms,
                                target.session_id, actor.tenant_id, actor.principal_id,
                                actor.session_id, actor.session_id, actor.session_id,
                                deadline_ms, deadline_ms,
                                content_bytes, self.inbox_byte_limit,
                            ),
                        )
                        if srow is None:
                            raise _AcceptSlowPath("not_idle_fast")

                    descriptor = RuntimeDescriptor(**json.loads(srow["descriptor_json"]))
                    if can_execute is not None and not can_execute(descriptor):
                        raise _AcceptSlowPath("incapable")

                    send_idx = int(srow["send_index"])
                    new_inbox_ver = int(srow["inbox_version"])
                    new_wake_seq = int(srow["wake_inbox_seq"])
                    is_driver = srow["session_kind"] == "workflow_driver"
                    user_seq = None if is_driver else int(srow["next_seq"]) - 1
                    db_now = int(srow["updated_at"])

                    try:
                        await tx.execute(
                            f"INSERT INTO {self.q} ("
                            "session_id,item_id,inbox_version,kind,priority,wake,status,"
                            "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                            "meta_json,request_hash,result_kind,created_at,expires_at,"
                            "delivered_at,delivered_seq,delivered_send_index) "
                            "VALUES (?,?,?,?,?,1,'delivered',?,?,?,?,?,?,?,"
                            f"{now_sql},?,{now_sql},?,?)",
                            (
                                target.session_id, ref.message_id, new_inbox_ver, kind, priority,
                                reply_to_sid, reply_rk, content, content_bytes,
                                meta, digest, result_kind, deadline_ms, user_seq, send_idx,
                            ),
                        )
                    except Exception as exc:
                        if is_duplicate_key_error(exc):
                            raise _AcceptSlowPath("duplicate") from exc
                        raise

                    merged = self.merge_batch([{
                        "item_id": ref.message_id, "kind": kind,
                        "priority": priority, "content_json": content,
                    }])
                    merged_json = canonical(merged)
                    msg_meta = {"content_encoding": "json", "input_refs": [plain(ref)]}
                    if not is_driver:
                        await self._insert_message(
                            tx, target.session_id, user_seq, "user", merged_json,
                            msg_meta, send_idx, db_now,
                        )
                    user_row = StoredMessageRow(
                        session_id=target.session_id, seq=user_seq or 0, role="user", name=None,
                        content=merged_json, tool_calls=None, tool_call_id=None, round_index=0,
                        state=MessageState.ACTIVE, meta=msg_meta, created_at=db_now,
                        send_index=send_idx,
                    )
                    batch = ClaimedBatch(
                        target=target,
                        send_index=send_idx,
                        message_seq=user_seq or 0,
                        input_refs=(ref,),
                        owner_epoch=int(srow["owner_epoch"]),
                        owner_node_id=node_id,
                        owner_boot_id=boot_id,
                        descriptor=descriptor,
                        tenant_id=srow["tenant_id"],
                        principal_id=srow["principal_id"],
                        parent_session_id=srow["parent_session_id"],
                        user_row=user_row,
                        result_route=(
                            json.loads(srow["result_route_json"])
                            if srow["result_route_json"] else None
                        ),
                        session_updated_at=db_now,
                        inbox_version=new_inbox_ver,
                        background_version=int(srow["background_version"]),
                        runtime_version=int(srow["runtime_version"]),
                        wake_inbox_seq=new_wake_seq,
                        cancel_seq=int(srow["cancel_seq"]),
                        start_input_message_id=srow["start_input_message_id"],
                        root_session_id=str(srow["root_session_id"]),
                        depth=int(srow["depth"]),
                        reply_to_session_id=reply_to_sid,
                        reply_result_kind=reply_rk,
                    )
                receipt = Receipt(ref, target, ref.message_id, True, "accepted", send_idx, "not_needed")
                return receipt, batch, False
            except _AcceptSlowPath:
                receipt, need_wake = await self.accept(
                    target, body, message_id=message_id, actor=actor, priority=priority,
                    deadline_ms=deadline_ms, kind=kind, result_kind=result_kind,
                    wake=wake, reply_route=reply_route, public=public, _with_activation=True,
                )
                return receipt, None, need_wake
        except RuntimeRequestError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            raise StoreError(ref) from exc

    # ── Standard Claim & 1-SQL Pre-Staged Child Claim (Probe 8 Entries #2 & #3) ──

    async def claim(
        self, target, *, node_id: str, boot_id: str, ttl_s: float = 30.0,
        can_execute=None,
    ) -> ClaimedBatch | None:
        """Claim queued inputs atomically (2 control SQL on SQLite/PG; 3 on MySQL).

        - Pre-staged cross-container child from ``create_many`` takes 1 SQL (SQLite/PG)
          or 2 SQL (MySQL) with 0 message inserts.
        - Promotes ``next_descriptor_json/hash`` atomically in the same ``UPDATE sessions``.
        - Repairs ``pending_tool_seq`` (if non-NULL) before inserting the new merged user row.
        """
        self.check_store(target)
        if ttl_s <= 0:
            raise RuntimeRequestError("invalid_options", "ttl_s")
        now_sql = self._now_sql()
        ttl_ms = int(ttl_s * 1000)
        context: dict[str, Any] = {}
        try:
            async with self.db.transaction() as tx:
                if self._dialect == "mysql":
                    # MySQL evaluates SET left-to-right: read pending_count & ready_item_json
                    # BEFORE zeroing pending_count, and promote descriptor BEFORE clearing next_descriptor!
                    rc = await tx.execute(
                        f"UPDATE {self.s} SET "
                        "execution_state='running',"
                        "owner_node_id=?,owner_boot_id=?,"
                        "owner_epoch=owner_epoch+1,"
                        "send_index=send_index+1,"
                        "next_seq=CASE "
                        "WHEN session_kind='workflow_driver' THEN next_seq "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN next_seq "
                        "ELSE next_seq+1 END,"
                        "cancel_by=CASE "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN cancel_by "
                        "ELSE NULL END,"
                        "cancel_reason=CASE "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN cancel_reason "
                        "ELSE NULL END,"
                        "ready_item_json=CASE "
                        "WHEN pending_count=0 THEN ready_item_json "
                        "WHEN ready_item_json IS NOT NULL THEN '' "
                        "ELSE NULL END,"
                        "descriptor_json=COALESCE(next_descriptor_json,descriptor_json),"
                        "descriptor_hash=COALESCE(next_descriptor_hash,descriptor_hash),"
                        "next_descriptor_json=NULL,next_descriptor_hash=NULL,"
                        f"owner_lease_until_ms={now_sql}+?,"
                        "wake_pending_count=0,pending_count=0,pending_bytes=0,"
                        f"updated_at={now_sql} "
                        "WHERE session_id=? AND lifecycle='open' "
                        "AND execution_state='queued' AND wake_pending_count>0",
                        (node_id, boot_id, ttl_ms, target.session_id),
                    )
                    if rc == 0:
                        return None
                    joined = await tx.fetchall(
                        "SELECT s.owner_epoch,s.send_index,s.inbox_version,s.background_version,"
                        "s.runtime_version,s.wake_inbox_seq,"
                        "s.next_seq,s.cancel_seq,s.cancel_by,s.pending_tool_seq,s.depth,"
                        "s.session_kind,s.tenant_id,s.principal_id,s.parent_session_id,"
                        "s.root_session_id,s.result_route_json,s.start_input_message_id,"
                        "s.ready_item_json,s.descriptor_json,s.updated_at,"
                        "q.id AS q_id,q.item_id AS q_item_id,q.kind AS q_kind,"
                        "q.priority AS q_priority,q.content_json AS q_content_json,"
                        "q.meta_json AS q_meta_json,q.cancel_by AS q_cancel_by,"
                        "q.reply_to_session_id AS q_reply_to_session_id,"
                        "q.reply_result_kind AS q_reply_result_kind "
                        f"FROM {self.s} s LEFT JOIN {self.q} q "
                        "ON q.session_id=s.session_id AND q.status='pending' "
                        "WHERE s.session_id=? "
                        "ORDER BY CASE q.priority WHEN 'urgent' THEN 0 ELSE 1 END,q.id",
                        (target.session_id,),
                    )
                    if not joined:
                        raise _ClaimRollback()
                    srow = joined[0]
                    descriptor = RuntimeDescriptor(**json.loads(srow["descriptor_json"]))
                    if can_execute is not None and not can_execute(descriptor):
                        raise _ClaimRollback()
                    # 1-SQL (2 SQL on MySQL) pre-staged cross-container child claim!
                    if srow["ready_item_json"] and srow["q_id"] is None and srow["pending_tool_seq"] is None:
                        return self._build_prestaged_batch(
                            target, srow, descriptor, node_id, boot_id,
                        )
                    rows = [
                        {
                            "id": r["q_id"], "item_id": r["q_item_id"], "kind": r["q_kind"],
                            "priority": r["q_priority"], "content_json": r["q_content_json"],
                            "meta_json": r["q_meta_json"], "cancel_by": r["q_cancel_by"],
                            "reply_to_session_id": r["q_reply_to_session_id"],
                            "reply_result_kind": r["q_reply_result_kind"],
                        }
                        for r in joined if r["q_id"] is not None
                    ]
                    if not rows:
                        raise _ClaimRollback()
                    pending_already_delivered = False
                else:
                    srow = await tx.fetchone(
                        f"UPDATE {self.s} SET "
                        "execution_state='running',"
                        "owner_node_id=?,owner_boot_id=?,"
                        "owner_epoch=owner_epoch+1,"
                        "send_index=send_index+1,"
                        "next_seq=CASE "
                        "WHEN session_kind='workflow_driver' THEN next_seq "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN next_seq "
                        "ELSE next_seq+1 END,"
                        "cancel_by=CASE "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN cancel_by "
                        "ELSE NULL END,"
                        "cancel_reason=CASE "
                        "WHEN pending_count=0 AND ready_item_json IS NOT NULL THEN cancel_reason "
                        "ELSE NULL END,"
                        "ready_item_json=CASE "
                        "WHEN pending_count=0 THEN ready_item_json "
                        "WHEN ready_item_json IS NOT NULL THEN '' "
                        "ELSE NULL END,"
                        "descriptor_json=COALESCE(next_descriptor_json,descriptor_json),"
                        "descriptor_hash=COALESCE(next_descriptor_hash,descriptor_hash),"
                        "next_descriptor_json=NULL,next_descriptor_hash=NULL,"
                        f"owner_lease_until_ms={now_sql}+?,"
                        "wake_pending_count=0,pending_count=0,pending_bytes=0,"
                        f"updated_at={now_sql} "
                        "WHERE session_id=? AND lifecycle='open' "
                        "AND execution_state='queued' AND wake_pending_count>0 "
                        "RETURNING owner_epoch,send_index,inbox_version,background_version,"
                        "runtime_version,wake_inbox_seq,"
                        "next_seq,cancel_seq,cancel_by,pending_tool_seq,depth,"
                        "session_kind,tenant_id,principal_id,parent_session_id,"
                        "root_session_id,result_route_json,start_input_message_id,"
                        "ready_item_json,descriptor_json,updated_at",
                        (node_id, boot_id, ttl_ms, target.session_id),
                    )
                    if srow is None:
                        return None
                    descriptor = RuntimeDescriptor(**json.loads(srow["descriptor_json"]))
                    if can_execute is not None and not can_execute(descriptor):
                        raise _ClaimRollback()
                    # Optimization 6: 1-SQL Claim for pre-staged cross-container child!
                    if srow["ready_item_json"] and srow["pending_tool_seq"] is None:
                        return self._build_prestaged_batch(
                            target, srow, descriptor, node_id, boot_id,
                        )
                    send_idx_tmp = int(srow["send_index"])
                    is_drv_tmp = srow["session_kind"] == "workflow_driver"
                    user_seq_tmp = None if is_drv_tmp else int(srow["next_seq"]) - 1
                    rows = await tx.fetchall(
                        f"UPDATE {self.q} SET status='delivered',delivered_send_index=?,"
                        f"delivered_seq=?,delivered_at={now_sql} "
                        "WHERE session_id=? AND status='pending' "
                        "RETURNING id,item_id,kind,priority,content_json,meta_json,"
                        "cancel_by,reply_to_session_id,reply_result_kind",
                        (send_idx_tmp, user_seq_tmp, target.session_id),
                    )
                    if not rows:
                        raise _ClaimRollback()
                    rows.sort(key=lambda r: (0 if r["priority"] == "urgent" else 1, int(r["id"])))
                    pending_already_delivered = True

                send_index = int(srow["send_index"])
                is_driver = srow["session_kind"] == "workflow_driver"
                db_now = int(srow["updated_at"])
                seq = int(srow["next_seq"]) - (0 if is_driver else 1)

                # Rare path: repair dangling pending_tool_seq from a previous interrupted send
                if srow["pending_tool_seq"] is not None:
                    # Rewind next_seq to pre-user allocation, repair old send's tools, then allocate user seq!
                    pre_user_next_seq = seq
                    repaired_state = {
                        "pending_tool_seq": srow["pending_tool_seq"],
                        "next_seq": pre_user_next_seq,
                    }
                    repaired = await self._repair_pending_tx(
                        tx, target.session_id, send_index - 1, repaired_state,
                    )
                    if repaired:
                        seq = pre_user_next_seq + repaired
                        new_next_seq = seq + (0 if is_driver else 1)
                        await tx.execute(
                            f"UPDATE {self.s} SET next_seq=?,pending_tool_seq=NULL WHERE session_id=?",
                            (new_next_seq, target.session_id),
                        )
                        if pending_already_delivered and not is_driver:
                            ids = [int(r["id"]) for r in rows]
                            await tx.execute(
                                f"UPDATE {self.q} SET delivered_seq=? "
                                f"WHERE session_id=? AND id IN ({','.join('?' for _ in ids)})",
                                (seq, target.session_id, *ids),
                            )

                # Rare path: extra inputs arrived on a pre-staged child before first claim
                if srow["ready_item_json"] == "":
                    await tx.execute(
                        f"UPDATE {self.s} SET ready_item_json=NULL WHERE session_id=?",
                        (target.session_id,),
                    )
                    prestaged = await tx.fetchall(
                        f"SELECT id,item_id,kind,priority,content_json,meta_json,cancel_by,"
                        f"reply_to_session_id,reply_result_kind FROM {self.q} "
                        "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                        "ORDER BY id ASC",
                        (target.session_id, send_index),
                    )
                    row_ids = {int(r["id"]) for r in rows}
                    extra_pre = [r for r in prestaged if int(r["id"]) not in row_ids]
                    if extra_pre:
                        rows = [*extra_pre, *rows]

                refs = tuple(
                    InputRef(target.store_id, target.session_id, r["item_id"]) for r in rows
                )
                context["refs"] = refs
                deliv_seq = None if is_driver else seq
                if not pending_already_delivered:
                    ids = [int(r["id"]) for r in rows]
                    delivered = await tx.execute(
                        f"UPDATE {self.q} SET status='delivered',delivered_at={now_sql},"
                        "delivered_seq=?,delivered_send_index=? "
                        f"WHERE session_id=? AND status='pending' AND id IN ({','.join('?' for _ in ids)})",
                        (deliv_seq, send_index, target.session_id, *ids),
                    )
                    if delivered != len(ids):
                        raise SessionLeaseLost(target.session_id)

                merged = self.merge_batch(rows)
                merged_json = canonical(merged)
                message_meta = {
                    "content_encoding": "json",
                    "input_refs": [plain(ref) for ref in refs],
                }
                if not is_driver:
                    await self._insert_message(
                        tx, target.session_id, seq, "user", merged_json, message_meta,
                        send_index, db_now,
                    )
                user_row = StoredMessageRow(
                    session_id=target.session_id, seq=seq, role="user", name=None,
                    content=merged_json, tool_calls=None, tool_call_id=None, round_index=0,
                    state=MessageState.ACTIVE, meta=message_meta, created_at=db_now,
                    send_index=send_index,
                )
                reply_to_sid = next(
                    (r.get("reply_to_session_id") for r in rows if r.get("reply_to_session_id")), None
                )
                reply_rk = next(
                    (r.get("reply_result_kind") for r in rows if r.get("reply_to_session_id")), None
                )
                batch = ClaimedBatch(
                    target=target,
                    send_index=send_index,
                    message_seq=seq,
                    input_refs=refs,
                    owner_epoch=int(srow["owner_epoch"]),
                    owner_node_id=node_id,
                    owner_boot_id=boot_id,
                    descriptor=descriptor,
                    tenant_id=srow["tenant_id"],
                    principal_id=srow["principal_id"],
                    parent_session_id=srow["parent_session_id"],
                    user_row=user_row,
                    result_route=(
                        json.loads(srow["result_route_json"])
                        if srow["result_route_json"] else None
                    ),
                    session_updated_at=db_now,
                    inbox_version=int(srow["inbox_version"]),
                    background_version=int(srow["background_version"]),
                    runtime_version=int(srow["runtime_version"]),
                    wake_inbox_seq=int(srow["wake_inbox_seq"]),
                    cancel_seq=int(srow["cancel_seq"]),
                    cancel_by=srow["cancel_by"],
                    start_input_message_id=srow["start_input_message_id"],
                    root_session_id=str(srow["root_session_id"]),
                    depth=int(srow["depth"]),
                    reply_to_session_id=reply_to_sid,
                    reply_result_kind=reply_rk,
                )
                context["batch"] = batch
                return batch
        except _ClaimRollback:
            return None
        except RuntimeRequestError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            recovered = await self._claim_after_unknown(target, context)
            if recovered is not None:
                return recovered
            ref = context.get("refs", (None,))[0]
            raise StoreError(ref) from exc

    def _build_prestaged_batch(
        self, target: TargetRef, srow, descriptor: RuntimeDescriptor, node_id: str, boot_id: str,
    ) -> ClaimedBatch:
        ready = json.loads(srow["ready_item_json"])
        ref = InputRef(target.store_id, target.session_id, ready["item_id"])
        send_index = int(srow["send_index"])
        db_now = int(srow["updated_at"])
        user_seq = int(ready.get("user_seq") or 1) if srow["session_kind"] != "workflow_driver" else 0
        user_content = ready.get("user_content") or canonical(self.merge_batch([ready]))
        msg_meta = {"content_encoding": "json", "input_refs": [plain(ref)]}
        user_row = StoredMessageRow(
            session_id=target.session_id, seq=user_seq, role="user", name=None,
            content=user_content, tool_calls=None, tool_call_id=None, round_index=0,
            state=MessageState.ACTIVE, meta=msg_meta, created_at=db_now, send_index=send_index,
        )
        return ClaimedBatch(
            target=target,
            send_index=send_index,
            message_seq=user_seq,
            input_refs=(ref,),
            owner_epoch=int(srow["owner_epoch"]),
            owner_node_id=node_id,
            owner_boot_id=boot_id,
            descriptor=descriptor,
            tenant_id=srow["tenant_id"],
            principal_id=srow["principal_id"],
            parent_session_id=srow["parent_session_id"],
            user_row=user_row,
            result_route=(
                json.loads(srow["result_route_json"]) if srow["result_route_json"] else None
            ),
            session_updated_at=db_now,
            inbox_version=int(srow["inbox_version"]),
            background_version=int(srow["background_version"]),
            runtime_version=int(srow["runtime_version"]),
            wake_inbox_seq=int(srow["wake_inbox_seq"]),
            cancel_seq=int(srow["cancel_seq"]),
            cancel_by=srow["cancel_by"],
            start_input_message_id=srow["start_input_message_id"],
            root_session_id=str(srow["root_session_id"]),
            depth=int(srow["depth"]),
            reply_to_session_id=ready.get("reply_to_session_id"),
            reply_result_kind=ready.get("reply_result_kind"),
        )

    async def release_claim(self, batch: ClaimedBatch):
        """Atomically release an unexecuted claim, honouring active cancellation."""
        self.check_store(batch.target)
        ids = [ref.message_id for ref in batch.input_refs]
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, batch.target.session_id)
            if (
                state["execution_state"] != "running"
                or int(state["send_index"]) != batch.send_index
                or int(state["owner_epoch"]) != batch.owner_epoch
            ):
                raise SessionLeaseLost(batch.target.session_id)
            if int(state["cancel_seq"]) > batch.cancel_seq or state["lifecycle"] in ("closing", "closed"):
                await self._final_tx(
                    tx, batch, batch.input_refs, "cancelled", "", {}, None,
                    state["cancel_reason"] or "cancelled", "cancelled",
                )
            else:
                qrows = await tx.fetchall(
                    f"SELECT wake,content_bytes FROM {self.q} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    f"AND item_id IN ({','.join('?' for _ in ids)})",
                    (batch.target.session_id, batch.send_index, *ids),
                )
                restored = await tx.execute(
                    f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                    "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                    f"AND delivered_send_index=? AND item_id IN ({','.join('?' for _ in ids)})",
                    (batch.target.session_id, batch.send_index, *ids),
                )
                if restored != len(ids):
                    raise SessionLeaseLost(batch.target.session_id)
                w_add = sum(int(r["wake"]) for r in qrows)
                b_add = sum(int(r["content_bytes"]) for r in qrows)
                changed = await tx.execute(
                    f"UPDATE {self.s} SET execution_state='queued',"
                    "send_index=CASE WHEN send_index>0 THEN send_index-1 ELSE 0 END,"
                    "next_seq=CASE WHEN next_seq>1 THEN next_seq-1 ELSE 1 END,"
                    "wake_pending_count=wake_pending_count+?,"
                    "pending_count=pending_count+?,"
                    "pending_bytes=pending_bytes+?,"
                    "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                    "cancel_reason=NULL,cancel_by=NULL "
                    "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                    (w_add, len(ids), b_add, batch.target.session_id, batch.owner_epoch),
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
        """Read current control boundary fields from ``sessions`` (slow/diagnostic path)."""
        self.check_store(target)
        return await self.db.fetchone(
            "SELECT execution_state,send_index,owner_epoch,lifecycle,"
            "inbox_version,wake_inbox_seq,background_version,runtime_version,"
            f"cancel_seq,cancel_by,cancel_reason FROM {self.s} WHERE session_id=?",
            (target.session_id,),
        )

    async def session_entry(self, target):
        """Session identity/control facts and DB time in one read."""
        self.check_store(target)
        return await self.db.fetchone(
            "SELECT lifecycle,descriptor_json,tenant_id,principal_id,parent_session_id,"
            f"execution_state,owner_lease_until_ms,owner_epoch,{self._now_sql()} AS db_now "
            f"FROM {self.s} WHERE session_id=?",
            (target.session_id,),
        )

    async def claim_follow_up(self, batch: ClaimedBatch):
        """Atomically drain ALL pending inbox items (normal + urgent) into the RUNNING send."""
        self.check_store(batch.target)
        now_sql = self._now_sql()
        from power_loop.runtime.store import send_reads
        holding = send_reads.current(self.store, batch.target.session_id)
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, batch.target.session_id)
            if (
                state["execution_state"] != "running"
                or int(state["send_index"]) != batch.send_index
                or int(state["owner_epoch"]) != batch.owner_epoch
            ):
                raise SessionLeaseLost(batch.target.session_id)
            rows = await tx.fetchall(
                "SELECT id,item_id,kind,priority,wake,content_json,content_bytes,"
                f"meta_json,cancel_by,reply_to_session_id,reply_result_kind,{now_sql} AS db_now "
                f"FROM {self.q} WHERE session_id=? AND status='pending' "
                "ORDER BY CASE priority WHEN 'urgent' THEN 0 ELSE 1 END,id",
                (batch.target.session_id,),
            )
            if not rows:
                return None
            refs = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in rows
            )
            db_next_seq = int(state["next_seq"])
            seq = max(db_next_seq, int(holding.next_seq)) if (holding is not None and holding.next_seq is not None) else db_next_seq
            database_now = int(rows[0]["db_now"])
            merged = self.merge_batch(rows)
            w_sub = sum(int(r["wake"]) for r in rows)
            p_sub = len(rows)
            b_sub = sum(int(r["content_bytes"]) for r in rows)
            changed = await tx.execute(
                f"UPDATE {self.s} SET next_seq=?,"
                "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                (
                    seq + 1, w_sub, w_sub, p_sub, p_sub, b_sub, b_sub,
                    batch.target.session_id, batch.owner_epoch,
                ),
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
            wake_seq = int(state["wake_inbox_seq"])
        if holding is not None:
            holding.next_seq = seq + 1
            holding.wake_inbox_seq = wake_seq
            holding.change_snapshot = None
        return refs, seq, merged, version

    async def _repair_pending_tx(self, tx, session_id: str, send_index: int, state) -> int:
        """Synthesize unknown tool results for unresolved tool calls (explicit pending_tool_seq OR messages scan)."""
        pts = state.get("pending_tool_seq") if isinstance(state, dict) else state["pending_tool_seq"]
        assistant = None
        tool_calls: list[dict] = []
        seq = int(state["next_seq"])
        if pts is not None:
            from .pending_tools import read_pending_tools
            # Ensure upper bound covers any messages written via fenced_fast without bumping next_seq
            max_row = await tx.fetchone(
                f"SELECT COALESCE(MAX(seq),0) AS max_seq FROM {self.store.t.messages} WHERE session_id=?",
                (session_id,),
            )
            seq = max(seq, int(max_row["max_seq"]) + 1) if max_row else seq
            assistant, tool_calls = await read_pending_tools(
                self.store, tx, session_id, {"assistant_seq": int(pts)}, seq,
            )
        else:
            # Also detect dangling assistant(tool_calls) written via fenced_fast without pending_tool_seq
            turn_msgs = await tx.fetchall(
                f"SELECT id,seq,role,tool_call_id,send_index,round_index,tool_calls_json "
                f"FROM {self.store.t.messages} WHERE session_id=? AND send_index=? ORDER BY seq ASC",
                (session_id, send_index),
            )
            if turn_msgs:
                seq = max(seq, max(int(m["seq"]) for m in turn_msgs) + 1)
                last_asst = None
                for m in turn_msgs:
                    if m["role"] == "assistant" and m["tool_calls_json"]:
                        last_asst = m
                if last_asst is not None:
                    calls = json.loads(last_asst["tool_calls_json"])
                    completed = {
                        m["tool_call_id"]
                        for m in turn_msgs
                        if m["role"] == "tool" and int(m["seq"]) > int(last_asst["seq"]) and m["tool_call_id"]
                    }
                    missing = [tc for tc in calls if str(tc.get("id") or "") not in completed]
                    if missing:
                        assistant = last_asst
                        tool_calls = missing
        if assistant is None or not tool_calls:
            if pts is not None:
                await tx.execute(
                    f"UPDATE {self.s} SET pending_tool_seq=NULL,next_seq=? WHERE session_id=?",
                    (seq, session_id),
                )
            return 0
        assistant_seq = int(assistant["seq"])
        round_index, asst_send_index = assistant["round_index"], assistant["send_index"]
        names = {
            str(tc.get("id") or ""): (tc.get("function", {}) or {}).get("name")
            if "function" in tc else tc.get("name")
            for tc in tool_calls
        }
        from power_loop.runtime.message_ref import MessageRef
        call_ref = MessageRef(
            self.store.store_id, session_id, assistant["id"], assistant_seq,
            assistant["role"], assistant["tool_call_id"], asst_send_index, round_index,
        )
        database_now = await self._db_now(tx)
        appended = 0
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            if not cid:
                continue
            content = "<aborted: 执行中断，结果未知>"
            meta: dict[str, Any] = {"interrupted": True, "_assistant_seq": assistant_seq}
            child = await tx.fetchone(
                f"SELECT session_id,start_input_message_id FROM {self.s} "
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
                "VALUES (?,?,'tool',?,?,?,?,'active',?,?,?)",
                (session_id, seq, names.get(cid), content, cid,
                 round_index, canonical(meta), asst_send_index, database_now),
            )
            seq += 1
            appended += 1
        await tx.execute(
            f"UPDATE {self.s} SET pending_tool_seq=NULL,next_seq=? WHERE session_id=?",
            (seq, session_id),
        )
        return appended

    async def _insert_message(self, tx, session_id, seq, role, content, meta, send_index, created_at):
        sql = (
            f"INSERT INTO {self.store.t.messages} (session_id,seq,role,content,meta_json,"
            "send_index,created_at) VALUES (?,?,?,?,?,?,?)"
        )
        await tx.execute(
            sql, (session_id, seq, role, content, canonical(meta), send_index, created_at)
        )

    async def _claim_after_unknown(self, target, context):
        if "batch" not in context:
            return None
        batch = context["batch"]
        state = await self.db.fetchone(
            f"SELECT * FROM {self.s} WHERE session_id=?", (target.session_id,)
        )
        if not state or int(state["send_index"]) != batch.send_index:
            return None
        if (
            int(state["owner_epoch"]), state["execution_state"],
            state["owner_node_id"], state["owner_boot_id"],
        ) != (batch.owner_epoch, "running", batch.owner_node_id, batch.owner_boot_id):
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

    # ── Final + River Same-Tx Non-Cascading Delivery (0 final_settle_inbox!) ─

    async def final(
        self, batch: ClaimedBatch, *, consumed=None, status="completed", final_text="",
        usage=None, structured_output=None, error=None, error_code=None, result_seq=None,
        assistant_meta=None, next_seq: int | None = None, result_json: dict | None = None,
        settle_start_input: bool = False,
    ) -> FinalBatch:
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
                    structured_output, error, error_code, result_seq, assistant_meta,
                    next_seq=next_seq, result_json=result_json,
                    settle_start_input=settle_start_input,
                )
        except (RuntimeRequestError, SessionLeaseLost):
            raise
        except (Exception, asyncio.CancelledError) as exc:
            recovered = await self._final_after_unknown(batch, consumed)
            if recovered is not None:
                return recovered
            raise StoreError(consumed[0] if consumed else batch.input_refs[0]) from exc

    async def _final_tx(
        self, tx, batch: ClaimedBatch, consumed, status, final_text, usage, structured_output,
        error, error_code, result_seq=None, assistant_meta=None, *, next_seq=None,
        result_json=None, settle_start_input=False,
    ) -> FinalBatch:
        sid = batch.target.session_id
        now_sql = self._now_sql()
        consumed_refs = tuple(batch.input_refs) if consumed is None else tuple(consumed)
        unconsumed_ids = [ref.message_id for ref in batch.input_refs if ref not in consumed_refs]
        has_unconsumed = bool(unconsumed_ids)

        # Compute effective next_seq before releasing the session row (so 1 UPDATE suffices!).
        # On abnormal/untracked final (exception, cancel, recovery), sync with MAX(seq)+1 so
        # fenced appends made before the failure never cause a next_seq collision on the next send.
        if next_seq is None or status != "completed":
            max_row = await tx.fetchone(
                f"SELECT COALESCE(MAX(seq),0) AS max_seq FROM {self.store.t.messages} WHERE session_id=?",
                (sid,),
            )
            if max_row is not None:
                next_seq = max(int(next_seq or 0), int(max_row["max_seq"]) + 1)
        write_message = result_seq is None and status == "completed"
        effective_next_seq = max(
            int(next_seq or 0),
            (int(result_seq) + 1) if result_seq is not None else 0,
            batch.message_seq + 1,
        )
        keep_pending_tool = status == "interrupted"

        # Step 1 (Deep-first lock order): Fenced UPDATE on child session FIRST (1 control SQL!).
        if self._dialect == "mysql":
            # 4-bit LAST_INSERT_ID packing in 1st SET slot:
            # bit 0: has_more_wake | bit 1: subtree_raced | bit 2: cancel_raced | bit 3: closed_raced
            rc, packed = await tx.execute_packed(
                f"UPDATE {self.s} SET "
                "owner_epoch=(LAST_INSERT_ID("
                "(owner_epoch<<4)"
                "|(CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) THEN 1 ELSE 0 END)"
                "|(CASE WHEN cancel_by='subtree' AND cancel_seq>? THEN 2 ELSE 0 END)"
                "|(CASE WHEN cancel_seq>? THEN 4 ELSE 0 END)"
                "|(CASE WHEN lifecycle IN ('closing','closed') THEN 8 ELSE 0 END)"
                ")>>4),"
                "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) "
                "THEN 'queued' ELSE 'idle' END,"
                "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                "next_seq=CASE WHEN lifecycle IN ('open','suspended') THEN "
                "(CASE WHEN ?>next_seq THEN ? ELSE next_seq END)+? ELSE next_seq END,"
                "pending_tool_seq=CASE WHEN ?=1 THEN pending_tool_seq ELSE NULL END,"
                "ready_item_json=NULL,"
                "cancel_by=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_by END,"
                "cancel_reason=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_reason END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND owner_epoch=? AND execution_state='running'",
                (
                    int(has_unconsumed), batch.cancel_seq, batch.cancel_seq,
                    int(has_unconsumed),
                    effective_next_seq, effective_next_seq, int(write_message),
                    int(keep_pending_tool), batch.cancel_seq, batch.cancel_seq,
                    sid, batch.owner_epoch,
                ),
            )
            if rc == 0:
                raise SessionLeaseLost(sid)
            has_more_wake = bool(packed & 1)
            subtree_raced = bool(packed & 2)
            cancel_raced = bool(packed & 4)
            closed_raced = bool(packed & 8)
            next_state = "queued" if has_more_wake else "idle"
            cancel_by_db = "subtree" if subtree_raced else None
            cancel_reason_db = None
            post_next_seq = effective_next_seq + int(write_message)
            database_now = None
            if closed_raced or cancel_raced or write_message:
                diag = await tx.fetchone(
                    f"SELECT next_seq,cancel_by,cancel_reason,updated_at FROM {self.s} WHERE session_id=?",
                    (sid,),
                )
                if diag is not None:
                    post_next_seq = int(diag["next_seq"])
                    cancel_by_db = diag["cancel_by"]
                    cancel_reason_db = diag["cancel_reason"]
                    database_now = int(diag["updated_at"])
        else:
            srow = await tx.fetchone(
                f"UPDATE {self.s} SET "
                "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) "
                "THEN 'queued' ELSE 'idle' END,"
                "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                "next_seq=CASE WHEN lifecycle IN ('open','suspended') THEN "
                "(CASE WHEN ?>next_seq THEN ? ELSE next_seq END)+? ELSE next_seq END,"
                "pending_tool_seq=CASE WHEN ?=1 THEN pending_tool_seq ELSE NULL END,"
                "ready_item_json=NULL,"
                "cancel_by=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_by END,"
                "cancel_reason=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_reason END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND owner_epoch=? AND execution_state='running' "
                "RETURNING execution_state,lifecycle,wake_pending_count,cancel_seq,"
                "cancel_by,cancel_reason,next_seq,updated_at",
                (
                    int(has_unconsumed),
                    effective_next_seq, effective_next_seq, int(write_message),
                    int(keep_pending_tool), batch.cancel_seq, batch.cancel_seq,
                    sid, batch.owner_epoch,
                ),
            )
            if srow is None:
                raise SessionLeaseLost(sid)
            next_state = str(srow["execution_state"])
            closed_raced = srow["lifecycle"] in ("closing", "closed")
            cancel_raced = int(srow["cancel_seq"]) > batch.cancel_seq
            subtree_raced = cancel_raced and srow["cancel_by"] == "subtree"
            cancel_by_db = srow["cancel_by"]
            cancel_reason_db = srow["cancel_reason"]
            post_next_seq = int(srow["next_seq"])
            database_now = int(srow["updated_at"])

        effective_status = status
        effective_cancel_by = batch.cancel_by or cancel_by_db
        effective_error = error
        effective_error_code = error_code
        if closed_raced:
            effective_status = "cancelled"
            effective_cancel_by = effective_cancel_by or "removed"
            effective_error = cancel_reason_db or "cancelled"
            effective_error_code = "cancelled"
            write_message = False
        elif cancel_raced and effective_status == "completed":
            effective_status = "cancelled"
            effective_error = cancel_reason_db or "cancelled"
            effective_error_code = "cancelled"
            write_message = False

        if write_message:
            result_seq = post_next_seq - 1

        # Resolve consumed inputs if caller passed consumed=None and we need exact list for abnormal settle or River
        if consumed is None and (effective_status != "completed" or batch.result_route is not None or batch.reply_to_session_id is not None):
            delivered = await tx.fetchall(
                f"SELECT item_id FROM {self.q} WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id ASC",
                (sid, batch.send_index),
            )
            if delivered:
                consumed_refs = tuple(
                    InputRef(batch.target.store_id, sid, row["item_id"]) for row in delivered
                )

        # Restore any unconsumed items back to pending (rare partial-consume path)
        if unconsumed_ids:
            urows = await tx.fetchall(
                f"SELECT wake,content_bytes FROM {self.q} WHERE session_id=? AND status='delivered' "
                f"AND delivered_send_index=? AND item_id IN ({','.join('?' for _ in unconsumed_ids)})",
                (sid, batch.send_index, *unconsumed_ids),
            )
            await tx.execute(
                f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                f"AND delivered_send_index=? AND item_id IN ({','.join('?' for _ in unconsumed_ids)})",
                (sid, batch.send_index, *unconsumed_ids),
            )
            w_add = sum(int(r["wake"]) for r in urows)
            b_add = sum(int(r["content_bytes"]) for r in urows)
            await tx.execute(
                f"UPDATE {self.s} SET wake_pending_count=wake_pending_count+?,"
                "pending_count=pending_count+?,pending_bytes=pending_bytes+? WHERE session_id=?",
                (w_add, len(urows), b_add, sid),
            )

        # Step 2 (Deep-first lock order): River delivery to parent AFTER child UPDATE (0 final_lock_asc!).
        result_ref, delivery_error = None, None
        activations: list[TargetRef] = []
        route = batch.result_route
        deliver_parent_sid = None
        deliver_res_kind = None
        deliver_start_ref = None
        if not closed_raced and consumed_refs:
            if route is not None:
                deliver_parent_sid = route["target"]["session_id"]
                deliver_res_kind = route["result_kind"]
                deliver_start_ref = InputRef(**route["input_ref"])
            elif batch.reply_to_session_id is not None:
                deliver_parent_sid = batch.reply_to_session_id
                deliver_res_kind = batch.reply_result_kind or "agent_result"
                deliver_start_ref = (
                    InputRef(batch.target.store_id, sid, batch.start_input_message_id)
                    if batch.start_input_message_id else consumed_refs[0]
                )

        if deliver_parent_sid is not None and deliver_res_kind is not None:
            child_wake = 0 if (effective_cancel_by == "subtree" or subtree_raced) else 1
            route_result = Result(
                consumed_refs[0], batch.target, effective_status, batch.send_index, result_seq,
                final_text, usage, structured_output, effective_error, effective_error_code,
            )
            result_ref, delivery_error = await self._river_deliver_tx(
                tx,
                child_target=batch.target,
                child_tenant_id=batch.tenant_id,
                child_principal_id=batch.principal_id,
                parent_session_id=deliver_parent_sid,
                start_ref=deliver_start_ref or consumed_refs[0],
                result=route_result,
                result_kind=deliver_res_kind,
                child_wake=child_wake,
                activations=activations,
            )

        # Step 3: Assistant message write / meta stamp (skipped when already written cleanly by sink!)
        result_meta = {
            "usage": plain(usage),
            "structured_output": plain(structured_output),
            "error": effective_error,
            "error_code": effective_error_code,
            "result_delivery_error": delivery_error,
        }
        if write_message:
            if database_now is None:
                database_now = await self._db_now(tx)
            await self._insert_message(
                tx, sid, result_seq, "assistant", final_text,
                result_meta, batch.send_index, database_now,
            )
        elif result_seq is not None and not closed_raced:
            need_meta_update = (
                structured_output is not None
                or effective_error is not None
                or effective_error_code is not None
                or delivery_error is not None
                or assistant_meta is None
                or assistant_meta.get("usage") != plain(usage)
            )
            if need_meta_update:
                if assistant_meta is None:
                    mrow = await tx.fetchone(
                        f"SELECT meta_json FROM {self.store.t.messages} WHERE session_id=? AND seq=?",
                        (sid, result_seq),
                    )
                    meta = json.loads(mrow["meta_json"] or "{}") if mrow is not None else {}
                else:
                    meta = dict(assistant_meta)
                meta.update(result_meta)
                await tx.execute(
                    f"UPDATE {self.store.t.messages} SET meta_json=? WHERE session_id=? AND seq=?",
                    (canonical(meta), sid, result_seq),
                )

        # Step 4: 0 UPDATE follow_up_queue on normal completion!
        # Only update follow_up_queue on abnormal status, delivery_error, or explicit settle_start_input.
        res_json_str = canonical(result_json) if result_json is not None else None
        if settle_start_input and batch.start_input_message_id is not None:
            await tx.execute(
                f"UPDATE {self.q} SET terminal_status=?,cancel_by=?,result_seq=?,"
                "result_send_index=?,result_json=?,error_code=?,error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} "
                "WHERE session_id=? AND item_id=?",
                (
                    effective_status, effective_cancel_by, result_seq, batch.send_index,
                    res_json_str, effective_error_code, effective_error, delivery_error,
                    sid, batch.start_input_message_id,
                ),
            )
        elif (
            effective_status != "completed"
            or delivery_error is not None
            or effective_cancel_by is not None
            or result_json is not None
        ):
            consumed_ids = [ref.message_id for ref in consumed_refs]
            if consumed_ids:
                changed = await tx.execute(
                    f"UPDATE {self.q} SET terminal_status=?,cancel_by=?,result_seq=?,"
                    "result_send_index=?,result_json=?,error_code=?,error_detail=?,"
                    f"result_delivery_error=?,completed_at={now_sql} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    f"AND item_id IN ({','.join('?' for _ in consumed_ids)})",
                    (
                        effective_status, effective_cancel_by, result_seq, batch.send_index,
                        res_json_str, effective_error_code, effective_error, delivery_error,
                        sid, batch.send_index, *consumed_ids,
                    ),
                )
                if changed != len(consumed_ids):
                    raise SessionLeaseLost(sid)

        results = tuple(
            Result(
                ref, batch.target, effective_status, batch.send_index, result_seq, final_text,
                usage, structured_output, effective_error, effective_error_code, delivery_error,
            )
            for ref in consumed_refs
        )
        return FinalBatch(
            next_state, results, result_ref,
            activations[0] if activations else None,
            delivery_error,
        )

    async def _river_deliver_tx(
        self,
        tx,
        *,
        child_target: TargetRef,
        child_tenant_id: str,
        child_principal_id: str,
        parent_session_id: str,
        start_ref: InputRef,
        result: Result,
        result_kind: str,
        child_wake: int,
        activations: list[TargetRef] | None = None,
    ) -> tuple[InputRef, str | None]:
        """Same-tx non-cascading River delivery (parent lifecycle IN ('open','suspended')).

        - Parent closed / missing / over-capacity NEVER rolls back child result; returns error code.
        - Parent suspended accepts the receipt into inbox without waking (keeps execution_state='idle').
        - Uses STABLE_API ``result_message_id`` for deterministic dedupe.
        """
        parent_target = TargetRef(self.store.store_id, parent_session_id)
        effective_start = result.input_ref if result.send_index is None else start_ref
        message_id = result_message_id(child_target, effective_start, result.send_index, result_kind)
        ref = InputRef(parent_target.store_id, parent_session_id, message_id)
        body = {
            "child_target": plain(child_target),
            "input_ref": plain(result.input_ref),
            "status": result.status,
            "final_text": result.final_text,
            "usage": plain(result.usage),
            "structured_output": plain(result.structured_output),
            "error": result.error,
            "error_code": result.error_code,
            "result_seq": result.result_seq,
        }
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        digest = self.request_hash(parent_target, body, "normal")
        now_sql = self._now_sql()

        if self._dialect == "mysql":
            p_rc, p_packed = await tx.execute_packed(
                f"UPDATE {self.s} SET "
                "inbox_version=(LAST_INSERT_ID("
                "((inbox_version+1)<<1)|"
                "(CASE WHEN lifecycle='open' AND execution_state='idle' AND ?=1 THEN 1 ELSE 0 END)"
                ")>>1),"
                "wake_inbox_seq=wake_inbox_seq+?,"
                "wake_pending_count=wake_pending_count+?,"
                "pending_count=pending_count+1,"
                "pending_bytes=pending_bytes+?,"
                "execution_state=CASE WHEN lifecycle='open' AND execution_state='idle' AND ?=1 "
                "THEN 'queued' ELSE execution_state END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                "AND tenant_id=? AND principal_id=? "
                "AND pending_count<? AND (pending_bytes+?)<=? "
                f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?)",
                (
                    child_wake, child_wake, child_wake, content_bytes, child_wake,
                    parent_session_id, child_tenant_id, child_principal_id,
                    self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                    parent_session_id, message_id,
                ),
            )
            parent_updated = p_rc == 1
            parent_need_wake = bool(p_packed & 1) if parent_updated else False
            parent_inbox_ver = (p_packed >> 1) if parent_updated else 0
        else:
            prow = await tx.fetchone(
                f"UPDATE {self.s} SET "
                "inbox_version=inbox_version+1,"
                "wake_inbox_seq=wake_inbox_seq+?,"
                "wake_pending_count=wake_pending_count+?,"
                "pending_count=pending_count+1,"
                "pending_bytes=pending_bytes+?,"
                "execution_state=CASE WHEN lifecycle='open' AND execution_state='idle' AND ?=1 "
                "THEN 'queued' ELSE execution_state END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                "AND tenant_id=? AND principal_id=? "
                "AND pending_count<? AND (pending_bytes+?)<=? "
                f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?) "
                "RETURNING inbox_version,lifecycle,execution_state,wake_pending_count",
                (
                    child_wake, child_wake, content_bytes, child_wake,
                    parent_session_id, child_tenant_id, child_principal_id,
                    self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                    parent_session_id, message_id,
                ),
            )
            parent_updated = prow is not None
            parent_inbox_ver = int(prow["inbox_version"]) if parent_updated else 0
            parent_need_wake = (
                parent_updated
                and prow["lifecycle"] == "open"
                and child_wake == 1
                and prow["execution_state"] == "queued"
                and int(prow["wake_pending_count"]) == 1
            )

        if parent_updated:
            await tx.execute(
                f"INSERT INTO {self.q} ("
                "session_id,item_id,inbox_version,kind,priority,wake,status,"
                "content_json,content_bytes,meta_json,request_hash,result_kind,created_at) "
                f"VALUES (?,?,?,?,'normal',?,'pending',?,?,?,?,?,{now_sql})",
                (
                    parent_session_id, message_id, parent_inbox_ver, result_kind,
                    child_wake, content, content_bytes, meta, digest, result_kind,
                ),
            )
            if activations is not None and parent_need_wake:
                activations.append(parent_target)
            return ref, None

        # Non-cascading diagnosis: never roll back child's result!
        pdiag = await tx.fetchone(
            "SELECT s.session_id,s.tenant_id,s.principal_id,s.lifecycle,"
            "s.pending_count,s.pending_bytes,"
            "q.id AS dup_id,q.request_hash AS dup_hash "
            f"FROM {self.s} s LEFT JOIN {self.q} q "
            "ON q.session_id=s.session_id AND q.item_id=? "
            "WHERE s.session_id=?",
            (message_id, parent_session_id),
        )
        if pdiag is None:
            return ref, "not_found"
        if pdiag["dup_id"] is not None:
            if pdiag["dup_hash"] != digest:
                return ref, "request_conflict"
            return ref, None
        if pdiag["tenant_id"] != child_tenant_id or pdiag["principal_id"] != child_principal_id:
            return ref, "forbidden"
        if pdiag["lifecycle"] not in ("open", "suspended"):
            return ref, "session_closed"
        return ref, "capacity_exceeded"

    async def accept_result(self, child_target, result: Result, *, result_kind):
        """Trusted delivery using only the child's frozen direct-parent route."""
        self.check_store(child_target)
        async with self.db.transaction() as tx:
            session = await self._session(tx, child_target.session_id)
            if not session["result_route_json"]:
                return None
            route = json.loads(session["result_route_json"])
            activations: list[TargetRef] = []
            ref, error = await self._river_deliver_tx(
                tx,
                child_target=child_target,
                child_tenant_id=session["tenant_id"],
                child_principal_id=session["principal_id"],
                parent_session_id=route["target"]["session_id"],
                start_ref=InputRef(**route["input_ref"]),
                result=result,
                result_kind=result_kind,
                child_wake=1,
                activations=activations,
            )
            return ref, error, activations[0] if activations else None

    async def _complete_managed_recovery_tx(
        self, tx, session, state, route, parent_state, ref, result_document, database_now,
    ):
        """Save a verified historical outcome; never queue or own a child send."""
        sid = ref.session_id
        target = TargetRef(ref.store_id, sid)
        previous_send = int(state["send_index"])
        send_index = previous_send + 1
        seq = int(state["next_seq"]) + await self._repair_pending_tx(tx, sid, previous_send, state)
        body = {"managed_effect": "recovered", "result": result_document}
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        result = Result(
            ref, target, "completed", send_index, seq + 1,
            result_document["display"], {}, result_document,
        )
        activations: list[TargetRef] = []
        parent_ref, error = await self._river_deliver_tx(
            tx,
            child_target=target,
            child_tenant_id=session["tenant_id"],
            child_principal_id=session["principal_id"],
            parent_session_id=route["target"]["session_id"],
            start_ref=InputRef(**route["input_ref"]),
            result=result,
            result_kind=route["result_kind"],
            child_wake=1,
            activations=activations,
        )
        changed = await tx.execute(
            f"UPDATE {self.s} SET send_index=?,next_seq=?,inbox_version=inbox_version+1,"
            "updated_at=? "
            "WHERE session_id=? AND execution_state=? AND send_index=? AND next_seq=? "
            "AND owner_node_id IS NULL",
            (send_index, seq + 2, database_now, sid, state["execution_state"], previous_send, seq),
        )
        if changed != 1:
            raise SessionLeaseLost(sid)
        await tx.execute(
            f"INSERT INTO {self.q} (session_id,item_id,inbox_version,kind,priority,wake,status,"
            "content_json,content_bytes,meta_json,request_hash,created_at,delivered_at,"
            "delivered_seq,delivered_send_index,terminal_status,result_seq,result_send_index,"
            "result_delivery_error,completed_at) "
            "VALUES (?,?,1,'user','normal',1,'delivered',?,?,?,?,?,?,?,?,'completed',?,?,?,?)",
            (
                sid, ref.message_id, content, content_bytes, meta,
                self.request_hash(target, body, "normal"), database_now, database_now, seq,
                send_index, seq + 1, send_index, error, database_now,
            ),
        )
        merged = self.merge_batch([{
            "item_id": ref.message_id, "kind": "user", "priority": "normal",
            "content_json": content,
        }])
        await self._insert_message(
            tx, sid, seq, "user", canonical(merged),
            {"content_encoding": "json", "input_refs": [plain(ref)]}, send_index, database_now,
        )
        await self._insert_message(
            tx, sid, seq + 1, "assistant", result.final_text,
            {"usage": {}, "structured_output": result_document, "error": None,
             "error_code": None, "result_delivery_error": error}, send_index, database_now,
        )
        return FinalBatch(
            state["execution_state"], (replace(result, result_delivery_error=error),),
            parent_ref, activations[0] if activations else None, error,
        )

    async def _final_after_unknown(self, batch, consumed):
        state = await self.db.fetchone(
            f"SELECT * FROM {self.s} WHERE session_id=?", (batch.target.session_id,)
        )
        if not state or state["execution_state"] not in {"idle", "queued"}:
            return None
        if int(state["send_index"]) < batch.send_index:
            return None
        if consumed is None:
            rows = await self.db.fetchall(
                f"SELECT item_id FROM {self.q} WHERE session_id=? AND delivered_send_index=? "
                "AND status='delivered'",
                (batch.target.session_id, batch.send_index),
            )
            consumed = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in rows
            )
        results = []
        for ref in consumed:
            info = await self.inspect_input(ref)
            if info["terminal_status"] is None or info["result_send_index"] != batch.send_index:
                return None
            final_text, usage, structured = "", {}, None
            if info["result_seq"] is not None:
                message = await self.db.fetchone(
                    f"SELECT content,meta_json FROM {self.store.t.messages} "
                    "WHERE session_id=? AND seq=? AND send_index=?",
                    (ref.session_id, info["result_seq"], info["result_send_index"]),
                )
                if message is not None:
                    final_text = message["content"] or ""
                    meta = json.loads(message["meta_json"] or "{}")
                    usage = meta.get("usage") or {}
                    structured = meta.get("structured_output")
            results.append(Result(
                ref, batch.target, info["terminal_status"], info["result_send_index"],
                info["result_seq"], final_text, usage, structured, info["error_detail"],
                info["error_code"], info["result_delivery_error"],
            ))
        return FinalBatch(state["execution_state"], tuple(results), None)

    # ── Crash Takeover & Periodic Node SQL (3 / 3 / 4 Budget) ────────────────

    async def batch_renew(self, *, node_id: str, boot_id: str, ttl_s: float = 30.0) -> list[dict[str, Any]]:
        """1 SQL (SQLite/PG) or 2 SQL (MySQL) per node to renew all active leases (idx_sessions_node_active)."""
        now_sql = self._now_sql()
        ttl_ms = int(ttl_s * 1000)
        if self._dialect == "mysql":
            await self.db.execute(
                f"UPDATE {self.s} SET owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
                "WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=?",
                (ttl_ms, node_id, boot_id),
            )
            rows = await self.db.fetchall(
                "SELECT session_id,owner_epoch,cancel_seq,cancel_by,lifecycle "
                f"FROM {self.s} WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=?",
                (node_id, boot_id),
            )
            return [dict(r) for r in rows]
        rows = await self.db.fetchall(
            f"UPDATE {self.s} SET owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
            "WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=? "
            "RETURNING session_id,owner_epoch,cancel_seq,cancel_by,lifecycle",
            (ttl_ms, node_id, boot_id),
        )
        return [dict(r) for r in rows]

    async def discover_queued(self, *, limit: int = 32) -> list[str]:
        """1 bounded index scan on ``idx_session_recovery``."""
        if not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options", "limit")
        rows = await self.db.fetchall(
            f"SELECT session_id FROM {self.s} "
            "WHERE execution_state='queued' AND lifecycle='open' AND wake_pending_count>0 "
            "ORDER BY session_id ASC LIMIT ?",
            (limit,),
        )
        return [str(r["session_id"]) for r in rows]

    async def graceful_release(self, *, node_id: str, boot_id: str) -> int:
        """1-SQL batch lease release on graceful node shutdown (idx_sessions_node_active)."""
        now_sql = self._now_sql()
        return await self.db.execute(
            f"UPDATE {self.s} SET owner_lease_until_ms=0,updated_at={now_sql} "
            "WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=?",
            (node_id, boot_id),
        )

    async def sweep_expired_leases(
        self, *, node_id: str, boot_id: str, ttl_s: float = 30.0, limit: int = 32,
    ) -> list[ClaimedBatch]:
        """1 bounded index scan on ``idx_session_recovery``; reclaims expired running sessions."""
        now_sql = self._now_sql()
        expired = await self.db.fetchall(
            f"SELECT session_id,owner_epoch FROM {self.s} "
            f"WHERE execution_state='running' AND owner_lease_until_ms<{now_sql} "
            "ORDER BY owner_lease_until_ms ASC LIMIT ?",
            (limit,),
        )
        reclaimed: list[ClaimedBatch] = []
        for row in expired:
            batch = await self.reclaim_running(
                TargetRef(self.store.store_id, row["session_id"]),
                expected_epoch=int(row["owner_epoch"]),
                node_id=node_id,
                boot_id=boot_id,
                ttl_s=ttl_s,
            )
            if batch is not None:
                reclaimed.append(batch)
        return reclaimed

    async def reclaim_running(
        self, target: TargetRef, *, expected_epoch: int | None = None, node_id: str, boot_id: str,
        ttl_s: float = 30.0, can_execute=None,
    ) -> ClaimedBatch | None:
        """Atomically take over a crashed/expired running session (supports 0-assistant rerun!)."""
        self.check_store(target)
        now_sql = self._now_sql()
        ttl_ms = int(ttl_s * 1000)
        sid = target.session_id
        epoch_cond = (
            "AND owner_epoch=?"
            if expected_epoch is not None else f"AND owner_lease_until_ms<={now_sql}"
        )
        extra_params = (expected_epoch,) if expected_epoch is not None else ()
        try:
            async with self.db.transaction() as tx:
                if self._dialect == "mysql":
                    rc = await tx.execute(
                        f"UPDATE {self.s} SET owner_node_id=?,owner_boot_id=?,"
                        f"owner_epoch=owner_epoch+1,owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
                        f"WHERE session_id=? AND execution_state='running' {epoch_cond}",
                        (node_id, boot_id, ttl_ms, sid, *extra_params),
                    )
                    if rc == 0:
                        return None
                    srow = await tx.fetchone(f"SELECT * FROM {self.s} WHERE session_id=?", (sid,))
                else:
                    srow = await tx.fetchone(
                        f"UPDATE {self.s} SET owner_node_id=?,owner_boot_id=?,"
                        f"owner_epoch=owner_epoch+1,owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
                        f"WHERE session_id=? AND execution_state='running' {epoch_cond} "
                        "RETURNING *",
                        (node_id, boot_id, ttl_ms, sid, *extra_params),
                    )
                if srow is None:
                    return None
                if can_execute is not None:
                    desc_check = RuntimeDescriptor(**json.loads(srow["descriptor_json"]))
                    if not can_execute(desc_check):
                        raise _ClaimRollback()
                cur_send_index = int(srow["send_index"])
                turn_msgs = await tx.fetchall(
                    f"SELECT id,seq,role,name,content,tool_calls_json,tool_call_id,"
                    f"round_index,state,meta_json,send_index,created_at "
                    f"FROM {self.store.t.messages} WHERE session_id=? AND send_index=? ORDER BY seq ASC",
                    (sid, cur_send_index),
                )
                has_assistant = any(m["role"] == "assistant" for m in turn_msgs)
                await self._repair_pending_tx(tx, sid, cur_send_index, srow)
                qrows = await tx.fetchall(
                    f"SELECT item_id,reply_to_session_id,reply_result_kind FROM {self.q} "
                    "WHERE session_id=? AND delivered_send_index=? AND status='delivered' ORDER BY id ASC",
                    (sid, cur_send_index),
                )
        except _ClaimRollback:
            return None
        refs = tuple(InputRef(target.store_id, sid, r["item_id"]) for r in qrows)
        first_user = next((m for m in turn_msgs if m["role"] == "user"), None)
        if first_user is not None:
            user_row = StoredMessageRow(
                session_id=sid, seq=int(first_user["seq"]), role="user",
                name=first_user["name"], content=first_user["content"],
                tool_calls=None, tool_call_id=None, round_index=first_user["round_index"],
                state=MessageState(first_user["state"]),
                meta=json.loads(first_user["meta_json"] or "{}"),
                created_at=int(first_user["created_at"]), send_index=cur_send_index,
            )
            msg_seq = max(int(m["seq"]) for m in turn_msgs)
        else:
            msg_seq = max(0, int(srow["next_seq"]) - 1)
            user_row = StoredMessageRow(
                session_id=sid, seq=msg_seq, role="user", name=None, content="",
                tool_calls=None, tool_call_id=None, round_index=0,
                state=MessageState.ACTIVE, meta={}, created_at=int(srow["updated_at"]),
                send_index=cur_send_index,
            )
        reply_to_sid = next((r["reply_to_session_id"] for r in qrows if r["reply_to_session_id"]), None)
        reply_rk = next((r["reply_result_kind"] for r in qrows if r["reply_to_session_id"]), None)
        return ClaimedBatch(
            target=target,
            send_index=cur_send_index,
            message_seq=msg_seq,
            input_refs=refs,
            owner_epoch=int(srow["owner_epoch"]),
            owner_node_id=node_id,
            owner_boot_id=boot_id,
            descriptor=RuntimeDescriptor(**json.loads(srow["descriptor_json"])),
            tenant_id=srow["tenant_id"],
            principal_id=srow["principal_id"],
            parent_session_id=srow["parent_session_id"],
            user_row=user_row,
            result_route=(
                json.loads(srow["result_route_json"]) if srow["result_route_json"] else None
            ),
            session_updated_at=int(srow["updated_at"]),
            inbox_version=int(srow["inbox_version"]),
            background_version=int(srow["background_version"]),
            runtime_version=int(srow["runtime_version"]),
            wake_inbox_seq=int(srow["wake_inbox_seq"]),
            cancel_seq=int(srow["cancel_seq"]),
            cancel_by=srow["cancel_by"],
            start_input_message_id=srow["start_input_message_id"],
            root_session_id=str(srow["root_session_id"]),
            depth=int(srow["depth"]),
            reply_to_session_id=reply_to_sid,
            reply_result_kind=reply_rk,
            is_reclaimed=True,
            reclaimed_has_assistant=has_assistant,
            reclaimed_cancelled=bool(
                srow["cancel_by"] is not None or srow["lifecycle"] in ("closing", "closed")
            ),
        )

    async def recover(self, target, *, node_id: str, boot_id: str, ttl_s: float = 30.0, ownership=None) -> FinalBatch | None:
        """Atomically take over an expired running session without any 'recovering' state."""
        self.check_store(target)
        now_sql = self._now_sql()
        sid = target.session_id
        async with self.db.transaction() as tx:
            try:
                own_state = await self.db.dialect.lock_state(tx, self.s, sid)
            except ValueError:
                return None
            if own_state["execution_state"] != "running":
                return None
            database_now = await self._db_now(tx)
            expired = (
                own_state["owner_lease_until_ms"] is not None
                and int(own_state["owner_lease_until_ms"]) <= database_now
            )
            active = int(own_state["send_index"])
            if ownership is not None:
                ownership.require_owned()
                if (
                    int(own_state["owner_epoch"]) != int(ownership.epoch)
                    or own_state["owner_node_id"] != node_id
                    or own_state["owner_boot_id"] != boot_id
                ):
                    raise SessionLeaseLost(sid)
                epoch = int(own_state["owner_epoch"])
            else:
                if not expired:
                    return None
                epoch = int(own_state["owner_epoch"]) + 1
                changed = await tx.execute(
                    f"UPDATE {self.s} SET owner_epoch=?,owner_node_id=?,owner_boot_id=?,"
                    f"owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
                    "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                    (epoch, node_id, boot_id, int(ttl_s * 1000), sid, own_state["owner_epoch"]),
                )
                if changed != 1:
                    return None

            await self._repair_pending_tx(tx, sid, active, own_state)

            rows = await tx.fetchall(
                f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id ASC",
                (sid, active),
            )
            cancelled = own_state["cancel_by"] is not None or own_state["lifecycle"] in ("closing", "closed")
            terminal = "cancelled" if cancelled else "interrupted"
            detail = own_state["cancel_reason"] if cancelled else "execution owner lost"
            delivery_error = None
            result_ref = None
            activations: list[TargetRef] = []
            route = json.loads(own_state["result_route_json"]) if own_state["result_route_json"] else None
            if rows and route is not None:
                first = self.record(rows[0])
                route_result = Result(
                    first.ref, target, terminal, active, None,
                    error=detail, error_code=terminal,
                )
                result_ref, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=target,
                    child_tenant_id=own_state["tenant_id"],
                    child_principal_id=own_state["principal_id"],
                    parent_session_id=route["target"]["session_id"],
                    start_ref=InputRef(**route["input_ref"]),
                    result=route_result,
                    result_kind=route["result_kind"],
                    child_wake=0 if own_state["cancel_by"] == "subtree" else 1,
                    activations=activations,
                )
            if rows:
                await tx.execute(
                    f"UPDATE {self.q} SET terminal_status=?,result_send_index=?,"
                    "error_code=?,error_detail=?,result_delivery_error=?,completed_at=? "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    "AND terminal_status IS NULL",
                    (terminal, active, terminal, detail, delivery_error, database_now, sid, active),
                )
            next_state = (
                "queued"
                if own_state["lifecycle"] == "open" and int(own_state["wake_pending_count"]) > 0
                else "idle"
            )
            changed = await tx.execute(
                f"UPDATE {self.s} SET execution_state=?,owner_node_id=NULL,owner_boot_id=NULL,"
                "owner_lease_until_ms=NULL,pending_tool_seq=NULL,ready_item_json=NULL,"
                f"cancel_by=NULL,cancel_reason=NULL,updated_at={now_sql} "
                "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                (next_state, sid, epoch),
            )
            if changed != 1:
                raise SessionLeaseLost(sid)
            results = tuple(
                Result(
                    self.record(row).ref, target, terminal, active, None,
                    error=detail, error_code=terminal, result_delivery_error=delivery_error,
                )
                for row in rows
            )
            return FinalBatch(
                next_state, results, result_ref,
                activations[0] if activations else None, delivery_error,
            )

    async def discover_unfinished(self, *, limit: int = 100):
        """Bounded discovery of queued or expired running sessions via ``idx_session_recovery``."""
        if not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options", "limit")
        columns = "session_id,execution_state,owner_lease_until_ms"
        order = "execution_state,owner_lease_until_ms,session_id"
        now_sql = self._now_sql()
        branches = [
            f"SELECT {columns} FROM (SELECT {columns} FROM {self.s} "
            f"WHERE execution_state='queued' AND lifecycle='open' AND wake_pending_count>0 "
            f"ORDER BY {order} LIMIT ?) AS queued_candidates",
            f"SELECT {columns} FROM (SELECT {columns} FROM {self.s} "
            f"WHERE execution_state='running' AND owner_lease_until_ms<={now_sql} "
            f"ORDER BY {order} LIMIT ?) AS running_candidates",
        ]
        rows = await self.db.fetchall(
            "SELECT session_id FROM (" + " UNION ALL ".join(branches)
            + f") AS candidates ORDER BY {order} LIMIT ?",
            (limit, limit, limit),
        )
        return tuple(TargetRef(self.store.store_id, row["session_id"]) for row in rows)

    # ── Stop / Cancel (cancel_seq + cancel_by + active-only depth DESC locking) ──

    async def _collect_active_descendants(
        self, tx, *, session_id: str, root_session_id: str, target_depth: int,
        only_linked: bool = True,
    ) -> list[tuple[str, int, bool]]:
        """Bounded active-only subtree scan using ``idx_sessions_root`` (0 idle rows scanned!)."""
        active_rows = await tx.fetchall(
            "SELECT session_id,parent_session_id,depth,send_index,ready_item_json,cancel_propagation "
            f"FROM {self.s} WHERE root_session_id=? AND execution_state IN ('queued','running')",
            (root_session_id,),
        )
        if session_id == root_session_id and not only_linked:
            return [
                (
                    str(r["session_id"]), int(r["depth"]),
                    int(r["send_index"]) == 0 and bool(r["ready_item_json"]),
                )
                for r in active_rows if r["session_id"] != session_id
            ]
        deeper = [r for r in active_rows if int(r["depth"]) > target_depth]
        by_id = {r["session_id"]: r for r in active_rows}
        result: list[tuple[str, int, bool]] = []
        for r in deeper:
            cur_id = r["session_id"]
            cur_parent = r["parent_session_id"]
            cur_prop = r["cancel_propagation"]
            matched = False
            visited: set[str] = set()
            while cur_parent is not None and cur_parent not in visited:
                if only_linked and cur_prop != "linked":
                    break
                if cur_parent == session_id:
                    matched = True
                    break
                visited.add(cur_id)
                if cur_parent in by_id:
                    cur_id = cur_parent
                    cur_prop = by_id[cur_parent]["cancel_propagation"]
                    cur_parent = by_id[cur_parent]["parent_session_id"]
                else:
                    prow = await tx.fetchone(
                        f"SELECT parent_session_id,cancel_propagation FROM {self.s} WHERE session_id=?",
                        (cur_parent,),
                    )
                    if prow is None:
                        break
                    cur_id = cur_parent
                    cur_prop = prow["cancel_propagation"]
                    cur_parent = prow["parent_session_id"]
            if matched:
                result.append((
                    str(r["session_id"]), int(r["depth"]),
                    int(r["send_index"]) == 0 and bool(r["ready_item_json"]),
                ))
        return result

    async def cancel(
        self, ref, *, actor, scope="self", reason="cancelled",
        cancel_by: str | None = None, only_linked: bool = True,
        _with_activation: bool = False,
    ):
        """Cancel/stop an input, a single session, or its active subtree.

        - ``scope='subtree'`` scans ONLY active descendants (``execution_state IN ('queued','running')``)
          via ``idx_sessions_root`` and locks level-by-level in ``depth DESC``, within each level
          ``ORDER BY session_id ASC FOR UPDATE``.
        - Cascaded descendants receive ``cancel_by='subtree'`` so their River receipts use ``wake=0``.
        """
        if scope not in {"self", "subtree"}:
            raise RuntimeRequestError("invalid_options", "scope")
        if isinstance(ref, InputRef) and scope == "subtree":
            raise RuntimeRequestError("invalid_options", "InputRef subtree")
        target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
        eff_cancel_by = cancel_by or ("agent" if actor.session_id is not None else "user")
        now_sql = self._now_sql()
        activations: list[TargetRef] = []
        completed_refs: list[InputRef] = []
        stopped_descendants: list[str] = []

        async with self.db.transaction() as tx:
            session = await self.authorize(tx, target, actor)
            desc_by_depth: dict[int, list[str]] = {}
            prestaged_desc: list[str] = []
            if scope == "subtree":
                active_desc = await self._collect_active_descendants(
                    tx,
                    session_id=target.session_id,
                    root_session_id=str(session["root_session_id"]),
                    target_depth=int(session["depth"]),
                    only_linked=only_linked,
                )
                for d_sid, d_val, d_pre in active_desc:
                    desc_by_depth.setdefault(d_val, []).append(d_sid)
                    if d_pre:
                        prestaged_desc.append(d_sid)
                for lvl_ids in desc_by_depth.values():
                    lvl_ids.sort()

            # Level-by-level deep-first locking (depth DESC, ORDER BY session_id ASC FOR UPDATE)
            cascaded_delivered: dict[str, set[str]] = {}
            for d_val in sorted(desc_by_depth, reverse=True):
                lvl_ids = desc_by_depth[d_val]
                stopped_descendants.extend(lvl_ids)
                ph = ",".join("?" for _ in lvl_ids)
                lock_suffix = " FOR UPDATE" if self._dialect in ("postgres", "mysql") else ""
                lvl_rows = await tx.fetchall(
                    f"SELECT * FROM {self.s} WHERE session_id IN ({ph}) "
                    f"ORDER BY session_id ASC{lock_suffix}",
                    tuple(lvl_ids),
                )
                for drow in lvl_rows:
                    await self._cancel_single_session_tx(
                        tx, drow, reason=reason, cancel_by="subtree",
                        is_subtree_cascade=True, activations=activations,
                        completed_refs=completed_refs,
                        skip_item_ids=cascaded_delivered.get(drow["session_id"]),
                        cascaded_delivered=cascaded_delivered,
                    )

            # Now lock target itself (strictly shallower than all descendants)
            state = await self.db.dialect.lock_state(tx, self.s, target.session_id)
            if isinstance(ref, InputRef):
                record = await self.input(tx, ref)
                if record is None:
                    raise RuntimeRequestError("not_found")
                receipt = await self._cancel_input_tx(
                    tx, state, record, reason=reason, cancel_by=eff_cancel_by,
                    activations=activations, completed_refs=completed_refs,
                )
            else:
                receipt = await self._cancel_single_session_tx(
                    tx, state, reason=reason, cancel_by=eff_cancel_by,
                    is_subtree_cascade=False, activations=activations,
                    completed_refs=completed_refs,
                    skip_item_ids=cascaded_delivered.get(target.session_id),
                )

            if prestaged_desc:
                pre_ph = ",".join("?" for _ in prestaged_desc)
                await tx.execute(
                    f"UPDATE {self.q} SET cancel_by='subtree' "
                    f"WHERE session_id IN ({pre_ph}) AND status='delivered' "
                    "AND delivered_send_index=1 AND terminal_status IS NULL",
                    tuple(prestaged_desc),
                )

            _ = now_sql  # used via helpers

        if _with_activation:
            return receipt, activations, completed_refs
        return receipt

    async def _cancel_single_session_tx(
        self, tx, state, *, reason: str, cancel_by: str, is_subtree_cascade: bool,
        activations: list[TargetRef], completed_refs: list[InputRef],
        skip_item_ids: set[str] | None = None,
        cascaded_delivered: dict[str, set[str]] | None = None,
    ) -> Receipt:
        sid = state["session_id"]
        target = TargetRef(self.store.store_id, sid)
        now_sql = self._now_sql()
        route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
        child_wake = 0 if is_subtree_cascade else 1

        if state["execution_state"] == "running":
            await tx.execute(
                f"UPDATE {self.s} SET cancel_seq=cancel_seq+1,cancel_by=?,cancel_reason=?,"
                f"updated_at={now_sql} WHERE session_id=?",
                (cancel_by, reason, sid),
            )
            return Receipt(None, target, None, False, "cancel_requested", int(state["send_index"]), "not_needed", reason)

        # Idle or queued: void pending items & pre-staged delivered turn-1 input if any
        unfinished = await tx.fetchall(
            f"SELECT * FROM {self.q} WHERE session_id=? "
            "AND status IN ('pending','waiting_continuation') ORDER BY id",
            (sid,),
        )
        skip = skip_item_ids or set()
        pending_rows = [r for r in unfinished if r["status"] == "pending" and r["item_id"] not in skip]
        waiting_by_item = {r["item_id"]: r for r in unfinished if r["status"] == "waiting_continuation"}
        changed = 0
        voided_bytes = 0
        voided_wake = 0
        for row in pending_rows:
            rec = self.record(row)
            delivery_error = None
            if route is not None:
                cancelled_res = Result(
                    rec.ref, target, "cancelled", None, None,
                    error=reason, error_code="cancelled",
                )
                parent_ref, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=target,
                    child_tenant_id=state["tenant_id"],
                    child_principal_id=state["principal_id"],
                    parent_session_id=route["target"]["session_id"],
                    start_ref=InputRef(**route["input_ref"]),
                    result=cancelled_res,
                    result_kind=route["result_kind"],
                    child_wake=child_wake,
                    activations=activations,
                )
                if delivery_error is None and cascaded_delivered is not None:
                    cascaded_delivered.setdefault(parent_ref.session_id, set()).add(parent_ref.message_id)
            affected = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "error_code='cancelled',error_detail=?,result_delivery_error=?,"
                f"completed_at={now_sql} WHERE session_id=? AND item_id=? AND status='pending'",
                (cancel_by, reason, delivery_error, sid, rec.ref.message_id),
            )
            changed += affected
            if affected:
                voided_bytes += int(row["content_bytes"])
                voided_wake += int(row["wake"])
                completed_refs.append(rec.ref)
                parent_id = row["continuation_of"]
                while parent_id in waiting_by_item:
                    parent_row = waiting_by_item.pop(parent_id)
                    completed_refs.append(InputRef(self.store.store_id, sid, parent_id))
                    parent_id = parent_row["continuation_of"]

        # Handle queued pre-staged child (send_index=0 and ready_item_json IS NOT NULL)
        is_prestaged = int(state["send_index"]) == 0 and bool(state["ready_item_json"])
        if is_prestaged:
            ready = json.loads(state["ready_item_json"])
            pre_ref = InputRef(self.store.store_id, sid, ready["item_id"])
            delivery_error = None
            if route is not None:
                cancelled_res = Result(
                    pre_ref, target, "cancelled", None, None,
                    error=reason, error_code="cancelled",
                )
                parent_ref, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=target,
                    child_tenant_id=state["tenant_id"],
                    child_principal_id=state["principal_id"],
                    parent_session_id=route["target"]["session_id"],
                    start_ref=InputRef(**route["input_ref"]),
                    result=cancelled_res,
                    result_kind=route["result_kind"],
                    child_wake=child_wake,
                    activations=activations,
                )
                if delivery_error is None and cascaded_delivered is not None:
                    cascaded_delivered.setdefault(parent_ref.session_id, set()).add(parent_ref.message_id)
            affected = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "delivered_send_index=NULL,delivered_seq=NULL,error_code='cancelled',"
                f"error_detail=?,result_delivery_error=?,completed_at={now_sql} "
                "WHERE session_id=? AND item_id=? AND status='delivered' AND delivered_send_index=1",
                (cancel_by, reason, delivery_error, sid, ready["item_id"]),
            )
            changed += affected
            if affected:
                voided_wake += 1
                completed_refs.append(pre_ref)

        if not pending_rows and not is_prestaged and route is not None:
            start = await self.input(tx, InputRef(**route["input_ref"]))
            if start is not None and start.row["terminal_status"] is not None:
                await self._route_stored_result_tx(tx, state, route, start, activations)

        if state["execution_state"] == "queued" or changed > 0:
            rem_count = max(0, int(state["pending_count"]) - (changed - (1 if is_prestaged and changed else 0)))
            rem_bytes = max(0, int(state["pending_bytes"]) - voided_bytes)
            rem_wake = max(0, int(state["wake_pending_count"]) - voided_wake)
            await tx.execute(
                f"UPDATE {self.s} SET execution_state='idle',cancel_seq=cancel_seq+1,"
                "cancel_by=?,cancel_reason=?,wake_pending_count=?,pending_count=?,"
                f"pending_bytes=?,ready_item_json=NULL,updated_at={now_sql} "
                "WHERE session_id=? AND execution_state IN ('idle','queued')",
                (cancel_by, reason, rem_wake, rem_count, rem_bytes, sid),
            )
        return Receipt(
            None, target, None, False,
            "voided" if changed else "already_finished",
            None, "not_needed", reason if changed else None,
        )

    async def _cancel_input_tx(
        self, tx, state, record: InputRecord, *, reason: str, cancel_by: str,
        activations: list[TargetRef], completed_refs: list[InputRef],
    ) -> Receipt:
        row = record.row
        ref = record.ref
        sid = ref.session_id
        now_sql = self._now_sql()
        route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
        # Check derived terminal status for normal completed rows (no final_settle_inbox)
        derived = await self._derive_input_terminal(tx, state, row)
        if derived["terminal_status"] is not None:
            if route is not None and row["terminal_status"] is not None:
                await self._route_stored_result_tx(tx, state, route, record, activations)
            return self.receipt(record, state="already_finished")
        if row["status"] == "pending":
            delivery_error = None
            if route is not None:
                cancelled = Result(
                    ref, TargetRef(self.store.store_id, sid), "cancelled", None, None,
                    error=reason, error_code="cancelled",
                )
                _, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=TargetRef(self.store.store_id, sid),
                    child_tenant_id=state["tenant_id"],
                    child_principal_id=state["principal_id"],
                    parent_session_id=route["target"]["session_id"],
                    start_ref=InputRef(**route["input_ref"]),
                    result=cancelled,
                    result_kind=route["result_kind"],
                    child_wake=1,
                    activations=activations,
                )
            changed = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "error_code='cancelled',error_detail=?,result_delivery_error=?,"
                f"completed_at={now_sql} WHERE session_id=? AND item_id=? AND status='pending'",
                (cancel_by, reason, delivery_error, sid, ref.message_id),
            )
            if changed:
                w_val = int(row["wake"])
                b_val = int(row["content_bytes"])
                await tx.execute(
                    f"UPDATE {self.s} SET "
                    "execution_state=CASE WHEN execution_state='queued' AND (wake_pending_count-?)<=0 "
                    "THEN 'idle' ELSE execution_state END,"
                    "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                    "pending_count=CASE WHEN pending_count>=1 THEN pending_count-1 ELSE 0 END,"
                    "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END,"
                    f"updated_at={now_sql} WHERE session_id=?",
                    (w_val, w_val, w_val, b_val, b_val, sid),
                )
                completed_refs.append(ref)
                new_row = dict(
                    row, status="void", terminal_status="cancelled", error_detail=reason,
                    result_delivery_error=delivery_error,
                )
                return self.receipt(InputRecord(ref, new_row, record.meta), state="voided", reason=reason)
        if (
            row["status"] == "delivered"
            and state["execution_state"] == "running"
            and row["delivered_send_index"] == int(state["send_index"])
        ):
            await tx.execute(
                f"UPDATE {self.s} SET cancel_seq=cancel_seq+1,cancel_by=?,cancel_reason=?,"
                f"updated_at={now_sql} WHERE session_id=? AND execution_state='running'",
                (cancel_by, reason, sid),
            )
            return self.receipt(record, state="cancel_requested", reason=reason)
        return self.receipt(record, state="already_finished")

    async def _route_stored_result_tx(
        self, tx, session, route, record, activations=None, *, parent_state=None,
    ):
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
        existing = Result(
            record.ref, TargetRef(self.store.store_id, record.ref.session_id),
            row["terminal_status"], row["result_send_index"], row["result_seq"],
            final_text, usage, structured, row["error_detail"], row["error_code"],
            row["result_delivery_error"],
        )
        result_ref, delivery_error = await self._river_deliver_tx(
            tx,
            child_target=TargetRef(self.store.store_id, record.ref.session_id),
            child_tenant_id=session["tenant_id"],
            child_principal_id=session["principal_id"],
            parent_session_id=route["target"]["session_id"],
            start_ref=InputRef(**route["input_ref"]),
            result=existing,
            result_kind=route["result_kind"],
            child_wake=1,
            activations=activations,
        )
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

    async def cancel_active(self, batch: ClaimedBatch, *, reason="cancelled", cancel_by="agent"):
        now_sql = self._now_sql()
        changed = await self.db.execute(
            f"UPDATE {self.s} SET cancel_seq=cancel_seq+1,cancel_by=?,cancel_reason=?,"
            f"updated_at={now_sql} WHERE session_id=? AND owner_epoch=? AND execution_state='running'",
            (cancel_by, reason, batch.target.session_id, batch.owner_epoch),
        )
        if changed != 1:
            raise SessionLeaseLost(batch.target.session_id)
        return Receipt(
            None, batch.target, None, False, "cancel_requested", batch.send_index,
            "not_needed", reason,
        )

    async def close(self, target, *, actor):
        from power_loop.contracts.errors import SessionBusy

        self.check_store(target)
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, target.session_id)
            await self._authorize_known(tx, target, actor, state)
            if state["execution_state"] == "running" or int(state["wake_pending_count"]) > 0:
                raise SessionBusy(target.session_id)
            pending = await tx.fetchone(
                f"SELECT COUNT(*) AS n FROM {self.q} WHERE session_id=? "
                "AND status IN ('pending','waiting_continuation')", (target.session_id,),
            )
            if int(pending["n"]):
                raise SessionBusy(target.session_id)
            linked_child = await tx.fetchone(
                f"SELECT session_id FROM {self.s} "
                "WHERE parent_session_id=? AND cancel_propagation='linked' "
                "AND lifecycle IN ('open','suspended') LIMIT 1",
                (target.session_id,),
            )
            if linked_child is not None:
                raise SessionBusy(target.session_id)
            changed = await tx.execute(
                f"UPDATE {self.s} SET lifecycle='closed',updated_at=? "
                "WHERE session_id=? AND lifecycle IN ('open','suspended')",
                (now_ms(), target.session_id),
            )
            if changed != 1:
                raise RuntimeRequestError("session_closed")

    # ── Unified inspect_input / inspect / read_result_state / get_result ─────

    async def _derive_input_terminal(self, reader, session_row: dict, input_row: dict) -> dict:
        """Derive terminal_status / result_send_index / result_seq when normal Final skipped final_settle_inbox.

        Probe 1 invariant: for ``session_kind == 'workflow_driver'``, ``start_input_message_id``
        MUST NEVER derive premature ``'completed'`` from ``delivered_send_index <= send_index`` —
        only an explicit ``terminal_status`` written at the driver's final commit counts!
        """
        out = dict(input_row)
        if out["status"] != "delivered" or out["terminal_status"] is not None:
            return out
        dsi = out.get("delivered_send_index")
        if dsi is None:
            return out
        is_driver = session_row["session_kind"] == "workflow_driver"
        start_id = session_row.get("start_input_message_id")
        if is_driver and start_id is not None and out["item_id"] == start_id:
            return out
        cur_si = int(session_row["send_index"])
        is_running = session_row["execution_state"] == "running"
        dsi_i = int(dsi)
        if dsi_i <= cur_si and not (is_running and dsi_i == cur_si):
            out["terminal_status"] = "completed"
            out["result_send_index"] = dsi_i
            if out.get("result_seq") is None and not is_driver:
                msg = await reader.fetchone(
                    f"SELECT seq FROM {self.store.t.messages} "
                    "WHERE session_id=? AND send_index=? AND role='assistant' "
                    "ORDER BY seq DESC LIMIT 1",
                    (session_row["session_id"], dsi_i),
                )
                if msg is not None:
                    out["result_seq"] = int(msg["seq"])
                else:
                    proj = await reader.fetchone(
                        f"SELECT MAX(source_seq_hi) AS max_hi FROM {self.store.t.project_messages} "
                        "WHERE session_id=? AND send_index=?",
                        (session_row["session_id"], dsi_i),
                    )
                    if proj is not None and proj["max_hi"] is not None:
                        out["result_seq"] = int(proj["max_hi"])
        return out

    async def inspect_input(self, ref: InputRef, *, actor=None) -> dict[str, Any]:
        """Unified point lookup for one InputRef with derived normal-completion state."""
        self.check_store(ref)
        target = TargetRef(ref.store_id, ref.session_id)
        async with self.db.read_snapshot() as reader:
            srow = await self._session(reader, ref.session_id)
            if actor is not None:
                await self._authorize_known(reader, target, actor, srow)
            irow = await reader.fetchone(
                f"SELECT * FROM {self.q} WHERE session_id=? AND item_id=?",
                (ref.session_id, ref.message_id),
            )
            if irow is None:
                raise RuntimeRequestError("not_found")
            return await self._derive_input_terminal(reader, dict(srow), dict(irow))

    async def inspect(self, ref, *, actor):
        target = ref if isinstance(ref, TargetRef) else TargetRef(ref.store_id, ref.session_id)
        async with self.db.read_snapshot() as reader:
            state = await self.authorize(reader, target, actor)
            is_running = state["execution_state"] == "running"
            active_send = int(state["send_index"]) if is_running else None
            cancel_active = is_running and (
                state["cancel_by"] is not None or state["lifecycle"] in ("closing", "closed")
            )
            if isinstance(ref, TargetRef):
                send_index = int(state["send_index"]) or None
                return Status(
                    None, target, state["execution_state"], None, None, send_index, False,
                    None, cancel_active, state["cancel_reason"], None, None, None, None,
                )
            record = await self.input(reader, ref)
            if record is None:
                raise RuntimeRequestError("not_found")
            row = await self._derive_input_terminal(reader, dict(state), record.row)
            return Status(
                ref, target, state["execution_state"], row["status"], row["terminal_status"],
                row["result_send_index"] or row["delivered_send_index"],
                row["terminal_status"] is not None, row["result_seq"],
                bool(cancel_active and row["delivered_send_index"] == active_send),
                row["error_detail"] or (
                    state["cancel_reason"]
                    if cancel_active and row["delivered_send_index"] == active_send else None
                ),
                row["error_code"], row["result_delivery_error"], row["continuation_of"],
                row["continuation_index"],
            )

    async def _continuation_child(self, reader, record):
        row = await reader.fetchone(
            f"SELECT session_id,item_id,kind,status,terminal_status,result_seq,"
            "result_send_index,error_code,error_detail,result_delivery_error,"
            "delivered_send_index,continuation_of,result_json,cancel_by "
            f"FROM {self.q} WHERE session_id=? AND continuation_of=? "
            "ORDER BY continuation_index DESC LIMIT 1",
            (record.ref.session_id, record.ref.message_id),
        )
        if row is None:
            return None
        return ResultInput(
            InputRef(self.store.store_id, row["session_id"], row["item_id"]),
            {name: row[name] for name in _RESULT_INPUT_COLUMNS},
        )

    async def _latest_continuation(self, reader, record):
        current = record
        for _ in range(10000):
            if current.row["status"] != "waiting_continuation":
                return current
            child = await self._continuation_child(reader, current)
            if child is None:
                return current
            current = child
        raise CapacityError()

    async def _result_state(self, reader, ref: InputRef, target: TargetRef, actor, *, follow_continuation: bool):
        row = await reader.fetchone(
            "SELECT s.session_id,s.tenant_id,s.principal_id,s.parent_session_id,"
            "s.session_kind,s.execution_state,s.send_index,s.start_input_message_id,"
            "q.item_id AS input_item_id,q.kind AS input_kind,q.status AS input_status,"
            "q.terminal_status AS input_terminal_status,q.result_seq AS input_result_seq,"
            "q.result_send_index AS input_result_send_index,q.error_code AS input_error_code,"
            "q.error_detail AS input_error_detail,q.result_delivery_error AS input_result_delivery_error,"
            "q.delivered_send_index AS input_delivered_send_index,q.continuation_of AS input_continuation_of,"
            "q.result_json AS input_result_json,q.cancel_by AS input_cancel_by,"
            "m.seq AS message_seq,m.content AS message_content,m.meta_json AS message_meta_json "
            f"FROM {self.s} s LEFT JOIN {self.q} q "
            "ON q.session_id=s.session_id AND q.item_id=? "
            f"LEFT JOIN {self.store.t.messages} m "
            "ON m.session_id=s.session_id AND m.seq=COALESCE("
            "q.result_seq,"
            f"(SELECT MAX(m2.seq) FROM {self.store.t.messages} m2 "
            "WHERE m2.session_id=s.session_id AND m2.send_index=q.delivered_send_index "
            "AND m2.role='assistant')) "
            "WHERE s.session_id=?",
            (ref.message_id, ref.session_id),
        )
        if row is None:
            raise RuntimeRequestError("not_found")
        session = {
            "session_id": row["session_id"],
            "tenant_id": row["tenant_id"],
            "principal_id": row["principal_id"],
            "parent_session_id": row["parent_session_id"],
            "session_kind": row["session_kind"],
            "execution_state": row["execution_state"],
            "send_index": row["send_index"],
            "start_input_message_id": row["start_input_message_id"],
        }
        await self._authorize_known(reader, target, actor, session)
        if row["input_item_id"] is None:
            return ResultRead(ref, target, session, None, None, None), False
        input_dict = {
            "session_id": row["session_id"],
            "item_id": row["input_item_id"],
            "kind": row["input_kind"],
            "status": row["input_status"],
            "terminal_status": row["input_terminal_status"],
            "result_seq": row["input_result_seq"],
            "result_send_index": row["input_result_send_index"],
            "error_code": row["input_error_code"],
            "error_detail": row["input_error_detail"],
            "result_delivery_error": row["input_result_delivery_error"],
            "delivered_send_index": row["input_delivered_send_index"],
            "continuation_of": row["input_continuation_of"],
            "result_json": row["input_result_json"],
            "cancel_by": row["input_cancel_by"],
        }
        record = ResultInput(ref, input_dict)
        continuation = (
            record.row["status"] == "waiting_continuation"
            or record.row["kind"] == "workflow_continue"
        )
        if continuation and not follow_continuation:
            return ResultRead(ref, target, session, record, record, None), True
        if not continuation:
            # Pure in-memory derivation from the single joined row (0 extra SQL!).
            derived = dict(input_dict)
            if derived["status"] == "delivered" and derived["terminal_status"] is None:
                dsi = derived.get("delivered_send_index")
                is_driver = session["session_kind"] == "workflow_driver"
                start_id = session.get("start_input_message_id")
                if dsi is not None and not (is_driver and start_id is not None and derived["item_id"] == start_id):
                    cur_si = int(session["send_index"])
                    is_running = session["execution_state"] == "running"
                    dsi_i = int(dsi)
                    if dsi_i <= cur_si and not (is_running and dsi_i == cur_si):
                        derived["terminal_status"] = "completed"
                        derived["result_send_index"] = dsi_i
                        if derived.get("result_seq") is None and row["message_seq"] is not None:
                            derived["result_seq"] = int(row["message_seq"])
            resolved = ResultInput(ref, derived)
            joined_msg = (
                {"content": row["message_content"], "meta_json": row["message_meta_json"]}
                if row["message_seq"] is not None else None
            )
            result = await self._terminal_result(reader, ref, target, resolved, message_row=joined_msg)
            return ResultRead(ref, target, session, record, resolved, result), False
        resolved = await self._latest_continuation(reader, record)
        derived = await self._derive_input_terminal(reader, session, resolved.row)
        resolved = ResultInput(resolved.ref, derived)
        result = await self._terminal_result(reader, ref, target, resolved)
        return ResultRead(ref, target, session, record, resolved, result), False

    async def _terminal_result(self, reader, ref, target, resolved: ResultInput, *, message_row=None):
        terminal = resolved.row["terminal_status"]
        if terminal is None:
            return None
        final_text, usage, structured = "", {}, None
        result_seq = resolved.row["result_seq"]
        result_si = resolved.row["result_send_index"]
        if message_row is not None and result_seq is not None:
            final_text = message_row["content"] or ""
            meta = json.loads(message_row["meta_json"] or "{}")
            usage = meta.get("usage") or {}
            structured = meta.get("structured_output")
        elif result_seq is not None and result_si is not None:
            message = await reader.fetchone(
                f"SELECT content,meta_json FROM {self.store.t.messages} "
                "WHERE session_id=? AND seq=? AND send_index=?",
                (ref.session_id, result_seq, result_si),
            )
            if message is not None:
                final_text = message["content"] or ""
                meta = json.loads(message["meta_json"] or "{}")
                usage = meta.get("usage") or {}
                structured = meta.get("structured_output")
        if structured is None and resolved.row.get("result_json"):
            structured = json.loads(resolved.row["result_json"])
        return Result(
            ref, target, terminal, result_si, result_seq,
            final_text, usage, structured, resolved.row["error_detail"],
            resolved.row["error_code"], resolved.row["result_delivery_error"],
        )

    async def read_result_state(self, ref: InputRef, *, actor) -> ResultRead:
        self.check_store(ref)
        target = TargetRef(ref.store_id, ref.session_id)
        if actor.session_id is None or actor.session_id == ref.session_id:
            state, continuation = await self._result_state(
                self.db, ref, target, actor, follow_continuation=False,
            )
            if not continuation:
                return state
        async with self.db.read_snapshot() as reader:
            state, _ = await self._result_state(
                reader, ref, target, actor, follow_continuation=True,
            )
        return state

    async def get_result(self, ref: InputRef, *, actor):
        state = await self.read_result_state(ref, actor=actor)
        if state.record is None:
            raise RuntimeRequestError("not_found")
        return state.result

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
                f"SELECT session_id FROM {self.s} WHERE parent_session_id=? "
                "AND session_id>? ORDER BY session_id LIMIT ?",
                (target.session_id, cursor or "", limit),
            )
        items = tuple(TargetRef(self.store.store_id, row["session_id"]) for row in rows)
        return SessionPage(items, items[-1].session_id if len(items) == limit else None)

    async def poll_notifications(self, binding, *, cursor=None, limit=100):
        """Bounded shared-store discovery (supports both explicit terminal and derived normal completion)."""
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

        settled_cond = (
            "(q.terminal_status IS NOT NULL OR ("
            "q.status='delivered' AND q.terminal_status IS NULL AND q.delivered_send_index IS NOT NULL "
            "AND q.delivered_send_index<=s.send_index "
            "AND NOT (s.execution_state='running' AND q.delivered_send_index=s.send_index) "
            "AND NOT (s.session_kind='workflow_driver' AND s.start_input_message_id IS NOT NULL "
            "AND q.item_id=s.start_input_message_id)))"
        )
        select = (
            "SELECT q.id,q.session_id,q.item_id,"
            "COALESCE(q.terminal_status,'completed') AS terminal_status,"
            "COALESCE(q.result_send_index,q.delivered_send_index) AS result_send_index,"
            "COALESCE(q.completed_at,q.delivered_at,q.created_at) AS done_at "
            f"FROM {self.q} q JOIN {self.s} s ON s.session_id=q.session_id WHERE "
        )
        if binding.discovery_message_id is not None:
            rows = await self.db.fetchall(
                select + f"q.session_id=? AND q.item_id=? AND {settled_cond}",
                (binding.session_id, binding.discovery_message_id),
            )
        else:
            where, params = [settled_cond], []
            if cursor is not None:
                done_at, row_id = cursor
                where.append(
                    "(COALESCE(q.completed_at,q.delivered_at,q.created_at)>? "
                    "OR (COALESCE(q.completed_at,q.delivered_at,q.created_at)=? AND q.id>?))"
                )
                params.extend((done_at, done_at, row_id))
            if binding.session_id is not None:
                where.append("q.session_id=?")
                params.append(binding.session_id)
            params.append(limit)
            rows = await self.db.fetchall(
                select + f"{' AND '.join(where)} ORDER BY done_at,q.id LIMIT ?",
                tuple(params),
            )
        packets = []
        next_cursor = cursor
        for row in rows:
            ref = InputRef(self.store.store_id, row["session_id"], row["item_id"])
            packets.append(TransportPacket(
                PROTOCOL_VERSION, binding.namespace, row["item_id"],
                TargetRef(self.store.store_id, row["session_id"]), "output", ref,
                {"kind": "completed", "session_id": row["session_id"],
                 "send_index": row["result_send_index"]},
            ))
            next_cursor = (row["done_at"], row["id"])
        return packets, next_cursor
