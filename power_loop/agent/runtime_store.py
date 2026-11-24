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
    ContextBudget,
    InputRef,
    MessagePage,
    MessageRow,
    RebindResult,
    Receipt,
    RepairResult,
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
_PUBLIC_REPLY_RESULT_KINDS = frozenset({"agent_result"})
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


def _extract_result_usage(raw_meta: str | None) -> dict[str, Any] | None:
    if not raw_meta:
        return None
    usage = json.loads(raw_meta).get("result_usage")
    return dict(usage) if usage else None


@dataclass(frozen=True)
class ResultInput:
    """Narrow input view needed to resolve and report a durable result."""
    ref: InputRef
    row: dict
    result_usage: dict[str, Any] | None = None


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
class PendingProjection:
    """Deferred double-row ('user'+'project') projection + inline fold_stats_json for Final."""
    rows: tuple[tuple[str, Any, str | None], ...]
    source_seq_lo: int | None
    source_seq_hi: int | None
    projector_version: int
    fold_stats_json: str | None = None
    expected_fold_stats_json: str | None = None
    projector: Any = None
    policy: Any = None


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
    reply_to_item_id: str | None = None
    fold_stats_json: str | None = None
    is_reclaimed: bool = False
    reclaimed_has_assistant: bool = False
    reclaimed_cancelled: bool = False
    reclaimed_next_round_index: int = 0
    reclaimed_pending_tool_seq: int | None = None
    max_rounds_override: int | None = None


def _extract_max_rounds_override(rows) -> int | None:
    override: int | None = None
    for row in rows:
        raw_meta = row.get("meta_json")
        if not raw_meta:
            continue
        try:
            parsed = json.loads(raw_meta)
        except Exception:
            continue
        if isinstance(parsed, dict):
            val = parsed.get("max_rounds")
            if isinstance(val, int) and not isinstance(val, bool) and val >= 1:
                override = val
    return override


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


class _SubtreeChangedRetry(Exception):
    pass


class _RebindChangedRetry(Exception):
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
        hooks=None,
        hook_timeout_s: float = 1.0,
        hooks_provider=None,
        on_repair_interrupted_turn=None,
        on_session_takeover=None,
        on_result_undeliverable=None,
        on_local_claim=None,
        on_local_claim_cancel=None,
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
        self.hooks = hooks
        self.hook_timeout_s = hook_timeout_s
        self._hooks_provider = hooks_provider
        self.on_repair_interrupted_turn = on_repair_interrupted_turn
        self.on_session_takeover = on_session_takeover
        self.on_result_undeliverable = on_result_undeliverable
        self._on_local_claim = on_local_claim
        self._on_local_claim_cancel = on_local_claim_cancel
        waiters = getattr(store, "_driver_waiters", None)
        if waiters is None:
            waiters = {}
            store._driver_waiters = waiters
        self._driver_waiters: dict[str, tuple[asyncio.Queue, asyncio.Event, list[bool]]] = waiters

    def register_driver_waiter(self, sid: str) -> tuple[asyncio.Queue, asyncio.Event, list[bool]]:
        q: asyncio.Queue = asyncio.Queue()
        sig = asyncio.Event()
        subtree_flag = [False]
        self._driver_waiters[sid] = (q, sig, subtree_flag)
        return q, sig, subtree_flag

    def unregister_driver_waiter(self, sid: str) -> None:
        self._driver_waiters.pop(sid, None)

    def notify_driver_wake(self, sid: str) -> None:
        entry = self._driver_waiters.get(sid)
        if entry is not None:
            q, sig, _ = entry
            q.put_nowait({"type": "remote_wake"})
            sig.set()

    def notify_driver_cancel(
        self, sid: str, *, cancel_by: str | None = None, reason: str = "cancelled",
        subtree_stopped: bool = False,
    ) -> None:
        entry = self._driver_waiters.get(sid)
        if entry is not None:
            q, sig, subtree_flag = entry
            if subtree_stopped:
                subtree_flag[0] = True
            q.put_nowait({"type": "cancel", "cancel_by": cancel_by, "reason": reason})
            sig.set()

    def _push_driver_leaf_result(self, driver_sid: str, item_id: str, payload: dict[str, Any]) -> None:
        entry = self._driver_waiters.get(driver_sid)
        if entry is not None:
            q, sig, _ = entry
            q.put_nowait({"type": "leaf_result", "item_id": item_id, "payload": payload})
            sig.set()

    async def poll_pending_driver_results(self, sid: str, *, after_id: int = 0):
        """Single indexed SELECT on ``idx_inbox_session_status`` for cross-process leaf results."""
        return await self.db.fetchall(
            f"SELECT id,item_id,content_json FROM {self.q} "
            "WHERE session_id=? AND status='pending' AND kind='workflow_node_result' AND id>? "
            "ORDER BY id ASC LIMIT 1000",
            (sid, int(after_id)),
        )

    async def poll_driver_results_for_send(self, sid: str, send_index: int, *, after_id: int = 0):
        """Crash-replay poll: includes pending AND node results already claimed into ``send_index``."""
        return await self.db.fetchall(
            f"SELECT id,item_id,content_json FROM {self.q} "
            "WHERE session_id=? AND kind='workflow_node_result' AND terminal_status IS NULL "
            "AND (status='pending' OR (status='delivered' AND delivered_send_index=?)) AND id>? "
            "ORDER BY id ASC LIMIT 1000",
            (sid, int(send_index), int(after_id)),
        )

    def _effective_hooks(self):
        if self.hooks is not None:
            return self.hooks
        if self._hooks_provider is not None:
            return self._hooks_provider()
        return None

    def _turn_repair_callback(self):
        """Bridge ``on_repair_interrupted_turn`` and ``HookPoint.TURN_REPAIR``."""
        hooks = self._effective_hooks()
        direct = self.on_repair_interrupted_turn
        if direct is None and hooks is None:
            return None

        async def _repair_hook(session_id: str, dangling: list[dict[str, Any]]) -> dict[str, str]:
            import inspect
            import logging

            recovered: dict[str, str] = {}
            if direct is not None:
                res = direct(session_id, dangling)
                if inspect.isawaitable(res):
                    res = await res
                if res:
                    recovered.update({str(k): str(v) for k, v in res.items()})
            if hooks is not None:
                from power_loop.contracts.hooks import HookContext, HookPoint
                ctx = HookContext({
                    "session_id": session_id,
                    "dangling": dangling,
                    "recovered": dict(recovered),
                })
                try:
                    async with asyncio.timeout(self.hook_timeout_s):
                        hr = await hooks.run_async(HookPoint.TURN_REPAIR, ctx)
                    hook_rec = hr.context.values.get("recovered")
                    if isinstance(hook_rec, dict):
                        recovered.update({str(k): str(v) for k, v in hook_rec.items()})
                except Exception:
                    logging.getLogger(__name__).exception(
                        "execution notification failed: %s", HookPoint.TURN_REPAIR
                    )
            return recovered

        return _repair_hook

    async def _fire_session_takeover(
        self,
        *,
        session_id: str,
        target: TargetRef,
        prev_node_id: str | None,
        prev_boot_id: str | None,
        new_node_id: str,
        new_boot_id: str,
        owner_epoch: int,
        send_index: int,
    ) -> None:
        import inspect
        import logging

        if self.on_session_takeover is not None:
            try:
                res = self.on_session_takeover(session_id, prev_node_id, prev_boot_id, owner_epoch)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logging.getLogger(__name__).exception("on_session_takeover callback failed for %s", session_id)
        hooks = self._effective_hooks()
        if hooks is not None:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(
                hooks,
                HookPoint.SESSION_TAKEOVER,
                {
                    "session_id": session_id,
                    "target": target,
                    "previous_node_id": prev_node_id,
                    "previous_boot_id": prev_boot_id,
                    "new_node_id": new_node_id,
                    "new_boot_id": new_boot_id,
                    "owner_epoch": owner_epoch,
                    "send_index": send_index,
                },
                timeout_s=self.hook_timeout_s,
            )

    async def _fire_result_undeliverable(
        self,
        *,
        child_target: TargetRef,
        parent_session_id: str,
        result: Result,
        result_kind: str,
        reason: str,
    ) -> None:
        import inspect
        import logging

        if self.on_result_undeliverable is not None:
            try:
                res = self.on_result_undeliverable(
                    child_target.session_id, parent_session_id, plain(result), reason
                )
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logging.getLogger(__name__).exception(
                    "on_result_undeliverable callback failed for %s", child_target.session_id
                )
        hooks = self._effective_hooks()
        if hooks is not None:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(
                hooks,
                HookPoint.RESULT_UNDELIVERABLE,
                {
                    "child_session_id": child_target.session_id,
                    "child_target": child_target,
                    "parent_session_id": parent_session_id,
                    "parent_target": TargetRef(self.store.store_id, parent_session_id),
                    "result": result,
                    "result_kind": result_kind,
                    "reason": reason,
                },
                timeout_s=self.hook_timeout_s,
            )

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

        registered_claims: list[tuple[str, int]] = []
        try:
            now_sql = self._now_sql()
            lock_suffix = " FOR UPDATE" if self._dialect in ("postgres", "mysql") else ""
            try:
                async with self.db.transaction() as tx:
                    parent_row = await tx.fetchone(
                        f"SELECT *,{now_sql} AS _db_now FROM {self.s} WHERE session_id=?{lock_suffix}",
                        (parent.session_id,),
                    )
                    if parent_row is None:
                        raise ValueError(f"unknown session: {parent.session_id}")
                    await self._authorize_known(tx, parent, actor, parent_row)
                    if parent_owner_epoch is not None and (
                        int(parent_row["owner_epoch"]) != int(parent_owner_epoch)
                        or parent_row["execution_state"] != "running"
                    ):
                        raise SessionLeaseLost(parent.session_id)
                    if parent_row["lifecycle"] != "open":
                        raise RuntimeRequestError("session_closed")
                    if parent_row["execution_state"] == "running" and parent_row["cancel_by"] is not None:
                        raise RuntimeRequestError("cancelled", parent_row["cancel_reason"] or "parent_cancelled")
                    if parent_cancel_seq is not None and int(parent_row["cancel_seq"]) > int(parent_cancel_seq):
                        raise RuntimeRequestError("cancelled", "parent_cancelled")
                    if any(sp.descriptor.kind == "workflow_driver" for sp in specs) and (
                        parent_row["parent_session_id"] is not None or parent_row["session_kind"] != "agent"
                    ):
                        raise RuntimeRequestError("forbidden", "execution_role_forbids_workflow")

                    database_now = int(parent_row["_db_now"])
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
                    sess_rows_sql: list[str] = []
                    sess_params: list[Any] = []
                    queue_rows_sql: list[str] = []
                    queue_params: list[Any] = []
                    msg_rows_sql: list[str] = []
                    msg_params: list[Any] = []

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

                        sess_rows_sql.append(
                            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                        )
                        sess_params.extend((
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
                        ))
                        deliv_seq = None if is_driver else 1
                        queue_rows_sql.append(
                            "(?,?,1,'user','normal',1,'delivered',?,?,?,?,?,?,NULL,?,?,?,1)"
                        )
                        queue_params.extend((
                            sp.session_id, sp.message_id, reply_to, sp.result_kind,
                            content, content_bytes, meta, req_hash, database_now,
                            database_now, deliv_seq,
                        ))
                        if not is_driver:
                            msg_rows_sql.append("(?,?,?,?,?,?,?)")
                            msg_params.extend((
                                sp.session_id, 1, "user", merged_json,
                                canonical(msg_meta), 1, database_now,
                            ))
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
                                reply_to_item_id=sp.message_id,
                            ))
                        else:
                            queued.append(child_target)

                    await tx.execute(
                        f"INSERT INTO {self.s} ("
                        "session_id,parent_session_id,created_by_session_id,root_session_id,depth,"
                        "session_kind,descriptor_json,descriptor_hash,creation_identity_hash,"
                        "creation_call_ref_json,creation_tool_call_id,start_input_message_id,"
                        "tenant_id,principal_id,result_route_json,cancel_propagation,"
                        "next_seq,send_index,execution_state,owner_epoch,owner_node_id,"
                        "owner_boot_id,owner_lease_until_ms,inbox_version,wake_inbox_seq,"
                        "wake_pending_count,pending_count,pending_bytes,ready_item_json,"
                        "created_at,updated_at) VALUES " + ",".join(sess_rows_sql),
                        tuple(sess_params),
                    )
                    await tx.execute(
                        f"INSERT INTO {self.q} ("
                        "session_id,item_id,inbox_version,kind,priority,wake,status,"
                        "reply_to_session_id,reply_result_kind,content_json,content_bytes,"
                        "meta_json,request_hash,result_kind,created_at,delivered_at,"
                        "delivered_seq,delivered_send_index) VALUES " + ",".join(queue_rows_sql),
                        tuple(queue_params),
                    )
                    if msg_rows_sql:
                        await tx.execute(
                            f"INSERT INTO {self.store.t.messages} ("
                            "session_id,seq,role,content,meta_json,send_index,created_at) VALUES "
                            + ",".join(msg_rows_sql),
                            tuple(msg_params),
                        )
                    if preclaimed and self._on_local_claim is not None:
                        registered_claims = [(b.target.session_id, b.owner_epoch) for b in preclaimed]
                        self._on_local_claim(node_id, boot_id, registered_claims)
                return CreateManyResult(preclaimed=tuple(preclaimed), queued_targets=tuple(queued))
            except BaseException:
                if registered_claims and self._on_local_claim_cancel is not None:
                    self._on_local_claim_cancel(node_id, boot_id, registered_claims)
                raise
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
                    is_driver_launch = descriptor.kind == "workflow_driver" or result_kind == "workflow_result"
                    is_main_agent = (
                        parent_session["parent_session_id"] is None
                        and parent_session["session_kind"] == "agent"
                    )
                    if is_driver_launch:
                        if not is_main_agent:
                            raise RuntimeRequestError("forbidden", "execution_role_forbids_workflow")
                    elif parent_session["session_kind"] != "workflow_driver" and not is_main_agent:
                        raise RuntimeRequestError("forbidden", "execution_role_forbids_workflow")
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

    def _parse_reply_route(self, reply_route, *, public: bool = True) -> tuple[str | None, str | None]:
        if reply_route is None:
            return None, None
        if not isinstance(reply_route, dict):
            raise RuntimeRequestError("invalid_options", "reply_route must be a mapping")
        target = reply_route.get("target") or reply_route.get("reply_to_session_id")
        if isinstance(target, TargetRef):
            self.check_store(target)
            reply_sid = target.session_id
        elif isinstance(target, dict) and "session_id" in target:
            if target.get("store_id") is not None and target["store_id"] != self.store.store_id:
                raise RuntimeRequestError("forbidden", "reply_route store mismatch")
            reply_sid = str(target["session_id"])
            if not reply_sid:
                raise RuntimeRequestError("invalid_options", "reply_route target required")
        elif isinstance(target, str) and target:
            reply_sid = target
        else:
            raise RuntimeRequestError("invalid_options", "reply_route target required")
        rk = reply_route.get("result_kind") or reply_route.get("reply_result_kind") or "agent_result"
        if rk not in _VALID_RESULT_KINDS or (public and rk not in _PUBLIC_REPLY_RESULT_KINDS):
            raise RuntimeRequestError("invalid_options", "reply_route result_kind")
        return reply_sid, rk

    async def _validate_public_reply_route(
        self, reader, target: TargetRef, reply_to_sid: str, actor, *, target_row=None,
    ) -> None:
        """Ensure public reply_route points to an authorized shallower ancestor of target."""
        if reply_to_sid == target.session_id:
            raise RuntimeRequestError("forbidden", "reply_route target must be a shallower ancestor")
        reply_row = await reader.fetchone(
            f"SELECT session_id,parent_session_id,depth,tenant_id,principal_id "
            f"FROM {self.s} WHERE session_id=?",
            (reply_to_sid,),
        )
        if reply_row is None:
            raise RuntimeRequestError("not_found", "reply_route target")
        await self._authorize_known(
            reader, TargetRef(self.store.store_id, reply_to_sid), actor, reply_row,
        )
        if target_row is None:
            target_row = await reader.fetchone(
                f"SELECT session_id,parent_session_id,depth,tenant_id,principal_id "
                f"FROM {self.s} WHERE session_id=?",
                (target.session_id,),
            )
            if target_row is None:
                raise RuntimeRequestError("not_found")
        if int(reply_row["depth"]) >= int(target_row["depth"]):
            raise RuntimeRequestError("forbidden", "reply_route target must be a shallower ancestor")
        parent_sid = target_row["parent_session_id"]
        if parent_sid != reply_to_sid and not await self._is_ancestor(reader, parent_sid, reply_to_sid):
            raise RuntimeRequestError("forbidden", "reply_route target must be a shallower ancestor")

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
        self, target, body, *, message_id, priority, kind, wake, reply_route, public,
        result_kind: str | None = None, max_rounds: int | None = None,
    ):
        if priority not in {"normal", "urgent"}:
            raise RuntimeRequestError("invalid_options", "priority")
        if public and isinstance(message_id, str) and message_id.startswith(RESERVED_INPUT_PREFIXES):
            raise RuntimeRequestError("invalid_options", "reserved message_id prefix")
        if kind not in _VALID_INPUT_KINDS or (public and kind not in _PUBLIC_INPUT_KINDS):
            raise RuntimeRequestError("invalid_options", "kind")
        if public and result_kind is not None:
            raise RuntimeRequestError("invalid_options", "result_kind")
        if max_rounds is not None:
            if public or not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1:
                raise RuntimeRequestError("invalid_options", "max_rounds")
        self.check_store(target)
        ref = InputRef(target.store_id, target.session_id, message_id)
        reply_to_sid, reply_rk = self._parse_reply_route(reply_route, public=public)
        if public and reply_to_sid is not None and reply_to_sid == target.session_id:
            raise RuntimeRequestError("forbidden", "reply_route target must be a shallower ancestor")
        content = canonical(body)
        meta_dict: dict[str, Any] = {"accepted_protocol": PROTOCOL_VERSION}
        if max_rounds is not None:
            meta_dict["max_rounds"] = max_rounds
        meta = canonical(meta_dict)
        content_bytes = len(content.encode()) + len(meta.encode())
        if max_rounds is None:
            digest = self.request_hash(target, body, priority)
        else:
            digest = sha256_canonical([
                PROTOCOL_VERSION, target.store_id, target.session_id, body, priority,
                {"max_rounds": max_rounds},
            ])
        wake_int = 1 if wake else 0
        return ref, digest, content, meta, content_bytes, wake_int, reply_to_sid, reply_rk

    async def accept(
        self, target, body, *, message_id, actor, priority="normal", deadline_ms=None,
        kind="user", result_kind=None, wake: bool = True, reply_route: dict | None = None,
        public: bool = True, max_rounds: int | None = None, _with_activation: bool = False,
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
                result_kind=result_kind, max_rounds=max_rounds,
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
                    if public and reply_to_sid is not None:
                        await self._validate_public_reply_route(tx, target, reply_to_sid, actor)
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
                    public=public,
                )
                return (receipt, need_wake) if _with_activation else receipt
        except RuntimeRequestError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            raise StoreError(ref) from exc

    async def _accept_slow_path(
        self, target, ref, digest, content, meta, content_bytes, *,
        actor, priority, deadline_ms, kind, result_kind, wake_int, limit,
        reply_to_sid, reply_rk, public: bool = True,
    ) -> tuple[Receipt, bool]:
        now_sql = self._now_sql()
        diag = await self.db.fetchone(
            "SELECT s.session_id,s.tenant_id,s.principal_id,s.parent_session_id,s.depth,"
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
        if public and reply_to_sid is not None:
            await self._validate_public_reply_route(
                self.db, target, reply_to_sid, actor, target_row=diag,
            )
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
                            "cancel_by=CASE WHEN execution_state='idle' AND ?=1 THEN NULL ELSE cancel_by END,"
                            "wake_pending_count=wake_pending_count+?,"
                            "pending_count=pending_count+1,"
                            "pending_bytes=pending_bytes+?,"
                            "execution_state=CASE WHEN execution_state='idle' AND ?=1 THEN 'queued' ELSE execution_state END,"
                            f"updated_at={now_sql} "
                            "WHERE session_id=? AND lifecycle='open' "
                            f"AND (? IS NULL OR ?>{now_sql}) "
                            "AND pending_count<? AND (pending_bytes+?)<=?",
                            (
                                wake_int, wake_int, wake_int, wake_int, content_bytes, wake_int,
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
                            "cancel_by=CASE WHEN execution_state='idle' AND ?=1 THEN NULL ELSE cancel_by END,"
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
                                wake_int, wake_int, wake_int, content_bytes, wake_int,
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
                    # Re-verify actor ancestry and public reply_route under the locked target row
                    # inside this write transaction so concurrent rebind_children cannot bypass them.
                    locked_target = await tx.fetchone(
                        f"SELECT session_id,parent_session_id,depth,tenant_id,principal_id "
                        f"FROM {self.s} WHERE session_id=?",
                        (target.session_id,),
                    )
                    if locked_target is None:
                        raise RuntimeRequestError("not_found")
                    await self._authorize_known(tx, target, actor, locked_target)
                    if public and reply_to_sid is not None:
                        await self._validate_public_reply_route(
                            tx, target, reply_to_sid, actor, target_row=locked_target,
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
        max_rounds: int | None = None,
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
                result_kind=result_kind, max_rounds=max_rounds,
            )
        )
        if not wake:
            receipt, need_wake = await self.accept(
                target, body, message_id=message_id, actor=actor, priority=priority,
                deadline_ms=deadline_ms, kind=kind, result_kind=result_kind,
                wake=False, reply_route=reply_route, public=public,
                max_rounds=max_rounds, _with_activation=True,
            )
            return receipt, None, need_wake

        now_sql = self._now_sql()
        pt, pi = self._p_text(), self._p_int()
        ttl_ms = int(ttl_s * 1000)
        registered_claims: list[tuple[str, int]] = []
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
                            f"descriptor_json,fold_stats_json,updated_at FROM {self.s} WHERE session_id=?",
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
                            "descriptor_json,fold_stats_json,updated_at",
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
                    if public and reply_to_sid is not None:
                        await self._validate_public_reply_route(
                            tx, target, reply_to_sid, actor, target_row=srow,
                        )

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
                        reply_to_item_id=(ref.message_id if reply_to_sid is not None else None),
                        fold_stats_json=srow["fold_stats_json"],
                        max_rounds_override=max_rounds,
                    )
                    if self._on_local_claim is not None:
                        registered_claims = [(target.session_id, batch.owner_epoch)]
                        self._on_local_claim(node_id, boot_id, registered_claims)
                receipt = Receipt(ref, target, ref.message_id, True, "accepted", send_idx, "not_needed")
                return receipt, batch, False
            except _AcceptSlowPath:
                if registered_claims and self._on_local_claim_cancel is not None:
                    self._on_local_claim_cancel(node_id, boot_id, registered_claims)
                receipt, need_wake = await self.accept(
                    target, body, message_id=message_id, actor=actor, priority=priority,
                    deadline_ms=deadline_ms, kind=kind, result_kind=result_kind,
                    wake=wake, reply_route=reply_route, public=public,
                    max_rounds=max_rounds, _with_activation=True,
                )
                return receipt, None, need_wake
            except BaseException:
                if registered_claims and self._on_local_claim_cancel is not None:
                    self._on_local_claim_cancel(node_id, boot_id, registered_claims)
                raise
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
                        "s.ready_item_json,s.descriptor_json,s.fold_stats_json,s.updated_at,"
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
                        "ready_item_json,descriptor_json,fold_stats_json,updated_at",
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
                prestaged_ids: set[int] = set()
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
                        prestaged_ids = {int(r["id"]) for r in extra_pre}
                        rows = [*extra_pre, *rows]

                # Probe 4B/4C: filter pending items that already carry cancel_by (returned by release_claim)
                cancelled_rows = [r for r in rows if r.get("cancel_by") is not None]
                active_rows = [r for r in rows if r.get("cancel_by") is None]
                batch_cancel_by = srow["cancel_by"]
                if cancelled_rows and active_rows and not is_driver:
                    c_ids = [int(r["id"]) for r in cancelled_rows]
                    c_ph = ",".join("?" for _ in c_ids)
                    await tx.execute(
                        f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                        "delivered_at=NULL,delivered_seq=NULL,delivered_send_index=NULL,"
                        f"error_code='cancelled',error_detail='cancelled',completed_at={now_sql} "
                        f"WHERE session_id=? AND id IN ({c_ph})",
                        (target.session_id, *c_ids),
                    )
                    rows = active_rows
                elif cancelled_rows and not active_rows:
                    # All items were marked cancelled via release_claim; propagate cancel_by onto batch
                    batch_cancel_by = batch_cancel_by or cancelled_rows[0]["cancel_by"]

                refs = tuple(
                    InputRef(target.store_id, target.session_id, r["item_id"]) for r in rows
                )
                context["refs"] = refs
                deliv_seq = None if is_driver else seq
                if not pending_already_delivered:
                    ids = [int(r["id"]) for r in rows if int(r["id"]) not in prestaged_ids]
                    if ids:
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
                reply_row = (
                    next(
                        (
                            r for r in rows
                            if r.get("reply_to_session_id")
                            and r.get("reply_result_kind") == "workflow_node_result"
                        ),
                        None,
                    )
                    or next((r for r in rows if r.get("reply_to_session_id")), None)
                )
                reply_to_sid = reply_row.get("reply_to_session_id") if reply_row is not None else None
                reply_rk = reply_row.get("reply_result_kind") if reply_row is not None else None
                reply_item_id = reply_row.get("item_id") if reply_row is not None else None
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
                    cancel_by=batch_cancel_by,
                    start_input_message_id=srow["start_input_message_id"],
                    root_session_id=str(srow["root_session_id"]),
                    depth=int(srow["depth"]),
                    reply_to_session_id=reply_to_sid,
                    reply_result_kind=reply_rk,
                    reply_to_item_id=reply_item_id,
                    fold_stats_json=srow["fold_stats_json"],
                    reclaimed_cancelled=bool(batch_cancel_by is not None and not active_rows),
                    max_rounds_override=_extract_max_rounds_override(rows),
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
            reply_to_item_id=(ready["item_id"] if ready.get("reply_to_session_id") else None),
            fold_stats_json=srow["fold_stats_json"],
            reclaimed_cancelled=bool(srow["cancel_by"] is not None),
        )

    async def release_claim(self, batch: ClaimedBatch, *, _on_committed=None):
        """Atomically release an unexecuted claim with Probe 4B lifecycle/cancel fences.

        - ``closing`` / ``closed``: stays ``execution_state='idle'`` (NEVER resurrects to
          ``queued``) and marks returned inputs ``status='void', terminal_status='cancelled'``.
        - ``suspended``: sets ``execution_state='idle'``, preserves inputs as ``status='pending'``,
          propagates ``cancel_by = COALESCE(cancel_by, sessions.cancel_by)``, and clears
          ``ready_item_json=NULL``.
        - ``open``:
          * if a matching stop arrived after claim (``cancel_seq > batch.cancel_seq``), settles
            immediately as ``cancelled`` via ``_final_tx`` (preserving River delivery and
            ``sessions.cancel_by``);
          * otherwise returns inputs to ``status='pending'`` with
            ``cancel_by = COALESCE(cancel_by, sessions.cancel_by)`` and clears
            ``ready_item_json=NULL`` without wiping ``sessions.cancel_by``.
        """
        self.check_store(batch.target)
        ids = [ref.message_id for ref in batch.input_refs]
        now_sql = self._now_sql()
        undeliverable_out: list[tuple[TargetRef, str, Result, str, str]] = []
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, batch.target.session_id)
            if (
                state["execution_state"] != "running"
                or int(state["send_index"]) != batch.send_index
                or int(state["owner_epoch"]) != batch.owner_epoch
            ):
                raise SessionLeaseLost(batch.target.session_id)
            lifecycle = str(state["lifecycle"])
            is_driver = state["session_kind"] == "workflow_driver"
            id_ph = ",".join("?" for _ in ids)
            if lifecycle in ("closing", "closed"):
                eff_cb = state["cancel_by"] or "removed"
                eff_reason = state["cancel_reason"] or "cancelled"
                await tx.execute(
                    f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                    "cancel_by=COALESCE(cancel_by,?),delivered_at=NULL,delivered_seq=NULL,"
                    "delivered_send_index=NULL,error_code='cancelled',error_detail=?,"
                    f"completed_at={now_sql} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    f"AND item_id IN ({id_ph})",
                    (eff_cb, eff_reason, batch.target.session_id, batch.send_index, *ids),
                )
                changed = await tx.execute(
                    f"UPDATE {self.s} SET execution_state='idle',"
                    "send_index=CASE WHEN send_index>0 THEN send_index-1 ELSE 0 END,"
                    "next_seq=CASE WHEN ?=1 THEN next_seq WHEN next_seq>1 THEN next_seq-1 ELSE 1 END,"
                    "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                    f"ready_item_json=NULL,updated_at={now_sql} "
                    "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                    (int(is_driver), batch.target.session_id, batch.owner_epoch),
                )
                if changed != 1:
                    raise SessionLeaseLost(batch.target.session_id)
            elif lifecycle == "open" and int(state["cancel_seq"]) > batch.cancel_seq:
                await self._final_tx(
                    tx, batch, batch.input_refs, "cancelled", "", {}, None,
                    state["cancel_reason"] or "cancelled", "cancelled",
                    _undeliverable_out=undeliverable_out,
                )
            else:
                qrows = await tx.fetchall(
                    f"SELECT wake,content_bytes FROM {self.q} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    f"AND item_id IN ({id_ph})",
                    (batch.target.session_id, batch.send_index, *ids),
                )
                restored = await tx.execute(
                    f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                    "delivered_send_index=NULL,cancel_by=COALESCE(cancel_by,?) "
                    "WHERE session_id=? AND status='delivered' "
                    f"AND delivered_send_index=? AND item_id IN ({id_ph})",
                    (state["cancel_by"], batch.target.session_id, batch.send_index, *ids),
                )
                if restored != len(ids):
                    raise SessionLeaseLost(batch.target.session_id)
                w_add = sum(int(r["wake"]) for r in qrows)
                b_add = sum(int(r["content_bytes"]) for r in qrows)
                next_exec = "idle" if lifecycle == "suspended" else "queued"
                changed = await tx.execute(
                    f"UPDATE {self.s} SET execution_state=?,"
                    "send_index=CASE WHEN send_index>0 THEN send_index-1 ELSE 0 END,"
                    "next_seq=CASE WHEN ?=1 THEN next_seq WHEN next_seq>1 THEN next_seq-1 ELSE 1 END,"
                    "wake_pending_count=wake_pending_count+?,"
                    "pending_count=pending_count+?,"
                    "pending_bytes=pending_bytes+?,"
                    "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                    f"ready_item_json=NULL,updated_at={now_sql} "
                    "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                    (
                        next_exec, int(is_driver), w_add, len(ids), b_add,
                        batch.target.session_id, batch.owner_epoch,
                    ),
                )
                if changed != 1:
                    raise SessionLeaseLost(batch.target.session_id)
            await tx.execute(
                f"DELETE FROM {self.store.t.messages} WHERE session_id=? AND seq=? "
                "AND role='user' AND send_index=?",
                (batch.target.session_id, batch.message_seq, batch.send_index),
            )
        if _on_committed is not None:
            _on_committed()
        for c_target, p_sid, r_res, r_kind, r_reason in undeliverable_out:
            await self._fire_result_undeliverable(
                child_target=c_target,
                parent_session_id=p_sid,
                result=r_res,
                result_kind=r_kind,
                reason=r_reason,
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
        """Atomically drain ALL pending inbox items (normal + urgent) into the RUNNING send.

        Probe 4C fences:
        - Permits ``lifecycle IN ('open', 'suspended')`` (refuses ``closing``/``closed``).
        - Any pending row already marked ``cancel_by IS NOT NULL`` is settled as
          ``status='void', terminal_status='cancelled'`` and subtracted from counts without
          being delivered to the model.
        - Only ``cancel_by IS NULL`` rows are merged and delivered into the active turn.
        """
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
            if state["lifecycle"] not in ("open", "suspended"):
                return None
            rows = await tx.fetchall(
                "SELECT id,item_id,kind,priority,wake,content_json,content_bytes,"
                f"meta_json,cancel_by,reply_to_session_id,reply_result_kind,{now_sql} AS db_now "
                f"FROM {self.q} WHERE session_id=? AND status='pending' "
                "ORDER BY CASE priority WHEN 'urgent' THEN 0 ELSE 1 END,id",
                (batch.target.session_id,),
            )
            if not rows:
                return None
            database_now = int(rows[0]["db_now"])
            cancelled_rows = [r for r in rows if r["cancel_by"] is not None]
            active_rows = [r for r in rows if r["cancel_by"] is None]
            w_sub = sum(int(r["wake"]) for r in rows)
            p_sub = len(rows)
            b_sub = sum(int(r["content_bytes"]) for r in rows)
            if cancelled_rows:
                c_ids = [int(r["id"]) for r in cancelled_rows]
                c_ph = ",".join("?" for _ in c_ids)
                await tx.execute(
                    f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                    "error_code='cancelled',error_detail='cancelled',completed_at=? "
                    f"WHERE session_id=? AND status='pending' AND id IN ({c_ph})",
                    (database_now, batch.target.session_id, *c_ids),
                )
            if not active_rows:
                changed = await tx.execute(
                    f"UPDATE {self.s} SET "
                    "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                    "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                    "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END,"
                    f"updated_at={now_sql} "
                    "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                    (w_sub, w_sub, p_sub, p_sub, b_sub, b_sub, batch.target.session_id, batch.owner_epoch),
                )
                if changed != 1:
                    raise SessionLeaseLost(batch.target.session_id)
                return None
            refs = tuple(
                InputRef(batch.target.store_id, batch.target.session_id, row["item_id"])
                for row in active_rows
            )
            db_next_seq = int(state["next_seq"])
            seq = (
                max(db_next_seq, int(holding.next_seq))
                if (holding is not None and holding.next_seq is not None)
                else db_next_seq
            )
            merged = self.merge_batch(active_rows)
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
            ids = [int(row["id"]) for row in active_rows]
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

    async def consume_pending_inbox(self, batch: ClaimedBatch):
        """Alias for ``claim_follow_up`` (Design 217 §4.2 / Probe 4C)."""
        return await self.claim_follow_up(batch)

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
        recovered_by_hook: dict[str, str] = {}
        repair_cb = self._turn_repair_callback()
        if repair_cb is not None:
            dangling_payload = [
                {
                    **(dict(tc) if isinstance(tc, dict) else {"id": str(tc)}),
                    "_assistant_seq": assistant_seq,
                    "_assistant_round_index": int(round_index or 0),
                }
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            ]
            if dangling_payload:
                rec = await repair_cb(session_id, dangling_payload)
                if rec:
                    recovered_by_hook = {str(k): str(v) for k, v in rec.items()}
        database_now = await self._db_now(tx)
        appended = 0
        for tc in tool_calls:
            cid = str(tc.get("id") or "")
            if not cid:
                continue
            if cid in recovered_by_hook:
                content = recovered_by_hook[cid]
                meta: dict[str, Any] = {
                    "interrupted": True, "repaired": True,
                    "repair_source": "host_recovered", "_assistant_seq": assistant_seq,
                }
            else:
                content = "<aborted: 执行中断，结果未知>"
                meta = {
                    "interrupted": True, "repaired": True,
                    "repair_source": "sdk_auto_closed", "_assistant_seq": assistant_seq,
                }
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
        settle_start_input: bool = False, settle_chain_ids: tuple[str, ...] | None = None,
        drain_driver_node_results: bool = False, deliver_to_parent: bool = True,
        write_driver_message: bool = False, tx_hook=None,
        projection: PendingProjection | None = None,
        _on_committed=None,
    ) -> FinalBatch:
        if consumed is not None:
            consumed = tuple(consumed)
            if any(ref.session_id != batch.target.session_id for ref in consumed):
                raise RuntimeRequestError("invalid_options", "consumed input outside this session")
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise RuntimeRequestError("invalid_options", "terminal status")
        fold_conflict_out = [False]
        undeliverable_out: list[tuple[TargetRef, str, Result, str, str]] = []
        driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
        try:
            async with self.db.transaction() as tx:
                final_batch = await self._final_tx(
                    tx, batch, consumed, status, final_text, usage or {},
                    structured_output, error, error_code, result_seq, assistant_meta,
                    next_seq=next_seq, result_json=result_json,
                    settle_start_input=settle_start_input,
                    settle_chain_ids=settle_chain_ids,
                    drain_driver_node_results=drain_driver_node_results,
                    deliver_to_parent=deliver_to_parent,
                    write_driver_message=write_driver_message,
                    tx_hook=tx_hook,
                    projection=projection,
                    _fold_conflict_out=fold_conflict_out,
                    _undeliverable_out=undeliverable_out,
                    _driver_events_out=driver_events_out,
                )
        except (RuntimeRequestError, SessionLeaseLost):
            raise
        except (Exception, asyncio.CancelledError) as exc:
            recovered = await self._final_after_unknown(batch, consumed)
            if recovered is not None:
                if _on_committed is not None:
                    _on_committed()
                return recovered
            raise StoreError(consumed[0] if consumed else batch.input_refs[0]) from exc

        if _on_committed is not None:
            _on_committed()

        for drv_sid, item_id, payload in driver_events_out:
            self._push_driver_leaf_result(drv_sid, item_id, payload)

        # Post-commit RESULT_UNDELIVERABLE hook (Design 217 §4.4 / §5.4)
        for c_target, p_sid, r_res, r_kind, r_reason in undeliverable_out:
            await self._fire_result_undeliverable(
                child_target=c_target,
                parent_session_id=p_sid,
                result=r_res,
                result_kind=r_kind,
                reason=r_reason,
            )

        # Rare conflict recovery (0 SQL on normal path): Final preserved invalid=True without
        # failing business commit or misreporting SessionLeaseLost; rebuild exact stats unlocked.
        if (
            fold_conflict_out[0]
            and projection is not None
            and projection.projector is not None
            and projection.policy is not None
        ):
            await self._rebuild_fold_stats_after_conflict(
                batch.target.session_id, projection.projector, projection.policy,
            )
        return final_batch

    async def _rebuild_fold_stats_after_conflict(self, sid: str, projector: Any, policy: Any) -> None:
        """Unlocked snapshot read + revision-fenced rebuild after a concurrent fold_stats_json conflict.

        Never raises into the committed business result: if a second concurrent mutation races or
        fails here, ``sessions.fold_stats_json`` remains ``invalid=True`` and the next send rebuilds.
        """
        from power_loop.runtime.store import projection_fold as pf

        try:
            snapshot = await pf.read_snapshot(self.store, sid)
            if snapshot is not None:
                await pf._rebuild(self.store, sid, projector, policy, snapshot)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "post-final fold_stats_json rebuild failed for %s (left invalid for next send)",
                sid, exc_info=True,
            )

    async def _insert_project_messages_tx(
        self, tx, sid: str, send_index: int, projection: PendingProjection, created_at: int,
    ) -> None:
        """Write double-row ('user' + 'project') projections in ONE multi-value INSERT (Design 217 §5.1)."""
        if not projection.rows:
            return
        from power_loop.runtime.budget import estimate_text_tokens
        value_groups: list[str] = []
        params: list[Any] = []
        for kind, content, rendered_text in projection.rows:
            content_json = (
                content if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            )
            tok = (
                estimate_text_tokens(rendered_text)
                if rendered_text
                else max(1, len(content_json) // 4)
            )
            value_groups.append("(?,?,?,?,?,?,?,NULL,NULL,?,?,?,?)")
            params.extend((
                sid, int(send_index), str(kind), content_json, rendered_text,
                projection.source_seq_lo, projection.source_seq_hi,
                int(projection.projector_version), int(tok), int(created_at), int(send_index),
            ))
        cols = (
            "session_id,send_index,kind,content_json,rendered_text,"
            "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
            "projector_version,token_estimate,created_at,completed_through_send_index"
        )
        if self._dialect == "mysql":
            upsert = (
                " AS _new ON DUPLICATE KEY UPDATE "
                "content_json=_new.content_json,rendered_text=_new.rendered_text,"
                "source_seq_lo=_new.source_seq_lo,source_seq_hi=_new.source_seq_hi,"
                "projector_version=_new.projector_version,token_estimate=_new.token_estimate,"
                "created_at=_new.created_at,"
                "completed_through_send_index=_new.completed_through_send_index"
            )
        else:
            upsert = (
                " ON CONFLICT(session_id,send_index,kind) DO UPDATE SET "
                "content_json=excluded.content_json,rendered_text=excluded.rendered_text,"
                "source_seq_lo=excluded.source_seq_lo,source_seq_hi=excluded.source_seq_hi,"
                "projector_version=excluded.projector_version,token_estimate=excluded.token_estimate,"
                "created_at=excluded.created_at,"
                "completed_through_send_index=excluded.completed_through_send_index"
            )
        await tx.execute(
            f"INSERT INTO {self.store.t.project_messages} ({cols}) "
            f"VALUES {','.join(value_groups)}{upsert}",
            tuple(params),
        )

    @staticmethod
    def _human_chat_notice_id(sid: str, send_index: int) -> str:
        raw_notice_id = f"sys_chat_{sid}_s{send_index}"
        if len(raw_notice_id) <= 64:
            return raw_notice_id
        suffix = f"_s{send_index}"
        digest_len = max(1, 64 - len("sys_chat_") - len(suffix))
        return f"sys_chat_{sha256_canonical([sid])[:digest_len]}{suffix}"

    async def _drain_pending_driver_node_results_tx(
        self, tx, sid: str, send_index: int,
    ) -> tuple[int, int, int]:
        """Drain pending workflow_node_result rows AFTER locking ``sessions(sid)``.

        Caller MUST already hold the exclusive row lock on ``sessions(sid)`` so lock order
        stays strictly ``sessions(sid) -> follow_up_queue(sid)`` (never AB-BA with ``recover``,
        ``set_lifecycle('closed')``, or ``_river_deliver_tx``). On MySQL, updates ONLY the exact
        locked ``id`` batch summed into ``(w_drain, p_drain, b_drain)``.
        """
        now_sql = self._now_sql()
        if self._dialect == "mysql":
            rows = await tx.fetchall(
                f"SELECT id,wake,content_bytes FROM {self.q} "
                "WHERE session_id=? AND status='pending' AND kind='workflow_node_result' "
                "ORDER BY id ASC FOR UPDATE",
                (sid,),
            )
            if not rows:
                return 0, 0, 0
            ids = [int(r["id"]) for r in rows]
            w_drain = sum(int(r["wake"]) for r in rows)
            p_drain = len(rows)
            b_drain = sum(int(r["content_bytes"]) for r in rows)
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                ph = ",".join("?" for _ in chunk)
                await tx.execute(
                    f"UPDATE {self.q} SET status='delivered',delivered_send_index=?,"
                    f"delivered_at={now_sql},terminal_status='completed',"
                    f"result_send_index=?,completed_at={now_sql} "
                    f"WHERE session_id=? AND status='pending' AND id IN ({ph})",
                    (send_index, send_index, sid, *chunk),
                )
            return w_drain, p_drain, b_drain
        drained_rows = await tx.fetchall(
            f"UPDATE {self.q} SET status='delivered',delivered_send_index=?,"
            f"delivered_at={now_sql},terminal_status='completed',"
            f"result_send_index=?,completed_at={now_sql} "
            "WHERE session_id=? AND status='pending' AND kind='workflow_node_result' "
            "RETURNING wake,content_bytes",
            (send_index, send_index, sid),
        )
        if not drained_rows:
            return 0, 0, 0
        return (
            sum(int(r["wake"]) for r in drained_rows),
            len(drained_rows),
            sum(int(r["content_bytes"]) for r in drained_rows),
        )

    async def _final_tx(
        self, tx, batch: ClaimedBatch, consumed, status, final_text, usage, structured_output,
        error, error_code, result_seq=None, assistant_meta=None, *, next_seq=None,
        result_json=None, settle_start_input=False, settle_chain_ids: tuple[str, ...] | None = None,
        drain_driver_node_results: bool = False, deliver_to_parent: bool = True,
        write_driver_message: bool = False, tx_hook=None,
        projection: PendingProjection | None = None,
        _fold_conflict_out: list[bool] | None = None,
        _undeliverable_out: list[tuple[TargetRef, str, Result, str, str]] | None = None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> FinalBatch:
        sid = batch.target.session_id
        now_sql = self._now_sql()
        consumed_refs = tuple(batch.input_refs) if consumed is None else tuple(consumed)
        unconsumed_ids = [ref.message_id for ref in batch.input_refs if ref not in consumed_refs]
        has_unconsumed = bool(unconsumed_ids)
        is_driver = batch.descriptor.kind == "workflow_driver"

        # Compute effective next_seq before releasing the session row (so 1 UPDATE suffices!).
        # On abnormal/untracked final (exception, cancel, recovery) of non-driver sessions, sync
        # with MAX(seq)+1 so fenced appends made before failure never collide on the next send.
        # Drivers never do mid-turn fenced appends, so they advance from sessions.next_seq (0 MAX(seq)).
        if not is_driver and (next_seq is None or status != "completed"):
            max_row = await tx.fetchone(
                f"SELECT COALESCE(MAX(seq),0) AS max_seq FROM {self.store.t.messages} WHERE session_id=?",
                (sid,),
            )
            if max_row is not None:
                next_seq = max(int(next_seq or 0), int(max_row["max_seq"]) + 1)
        write_message = (
            bool(write_driver_message) if is_driver else (result_seq is None and status == "completed")
        )
        effective_next_seq = max(
            int(next_seq or 0),
            (int(result_seq) + 1) if result_seq is not None else 0,
            batch.message_seq + (0 if is_driver else 1),
            1,
        )
        keep_pending_tool = status == "interrupted"
        proj_fold_json = (
            projection.fold_stats_json
            if (projection is not None and status == "completed" and projection.fold_stats_json is not None)
            else None
        )
        expected_fold_json = (
            projection.expected_fold_stats_json
            if (projection is not None and proj_fold_json is not None)
            else None
        )
        # Dialect-exact conditional write on sessions.fold_stats_json (covers both NULL and non-NULL
        # snapshots). Never placed in WHERE so a concurrent projection mutation cannot be misreported
        # as SessionLeaseLost or block the business commit.
        mysql_fold_conflict_bit = ""
        mysql_fold_conflict_params: tuple[Any, ...] = ()
        if proj_fold_json is None:
            fold_set_clause = ""
            fold_set_params: tuple[Any, ...] = ()
        elif expected_fold_json is None:
            fold_set_clause = (
                "fold_stats_json=CASE WHEN lifecycle IN ('open','suspended','closed') AND cancel_seq=? "
                "AND fold_stats_json IS NULL THEN ? ELSE fold_stats_json END,"
            )
            fold_set_params = (batch.cancel_seq, proj_fold_json)
            if self._dialect == "mysql":
                mysql_fold_conflict_bit = (
                    "|(CASE WHEN lifecycle IN ('open','suspended','closed') AND cancel_seq=? "
                    "AND fold_stats_json IS NOT NULL THEN 16 ELSE 0 END)"
                )
                mysql_fold_conflict_params = (batch.cancel_seq,)
        else:
            if self._dialect == "mysql":
                fold_set_clause = (
                    "fold_stats_json=CASE WHEN lifecycle IN ('open','suspended','closed') AND cancel_seq=? "
                    "AND CAST(fold_stats_json AS BINARY)=CAST(? AS BINARY) "
                    "THEN ? ELSE fold_stats_json END,"
                )
                mysql_fold_conflict_bit = (
                    "|(CASE WHEN lifecycle IN ('open','suspended','closed') AND cancel_seq=? "
                    "AND (fold_stats_json IS NULL "
                    "OR CAST(fold_stats_json AS BINARY)<>CAST(? AS BINARY)) THEN 16 ELSE 0 END)"
                )
                mysql_fold_conflict_params = (batch.cancel_seq, expected_fold_json)
            else:
                fold_set_clause = (
                    "fold_stats_json=CASE WHEN lifecycle IN ('open','suspended','closed') AND cancel_seq=? "
                    "AND fold_stats_json=? THEN ? ELSE fold_stats_json END,"
                )
            fold_set_params = (batch.cancel_seq, expected_fold_json, proj_fold_json)

        # Step 1 (Deep-first lock order): Fenced UPDATE on child/driver session FIRST (1 control SQL!).
        # Acquires the exclusive row lock on sessions(sid) BEFORE touching follow_up_queue(sid) or parent.
        post_fold_stats_json: str | None = None
        route_changed = False
        has_pending = False
        live_root_session_id = batch.root_session_id
        if self._dialect == "mysql":
            # 8-bit LAST_INSERT_ID packing in 1st SET slot (evaluated before fold_stats_json SET):
            # bit 0 (1):   has_more_wake  | bit 1 (2):   subtree_raced | bit 2 (4):  cancel_raced
            # bit 3 (8):   closed_raced   | bit 4 (16):  fold_conflict | bit 5 (32): route_changed
            # bit 6 (64):  has_pending (drain_driver_node_results)     | bit 7 (128): root_changed
            mysql_route_changed_bit = ""
            mysql_route_changed_params: tuple[Any, ...] = ()
            if batch.result_route is not None:
                mysql_route_changed_bit = (
                    "|(CASE WHEN NOT (result_route_json<=>?) THEN 32 ELSE 0 END)"
                )
                mysql_route_changed_params = (canonical(batch.result_route),)
            mysql_has_pending_bit = (
                "|(CASE WHEN pending_count>0 THEN 64 ELSE 0 END)"
                if drain_driver_node_results else ""
            )
            mysql_root_changed_bit = ""
            mysql_root_changed_params: tuple[Any, ...] = ()
            if batch.root_session_id is not None and batch.root_session_id != sid:
                mysql_root_changed_bit = (
                    "|(CASE WHEN NOT (root_session_id<=>?) THEN 128 ELSE 0 END)"
                )
                mysql_root_changed_params = (batch.root_session_id,)
            rc, packed = await tx.execute_packed(
                f"UPDATE {self.s} SET "
                "owner_epoch=(LAST_INSERT_ID("
                "(owner_epoch<<8)"
                "|(CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) THEN 1 ELSE 0 END)"
                "|(CASE WHEN cancel_by='subtree' AND cancel_seq>? THEN 2 ELSE 0 END)"
                "|(CASE WHEN cancel_seq>? THEN 4 ELSE 0 END)"
                "|(CASE WHEN lifecycle='closing' OR (lifecycle='closed' AND cancel_seq>?) THEN 8 ELSE 0 END)"
                f"{mysql_fold_conflict_bit}"
                f"{mysql_route_changed_bit}"
                f"{mysql_has_pending_bit}"
                f"{mysql_root_changed_bit}"
                ")>>8),"
                "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) "
                "THEN 'queued' ELSE 'idle' END,"
                "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                "next_seq=CASE WHEN lifecycle IN ('open','suspended') OR (lifecycle='closed' AND cancel_seq=?) THEN "
                "(CASE WHEN ?>next_seq THEN ? ELSE next_seq END)+? "
                "ELSE (CASE WHEN ?>next_seq THEN ? ELSE next_seq END) END,"
                "pending_tool_seq=CASE WHEN ?=1 THEN pending_tool_seq ELSE NULL END,"
                "ready_item_json=NULL,"
                f"{fold_set_clause}"
                "cancel_by=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_by END,"
                "cancel_reason=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_reason END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND owner_epoch=? AND execution_state='running'",
                (
                    int(has_unconsumed), batch.cancel_seq, batch.cancel_seq, batch.cancel_seq,
                    *mysql_fold_conflict_params,
                    *mysql_route_changed_params,
                    *mysql_root_changed_params,
                    int(has_unconsumed),
                    batch.cancel_seq,
                    effective_next_seq, effective_next_seq, int(write_message),
                    effective_next_seq, effective_next_seq,
                    int(keep_pending_tool), *fold_set_params, batch.cancel_seq, batch.cancel_seq,
                    sid, batch.owner_epoch,
                ),
            )
            if rc == 0:
                raise SessionLeaseLost(sid)
            has_more_wake = bool(packed & 1)
            subtree_raced = bool(packed & 2)
            cancel_raced = bool(packed & 4)
            closed_raced = bool(packed & 8)
            fold_conflict = bool(packed & 16)
            route_changed = bool(packed & 32)
            has_pending = bool(packed & 64)
            root_changed = bool(packed & 128)
            next_state = "queued" if has_more_wake else "idle"
            cancel_by_db = "subtree" if subtree_raced else None
            cancel_reason_db = None
            post_next_seq = effective_next_seq + int(write_message)
            database_now = None
            live_route_json = None
            if (
                closed_raced or cancel_raced or write_message
                or (route_changed and batch.result_route is not None)
                or root_changed
            ):
                diag = await tx.fetchone(
                    f"SELECT next_seq,cancel_by,cancel_reason,updated_at,"
                    f"result_route_json,root_session_id "
                    f"FROM {self.s} WHERE session_id=?",
                    (sid,),
                )
                if diag is not None:
                    post_next_seq = int(diag["next_seq"])
                    cancel_by_db = diag["cancel_by"]
                    cancel_reason_db = diag["cancel_reason"]
                    database_now = int(diag["updated_at"])
                    live_route_json = diag["result_route_json"]
                    if diag["root_session_id"]:
                        live_root_session_id = str(diag["root_session_id"])
        else:
            srow = await tx.fetchone(
                f"UPDATE {self.s} SET "
                "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>0 OR ?=1) "
                "THEN 'queued' ELSE 'idle' END,"
                "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
                "next_seq=CASE WHEN lifecycle IN ('open','suspended') OR (lifecycle='closed' AND cancel_seq=?) THEN "
                "(CASE WHEN ?>next_seq THEN ? ELSE next_seq END)+? "
                "ELSE (CASE WHEN ?>next_seq THEN ? ELSE next_seq END) END,"
                "pending_tool_seq=CASE WHEN ?=1 THEN pending_tool_seq ELSE NULL END,"
                "ready_item_json=NULL,"
                f"{fold_set_clause}"
                "cancel_by=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_by END,"
                "cancel_reason=CASE WHEN lifecycle IN ('open','suspended') AND cancel_seq=? THEN NULL ELSE cancel_reason END,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND owner_epoch=? AND execution_state='running' "
                "RETURNING execution_state,lifecycle,wake_pending_count,cancel_seq,"
                "cancel_by,cancel_reason,next_seq,fold_stats_json,updated_at,"
                "result_route_json,root_session_id,pending_count",
                (
                    int(has_unconsumed),
                    batch.cancel_seq,
                    effective_next_seq, effective_next_seq, int(write_message),
                    effective_next_seq, effective_next_seq,
                    int(keep_pending_tool), *fold_set_params, batch.cancel_seq, batch.cancel_seq,
                    sid, batch.owner_epoch,
                ),
            )
            if srow is None:
                raise SessionLeaseLost(sid)
            next_state = str(srow["execution_state"])
            cancel_raced = int(srow["cancel_seq"]) > batch.cancel_seq
            closed_raced = (
                srow["lifecycle"] == "closing"
                or (srow["lifecycle"] == "closed" and cancel_raced)
            )
            subtree_raced = cancel_raced and srow["cancel_by"] == "subtree"
            cancel_by_db = srow["cancel_by"]
            cancel_reason_db = srow["cancel_reason"]
            post_next_seq = int(srow["next_seq"])
            post_fold_stats_json = srow["fold_stats_json"]
            database_now = int(srow["updated_at"])
            live_route_json = srow["result_route_json"]
            if srow["root_session_id"]:
                live_root_session_id = str(srow["root_session_id"])
            has_pending = int(srow["pending_count"]) > 0
            fold_conflict = (
                proj_fold_json is not None
                and not closed_raced
                and not cancel_raced
                and post_fold_stats_json != proj_fold_json
            )

        # Step 1.5 (Plan B resident workflow_driver): drain pending workflow_node_result rows AFTER
        # acquiring the owned sessions(sid) row lock, deducting ONLY the exact drained batch.
        if drain_driver_node_results and not closed_raced and has_pending:
            w_drain, p_drain, b_drain = await self._drain_pending_driver_node_results_tx(
                tx, sid, batch.send_index,
            )
            if p_drain > 0:
                if self._dialect == "mysql":
                    _, d_packed = await tx.execute_packed(
                        f"UPDATE {self.s} SET "
                        "owner_epoch=(LAST_INSERT_ID("
                        "(owner_epoch<<1)"
                        "|(CASE WHEN lifecycle='open' AND (wake_pending_count>? OR ?=1) THEN 1 ELSE 0 END)"
                        ")>>1),"
                        "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>? OR ?=1) "
                        "THEN 'queued' ELSE 'idle' END,"
                        "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                        "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                        "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END "
                        "WHERE session_id=?",
                        (
                            w_drain, int(has_unconsumed),
                            w_drain, int(has_unconsumed),
                            w_drain, w_drain, p_drain, p_drain, b_drain, b_drain,
                            sid,
                        ),
                    )
                    next_state = "queued" if bool(d_packed & 1) else "idle"
                else:
                    drow = await tx.fetchone(
                        f"UPDATE {self.s} SET "
                        "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count>? OR ?=1) "
                        "THEN 'queued' ELSE 'idle' END,"
                        "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                        "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                        "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END "
                        "WHERE session_id=? "
                        "RETURNING execution_state",
                        (
                            w_drain, int(has_unconsumed),
                            w_drain, w_drain, p_drain, p_drain, b_drain, b_drain,
                            sid,
                        ),
                    )
                    if drow is not None:
                        next_state = str(drow["execution_state"])

        effective_status = status
        effective_cancel_by = batch.cancel_by or cancel_by_db
        if subtree_raced:
            effective_cancel_by = "subtree"
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
            effective_cancel_by = effective_cancel_by or "user"
            effective_error = cancel_reason_db or "cancelled"
            effective_error_code = "cancelled"
            if not write_driver_message:
                write_message = False
        if effective_status == "cancelled" and effective_cancel_by is None:
            effective_cancel_by = "user"

        if write_message:
            result_seq = post_next_seq - 1

        if tx_hook is not None:
            hook_override = await tx_hook(
                tx,
                {
                    "effective_status": effective_status,
                    "effective_cancel_by": effective_cancel_by,
                    "effective_error": effective_error,
                    "effective_error_code": effective_error_code,
                    "result_seq": result_seq,
                    "database_now": database_now,
                },
            )
            if isinstance(hook_override, dict):
                if "structured_output" in hook_override:
                    structured_output = hook_override["structured_output"]
                if "result_json" in hook_override:
                    result_json = hook_override["result_json"]
                if "final_text" in hook_override:
                    final_text = hook_override["final_text"]

        # Resolve consumed inputs if caller passed consumed=None and we need exact list for abnormal settle or River.
        delivered_routes: dict[str, tuple[str | None, str | None]] = {}
        if consumed is None and (effective_status != "completed" or batch.result_route is not None or batch.reply_to_session_id is not None):
            delivered = await tx.fetchall(
                f"SELECT item_id,reply_to_session_id,reply_result_kind FROM {self.q} "
                "WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id ASC",
                (sid, batch.send_index),
            )
            if delivered:
                consumed_refs = tuple(
                    InputRef(batch.target.store_id, sid, row["item_id"]) for row in delivered
                )
                for row in delivered:
                    delivered_routes[str(row["item_id"])] = (
                        row["reply_to_session_id"],
                        row["reply_result_kind"],
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
        if route is not None and live_route_json:
            parsed_live = json.loads(live_route_json)
            if isinstance(parsed_live, dict) and isinstance(parsed_live.get("target"), dict):
                if (
                    is_driver
                    and route.get("input_ref") is not None
                    and route.get("input_ref") != parsed_live.get("input_ref")
                ):
                    route = {**parsed_live, "input_ref": route["input_ref"]}
                else:
                    route = parsed_live
        # Design 217 §5.3: human direct chat with a finished workflow leaf (reply_to_session_id IS NULL)
        # replies ONLY to human + delivers system_notice(wake=0) to root_session_id (NEVER wakes Driver!).
        # Judge against the actually-consumed input subset so partial consume of only a human-chat
        # input does not inherit reply_to_session_id from an unconsumed input returned to pending.
        is_human_direct_leaf_chat = False
        if route is not None and route.get("result_kind") == "workflow_node_result":
            consumed_has_wf_route = False
            consumed_has_any_route = False
            if not has_unconsumed:
                consumed_has_wf_route = (
                    (
                        batch.reply_to_session_id is not None
                        and batch.reply_result_kind == "workflow_node_result"
                    )
                    or any(
                        to is not None and rk == "workflow_node_result"
                        for to, rk in delivered_routes.values()
                    )
                )
                consumed_has_any_route = (
                    consumed_has_wf_route
                    or batch.reply_to_session_id is not None
                    or any(to is not None for to, _ in delivered_routes.values())
                )
            elif consumed_refs:
                consumed_id_set_chk = {ref.message_id for ref in consumed_refs}
                if (
                    batch.reply_to_session_id is not None
                    and batch.reply_to_item_id is not None
                    and batch.reply_to_item_id in consumed_id_set_chk
                ):
                    consumed_has_any_route = True
                    consumed_has_wf_route = batch.reply_result_kind == "workflow_node_result"
                if not consumed_has_wf_route:
                    if delivered_routes:
                        for mid in consumed_id_set_chk:
                            to_sid, rk_val = delivered_routes.get(mid, (None, None))
                            if to_sid is not None:
                                consumed_has_any_route = True
                                if rk_val == "workflow_node_result":
                                    consumed_has_wf_route = True
                                    break
                    else:
                        c_ids_chk = [ref.message_id for ref in consumed_refs]
                        c_ph_chk = ",".join("?" for _ in c_ids_chk)
                        q_routes = await tx.fetchall(
                            f"SELECT reply_to_session_id,reply_result_kind FROM {self.q} "
                            f"WHERE session_id=? AND item_id IN ({c_ph_chk}) "
                            "AND reply_to_session_id IS NOT NULL",
                            (sid, *c_ids_chk),
                        )
                        if q_routes:
                            consumed_has_any_route = True
                            consumed_has_wf_route = any(
                                r["reply_result_kind"] == "workflow_node_result" for r in q_routes
                            )
            if not consumed_has_wf_route:
                route = None
                is_human_direct_leaf_chat = not consumed_has_any_route

        deliver_parent_sid = None
        deliver_res_kind = None
        deliver_start_ref = None
        deliver_source_ref = consumed_refs[0] if consumed_refs else None
        verify_source_item_id: str | None = None
        if deliver_to_parent and not closed_raced and consumed_refs:
            if route is not None:
                deliver_parent_sid = route["target"]["session_id"]
                deliver_res_kind = route["result_kind"]
                deliver_start_ref = InputRef(**route["input_ref"])
                if is_driver:
                    deliver_source_ref = deliver_start_ref
            elif not is_human_direct_leaf_chat and batch.reply_to_session_id is not None:
                consumed_id_set = {ref.message_id for ref in consumed_refs}
                reply_item_id = batch.reply_to_item_id
                effective_reply_to: str | None = batch.reply_to_session_id
                effective_reply_rk: str | None = batch.reply_result_kind
                if reply_item_id is not None and reply_item_id not in consumed_id_set:
                    c_ids = [ref.message_id for ref in consumed_refs]
                    c_ph = ",".join("?" for _ in c_ids)
                    q_alt = await tx.fetchone(
                        f"SELECT item_id,reply_to_session_id,reply_result_kind FROM {self.q} "
                        f"WHERE session_id=? AND item_id IN ({c_ph}) AND reply_to_session_id IS NOT NULL "
                        "ORDER BY id ASC LIMIT 1",
                        (sid, *c_ids),
                    )
                    if q_alt is not None:
                        reply_item_id = str(q_alt["item_id"])
                        effective_reply_to = str(q_alt["reply_to_session_id"])
                        effective_reply_rk = q_alt["reply_result_kind"]
                    else:
                        reply_item_id = None
                        effective_reply_to = None
                else:
                    if reply_item_id is None:
                        reply_item_id = consumed_refs[0].message_id
                    if reply_item_id in delivered_routes:
                        live_to, live_rk = delivered_routes[reply_item_id]
                        if live_to is not None:
                            effective_reply_to = str(live_to)
                            effective_reply_rk = live_rk or effective_reply_rk
                    else:
                        verify_source_item_id = reply_item_id
                if effective_reply_to is not None and reply_item_id is not None:
                    deliver_parent_sid = effective_reply_to
                    deliver_res_kind = effective_reply_rk or "agent_result"
                    deliver_source_ref = InputRef(batch.target.store_id, sid, reply_item_id)
                    deliver_start_ref = (
                        InputRef(batch.target.store_id, sid, batch.start_input_message_id)
                        if batch.start_input_message_id else deliver_source_ref
                    )

        if deliver_parent_sid is not None and deliver_res_kind is not None:
            child_wake = 0 if (effective_cancel_by == "subtree" or subtree_raced) else 1
            route_final_text = final_text
            route_structured = structured_output
            if effective_status == "cancelled" and result_seq is None and not write_message:
                route_final_text = ""
                route_structured = structured_output if result_json is not None else None
            route_result = Result(
                deliver_source_ref or consumed_refs[0], batch.target, effective_status,
                batch.send_index, result_seq,
                route_final_text, usage, route_structured, effective_error, effective_error_code,
                cancel_by=effective_cancel_by if effective_status == "cancelled" else None,
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
                verify_source_item_id=verify_source_item_id,
                _driver_events_out=_driver_events_out,
            )
            if delivery_error is not None and _undeliverable_out is not None:
                actual_parent_sid = (
                    result_ref.session_id if result_ref is not None else deliver_parent_sid
                )
                _undeliverable_out.append((
                    batch.target,
                    actual_parent_sid,
                    replace(route_result, result_delivery_error=delivery_error),
                    deliver_res_kind,
                    delivery_error,
                ))
        elif is_human_direct_leaf_chat and not closed_raced:
            if live_root_session_id is None:
                rrow = await tx.fetchone(
                    f"SELECT root_session_id FROM {self.s} WHERE session_id=?", (sid,),
                )
                if rrow is not None and rrow["root_session_id"]:
                    live_root_session_id = str(rrow["root_session_id"])
            if live_root_session_id and live_root_session_id != sid:
                notice_id = self._human_chat_notice_id(sid, batch.send_index)
                notice_body = {
                    "kind": "system_notice",
                    "event": "human_child_interaction",
                    "session_id": sid,
                    "send_index": batch.send_index,
                    "text": f"[system_notice · 用户直接与子会话 {sid} 进行了第 {batch.send_index} 轮对话]",
                }
                await self._deliver_system_notice_tx(
                    tx,
                    target_session_id=live_root_session_id,
                    tenant_id=batch.tenant_id,
                    principal_id=batch.principal_id,
                    item_id=notice_id,
                    body=notice_body,
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

        # Step 3.5: Single multi-value INSERT for double-row ('user' + 'project') projections (0 extra tx!).
        if (
            projection is not None
            and projection.rows
            and effective_status == "completed"
            and not closed_raced
            and not cancel_raced
        ):
            proj_ts = database_now if database_now is not None else now_ms()
            await self._insert_project_messages_tx(
                tx, sid, batch.send_index, projection, proj_ts,
            )
            if fold_conflict:
                from power_loop.runtime.store import projection_fold as pf
                if self._dialect == "mysql":
                    cur_state = await pf.read_state(self.store, tx, sid)
                else:
                    cur_state = pf.parse_state(post_fold_stats_json)
                if not (isinstance(cur_state, dict) and cur_state.get("invalid") is True):
                    await pf.invalidate(self.store, tx, sid, state=cur_state)
                if _fold_conflict_out is not None:
                    _fold_conflict_out[0] = True

        # Step 4: 0 UPDATE follow_up_queue on normal completion!
        # Only update follow_up_queue on abnormal status, delivery_error, or explicit settle_start_input.
        res_json_str = canonical(result_json) if result_json is not None else None
        usage_plain = plain(usage) if usage else {}
        persist_usage_in_meta = bool(usage_plain and result_seq is None and not write_message)
        delivered_source_id = (
            (deliver_source_ref or consumed_refs[0]).message_id
            if (deliver_parent_sid is not None and deliver_res_kind is not None and consumed_refs)
            else None
        )
        persist_source_in_meta = bool(
            delivered_source_id is not None
            and not is_driver
            and (delivery_error is not None or len(consumed_refs) > 1)
        )
        target_chain_ids: tuple[str, ...] | None = settle_chain_ids
        if target_chain_ids is None and settle_start_input and batch.start_input_message_id is not None:
            target_chain_ids = (batch.start_input_message_id,)
        if target_chain_ids:
            ph = ",".join("?" for _ in target_chain_ids)
            await tx.execute(
                f"UPDATE {self.q} SET status='delivered',terminal_status=?,cancel_by=?,result_seq=?,"
                "result_send_index=?,result_json=?,error_code=?,error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} "
                f"WHERE session_id=? AND item_id IN ({ph}) "
                "AND status IN ('delivered','waiting_continuation') AND terminal_status IS NULL",
                (
                    effective_status,
                    effective_cancel_by if effective_status == "cancelled" else None,
                    result_seq, batch.send_index,
                    res_json_str, effective_error_code, effective_error, delivery_error,
                    sid, *target_chain_ids,
                ),
            )
            if persist_usage_in_meta:
                mrows = await tx.fetchall(
                    f"SELECT item_id,meta_json FROM {self.q} WHERE session_id=? AND item_id IN ({ph})",
                    (sid, *target_chain_ids),
                )
                for mrow in mrows:
                    qmeta = json.loads(mrow["meta_json"] or "{}")
                    qmeta["result_usage"] = usage_plain
                    await tx.execute(
                        f"UPDATE {self.q} SET meta_json=? WHERE session_id=? AND item_id=?",
                        (canonical(qmeta), sid, mrow["item_id"]),
                    )
        elif (
            effective_status != "completed"
            or delivery_error is not None
            or effective_cancel_by is not None
            or result_json is not None
        ):
            consumed_ids = [ref.message_id for ref in consumed_refs]
            if consumed_ids:
                c_ph = ",".join("?" for _ in consumed_ids)
                meta_clause = ""
                meta_params: tuple[Any, ...] = ()
                per_item_metas: list[tuple[str, str]] = []
                if persist_usage_in_meta or persist_source_in_meta:
                    mrows = await tx.fetchall(
                        f"SELECT item_id,meta_json FROM {self.q} WHERE session_id=? AND item_id IN ({c_ph})",
                        (sid, *consumed_ids),
                    )
                    updated_metas: dict[str, str] = {}
                    for mrow in mrows:
                        qmeta = json.loads(mrow["meta_json"] or "{}")
                        if persist_usage_in_meta:
                            qmeta["result_usage"] = usage_plain
                        if persist_source_in_meta:
                            qmeta["result_source_item_id"] = delivered_source_id
                        updated_metas[str(mrow["item_id"])] = canonical(qmeta)
                    uniq_metas = set(updated_metas.values())
                    if len(uniq_metas) == 1:
                        meta_clause = ",meta_json=?"
                        meta_params = (next(iter(uniq_metas)),)
                    else:
                        per_item_metas = list(updated_metas.items())
                changed = await tx.execute(
                    f"UPDATE {self.q} SET terminal_status=?,cancel_by=?,result_seq=?,"
                    "result_send_index=?,result_json=?,error_code=?,error_detail=?,"
                    f"result_delivery_error=?,completed_at={now_sql}{meta_clause} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    f"AND item_id IN ({c_ph})",
                    (
                        effective_status,
                        effective_cancel_by if effective_status == "cancelled" else None,
                        result_seq, batch.send_index,
                        res_json_str, effective_error_code, effective_error, delivery_error,
                        *meta_params,
                        sid, batch.send_index, *consumed_ids,
                    ),
                )
                if changed != len(consumed_ids):
                    raise SessionLeaseLost(sid)
                for mid, mjson in per_item_metas:
                    await tx.execute(
                        f"UPDATE {self.q} SET meta_json=? WHERE session_id=? AND item_id=?",
                        (mjson, sid, mid),
                    )

        settle_refs = (
            tuple(InputRef(batch.target.store_id, sid, mid) for mid in target_chain_ids)
            if target_chain_ids else consumed_refs
        )
        results = tuple(
            Result(
                ref, batch.target, effective_status, batch.send_index, result_seq, final_text,
                usage, structured_output, effective_error, effective_error_code, delivery_error,
                cancel_by=effective_cancel_by if effective_status == "cancelled" else None,
            )
            for ref in settle_refs
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
        verify_source_item_id: str | None = None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> tuple[InputRef | None, str | None]:
        """Same-tx non-cascading River delivery (parent lifecycle IN ('open','suspended'))."""
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
            "send_index": result.send_index,
            "cancel_by": result.cancel_by,
        }
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        digest = self.request_hash(parent_target, body, "normal")
        now_sql = self._now_sql()
        source_route_sql = ""
        source_route_params: tuple[Any, ...] = ()
        if verify_source_item_id is not None:
            source_route_sql = (
                f" AND EXISTS (SELECT 1 FROM {self.q} "
                "WHERE session_id=? AND item_id=? AND reply_to_session_id=?)"
            )
            source_route_params = (
                child_target.session_id, verify_source_item_id, parent_session_id,
            )

        if self._dialect == "mysql":
            if verify_source_item_id is not None:
                # Materialize the 1-row source-route check via uq_inbox_item FIRST (LIMIT 1
                # disables derived_merge constant propagation onto _p.session_id, keeping _p
                # eq_ref instead of const). When an in-flight rebind retargeted the child's
                # route away from parent_session_id, _src yields 0 rows and MySQL never
                # probes or locks the stale old parent row (0 AB-BA deadlocks!).
                p_rc, p_packed = await tx.execute_packed(
                    "UPDATE ("
                    f"SELECT reply_to_session_id AS _target_sid FROM {self.q} "
                    "WHERE session_id=? AND item_id=? AND reply_to_session_id=? LIMIT 1"
                    f") AS _src STRAIGHT_JOIN {self.s} AS _p ON _p.session_id=_src._target_sid "
                    "SET "
                    "_p.inbox_version=(LAST_INSERT_ID("
                    "((_p.inbox_version+1)<<1)|"
                    "(CASE WHEN _p.lifecycle='open' AND (_p.execution_state='idle' "
                    "OR (_p.execution_state='running' AND _p.session_kind='workflow_driver')) "
                    "AND ?=1 THEN 1 ELSE 0 END)"
                    ")>>1),"
                    "_p.wake_inbox_seq=_p.wake_inbox_seq+?,"
                    "_p.wake_pending_count=_p.wake_pending_count+?,"
                    "_p.pending_count=_p.pending_count+1,"
                    "_p.pending_bytes=_p.pending_bytes+?,"
                    "_p.execution_state=CASE WHEN _p.lifecycle='open' AND _p.execution_state='idle' AND ?=1 "
                    "THEN 'queued' ELSE _p.execution_state END,"
                    f"_p.updated_at={now_sql} "
                    "WHERE _p.lifecycle IN ('open','suspended') "
                    "AND _p.tenant_id=? AND _p.principal_id=? "
                    "AND _p.pending_count<? AND (_p.pending_bytes+?)<=? "
                    f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=_p.session_id AND item_id=?)",
                    (
                        child_target.session_id, verify_source_item_id, parent_session_id,
                        child_wake, child_wake, child_wake, content_bytes, child_wake,
                        child_tenant_id, child_principal_id,
                        self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                        message_id,
                    ),
                )
            else:
                p_rc, p_packed = await tx.execute_packed(
                    f"UPDATE {self.s} SET "
                    "inbox_version=(LAST_INSERT_ID("
                    "((inbox_version+1)<<1)|"
                    "(CASE WHEN lifecycle='open' AND (execution_state='idle' "
                    "OR (execution_state='running' AND session_kind='workflow_driver')) "
                    "AND ?=1 THEN 1 ELSE 0 END)"
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
                f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?)"
                f"{source_route_sql} "
                "RETURNING inbox_version,lifecycle,execution_state,wake_pending_count,session_kind",
                (
                    child_wake, child_wake, content_bytes, child_wake,
                    parent_session_id, child_tenant_id, child_principal_id,
                    self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                    parent_session_id, message_id, *source_route_params,
                ),
            )
            parent_updated = prow is not None
            parent_inbox_ver = int(prow["inbox_version"]) if parent_updated else 0
            parent_need_wake = (
                parent_updated
                and prow["lifecycle"] == "open"
                and child_wake == 1
                and (
                    (prow["execution_state"] == "queued" and int(prow["wake_pending_count"]) == 1)
                    or (prow["execution_state"] == "running" and prow["session_kind"] == "workflow_driver")
                )
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
            if (
                result_kind == "workflow_node_result"
                and child_wake == 1
                and _driver_events_out is not None
            ):
                _driver_events_out.append((parent_session_id, message_id, body))
            return ref, None

        if verify_source_item_id is not None:
            q_live = await tx.fetchone(
                f"SELECT reply_to_session_id,reply_result_kind FROM {self.q} "
                "WHERE session_id=? AND item_id=?",
                (child_target.session_id, verify_source_item_id),
            )
            if q_live is not None:
                live_to = q_live["reply_to_session_id"]
                if live_to is None:
                    return None, None
                if str(live_to) != parent_session_id:
                    live_rk = q_live["reply_result_kind"] or result_kind
                    return await self._river_deliver_tx(
                        tx,
                        child_target=child_target,
                        child_tenant_id=child_tenant_id,
                        child_principal_id=child_principal_id,
                        parent_session_id=str(live_to),
                        start_ref=start_ref,
                        result=result,
                        result_kind=live_rk,
                        child_wake=child_wake,
                        activations=activations,
                        verify_source_item_id=None,
                        _driver_events_out=_driver_events_out,
                    )
            else:
                return await self._river_deliver_tx(
                    tx,
                    child_target=child_target,
                    child_tenant_id=child_tenant_id,
                    child_principal_id=child_principal_id,
                    parent_session_id=parent_session_id,
                    start_ref=start_ref,
                    result=result,
                    result_kind=result_kind,
                    child_wake=child_wake,
                    activations=activations,
                    verify_source_item_id=None,
                    _driver_events_out=_driver_events_out,
                )

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
        parent_sid = None
        async with self.db.transaction() as tx:
            session = await self._session(tx, child_target.session_id)
            if not session["result_route_json"]:
                return None
            route = json.loads(session["result_route_json"])
            parent_sid = route["target"]["session_id"]
            activations: list[TargetRef] = []
            ref, error = await self._river_deliver_tx(
                tx,
                child_target=child_target,
                child_tenant_id=session["tenant_id"],
                child_principal_id=session["principal_id"],
                parent_session_id=parent_sid,
                start_ref=InputRef(**route["input_ref"]),
                result=result,
                result_kind=result_kind,
                child_wake=1,
                activations=activations,
            )
        if error is not None and parent_sid is not None:
            await self._fire_result_undeliverable(
                child_target=child_target,
                parent_session_id=parent_sid,
                result=replace(result, result_delivery_error=error),
                result_kind=result_kind,
                reason=error,
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
                "WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=? "
                f"AND owner_lease_until_ms>{now_sql}",
                (ttl_ms, node_id, boot_id),
            )
            rows = await self.db.fetchall(
                "SELECT session_id,owner_epoch,cancel_seq,cancel_by,cancel_reason,lifecycle "
                f"FROM {self.s} WHERE owner_node_id=? AND execution_state='running' "
                f"AND owner_boot_id=? AND owner_lease_until_ms>{now_sql}",
                (node_id, boot_id),
            )
            return [dict(r) for r in rows]
        rows = await self.db.fetchall(
            f"UPDATE {self.s} SET owner_lease_until_ms={now_sql}+?,updated_at={now_sql} "
            "WHERE owner_node_id=? AND execution_state='running' AND owner_boot_id=? "
            f"AND owner_lease_until_ms>{now_sql} "
            "RETURNING session_id,owner_epoch,cancel_seq,cancel_by,cancel_reason,lifecycle",
            (ttl_ms, node_id, boot_id),
        )
        return [dict(r) for r in rows]

    async def release_orphan_lease(
        self,
        session_id: str,
        *,
        node_id: str,
        boot_id: str,
        owner_epoch: int,
    ) -> int:
        """Slow-path conditional lease release for a DB running row absent from local active set."""
        now_sql = self._now_sql()
        return await self.db.execute(
            f"UPDATE {self.s} SET owner_lease_until_ms=0,updated_at={now_sql} "
            "WHERE session_id=? AND owner_node_id=? AND owner_boot_id=? "
            "AND owner_epoch=? AND execution_state='running'",
            (session_id, node_id, boot_id, int(owner_epoch)),
        )

    async def discover_queued(self, *, limit: int = 32) -> list[str]:
        """1 bounded index scan on ``idx_session_recovery``."""
        if not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options", "limit")
        rows = await self.db.fetchall(
            f"SELECT session_id FROM {self.s} "
            "WHERE execution_state='queued' AND owner_lease_until_ms IS NULL "
            "AND lifecycle='open' AND wake_pending_count>0 "
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

    async def discover_expired_leases(self, *, limit: int = 32) -> list[dict[str, Any]]:
        """1 bounded index scan on ``idx_session_recovery`` returning expired running candidates."""
        now_sql = self._now_sql()
        rows = await self.db.fetchall(
            f"SELECT session_id,owner_epoch,session_kind,lifecycle,cancel_by FROM {self.s} "
            f"WHERE execution_state='running' AND owner_lease_until_ms<={now_sql} "
            "ORDER BY owner_lease_until_ms ASC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    async def sweep_expired_leases(
        self, *, node_id: str, boot_id: str, ttl_s: float = 30.0, limit: int = 32, can_execute=None,
    ) -> list[ClaimedBatch]:
        """1 bounded index scan on ``idx_session_recovery``; reclaims expired running sessions."""
        expired = await self.discover_expired_leases(limit=limit)
        reclaimed: list[ClaimedBatch] = []
        for row in expired:
            batch = await self.reclaim_running(
                TargetRef(self.store.store_id, row["session_id"]),
                expected_epoch=int(row["owner_epoch"]),
                node_id=node_id,
                boot_id=boot_id,
                ttl_s=ttl_s,
                can_execute=can_execute,
            )
            if batch is not None:
                reclaimed.append(batch)
        return reclaimed

    async def _reclaim_single_session(
        self,
        target: TargetRef,
        *,
        expected_epoch: int | None = None,
        node_id: str,
        boot_id: str,
        ttl_s: float = 30.0,
        can_execute=None,
    ) -> ClaimedBatch | None:
        """Atomically take over a crashed/expired running session (Defense 1 + S09-R5/R6 + Probe 7).

        - S09-R5: synthesized tool rows carry ``meta_json._assistant_seq`` and ``round_index``.
        - S09-R6: ``reclaimed_next_round_index = MAX(round_index) + 1``, ``next_seq = MAX(seq) + 1``,
          and ``reclaimed_pending_tool_seq`` is surfaced on ``ClaimedBatch``.
        - Probe 7: ``kind='repaired'`` projection uses same-tx ``DELETE + INSERT`` so a 2nd takeover
          on the same ``send_index`` never hits ``uq_project_send_kind``.
        - Line 1 seam: calls ``self.on_repair_interrupted_turn`` via ``build_interrupted_turn_repairs``
          (reserved for M1c ``HookPoint.TURN_REPAIR``).
        """
        from power_loop.runtime.budget import estimate_text_tokens
        from power_loop.runtime.history_sanitize import build_interrupted_turn_repairs
        from power_loop.runtime.message_ref import MessageRef

        self.check_store(target)
        now_sql = self._now_sql()
        ttl_ms = int(ttl_s * 1000)
        sid = target.session_id
        expiry_cond = f"AND owner_lease_until_ms<={now_sql}"
        epoch_cond = (
            f"{expiry_cond} AND owner_epoch=?"
            if expected_epoch is not None else expiry_cond
        )
        extra_params = (expected_epoch,) if expected_epoch is not None else ()
        prev_node_id: str | None = None
        prev_boot_id: str | None = None
        try:
            async with self.db.transaction() as tx:
                prev_row = await tx.fetchone(
                    f"SELECT owner_node_id,owner_boot_id FROM {self.s} "
                    "WHERE session_id=? AND execution_state='running'",
                    (sid,),
                )
                if prev_row is None:
                    return None
                prev_node_id = prev_row["owner_node_id"]
                prev_boot_id = prev_row["owner_boot_id"]
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
                database_now = int(srow["updated_at"])
                turn_msgs = await tx.fetchall(
                    f"SELECT id,seq,role,name,content,tool_calls_json,tool_call_id,"
                    f"round_index,state,meta_json,send_index,created_at "
                    f"FROM {self.store.t.messages} WHERE session_id=? AND send_index=? ORDER BY seq ASC",
                    (sid, cur_send_index),
                )
                max_row = await tx.fetchone(
                    f"SELECT COALESCE(MAX(seq),0) AS max_seq FROM {self.store.t.messages} WHERE session_id=?",
                    (sid,),
                )
                max_seq = int(max_row["max_seq"]) if max_row else 0
                next_seq = max(int(srow["next_seq"]), max_seq + 1)
                has_assistant = any(m["role"] == "assistant" for m in turn_msgs)
                round_indices = [int(m["round_index"]) for m in turn_msgs if m["round_index"] is not None]
                next_round_idx = (max(round_indices) + 1) if (has_assistant and round_indices) else 0
                raw_pending_tool_seq = (
                    int(srow["pending_tool_seq"]) if srow["pending_tool_seq"] is not None else None
                )

                normalized = [
                    {
                        "id": m["id"],
                        "seq": int(m["seq"]),
                        "round_index": int(m["round_index"]) if m["round_index"] is not None else 0,
                        "role": m["role"],
                        "name": m["name"],
                        "content": m["content"],
                        "tool_calls": json.loads(m["tool_calls_json"]) if m["tool_calls_json"] else None,
                        "tool_call_id": m["tool_call_id"],
                        "send_index": int(m["send_index"]) if m["send_index"] is not None else cur_send_index,
                    }
                    for m in turn_msgs
                ]
                asst_by_seq = {int(m["seq"]): m for m in turn_msgs if m["role"] == "assistant"}
                repairs = await build_interrupted_turn_repairs(
                    sid,
                    normalized,
                    on_repair_interrupted_turn=self._turn_repair_callback(),
                )
                reclaimed_pending_tool_seq = raw_pending_tool_seq
                if repairs:
                    first_asst_seq = (repairs[0].get("meta") or {}).get("_assistant_seq")
                    if first_asst_seq is not None:
                        reclaimed_pending_tool_seq = int(first_asst_seq)
                    for rep in repairs:
                        cid = str(rep.get("tool_call_id") or "")
                        meta = dict(rep.get("meta") or {})
                        asst_seq = meta.get("_assistant_seq")
                        content = rep.get("content") or "<aborted: 执行中断，结果未知>"
                        if asst_seq is not None and int(asst_seq) in asst_by_seq:
                            am = asst_by_seq[int(asst_seq)]
                            call_ref = MessageRef(
                                self.store.store_id, sid, am["id"], int(am["seq"]),
                                am["role"], am["tool_call_id"], am["send_index"], am["round_index"],
                            )
                            child = await tx.fetchone(
                                f"SELECT session_id,start_input_message_id FROM {self.s} "
                                "WHERE parent_session_id=? AND creation_call_ref_json=? "
                                "AND creation_tool_call_id=?",
                                (sid, canonical(call_ref.to_dict()), cid),
                            )
                            if child is not None and child["start_input_message_id"] is not None:
                                child_target = TargetRef(self.store.store_id, child["session_id"])
                                start = InputRef(
                                    self.store.store_id, child["session_id"],
                                    child["start_input_message_id"],
                                )
                                content = canonical({
                                    "creation_result_uncertain": True,
                                    "target_ref": plain(child_target),
                                    "input_ref": plain(start),
                                    "status": "accepted",
                                })
                                meta["creation_result_uncertain"] = True
                                rep["content"] = content
                        meta["repaired"] = True
                        await tx.execute(
                            f"INSERT INTO {self.store.t.messages} (session_id,seq,role,name,content,"
                            "tool_call_id,round_index,state,meta_json,send_index,created_at) "
                            "VALUES (?,?,'tool',?,?,?,?,'active',?,?,?)",
                            (
                                sid, next_seq, rep.get("name"), content, cid,
                                int(rep.get("round_index") or 0), canonical(meta),
                                cur_send_index, database_now,
                            ),
                        )
                        next_seq += 1

                    # Probe 7: same-tx DELETE + INSERT for kind='repaired' prevents DuplicateKeyError
                    # on 2nd crash takeover of the same send_index.
                    await tx.execute(
                        f"DELETE FROM {self.store.t.project_messages} "
                        "WHERE session_id=? AND send_index=? AND kind='repaired'",
                        (sid, cur_send_index),
                    )
                    rep_text = "\n".join(
                        f"[repaired:{r['tool_call_id']}] {r['content']}" for r in repairs
                    )
                    stitched_msgs = []
                    for nm in normalized:
                        entry_m: dict[str, Any] = {"role": nm["role"], "content": nm["content"]}
                        if nm.get("name"):
                            entry_m["name"] = nm["name"]
                        if nm.get("tool_calls"):
                            entry_m["tool_calls"] = nm["tool_calls"]
                        if nm.get("tool_call_id"):
                            entry_m["tool_call_id"] = nm["tool_call_id"]
                        stitched_msgs.append(entry_m)
                    for r in repairs:
                        stitched_msgs.append({
                            "role": "tool",
                            "tool_call_id": r["tool_call_id"],
                            "name": r.get("name"),
                            "content": r["content"],
                        })
                    lo_seq = int(turn_msgs[0]["seq"]) if turn_msgs else (next_seq - 1)
                    hi_seq = next_seq - 1
                    await tx.execute(
                        f"INSERT INTO {self.store.t.project_messages} ("
                        "session_id,send_index,kind,content_json,rendered_text,"
                        "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                        "projector_version,token_estimate,created_at,completed_through_send_index) "
                        "VALUES (?,?,'repaired',?,?,?,?,NULL,NULL,0,?,?,?)",
                        (
                            sid, cur_send_index,
                            canonical({"messages": stitched_msgs, "repaired_tools": repairs}),
                            rep_text, lo_seq, hi_seq,
                            estimate_text_tokens(rep_text), database_now, cur_send_index,
                        ),
                    )

                if next_seq != int(srow["next_seq"]) or srow["pending_tool_seq"] is not None:
                    await tx.execute(
                        f"UPDATE {self.s} SET next_seq=?,pending_tool_seq=NULL,updated_at={now_sql} "
                        "WHERE session_id=? AND owner_epoch=?",
                        (next_seq, sid, int(srow["owner_epoch"])),
                    )

                qrows = await tx.fetchall(
                    f"SELECT item_id,reply_to_session_id,reply_result_kind,meta_json FROM {self.q} "
                    "WHERE session_id=? AND delivered_send_index=? AND status='delivered' ORDER BY id ASC",
                    (sid, cur_send_index),
                )
        except _ClaimRollback:
            return None
        await self._fire_session_takeover(
            session_id=sid,
            target=target,
            prev_node_id=prev_node_id,
            prev_boot_id=prev_boot_id,
            new_node_id=node_id,
            new_boot_id=boot_id,
            owner_epoch=int(srow["owner_epoch"]),
            send_index=cur_send_index,
        )
        refs = tuple(InputRef(target.store_id, sid, r["item_id"]) for r in qrows)
        first_user = next((m for m in turn_msgs if m["role"] == "user"), None)
        msg_seq = max(0, next_seq - 1)
        if first_user is not None:
            user_row = StoredMessageRow(
                session_id=sid, seq=int(first_user["seq"]), role="user",
                name=first_user["name"], content=first_user["content"],
                tool_calls=None, tool_call_id=None, round_index=first_user["round_index"],
                state=MessageState(first_user["state"]),
                meta=json.loads(first_user["meta_json"] or "{}"),
                created_at=int(first_user["created_at"]), send_index=cur_send_index,
            )
        else:
            user_row = StoredMessageRow(
                session_id=sid, seq=msg_seq, role="user", name=None, content="",
                tool_calls=None, tool_call_id=None, round_index=0,
                state=MessageState.ACTIVE, meta={}, created_at=int(srow["updated_at"]),
                send_index=cur_send_index,
            )
        reply_row = (
            next(
                (
                    r for r in qrows
                    if r["reply_to_session_id"] and r["reply_result_kind"] == "workflow_node_result"
                ),
                None,
            )
            or next((r for r in qrows if r["reply_to_session_id"]), None)
        )
        reply_to_sid = reply_row["reply_to_session_id"] if reply_row is not None else None
        reply_rk = reply_row["reply_result_kind"] if reply_row is not None else None
        reply_item_id = reply_row["item_id"] if reply_row is not None else None
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
            cancel_by=srow["cancel_by"] or ("removed" if srow["lifecycle"] in ("closing", "closed") else None),
            start_input_message_id=srow["start_input_message_id"],
            root_session_id=str(srow["root_session_id"]),
            depth=int(srow["depth"]),
            reply_to_session_id=reply_to_sid,
            reply_result_kind=reply_rk,
            reply_to_item_id=reply_item_id,
            fold_stats_json=srow["fold_stats_json"],
            is_reclaimed=True,
            reclaimed_has_assistant=has_assistant,
            reclaimed_cancelled=bool(
                srow["cancel_by"] is not None or srow["lifecycle"] in ("closing", "closed")
            ),
            reclaimed_next_round_index=next_round_idx,
            reclaimed_pending_tool_seq=reclaimed_pending_tool_seq,
            max_rounds_override=_extract_max_rounds_override(qrows),
        )

    async def reclaim_running(
        self, target: TargetRef, *, expected_epoch: int | None = None, node_id: str, boot_id: str,
        ttl_s: float = 30.0, can_execute=None,
    ) -> ClaimedBatch | None:
        """Atomically take over a crashed/expired running session (delegates to ``_reclaim_single_session``)."""
        return await self._reclaim_single_session(
            target, expected_epoch=expected_epoch, node_id=node_id, boot_id=boot_id,
            ttl_s=ttl_s, can_execute=can_execute,
        )

    async def _select_workflow_origin_tx(
        self, tx, own_state, origins: list[InputRecord],
    ) -> InputRecord:
        """Select the true workflow run origin when a Driver send merged multiple inputs.

        Priority:
        1. Existing ``workflow_continue`` continuation chain item.
        2. The session's ``start_input_message_id`` (initial workflow start input).
        3. Explicit ``resume_run`` input matching saved journal ``definition_ref`` or
           carrying ``{"workflow_resume": ...}``.
        4. Fallback to ``origins[-1]`` (late inputs after completion).
        """
        for o in origins:
            if o.row["kind"] == "workflow_continue":
                root = await self._continuation_root(tx, o)
                if self._is_driver_run_input(own_state, root):
                    return o
        start_id = own_state["start_input_message_id"]
        if start_id is not None:
            for o in origins:
                if o.ref.message_id == start_id:
                    return o
            from power_loop.workflow import journal
            saved = await journal.get(
                self.store, tx, own_state["session_id"], journal.run_key(start_id),
            )
            def_ref = (saved or {}).get("definition_ref") if isinstance(saved, dict) else None
            def_msg_id = def_ref.get("message_id") if isinstance(def_ref, dict) else None
            if def_msg_id:
                for o in origins:
                    if o.ref.message_id == def_msg_id:
                        return o
        for o in origins:
            try:
                parsed = json.loads(o.row["content_json"])
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "workflow_resume" in parsed:
                return o
        return origins[-1]

    async def _close_workflow_run_journal_tx(
        self, tx, state, *, primary_ref: InputRef, root_ref: InputRef, cby: str, reason: str,
        database_now: int,
    ) -> dict[str, Any] | None:
        """Build cancelled WorkflowResult and close open run journal only if owned by ``root_ref``."""
        if state["session_kind"] != "workflow_driver":
            return None
        run_id = state["start_input_message_id"]
        if not run_id:
            return None
        from power_loop.workflow import journal

        sid = state["session_id"]
        saved = await journal.get(self.store, tx, sid, journal.run_key(run_id))
        desc_obj = json.loads(state["descriptor_json"])
        wf_name = (
            ((desc_obj.get("config") or {}).get("workflow") or {}).get("spec", {}).get("name")
            or run_id
        )
        agg_final = {
            "name": wf_name,
            "status": "cancelled",
            "results": {},
            "final": None,
            "usage": {},
            "errors": [reason] if reason and reason != "cancelled" else [],
            "tolerated": [],
            "cancel_by": cby,
        }
        def_ref = saved.get("definition_ref") if isinstance(saved, dict) else None
        def_mid = def_ref.get("message_id") if isinstance(def_ref, dict) else None
        matches_open_run = (
            saved is not None
            and not saved.get("aggregate_delivered")
            and (
                root_ref.message_id == run_id
                or (def_mid is not None and root_ref.message_id == def_mid)
                or (def_mid is None and state["execution_state"] != "running")
            )
        )
        if matches_open_run:
            saved.update(
                status="cancelled",
                cancel_by=cby,
                result=agg_final,
                result_seq=None,
                aggregate_delivered=True,
                aggregate_delivered_at=database_now,
                continuation={},
                last_input_ref=plain(primary_ref),
            )
            await journal.put(self.store, tx, sid, journal.run_key(run_id), saved)
        return agg_final

    async def _recover_workflow_tx(
        self, tx, own_state, target: TargetRef, active: int, epoch: int, database_now: int,
        undeliverable_out: list[tuple[TargetRef, str, Result, str, str]],
        driver_events_out: list[tuple[str, str, dict[str, Any]]],
    ) -> FinalBatch:
        """Continue an interrupted non-LLM workflow_driver via deterministic continuation."""
        sid = target.session_id
        now_sql = self._now_sql()
        rows = await tx.fetchall(
            f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
            "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id ASC",
            (sid, active),
        )
        cancelled = own_state["cancel_by"] is not None or own_state["lifecycle"] in ("closing", "closed")
        if cancelled:
            cby = own_state["cancel_by"] or ("removed" if own_state["lifecycle"] in ("closing", "closed") else "user")
            reason = own_state["cancel_reason"] or "cancelled"
            origins = [self.record(row) for row in rows if row["kind"] in {"user", "workflow_continue"}]
            delivery_error = None
            result_ref = None
            activations: list[TargetRef] = []
            settled_results: list[Result] = []
            seen_result_refs: set[InputRef] = set()
            if origins:
                primary = await self._select_workflow_origin_tx(tx, own_state, origins)
                route = json.loads(own_state["result_route_json"]) if own_state["result_route_json"] else None
                # Process primary first, then any remaining co-claimed origins (only possible on
                # whole-session/subtree cancel or close; single-input cancel already restored others).
                ordered_origins = [
                    primary, *(o for o in origins if o.ref.message_id != primary.ref.message_id),
                ]
                for orig in ordered_origins:
                    is_primary = orig.ref.message_id == primary.ref.message_id
                    orig_root = await self._continuation_root(tx, orig)
                    orig_is_run = self._is_driver_run_input(own_state, orig_root)
                    orig_agg: dict[str, Any] | None = None
                    if orig_is_run:
                        orig_agg = await self._close_workflow_run_journal_tx(
                            tx, own_state, primary_ref=orig.ref, root_ref=orig_root.ref,
                            cby=cby, reason=reason, database_now=database_now,
                        )
                    orig_res_json = canonical(orig_agg) if orig_agg is not None else None
                    orig_ref, orig_deliv_err = None, None
                    orig_send_idx = active if is_primary else None
                    if orig_is_run and route is not None:
                        cancelled_res = Result(
                            orig_root.ref, target, "cancelled", orig_send_idx, None,
                            structured_output=orig_agg,
                            error=reason, error_code="cancelled", cancel_by=cby,
                        )
                        orig_ref, orig_deliv_err = await self._river_deliver_tx(
                            tx,
                            child_target=target,
                            child_tenant_id=own_state["tenant_id"],
                            child_principal_id=own_state["principal_id"],
                            parent_session_id=route["target"]["session_id"],
                            start_ref=orig_root.ref,
                            result=cancelled_res,
                            result_kind=route["result_kind"],
                            child_wake=0 if cby == "subtree" else 1,
                            activations=activations,
                            _driver_events_out=driver_events_out,
                        )
                        if orig_deliv_err is not None:
                            undeliverable_out.append((
                                target,
                                route["target"]["session_id"],
                                replace(cancelled_res, result_delivery_error=orig_deliv_err),
                                route["result_kind"],
                                orig_deliv_err,
                            ))
                    if is_primary:
                        result_ref = orig_ref
                        delivery_error = orig_deliv_err
                    elif orig_ref is not None and result_ref is None:
                        result_ref = orig_ref
                    if is_primary:
                        await tx.execute(
                            f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',cancel_by=?,"
                            "result_send_index=?,result_json=?,error_code='cancelled',error_detail=?,"
                            "result_delivery_error=?,completed_at=? WHERE session_id=? "
                            "AND item_id=? AND delivered_send_index=? AND terminal_status IS NULL",
                            (
                                cby, active, orig_res_json, reason, orig_deliv_err,
                                database_now, sid, orig.ref.message_id, active,
                            ),
                        )
                    else:
                        await tx.execute(
                            f"UPDATE {self.q} SET status='void',delivered_at=NULL,delivered_seq=NULL,"
                            "delivered_send_index=NULL,terminal_status='cancelled',cancel_by=?,"
                            "result_send_index=NULL,result_json=?,error_code='cancelled',error_detail=?,"
                            "result_delivery_error=?,completed_at=? WHERE session_id=? "
                            "AND item_id=? AND delivered_send_index=? AND terminal_status IS NULL",
                            (
                                cby, orig_res_json, reason, orig_deliv_err,
                                database_now, sid, orig.ref.message_id, active,
                            ),
                        )
                    orig_refs: list[InputRef] = [orig.ref]
                    ancestors: list[str] = []
                    seen_anc: set[str] = {orig.ref.message_id}
                    current = orig
                    while current.row["continuation_of"] is not None:
                        anc_id = current.row["continuation_of"]
                        if anc_id in seen_anc:
                            break
                        current = await self.input(
                            tx, InputRef(current.ref.store_id, sid, anc_id),
                        )
                        if current is None:
                            break
                        seen_anc.add(anc_id)
                        ancestors.append(anc_id)
                        orig_refs.append(current.ref)
                    if ancestors:
                        aph = ",".join("?" for _ in ancestors)
                        await tx.execute(
                            f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',cancel_by=?,"
                            "result_send_index=?,result_json=?,error_code='cancelled',error_detail=?,"
                            "result_delivery_error=?,completed_at=? WHERE session_id=? "
                            f"AND item_id IN ({aph}) AND terminal_status IS NULL",
                            (
                                cby, orig_send_idx, orig_res_json, reason, orig_deliv_err,
                                database_now, sid, *ancestors,
                            ),
                        )
                    for ref in orig_refs:
                        if ref not in seen_result_refs:
                            seen_result_refs.add(ref)
                            settled_results.append(Result(
                                ref, target, "cancelled", orig_send_idx, None,
                                structured_output=orig_agg,
                                error=reason, error_code="cancelled",
                                result_delivery_error=orig_deliv_err,
                                cancel_by=cby,
                            ))
                if len(origins) < len(rows):
                    await tx.execute(
                        f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',cancel_by=?,"
                        "result_send_index=?,error_code='cancelled',error_detail=?,"
                        "completed_at=? WHERE session_id=? "
                        "AND delivered_send_index=? AND terminal_status IS NULL",
                        (cby, active, reason, database_now, sid, active),
                    )
            w_drain, p_drain, b_drain = await self._drain_pending_driver_node_results_tx(
                tx, sid, active,
            )
            rem_wake = max(0, int(own_state["wake_pending_count"]) - w_drain)
            next_state = (
                "queued"
                if own_state["lifecycle"] == "open" and rem_wake > 0
                else "idle"
            )
            changed = await tx.execute(
                f"UPDATE {self.s} SET execution_state=?,"
                "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END,"
                "owner_node_id=NULL,owner_boot_id=NULL,"
                "owner_lease_until_ms=NULL,pending_tool_seq=NULL,ready_item_json=NULL,"
                f"cancel_by=NULL,cancel_reason=NULL,updated_at={now_sql} "
                "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
                (
                    next_state,
                    w_drain, w_drain, p_drain, p_drain, b_drain, b_drain,
                    sid, epoch,
                ),
            )
            if changed != 1:
                raise SessionLeaseLost(sid)
            return FinalBatch(
                next_state, tuple(settled_results), result_ref,
                activations[0] if activations else None, delivery_error,
            )

        # Uncancelled crash: return unconsumed delivered workflow_node_result rows back to pending
        unconsumed_nodes = [
            r for r in rows
            if r["kind"] == "workflow_node_result" and r["terminal_status"] is None
        ]
        restored_wake = sum(int(r["wake"]) for r in unconsumed_nodes)
        restored_count = len(unconsumed_nodes)
        restored_bytes = sum(int(r["content_bytes"]) for r in unconsumed_nodes)
        if unconsumed_nodes:
            await tx.execute(
                f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                "delivered_send_index=NULL WHERE session_id=? AND kind='workflow_node_result' "
                "AND status='delivered' AND delivered_send_index=? AND terminal_status IS NULL",
                (sid, active),
            )

        origins = [self.record(row) for row in rows if row["kind"] in {"user", "workflow_continue"}]
        ver_bump = 0
        wake_seq_bump = 0
        if not origins:
            next_state = (
                "queued"
                if own_state["lifecycle"] == "open" and (int(own_state["wake_pending_count"]) + restored_wake) > 0
                else "idle"
            )
        else:
            origin = await self._select_workflow_origin_tx(tx, own_state, origins)
            root = await self._continuation_root(tx, origin)
            index = int(origin.row["continuation_index"] or 0) + 1
            message_id = "wfc_" + sha256_canonical([
                "workflow_continue", self.store.store_id, sid, root.ref.message_id, index,
            ])[:56]
            continuation = InputRef(self.store.store_id, sid, message_id)
            body = {
                "workflow_continue": {
                    "origin_input_ref": plain(root.ref),
                    "previous_input_ref": plain(origin.ref),
                    "continuation_index": index,
                },
            }
            content = canonical(body)
            meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
            cbytes = len(content.encode()) + len(meta.encode())
            req_hash = self.request_hash(target, body, "normal")
            existing = await self.input(tx, continuation)
            if existing is None:
                ver_bump = 1
                wake_seq_bump = 1
                new_ver = int(own_state["inbox_version"]) + 1
                await tx.execute(
                    f"INSERT INTO {self.q} ("
                    "session_id,item_id,inbox_version,kind,priority,wake,status,"
                    "content_json,content_bytes,meta_json,request_hash,created_at,"
                    "continuation_of,continuation_index) "
                    "VALUES (?,?,?,'workflow_continue','normal',1,'pending',?,?,?,?,?,?,?)",
                    (
                        sid, message_id, new_ver, content, cbytes, meta, req_hash,
                        database_now, origin.ref.message_id, index,
                    ),
                )
                restored_wake += 1
                restored_count += 1
                restored_bytes += cbytes
            elif existing.row["request_hash"] != req_hash:
                raise RuntimeRequestError("request_conflict")
            await tx.execute(
                f"UPDATE {self.q} SET status='waiting_continuation' WHERE session_id=? "
                "AND item_id=? AND status='delivered' AND delivered_send_index=? "
                "AND terminal_status IS NULL",
                (sid, origin.ref.message_id, active),
            )
            # Restore every other co-claimed user/workflow_continue input (e.g., secondary
            # formal handle.resume() or merged plain late input) back to pending so no input
            # is ever orphaned in delivered(terminal_status=NULL) after takeover!
            other_origins = [o for o in origins if o.ref.message_id != origin.ref.message_id]
            if other_origins:
                other_ids = [o.ref.message_id for o in other_origins]
                oph = ",".join("?" for _ in other_ids)
                await tx.execute(
                    f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                    "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                    f"AND delivered_send_index=? AND terminal_status IS NULL AND item_id IN ({oph})",
                    (sid, active, *other_ids),
                )
                restored_wake += sum(int(o.row["wake"]) for o in other_origins)
                restored_count += len(other_origins)
                restored_bytes += sum(int(o.row["content_bytes"]) for o in other_origins)
            next_state = "queued" if own_state["lifecycle"] == "open" else "idle"

        changed = await tx.execute(
            f"UPDATE {self.s} SET execution_state=?,inbox_version=inbox_version+?,"
            "wake_inbox_seq=wake_inbox_seq+?,wake_pending_count=wake_pending_count+?,"
            "pending_count=pending_count+?,pending_bytes=pending_bytes+?,"
            "owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
            f"pending_tool_seq=NULL,ready_item_json=NULL,cancel_by=NULL,cancel_reason=NULL,updated_at={now_sql} "
            "WHERE session_id=? AND execution_state='running' AND owner_epoch=?",
            (
                next_state, ver_bump, wake_seq_bump, restored_wake, restored_count, restored_bytes,
                sid, epoch,
            ),
        )
        if changed != 1:
            raise SessionLeaseLost(sid)
        return FinalBatch(next_state, ())

    async def recover(self, target, *, node_id: str, boot_id: str, ttl_s: float = 30.0, ownership=None) -> FinalBatch | None:
        """Atomically take over an expired running session without any 'recovering' state."""
        self.check_store(target)
        now_sql = self._now_sql()
        sid = target.session_id
        prev_node_id: str | None = None
        prev_boot_id: str | None = None
        new_epoch = 0
        active = 0
        undeliverables: list[tuple[TargetRef, str, Result, str, str]] = []
        driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
        async with self.db.transaction() as tx:
            try:
                own_state = await self.db.dialect.lock_state(tx, self.s, sid)
            except ValueError:
                return None
            if own_state["execution_state"] != "running":
                return None
            prev_node_id = own_state["owner_node_id"]
            prev_boot_id = own_state["owner_boot_id"]
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
            new_epoch = epoch

            if own_state["session_kind"] == "workflow_driver":
                final_batch = await self._recover_workflow_tx(
                    tx, own_state, target, active, epoch, database_now,
                    undeliverables, driver_events_out,
                )
            else:
                await self._repair_pending_tx(tx, sid, active, own_state)

                rows = await tx.fetchall(
                    f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
                    "AND delivered_send_index=? AND terminal_status IS NULL ORDER BY id ASC",
                    (sid, active),
                )
                cancelled = own_state["cancel_by"] is not None or own_state["lifecycle"] in ("closing", "closed")
                terminal = "cancelled" if cancelled else "interrupted"
                cby = (
                    (own_state["cancel_by"] or ("removed" if own_state["lifecycle"] in ("closing", "closed") else "user"))
                    if cancelled else None
                )
                detail = own_state["cancel_reason"] if cancelled else "execution owner lost"
                delivery_error = None
                result_ref = None
                activations: list[TargetRef] = []
                route = json.loads(own_state["result_route_json"]) if own_state["result_route_json"] else None
                live_root_sid = str(own_state["root_session_id"]) if own_state["root_session_id"] else None
                is_closed = own_state["lifecycle"] in ("closing", "closed")
                is_human_direct_leaf_chat = False
                if rows and route is not None and route.get("result_kind") == "workflow_node_result":
                    has_wf_route = any(
                        r["reply_to_session_id"] is not None
                        and r["reply_result_kind"] == "workflow_node_result"
                        for r in rows
                    )
                    if not has_wf_route:
                        route = None
                        is_human_direct_leaf_chat = not any(
                            r["reply_to_session_id"] is not None for r in rows
                        )

                deliver_parent_sid = None
                deliver_res_kind = None
                deliver_start_ref = None
                deliver_source_ref = None
                if rows and route is not None:
                    deliver_parent_sid = route["target"]["session_id"]
                    deliver_res_kind = route["result_kind"]
                    deliver_start_ref = InputRef(**route["input_ref"])
                    deliver_source_ref = self.record(rows[0]).ref

                if deliver_parent_sid is not None and deliver_res_kind is not None and deliver_source_ref is not None:
                    route_result = Result(
                        deliver_source_ref, target, terminal, active, None,
                        error=detail, error_code=terminal, cancel_by=cby,
                    )
                    result_ref, delivery_error = await self._river_deliver_tx(
                        tx,
                        child_target=target,
                        child_tenant_id=own_state["tenant_id"],
                        child_principal_id=own_state["principal_id"],
                        parent_session_id=deliver_parent_sid,
                        start_ref=deliver_start_ref or deliver_source_ref,
                        result=route_result,
                        result_kind=deliver_res_kind,
                        child_wake=0 if own_state["cancel_by"] == "subtree" else 1,
                        activations=activations,
                        _driver_events_out=driver_events_out,
                    )
                    if delivery_error is not None:
                        undeliverables.append((
                            target,
                            deliver_parent_sid,
                            replace(route_result, result_delivery_error=delivery_error),
                            deliver_res_kind,
                            delivery_error,
                        ))
                elif (
                    is_human_direct_leaf_chat
                    and not is_closed
                    and live_root_sid
                    and live_root_sid != sid
                ):
                    notice_id = self._human_chat_notice_id(sid, active)
                    notice_body = {
                        "kind": "system_notice",
                        "event": "human_child_interaction",
                        "session_id": sid,
                        "send_index": active,
                        "text": f"[system_notice · 用户直接与子会话 {sid} 进行了第 {active} 轮对话]",
                    }
                    await self._deliver_system_notice_tx(
                        tx,
                        target_session_id=live_root_sid,
                        tenant_id=own_state["tenant_id"],
                        principal_id=own_state["principal_id"],
                        item_id=notice_id,
                        body=notice_body,
                    )
                if rows:
                    await tx.execute(
                        f"UPDATE {self.q} SET terminal_status=?,cancel_by=?,result_send_index=?,"
                        "error_code=?,error_detail=?,result_delivery_error=?,completed_at=? "
                        "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                        "AND terminal_status IS NULL",
                        (terminal, cby, active, terminal, detail, delivery_error, database_now, sid, active),
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
                        cancel_by=cby,
                    )
                    for row in rows
                )
                final_batch = FinalBatch(
                    next_state, results, result_ref,
                    activations[0] if activations else None, delivery_error,
                )

        for drv_sid, item_id, payload in driver_events_out:
            self._push_driver_leaf_result(drv_sid, item_id, payload)

        await self._fire_session_takeover(
            session_id=sid,
            target=target,
            prev_node_id=prev_node_id,
            prev_boot_id=prev_boot_id,
            new_node_id=node_id,
            new_boot_id=boot_id,
            owner_epoch=new_epoch,
            send_index=active,
        )
        for c_target, p_sid, r_res, r_kind, r_reason in undeliverables:
            await self._fire_result_undeliverable(
                child_target=c_target,
                parent_session_id=p_sid,
                result=r_res,
                result_kind=r_kind,
                reason=r_reason,
            )
        return final_batch

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
        only_linked: bool = True, include_independent_drivers: bool = False,
    ) -> list[tuple[str, int, bool]]:
        """Bounded active-only subtree scan using ``idx_sessions_root`` (0 idle rows scanned!)."""
        active_rows = await tx.fetchall(
            "SELECT session_id,parent_session_id,depth,send_index,ready_item_json,"
            f"cancel_propagation,session_kind FROM {self.s} "
            "WHERE root_session_id=? AND execution_state IN ('queued','running')",
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
            cur_kind = r["session_kind"]
            matched = False
            visited: set[str] = set()
            while cur_parent is not None and cur_parent not in visited:
                if only_linked and cur_prop != "linked" and not (
                    include_independent_drivers and cur_kind == "workflow_driver"
                ):
                    break
                if cur_parent == session_id:
                    matched = True
                    break
                visited.add(cur_id)
                if cur_parent in by_id:
                    cur_id = cur_parent
                    cur_prop = by_id[cur_parent]["cancel_propagation"]
                    cur_kind = by_id[cur_parent]["session_kind"]
                    cur_parent = by_id[cur_parent]["parent_session_id"]
                else:
                    prow = await tx.fetchone(
                        f"SELECT parent_session_id,cancel_propagation,session_kind "
                        f"FROM {self.s} WHERE session_id=?",
                        (cur_parent,),
                    )
                    if prow is None:
                        break
                    cur_id = cur_parent
                    cur_prop = prow["cancel_propagation"]
                    cur_kind = prow["session_kind"]
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
        _with_activation: bool = False, _with_stopped: bool = False,
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

        for _attempt in range(8):
            activations: list[TargetRef] = []
            completed_refs: list[InputRef] = []
            stopped_descendants: list[str] = []
            undeliverables: list[tuple[TargetRef, str, Result, str, str]] = []
            driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
            target_is_driver = False
            try:
                async with self.db.transaction() as tx:
                    session = await self.authorize(tx, target, actor)
                    desc_by_depth: dict[int, list[str]] = {}
                    expected_depth: dict[str, int] = {}
                    if scope == "subtree":
                        active_desc = await self._collect_active_descendants(
                            tx,
                            session_id=target.session_id,
                            root_session_id=str(session["root_session_id"]),
                            target_depth=int(session["depth"]),
                            only_linked=only_linked,
                            include_independent_drivers=True,
                        )
                        for d_sid, d_val, _d_pre in active_desc:
                            expected_depth[d_sid] = d_val
                            desc_by_depth.setdefault(d_val, []).append(d_sid)
                        for lvl_ids in desc_by_depth.values():
                            lvl_ids.sort()

                    # Step B (Lock-only, strictly deep-first): lock all candidate descendants and any
                    # intermediate River route parents (> target depth) before locking target, WITHOUT
                    # writing any cancellation yet so moved/idle rows are never wrongly cancelled.
                    lock_suffix = " FOR UPDATE" if self._dialect in ("postgres", "mysql") else ""
                    locked_desc: dict[str, dict[str, Any]] = {}
                    extra_lock_by_depth: dict[int, set[str]] = {}
                    target_depth_snap = int(session["depth"])
                    if desc_by_depth:
                        max_d = max(desc_by_depth)
                        for d_val in range(max_d, target_depth_snap, -1):
                            lvl_set = set(desc_by_depth.get(d_val, ())) | extra_lock_by_depth.get(d_val, set())
                            if not lvl_set:
                                continue
                            lvl_ids = sorted(lvl_set)
                            ph = ",".join("?" for _ in lvl_ids)
                            lvl_rows = await tx.fetchall(
                                f"SELECT * FROM {self.s} WHERE session_id IN ({ph}) "
                                f"ORDER BY session_id ASC{lock_suffix}",
                                tuple(lvl_ids),
                            )
                            for drow in lvl_rows:
                                d_sid = str(drow["session_id"])
                                if d_sid in expected_depth:
                                    locked_desc[d_sid] = dict(drow)
                                    if drow.get("result_route_json") and d_val - 1 > target_depth_snap:
                                        r_obj = json.loads(drow["result_route_json"])
                                        p_sid = (
                                            r_obj["target"]["session_id"]
                                            if isinstance(r_obj, dict) and isinstance(r_obj.get("target"), dict)
                                            else None
                                        )
                                        if p_sid and p_sid != target.session_id:
                                            extra_lock_by_depth.setdefault(d_val - 1, set()).add(p_sid)

                    # Now lock target itself (strictly shallower than all descendants & intermediate parents)
                    state = await self.db.dialect.lock_state(tx, self.s, target.session_id)
                    target_is_driver = state["session_kind"] == "workflow_driver"

                    # Step C (Verify under locks): confirm target & every locked candidate still match
                    # the live subtree topology and active state; retry cleanly with 0 writes if changed!
                    prestaged_desc: list[str] = []
                    if scope == "subtree":
                        if (
                            str(state["root_session_id"]) != str(session["root_session_id"])
                            or int(state["depth"]) != target_depth_snap
                            or len(locked_desc) != len(expected_depth)
                        ):
                            raise _SubtreeChangedRetry()
                        recheck = await self._collect_active_descendants(
                            tx,
                            session_id=target.session_id,
                            root_session_id=str(state["root_session_id"]),
                            target_depth=int(state["depth"]),
                            only_linked=only_linked,
                            include_independent_drivers=True,
                        )
                        recheck_ids = {d_sid for d_sid, _, _ in recheck}
                        if recheck_ids != set(locked_desc.keys()):
                            raise _SubtreeChangedRetry()
                        for d_sid, drow in locked_desc.items():
                            if (
                                str(drow["root_session_id"]) != str(state["root_session_id"])
                                or int(drow["depth"]) != expected_depth[d_sid]
                                or drow["execution_state"] not in ("queued", "running")
                            ):
                                raise _SubtreeChangedRetry()
                            if int(drow["send_index"]) == 0 and bool(drow["ready_item_json"]):
                                prestaged_desc.append(d_sid)

                    # Step D (Verified writes in deep-first order)
                    cascaded_delivered: dict[str, set[str]] = {}
                    for d_val in sorted(desc_by_depth, reverse=True):
                        lvl_ids = desc_by_depth[d_val]
                        for d_sid in lvl_ids:
                            drow = locked_desc[d_sid]
                            d_rcpt = await self._cancel_single_session_tx(
                                tx, drow, reason=reason, cancel_by="subtree",
                                is_subtree_cascade=True, activations=activations,
                                completed_refs=completed_refs,
                                skip_item_ids=cascaded_delivered.get(d_sid),
                                cascaded_delivered=cascaded_delivered,
                                undeliverables=undeliverables,
                                _driver_events_out=driver_events_out,
                            )
                            if d_rcpt.state == "cancel_requested":
                                stopped_descendants.append(d_sid)
                    if isinstance(ref, InputRef):
                        record = await self.input(tx, ref)
                        if record is None:
                            raise RuntimeRequestError("not_found")
                        if (
                            state["session_kind"] == "workflow_driver"
                            or record.row["kind"] == "workflow_continue"
                            or record.row["status"] == "waiting_continuation"
                        ):
                            receipt = await self._cancel_workflow_input_tx(
                                tx, state, record, reason=reason, cancel_by=eff_cancel_by,
                                activations=activations, completed_refs=completed_refs,
                                undeliverables=undeliverables,
                                _driver_events_out=driver_events_out,
                            )
                        else:
                            receipt = await self._cancel_input_tx(
                                tx, state, record, reason=reason, cancel_by=eff_cancel_by,
                                activations=activations, completed_refs=completed_refs,
                                undeliverables=undeliverables,
                                _driver_events_out=driver_events_out,
                            )
                    else:
                        receipt = await self._cancel_single_session_tx(
                            tx, state, reason=reason, cancel_by=eff_cancel_by,
                            is_subtree_cascade=False, activations=activations,
                            completed_refs=completed_refs,
                            skip_item_ids=cascaded_delivered.get(target.session_id),
                            undeliverables=undeliverables,
                            _driver_events_out=driver_events_out,
                        )

                    if prestaged_desc:
                        pre_ph = ",".join("?" for _ in prestaged_desc)
                        await tx.execute(
                            f"UPDATE {self.q} SET cancel_by='subtree' "
                            f"WHERE session_id IN ({pre_ph}) AND status='delivered' "
                            "AND delivered_send_index=1 AND terminal_status IS NULL",
                            tuple(prestaged_desc),
                        )

                    # Design 217 §5.3: human stop on a child/driver notifies root_session_id
                    # with wake=0 system_notice keyed by sys_stop_{session_id}_c{new_cancel_seq}.
                    root_sid = str(state["root_session_id"])
                    if (
                        not isinstance(ref, InputRef)
                        and actor.session_id is None
                        and eff_cancel_by == "user"
                        and root_sid != target.session_id
                        and receipt.state in ("cancel_requested", "voided")
                    ):
                        new_cancel_seq = int(state["cancel_seq"]) + 1
                        raw_notice_id = f"sys_stop_{target.session_id}_c{new_cancel_seq}"
                        if len(raw_notice_id) <= 64:
                            notice_id = raw_notice_id
                        else:
                            suffix = f"_c{new_cancel_seq}"
                            digest_len = max(16, 64 - len("sys_stop_") - len(suffix))
                            notice_id = f"sys_stop_{sha256_canonical([target.session_id])[:digest_len]}{suffix}"
                        notice_body = {
                            "kind": "system_notice",
                            "event": "child_stopped_by_user",
                            "session_id": target.session_id,
                            "cancel_seq": new_cancel_seq,
                            "reason": reason,
                        }
                        await self._deliver_system_notice_tx(
                            tx,
                            target_session_id=root_sid,
                            tenant_id=state["tenant_id"],
                            principal_id=state["principal_id"],
                            item_id=notice_id,
                            body=notice_body,
                        )

                    _ = now_sql  # used via helpers
                break
            except _SubtreeChangedRetry:
                continue
        else:
            raise RuntimeRequestError("request_conflict", "concurrent subtree mutation")

        for drv_sid, item_id, payload in driver_events_out:
            self._push_driver_leaf_result(drv_sid, item_id, payload)
        if receipt.state == "cancel_requested" and target_is_driver:
            self.notify_driver_cancel(
                target.session_id,
                cancel_by=eff_cancel_by,
                reason=reason,
                subtree_stopped=(scope == "subtree"),
            )
        for d_sid in stopped_descendants:
            self.notify_driver_cancel(
                d_sid, cancel_by="subtree", reason=reason, subtree_stopped=True,
            )

        for c_target, p_sid, r_res, r_kind, r_reason in undeliverables:
            await self._fire_result_undeliverable(
                child_target=c_target,
                parent_session_id=p_sid,
                result=r_res,
                result_kind=r_kind,
                reason=r_reason,
            )

        if _with_stopped:
            return receipt, activations, completed_refs, stopped_descendants
        if _with_activation:
            return receipt, activations, completed_refs
        return receipt

    async def _deliver_system_notice_tx(
        self,
        tx,
        *,
        target_session_id: str,
        tenant_id: str,
        principal_id: str,
        item_id: str,
        body: dict[str, Any],
    ) -> bool:
        """Non-cascading wake=0 system_notice delivery obeying deep-first lock order."""
        target = TargetRef(self.store.store_id, target_session_id)
        content = canonical(body)
        meta = canonical({"accepted_protocol": PROTOCOL_VERSION})
        content_bytes = len(content.encode()) + len(meta.encode())
        digest = self.request_hash(target, body, "normal")
        now_sql = self._now_sql()
        if self._dialect == "mysql":
            rc, new_ver = await tx.execute_packed(
                f"UPDATE {self.s} SET inbox_version=LAST_INSERT_ID(inbox_version+1),"
                "pending_count=pending_count+1,pending_bytes=pending_bytes+?,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                "AND tenant_id=? AND principal_id=? "
                "AND pending_count<? AND (pending_bytes+?)<=? "
                f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?)",
                (
                    content_bytes, target_session_id, tenant_id, principal_id,
                    self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                    target_session_id, item_id,
                ),
            )
            if rc != 1:
                return False
        else:
            row = await tx.fetchone(
                f"UPDATE {self.s} SET inbox_version=inbox_version+1,"
                "pending_count=pending_count+1,pending_bytes=pending_bytes+?,"
                f"updated_at={now_sql} "
                "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                "AND tenant_id=? AND principal_id=? "
                "AND pending_count<? AND (pending_bytes+?)<=? "
                f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?) "
                "RETURNING inbox_version",
                (
                    content_bytes, target_session_id, tenant_id, principal_id,
                    self.hard_pending_limit, content_bytes, self.inbox_byte_limit,
                    target_session_id, item_id,
                ),
            )
            if row is None:
                return False
            new_ver = int(row["inbox_version"])
        await tx.execute(
            f"INSERT INTO {self.q} ("
            "session_id,item_id,inbox_version,kind,priority,wake,status,"
            "content_json,content_bytes,meta_json,request_hash,result_kind,created_at) "
            f"VALUES (?,?,?,'system_notice','normal',0,'pending',?,?,?,?,NULL,{now_sql})",
            (
                target_session_id, item_id, new_ver,
                content, content_bytes, meta, digest,
            ),
        )
        return True

    async def _cancel_single_session_tx(
        self, tx, state, *, reason: str, cancel_by: str, is_subtree_cascade: bool,
        activations: list[TargetRef], completed_refs: list[InputRef],
        skip_item_ids: set[str] | None = None,
        cascaded_delivered: dict[str, set[str]] | None = None,
        undeliverables: list[tuple[TargetRef, str, Result, str, str]] | None = None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> Receipt:
        sid = state["session_id"]
        target = TargetRef(self.store.store_id, sid)
        now_sql = self._now_sql()
        route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
        child_wake = 0 if (is_subtree_cascade or cancel_by == "subtree") else 1

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
            root_rec = await self._continuation_root(tx, rec) if row["continuation_of"] is not None else rec
            delivery_error = None
            is_driver_run = self._is_driver_run_input(state, root_rec)
            agg_void: dict[str, Any] | None = None
            if (
                state["session_kind"] == "workflow_driver"
                and is_driver_run
                and (
                    root_rec.ref.message_id == state["start_input_message_id"]
                    or row["continuation_of"] is not None
                )
            ):
                agg_void = await self._close_workflow_run_journal_tx(
                    tx, state, primary_ref=rec.ref, root_ref=root_rec.ref, cby=cancel_by, reason=reason,
                    database_now=now_ms(),
                )
            res_json_void = canonical(agg_void) if agg_void is not None else None
            eff_parent_sid, eff_rk, eff_start = self._resolve_input_delivery_route(
                state, route, row, root_rec.ref,
            )
            if (
                eff_parent_sid is not None
                and eff_rk is not None
                and eff_start is not None
                and is_driver_run
                and row["kind"] in ("user", "workflow_continue")
            ):
                cancelled_res = Result(
                    root_rec.ref, target, "cancelled", None, None,
                    structured_output=agg_void,
                    error=reason, error_code="cancelled", cancel_by=cancel_by,
                )
                parent_ref, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=target,
                    child_tenant_id=state["tenant_id"],
                    child_principal_id=state["principal_id"],
                    parent_session_id=eff_parent_sid,
                    start_ref=eff_start,
                    result=cancelled_res,
                    result_kind=eff_rk,
                    child_wake=child_wake,
                    activations=activations,
                    _driver_events_out=_driver_events_out,
                )
                if delivery_error is None and cascaded_delivered is not None:
                    cascaded_delivered.setdefault(parent_ref.session_id, set()).add(parent_ref.message_id)
                elif delivery_error is not None and undeliverables is not None:
                    undeliverables.append((
                        target,
                        eff_parent_sid,
                        replace(cancelled_res, result_delivery_error=delivery_error),
                        eff_rk,
                        delivery_error,
                    ))
            affected = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "result_json=COALESCE(result_json,?),error_code='cancelled',error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} WHERE session_id=? AND item_id=? AND status='pending'",
                (cancel_by, res_json_void, reason, delivery_error, sid, rec.ref.message_id),
            )
            changed += affected
            if affected:
                voided_bytes += int(row["content_bytes"])
                voided_wake += int(row["wake"])
                completed_refs.append(rec.ref)
                parent_id = row["continuation_of"]
                ancestors_to_close: list[str] = []
                while parent_id in waiting_by_item:
                    parent_row = waiting_by_item.pop(parent_id)
                    ancestors_to_close.append(parent_id)
                    completed_refs.append(InputRef(self.store.store_id, sid, parent_id))
                    parent_id = parent_row["continuation_of"]
                if ancestors_to_close:
                    await tx.execute(
                        f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',cancel_by=?,"
                        "result_send_index=delivered_send_index,result_json=COALESCE(result_json,?),"
                        "error_code='cancelled',error_detail=?,"
                        f"result_delivery_error=?,completed_at={now_sql} WHERE session_id=? "
                        f"AND item_id IN ({','.join('?' for _ in ancestors_to_close)}) "
                        "AND status='waiting_continuation' AND terminal_status IS NULL",
                        (cancel_by, res_json_void, reason, delivery_error, sid, *ancestors_to_close),
                    )

        # Handle queued pre-staged child (send_index=0 and ready_item_json IS NOT NULL)
        is_prestaged = int(state["send_index"]) == 0 and bool(state["ready_item_json"])
        if is_prestaged:
            ready = json.loads(state["ready_item_json"])
            pre_ref = InputRef(self.store.store_id, sid, ready["item_id"])
            delivery_error = None
            if route is not None:
                cancelled_res = Result(
                    pre_ref, target, "cancelled", None, None,
                    error=reason, error_code="cancelled", cancel_by=cancel_by,
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
                    _driver_events_out=_driver_events_out,
                )
                if delivery_error is None and cascaded_delivered is not None:
                    cascaded_delivered.setdefault(parent_ref.session_id, set()).add(parent_ref.message_id)
                elif delivery_error is not None and undeliverables is not None:
                    undeliverables.append((
                        target,
                        route["target"]["session_id"],
                        replace(cancelled_res, result_delivery_error=delivery_error),
                        route["result_kind"],
                        delivery_error,
                    ))
            affected = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "delivered_send_index=NULL,delivered_seq=NULL,error_code='cancelled',"
                f"error_detail=?,result_delivery_error=?,completed_at={now_sql} "
                "WHERE session_id=? AND item_id=? AND status='delivered' AND delivered_send_index=1",
                (cancel_by, reason, delivery_error, sid, ready["item_id"]),
            )
            prestaged_voided = affected
            changed += affected
            if affected:
                voided_wake += 1
                completed_refs.append(pre_ref)
        else:
            prestaged_voided = 0

        if not pending_rows and not is_prestaged and route is not None:
            start = await self.input(tx, InputRef(**route["input_ref"]))
            if start is not None and self._can_redeliver_stored_result(state, start.row):
                await self._route_stored_result_tx(
                    tx, state, route, start, activations,
                    _driver_events_out=_driver_events_out,
                )

        if state["execution_state"] == "queued" or changed > 0:
            # Deduct ONLY the rows actually voided in this step relative to the live locked
            # sessions row, preserving any wake=0 receipts delivered by deeper descendants
            # earlier in the same subtree-cancel transaction.
            voided_pending = max(0, changed - prestaged_voided)
            await tx.execute(
                f"UPDATE {self.s} SET "
                "execution_state=CASE WHEN lifecycle='open' AND (wake_pending_count-?)>0 "
                "THEN 'queued' ELSE 'idle' END,"
                "cancel_seq=cancel_seq+1,cancel_by=?,cancel_reason=?,"
                "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END,"
                f"ready_item_json=NULL,updated_at={now_sql} "
                "WHERE session_id=? AND execution_state IN ('idle','queued')",
                (
                    voided_wake, cancel_by, reason,
                    voided_wake, voided_wake,
                    voided_pending, voided_pending,
                    voided_bytes, voided_bytes,
                    sid,
                ),
            )
        return Receipt(
            None, target, None, False,
            "voided" if changed else "already_finished",
            None, "not_needed", reason if changed else None,
        )

    async def _continuation_root(self, reader, record: InputRecord) -> InputRecord:
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
        raise CapacityError()

    async def _cancel_workflow_input_tx(
        self, tx, state, record: InputRecord, *, reason: str, cancel_by: str,
        activations: list[TargetRef], completed_refs: list[InputRef],
        undeliverables: list[tuple[TargetRef, str, Result, str, str]] | None = None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> Receipt:
        """Resolve cancel(original/continuation) to the newest continuation and close its chain."""
        sid = record.ref.session_id
        target = TargetRef(self.store.store_id, sid)
        now_sql = self._now_sql()
        root = await self._continuation_root(tx, record)
        is_driver_run = self._is_driver_run_input(state, root)
        latest = await self._latest_continuation(
            tx,
            ResultInput(
                record.ref,
                {name: record.row[name] for name in _RESULT_INPUT_COLUMNS},
                result_usage=_extract_result_usage(record.row.get("meta_json")),
            ),
        )
        derived_latest = await self._derive_input_terminal(tx, dict(state), latest.row)
        if derived_latest["terminal_status"] is not None:
            route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
            if (
                route is not None
                and is_driver_run
                and self._can_redeliver_stored_result(state, latest.row)
            ):
                latest_full = await self.input(tx, latest.ref)
                if latest_full is not None:
                    await self._route_stored_result_tx(
                        tx, state, route, latest_full, activations,
                        _driver_events_out=_driver_events_out,
                    )
            return self.receipt(record, state="already_finished")
        was_pending = latest.row["status"] == "pending"
        is_deferred_secondary = False
        if (
            latest.row["status"] == "delivered"
            and state["execution_state"] == "running"
            and latest.row["delivered_send_index"] == int(state["send_index"])
        ):
            active_si = int(state["send_index"])
            del_rows = await tx.fetchall(
                f"SELECT * FROM {self.q} WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL "
                "AND kind IN ('user','workflow_continue') ORDER BY id ASC",
                (sid, active_si),
            )
            del_origins = [self.record(r) for r in del_rows]
            restore_others: list[InputRecord] = []
            if len(del_origins) > 1:
                primary = await self._select_workflow_origin_tx(tx, state, del_origins)
                primary_root = await self._continuation_root(tx, primary)
                if primary_root.ref.message_id != root.ref.message_id:
                    is_deferred_secondary = True
                else:
                    restore_others = [
                        o for o in del_origins if o.ref.message_id != primary.ref.message_id
                    ]
            if not is_deferred_secondary:
                w_add = sum(int(o.row["wake"]) for o in restore_others)
                c_add = len(restore_others)
                b_add = sum(int(o.row["content_bytes"]) for o in restore_others)
                if restore_others:
                    oph = ",".join("?" for _ in restore_others)
                    await tx.execute(
                        f"UPDATE {self.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                        "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                        f"AND delivered_send_index=? AND terminal_status IS NULL AND item_id IN ({oph})",
                        (sid, active_si, *(o.ref.message_id for o in restore_others)),
                    )
                await tx.execute(
                    f"UPDATE {self.s} SET cancel_seq=cancel_seq+1,cancel_by=?,cancel_reason=?,"
                    "wake_pending_count=wake_pending_count+?,pending_count=pending_count+?,"
                    f"pending_bytes=pending_bytes+?,updated_at={now_sql} "
                    "WHERE session_id=? AND execution_state='running'",
                    (cancel_by, reason, w_add, c_add, b_add, sid),
                )
                return Receipt(
                    record.ref, target, record.ref.message_id, False, "cancel_requested",
                    int(state["send_index"]), "not_needed", reason,
                )
        if not was_pending and not is_deferred_secondary:
            return self.receipt(record, state="already_finished")

        latest_full = await self.input(tx, latest.ref)
        w_val = int(latest_full.row["wake"]) if latest_full is not None else 1
        b_val = int(latest_full.row["content_bytes"]) if latest_full is not None else 0
        route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
        agg_void: dict[str, Any] | None = None
        if is_driver_run:
            agg_void = await self._close_workflow_run_journal_tx(
                tx, state, primary_ref=latest.ref, root_ref=root.ref, cby=cancel_by, reason=reason,
                database_now=now_ms(),
            )
        res_json_void = canonical(agg_void) if agg_void is not None else None
        delivery_error = None
        if is_driver_run and route is not None:
            cancelled = Result(
                root.ref, target, "cancelled", None, None,
                structured_output=agg_void,
                error=reason, error_code="cancelled", cancel_by=cancel_by,
            )
            _, delivery_error = await self._river_deliver_tx(
                tx,
                child_target=target,
                child_tenant_id=state["tenant_id"],
                child_principal_id=state["principal_id"],
                parent_session_id=route["target"]["session_id"],
                start_ref=root.ref,
                result=cancelled,
                result_kind=route["result_kind"],
                child_wake=0 if cancel_by == "subtree" else 1,
                activations=activations,
                _driver_events_out=_driver_events_out,
            )
            if delivery_error is not None and undeliverables is not None:
                undeliverables.append((
                    target,
                    route["target"]["session_id"],
                    replace(cancelled, result_delivery_error=delivery_error),
                    route["result_kind"],
                    delivery_error,
                ))
        if was_pending:
            changed = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "result_json=COALESCE(result_json,?),error_code='cancelled',error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} WHERE session_id=? AND item_id=? AND status='pending'",
                (cancel_by, res_json_void, reason, delivery_error, sid, latest.ref.message_id),
            )
        else:
            changed = await tx.execute(
                f"UPDATE {self.q} SET status='void',delivered_at=NULL,delivered_seq=NULL,"
                "delivered_send_index=NULL,terminal_status='cancelled',cancel_by=?,result_send_index=NULL,"
                "result_json=COALESCE(result_json,?),error_code='cancelled',error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} "
                "WHERE session_id=? AND item_id=? AND status='delivered' AND delivered_send_index=?",
                (cancel_by, res_json_void, reason, delivery_error, sid, latest.ref.message_id, int(state["send_index"])),
            )
        if changed != 1:
            raise SessionLeaseLost(sid)
        completed_refs.append(latest.ref)

        ancestors: list[str] = []
        closed_refs: list[InputRef] = []
        cur_id = latest.row["continuation_of"]
        while cur_id is not None:
            anc = await self.input(tx, InputRef(self.store.store_id, sid, cur_id))
            if anc is None:
                break
            ancestors.append(anc.ref.message_id)
            if anc.row["status"] == "waiting_continuation" and anc.row["terminal_status"] is None:
                closed_refs.append(anc.ref)
            cur_id = anc.row["continuation_of"]
        if record.ref.message_id not in ancestors:
            ancestors.append(record.ref.message_id)
            if record.row["status"] == "waiting_continuation" and record.row["terminal_status"] is None:
                closed_refs.append(record.ref)
        if ancestors:
            await tx.execute(
                f"UPDATE {self.q} SET status='delivered',terminal_status='cancelled',cancel_by=?,"
                "result_send_index=delivered_send_index,result_json=COALESCE(result_json,?),"
                "error_code='cancelled',error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} WHERE session_id=? "
                f"AND item_id IN ({','.join('?' for _ in ancestors)}) "
                "AND status='waiting_continuation' AND terminal_status IS NULL",
                (cancel_by, res_json_void, reason, delivery_error, sid, *ancestors),
            )
            completed_refs.extend(closed_refs)

        if was_pending:
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
        if is_deferred_secondary:
            return Receipt(
                record.ref, target, record.ref.message_id, False, "voided",
                None, "not_needed", reason,
            )
        return self.receipt(record, state="voided", reason=reason)

    async def _cancel_input_tx(
        self, tx, state, record: InputRecord, *, reason: str, cancel_by: str,
        activations: list[TargetRef], completed_refs: list[InputRef],
        undeliverables: list[tuple[TargetRef, str, Result, str, str]] | None = None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> Receipt:
        row = record.row
        ref = record.ref
        sid = ref.session_id
        now_sql = self._now_sql()
        route = json.loads(state["result_route_json"]) if state["result_route_json"] else None
        # Check derived terminal status for normal completed rows (no final_settle_inbox)
        eff_parent_sid, eff_rk, eff_start = self._resolve_input_delivery_route(state, route, row, ref)
        is_driver_run = self._is_driver_run_input(state, record)
        derived = await self._derive_input_terminal(tx, state, row)
        if derived["terminal_status"] is not None:
            if (
                eff_parent_sid is not None
                and eff_rk is not None
                and eff_start is not None
                and self._can_redeliver_stored_result(state, row)
                and is_driver_run
            ):
                eff_route = {
                    "target": {"store_id": self.store.store_id, "session_id": eff_parent_sid},
                    "input_ref": plain(eff_start),
                    "result_kind": eff_rk,
                }
                await self._route_stored_result_tx(
                    tx, state, eff_route, record, activations,
                    _driver_events_out=_driver_events_out,
                )
            return self.receipt(record, state="already_finished")
        if row["status"] == "pending":
            agg_void: dict[str, Any] | None = None
            if (
                state["session_kind"] == "workflow_driver"
                and is_driver_run
                and ref.message_id == state["start_input_message_id"]
            ):
                agg_void = await self._close_workflow_run_journal_tx(
                    tx, state, primary_ref=ref, root_ref=ref, cby=cancel_by, reason=reason,
                    database_now=now_ms(),
                )
            res_json_void = canonical(agg_void) if agg_void is not None else None
            delivery_error = None
            if (
                eff_parent_sid is not None
                and eff_rk is not None
                and eff_start is not None
                and is_driver_run
            ):
                cancelled = Result(
                    ref, TargetRef(self.store.store_id, sid), "cancelled", None, None,
                    structured_output=agg_void,
                    error=reason, error_code="cancelled", cancel_by=cancel_by,
                )
                _, delivery_error = await self._river_deliver_tx(
                    tx,
                    child_target=TargetRef(self.store.store_id, sid),
                    child_tenant_id=state["tenant_id"],
                    child_principal_id=state["principal_id"],
                    parent_session_id=eff_parent_sid,
                    start_ref=eff_start,
                    result=cancelled,
                    result_kind=eff_rk,
                    child_wake=0 if cancel_by == "subtree" else 1,
                    activations=activations,
                    _driver_events_out=_driver_events_out,
                )
                if delivery_error is not None and undeliverables is not None:
                    undeliverables.append((
                        TargetRef(self.store.store_id, sid),
                        eff_parent_sid,
                        replace(cancelled, result_delivery_error=delivery_error),
                        eff_rk,
                        delivery_error,
                    ))
            changed = await tx.execute(
                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',cancel_by=?,"
                "result_json=COALESCE(result_json,?),error_code='cancelled',error_detail=?,"
                f"result_delivery_error=?,completed_at={now_sql} WHERE session_id=? AND item_id=? AND status='pending'",
                (cancel_by, res_json_void, reason, delivery_error, sid, ref.message_id),
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
                    row, status="void", terminal_status="cancelled", cancel_by=cancel_by,
                    error_detail=reason, result_delivery_error=delivery_error,
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

    def _resolve_input_delivery_route(
        self,
        state: dict[str, Any],
        route: dict[str, Any] | None,
        row: dict[str, Any],
        fallback_ref: InputRef,
    ) -> tuple[str | None, str | None, InputRef | None]:
        """Resolve (parent_session_id, result_kind, start_ref) for an input row on cancel/redeliver.

        On workflow leaves (session-level ``workflow_node_result``), only inputs whose own
        ``reply_result_kind == 'workflow_node_result'`` may deliver ``workflow_node_result``;
        public routed inputs keep their own destination and ``reply_result_kind`` (never upgrading),
        and unrouted human inputs emit nothing.
        """
        if route is not None and route.get("result_kind") == "workflow_node_result":
            if (
                row.get("reply_to_session_id") is not None
                and row.get("reply_result_kind") == "workflow_node_result"
            ):
                return (
                    str(route["target"]["session_id"]),
                    "workflow_node_result",
                    InputRef(**route["input_ref"]),
                )
            if row.get("reply_to_session_id") is not None:
                start_ref = (
                    InputRef(self.store.store_id, str(state["session_id"]), str(state["start_input_message_id"]))
                    if state.get("start_input_message_id")
                    else fallback_ref
                )
                return (
                    str(row["reply_to_session_id"]),
                    str(row.get("reply_result_kind") or "agent_result"),
                    start_ref,
                )
            return None, None, None
        if route is not None:
            return (
                str(route["target"]["session_id"]),
                str(route["result_kind"]),
                InputRef(**route["input_ref"]),
            )
        return None, None, None

    @staticmethod
    def _is_driver_run_input(state, record: InputRecord) -> bool:
        """Return True for non-driver inputs or workflow_driver run/resume inputs (excluding plain late inputs)."""
        if state["session_kind"] != "workflow_driver":
            return True
        if record.row["kind"] == "workflow_continue" or record.row.get("continuation_of") is not None:
            return True
        if record.ref.message_id == state["start_input_message_id"]:
            return True
        if record.row["kind"] != "user":
            return False
        try:
            parsed = json.loads(record.row["content_json"])
        except Exception:
            return False
        return isinstance(parsed, dict) and "workflow_resume" in parsed

    @staticmethod
    def _can_redeliver_stored_result(state, row) -> bool:
        """Return False for locally-terminated closed-session inputs that never had delivery eligibility."""
        if row["terminal_status"] is None:
            return False
        if row.get("error_code") == "session_closed":
            return False
        if row.get("cancel_by") == "removed":
            return False
        if state["lifecycle"] in ("closing", "closed") and row.get("result_delivery_error") is None:
            return False
        return True

    async def _route_stored_result_tx(
        self, tx, session, route, record, activations=None, *, parent_state=None,
        _driver_events_out: list[tuple[str, str, dict[str, Any]]] | None = None,
    ):
        session_kind = dict(session).get("session_kind")
        if session_kind == "workflow_driver":
            # Follow any continuation chain (orig -> wfc_1 -> ...) to its terminal head so
            # repeat cancel(orig_ref) or cancel(driver_target) reuses the exact final send_index
            # (e.g. NULL when voided while pending) and never emits a second receipt.
            head = record
            for _ in range(10000):
                child = await tx.fetchone(
                    f"SELECT * FROM {self.q} WHERE session_id=? AND continuation_of=? "
                    "ORDER BY continuation_index DESC LIMIT 1",
                    (head.ref.session_id, head.ref.message_id),
                )
                if child is None or child["terminal_status"] is None:
                    break
                head = self.record(child)
            record = head
        row = record.row
        child_target = TargetRef(self.store.store_id, record.ref.session_id)
        parent_sid = route["target"]["session_id"]
        start_ref = InputRef(**route["input_ref"])
        result_si = row["result_send_index"]

        source_ref = record.ref
        first_meta_json: str | None = None
        if row.get("continuation_of") is not None:
            source_ref = (await self._continuation_root(tx, record)).ref
        elif session_kind == "workflow_driver":
            # Keep the terminal input's own identity (original start input OR explicit resume input).
            source_ref = record.ref
        elif result_si is not None:
            stored_source_id = record.meta.get("result_source_item_id")
            if stored_source_id:
                source_ref = InputRef(self.store.store_id, record.ref.session_id, str(stored_source_id))
            else:
                # Merged multi-input send fallback: match claim()'s [prestaged, pending] routed order.
                first_row = await tx.fetchone(
                    f"SELECT item_id,meta_json FROM {self.q} "
                    "WHERE session_id=? AND status='delivered' AND delivered_send_index=? "
                    "ORDER BY CASE WHEN reply_to_session_id IS NOT NULL THEN 0 ELSE 1 END, "
                    "CASE WHEN delivered_seq IS NOT NULL THEN delivered_seq ELSE 2147483647 END ASC, "
                    "CASE priority WHEN 'urgent' THEN 0 ELSE 1 END, id ASC LIMIT 1",
                    (record.ref.session_id, result_si),
                )
                if first_row is not None:
                    source_ref = InputRef(self.store.store_id, record.ref.session_id, str(first_row["item_id"]))
                    first_meta_json = first_row["meta_json"]

        route_start_ref = source_ref if session_kind == "workflow_driver" else start_ref
        effective_start = source_ref if result_si is None else route_start_ref
        message_id = result_message_id(child_target, effective_start, result_si, route["result_kind"])
        if row["result_delivery_error"] is None:
            existing_parent = await tx.fetchone(
                f"SELECT id FROM {self.q} WHERE session_id=? AND item_id=? LIMIT 1",
                (parent_sid, message_id),
            )
            if existing_parent is not None:
                return InputRef(self.store.store_id, parent_sid, message_id), None

        final_text, usage, structured = "", {}, None
        if row["result_seq"] is not None:
            message = await tx.fetchone(
                f"SELECT content,meta_json FROM {self.store.t.messages} "
                "WHERE session_id=? AND seq=? AND send_index=?",
                (record.ref.session_id, row["result_seq"], result_si),
            )
            if message is not None:
                final_text = message["content"] or ""
                meta = json.loads(message["meta_json"] or "{}")
                usage = meta.get("usage") or {}
                structured = meta.get("structured_output")
        else:
            stored_usage = record.meta.get("result_usage")
            if not stored_usage and first_meta_json:
                stored_usage = (json.loads(first_meta_json) or {}).get("result_usage")
            if stored_usage:
                usage = dict(stored_usage)
        if structured is None and row.get("result_json"):
            structured = json.loads(row["result_json"])
        stored_cancel_by = row.get("cancel_by") if row["terminal_status"] == "cancelled" else None
        existing = Result(
            source_ref, child_target,
            row["terminal_status"], result_si, row["result_seq"],
            final_text, usage, structured, row["error_detail"], row["error_code"],
            row["result_delivery_error"],
            cancel_by=stored_cancel_by,
        )
        child_wake = 0 if (row["terminal_status"] == "cancelled" and stored_cancel_by == "subtree") else 1
        result_ref, delivery_error = await self._river_deliver_tx(
            tx,
            child_target=child_target,
            child_tenant_id=session["tenant_id"],
            child_principal_id=session["principal_id"],
            parent_session_id=parent_sid,
            start_ref=route_start_ref,
            result=existing,
            result_kind=route["result_kind"],
            child_wake=child_wake,
            activations=activations,
            _driver_events_out=_driver_events_out,
        )
        if delivery_error != row["result_delivery_error"]:
            chain_ids: list[str] = [record.ref.message_id]
            cur_anc_id = row.get("continuation_of")
            seen_chain: set[str] = {record.ref.message_id}
            while cur_anc_id is not None and cur_anc_id not in seen_chain:
                seen_chain.add(cur_anc_id)
                chain_ids.append(cur_anc_id)
                anc_rec = await self.input(
                    tx, InputRef(record.ref.store_id, record.ref.session_id, cur_anc_id),
                )
                if anc_rec is None:
                    break
                cur_anc_id = anc_rec.row.get("continuation_of")
            if result_si is not None and delivery_error is None:
                await tx.execute(
                    f"UPDATE {self.q} SET result_delivery_error=NULL "
                    "WHERE session_id=? AND status='delivered' "
                    "AND (delivered_send_index=? OR result_send_index=?)",
                    (record.ref.session_id, result_si, result_si),
                )
            if len(chain_ids) > 1:
                c_ph = ",".join("?" for _ in chain_ids)
                await tx.execute(
                    f"UPDATE {self.q} SET result_delivery_error=? "
                    f"WHERE session_id=? AND item_id IN ({c_ph})",
                    (delivery_error, record.ref.session_id, *chain_ids),
                )
            elif not (result_si is not None and delivery_error is None):
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
        now_sql = self._now_sql()
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, target.session_id)
            await self._authorize_known(tx, target, actor, state)
            if state["lifecycle"] == "closed":
                return
            if state["execution_state"] != "idle" or int(state["wake_pending_count"]) > 0:
                raise SessionBusy(target.session_id)
            waiting = await tx.fetchone(
                f"SELECT id FROM {self.q} WHERE session_id=? "
                "AND status='waiting_continuation' LIMIT 1",
                (target.session_id,),
            )
            if waiting is not None:
                raise SessionBusy(target.session_id)
            linked_child = await tx.fetchone(
                f"SELECT session_id FROM {self.s} "
                "WHERE parent_session_id=? AND cancel_propagation='linked' "
                "AND lifecycle IN ('open','suspended') LIMIT 1",
                (target.session_id,),
            )
            if linked_child is not None:
                raise SessionBusy(target.session_id)
            await tx.execute(
                f"UPDATE {self.s} SET lifecycle='closed',pending_count=0,pending_bytes=0,"
                f"updated_at={now_sql} WHERE session_id=? AND lifecycle IN ('open','suspended','closing')",
                (target.session_id,),
            )
            if int(state["pending_count"]) > 0:
                await tx.execute(
                    f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                    f"cancel_by='removed',error_code='session_closed',completed_at={now_sql} "
                    "WHERE session_id=? AND status='pending'",
                    (target.session_id,),
                )

    # ── Unified Lifecycle & Child Subtree Rebinding (Design 217 §5.1–§5.2) ──

    async def set_lifecycle(
        self,
        target: TargetRef,
        lifecycle: str,
        *,
        actor,
        inflight: str = "abort",
        subtree: str = "none",
        reason: str | None = None,
        _with_effects: bool = False,
    ):
        """Unified session lifecycle state machine ('open' | 'suspended' | 'closed').

        - ``'closed'``: terminal soft-close (NEVER deletes ``sessions`` row); ``inflight='abort'``
          increments ``cancel_seq``; ``subtree`` cascades to active descendants level-by-level in
          ``depth DESC`` order (0 idle rows scanned).
        - ``'suspended'``: pauses new scheduling while allowing child River receipts to buffer in
          ``follow_up_queue`` with ``execution_state='idle'``; ``inflight='drain'`` does NOT bump
          ``cancel_seq`` so active tool rounds and ``Final`` complete without losing the reply.
        - ``'open'``: resumes a ``'suspended'`` session and transitions ``idle -> queued`` when
          ``wake_pending_count > 0``.
        """
        self.check_store(target)
        if lifecycle not in ("open", "suspended", "closed"):
            raise RuntimeRequestError("invalid_options", "lifecycle")
        if inflight not in ("abort", "drain"):
            raise RuntimeRequestError("invalid_options", "inflight")
        if subtree not in ("none", "linked", "all"):
            raise RuntimeRequestError("invalid_options", "subtree")

        sid = target.session_id
        now_sql = self._now_sql()
        eff_reason = reason or ("removed" if lifecycle == "closed" else lifecycle)

        for _attempt in range(8):
            stopped_map: dict[str, str] = {}
            descendants: list[str] = []
            completed_refs: list[InputRef] = []
            voided = 0
            need_wake = False
            try:
                async with self.db.transaction() as tx:
                    session = await self.authorize(tx, target, actor)
                    if session["lifecycle"] in ("closing", "closed") and lifecycle != "closed":
                        raise RuntimeRequestError("session_closed")

                    if lifecycle == "open":
                        if self._dialect == "mysql":
                            rc, packed = await tx.execute_packed(
                                f"UPDATE {self.s} SET "
                                "lifecycle=(CASE WHEN LAST_INSERT_ID("
                                "2|(CASE WHEN execution_state='idle' AND wake_pending_count>0 THEN 1 ELSE 0 END)"
                                ")>=0 THEN 'open' ELSE 'open' END),"
                                "execution_state=CASE WHEN execution_state='idle' AND wake_pending_count>0 "
                                "THEN 'queued' ELSE execution_state END,"
                                "cancel_by=CASE WHEN execution_state<>'running' THEN NULL ELSE cancel_by END,"
                                "cancel_reason=CASE WHEN execution_state<>'running' THEN NULL ELSE cancel_reason END,"
                                f"updated_at={now_sql} "
                                "WHERE session_id=? AND lifecycle='suspended'",
                                (sid,),
                            )
                            need_wake = rc == 1 and bool(packed & 1)
                            rrow = await tx.fetchone(
                                f"SELECT lifecycle,execution_state,send_index,cancel_by,cancel_reason "
                                f"FROM {self.s} WHERE session_id=?",
                                (sid,),
                            )
                            if rrow is None:
                                raise RuntimeRequestError("not_found")
                            if rrow["lifecycle"] in ("closing", "closed"):
                                raise RuntimeRequestError("session_closed")
                        else:
                            rrow = await tx.fetchone(
                                f"UPDATE {self.s} SET "
                                "lifecycle='open',"
                                "execution_state=CASE WHEN execution_state='idle' AND wake_pending_count>0 "
                                "THEN 'queued' ELSE execution_state END,"
                                "cancel_by=CASE WHEN execution_state<>'running' THEN NULL ELSE cancel_by END,"
                                "cancel_reason=CASE WHEN execution_state<>'running' THEN NULL ELSE cancel_reason END,"
                                f"updated_at={now_sql} "
                                "WHERE session_id=? AND lifecycle='suspended' "
                                "RETURNING lifecycle,execution_state,send_index,wake_pending_count,cancel_by,cancel_reason",
                                (sid,),
                            )
                            if rrow is None:
                                rrow = await tx.fetchone(
                                    f"SELECT lifecycle,execution_state,send_index,wake_pending_count,cancel_by,cancel_reason "
                                    f"FROM {self.s} WHERE session_id=?",
                                    (sid,),
                                )
                                if rrow is None:
                                    raise RuntimeRequestError("not_found")
                                if rrow["lifecycle"] in ("closing", "closed"):
                                    raise RuntimeRequestError("session_closed")
                                need_wake = False
                            else:
                                need_wake = (
                                    rrow["execution_state"] == "queued"
                                    and int(rrow["wake_pending_count"]) > 0
                                )
                        status_obj = Status(
                            None, target, rrow["execution_state"], None, None,
                            int(rrow["send_index"]) or None, False, None,
                            bool(rrow["execution_state"] == "running" and rrow["cancel_by"] is not None),
                            rrow["cancel_reason"], None, None, None, None,
                            lifecycle=rrow["lifecycle"],
                            need_wake=need_wake,
                            stopped_descendant_ids=(),
                            voided_pending_inputs=0,
                        )
                        return (status_obj, stopped_map, completed_refs) if _with_effects else status_obj

                    # lifecycle in ('suspended', 'closed')
                    desc_by_depth: dict[int, list[str]] = {}
                    expected_depth: dict[str, int] = {}
                    target_depth_snap = int(session["depth"])
                    if subtree in ("linked", "all"):
                        active_desc = await self._collect_active_descendants(
                            tx,
                            session_id=sid,
                            root_session_id=str(session["root_session_id"]),
                            target_depth=target_depth_snap,
                            only_linked=(subtree == "linked"),
                        )
                        for d_sid, d_val, _d_pre in active_desc:
                            expected_depth[d_sid] = d_val
                            desc_by_depth.setdefault(d_val, []).append(d_sid)
                        for lvl_ids in desc_by_depth.values():
                            lvl_ids.sort()

                    # Step B (Lock-only, strictly deep-first): lock all candidate descendants
                    # (depth DESC, ORDER BY session_id ASC FOR UPDATE) WITHOUT writing yet so
                    # rebound or already-idle sessions are never wrongly cancelled!
                    lock_suffix = " FOR UPDATE" if self._dialect in ("postgres", "mysql") else ""
                    locked_desc: dict[str, dict[str, Any]] = {}
                    for d_val in sorted(desc_by_depth, reverse=True):
                        lvl_ids = desc_by_depth[d_val]
                        ph = ",".join("?" for _ in lvl_ids)
                        lrows = await tx.fetchall(
                            f"SELECT session_id,parent_session_id,root_session_id,depth,"
                            f"execution_state,send_index,ready_item_json,cancel_propagation "
                            f"FROM {self.s} WHERE session_id IN ({ph}) "
                            f"ORDER BY session_id ASC{lock_suffix}",
                            tuple(lvl_ids),
                        )
                        for lr in lrows:
                            locked_desc[str(lr["session_id"])] = dict(lr)

                    # Now lock target itself (strictly shallower than all descendants)
                    locked = await self.db.dialect.lock_state(tx, self.s, sid)
                    if locked["lifecycle"] in ("closing", "closed") and lifecycle != "closed":
                        raise RuntimeRequestError("session_closed")

                    # Step C (Verify under locks): check target topology, re-scan active descendants
                    # via idx_sessions_root, and verify EVERY locked candidate is still active and
                    # still belongs to this subtree. Any mismatch rolls back with 0 writes and retries!
                    prestaged_descendants: list[str] = []
                    if subtree in ("linked", "all"):
                        if (
                            str(locked["root_session_id"]) != str(session["root_session_id"])
                            or int(locked["depth"]) != target_depth_snap
                            or len(locked_desc) != len(expected_depth)
                        ):
                            raise _SubtreeChangedRetry()
                        recheck = await self._collect_active_descendants(
                            tx,
                            session_id=sid,
                            root_session_id=str(locked["root_session_id"]),
                            target_depth=int(locked["depth"]),
                            only_linked=(subtree == "linked"),
                        )
                        recheck_ids = {d_sid for d_sid, _, _ in recheck}
                        if recheck_ids != set(locked_desc.keys()):
                            raise _SubtreeChangedRetry()
                        for d_sid, drow in locked_desc.items():
                            if (
                                str(drow["root_session_id"]) != str(locked["root_session_id"])
                                or int(drow["depth"]) != expected_depth[d_sid]
                                or drow["execution_state"] not in ("queued", "running")
                            ):
                                raise _SubtreeChangedRetry()
                            if int(drow["send_index"]) == 0 and bool(drow["ready_item_json"]):
                                prestaged_descendants.append(d_sid)
                        for d_val in sorted(desc_by_depth, reverse=True):
                            descendants.extend(desc_by_depth[d_val])

                    # Step D (Verified writes): cancel ONLY verified locked active descendants
                    for d_val in sorted(desc_by_depth, reverse=True):
                        lvl_ids = desc_by_depth[d_val]
                        ph = ",".join("?" for _ in lvl_ids)
                        await tx.execute(
                            f"UPDATE {self.s} SET cancel_seq=cancel_seq+1,cancel_by='subtree',"
                            f"cancel_reason=?,updated_at={now_sql} "
                            f"WHERE session_id IN ({ph}) AND root_session_id=? "
                            "AND execution_state IN ('queued','running')",
                            (eff_reason, *lvl_ids, str(locked["root_session_id"])),
                        )

                    if lifecycle == "suspended":
                        if inflight == "abort":
                            await tx.execute(
                                f"UPDATE {self.s} SET lifecycle='suspended',"
                                "cancel_seq=cancel_seq+1,cancel_by='suspended',cancel_reason=?,"
                                "execution_state=CASE WHEN execution_state='queued' THEN 'idle' ELSE execution_state END,"
                                f"updated_at={now_sql} "
                                "WHERE session_id=? AND lifecycle IN ('open','suspended')",
                                (eff_reason, sid),
                            )
                            stopped_map[sid] = "suspended"
                        else:
                            # inflight == 'drain': do NOT bump cancel_seq or signal local stop!
                            await tx.execute(
                                f"UPDATE {self.s} SET lifecycle='suspended',"
                                "execution_state=CASE WHEN execution_state='queued' THEN 'idle' ELSE execution_state END,"
                                f"updated_at={now_sql} "
                                "WHERE session_id=? AND lifecycle IN ('open','suspended')",
                                (sid,),
                            )
                    else:
                        # lifecycle == 'closed': soft-close, NEVER delete sessions row
                        if inflight == "abort":
                            await tx.execute(
                                f"UPDATE {self.s} SET lifecycle='closed',"
                                "cancel_seq=cancel_seq+1,cancel_by='removed',cancel_reason=?,"
                                "execution_state=CASE WHEN execution_state='queued' THEN 'idle' ELSE execution_state END,"
                                "wake_pending_count=0,pending_count=0,pending_bytes=0,ready_item_json=NULL,"
                                f"updated_at={now_sql} WHERE session_id=?",
                                (eff_reason, sid),
                            )
                            stopped_map[sid] = "removed"
                        else:
                            await tx.execute(
                                f"UPDATE {self.s} SET lifecycle='closed',"
                                "execution_state=CASE WHEN execution_state='queued' THEN 'idle' ELSE execution_state END,"
                                "wake_pending_count=0,pending_count=0,pending_bytes=0,ready_item_json=NULL,"
                                f"updated_at={now_sql} WHERE session_id=?",
                                (sid,),
                            )

                    if descendants:
                        ph = ",".join("?" for _ in descendants)
                        if prestaged_descendants:
                            pre_ph = ",".join("?" for _ in prestaged_descendants)
                            await tx.execute(
                                f"UPDATE {self.q} SET cancel_by='subtree' "
                                f"WHERE (session_id IN ({ph}) AND status='pending') "
                                f"OR (session_id IN ({pre_ph}) AND status='delivered' "
                                "AND delivered_send_index=1 AND terminal_status IS NULL)",
                                (*descendants, *prestaged_descendants),
                            )
                        else:
                            await tx.execute(
                                f"UPDATE {self.q} SET cancel_by='subtree' "
                                f"WHERE session_id IN ({ph}) AND status='pending'",
                                tuple(descendants),
                            )
                        for d_sid in descendants:
                            stopped_map[d_sid] = "subtree"

                    if lifecycle == "closed":
                        target_is_prestaged = int(locked["send_index"]) == 0 and bool(locked["ready_item_json"])
                        if target_is_prestaged:
                            void_rows = await tx.fetchall(
                                f"SELECT item_id FROM {self.q} WHERE session_id=? "
                                "AND (status='pending' OR (status='delivered' AND delivered_send_index=1 "
                                "AND terminal_status IS NULL))",
                                (sid,),
                            )
                            voided = await tx.execute(
                                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                                "cancel_by='removed',error_code='session_closed',error_detail=?,"
                                f"delivered_send_index=NULL,delivered_seq=NULL,delivered_at=NULL,completed_at={now_sql} "
                                "WHERE session_id=? "
                                "AND (status='pending' OR (status='delivered' AND delivered_send_index=1 "
                                "AND terminal_status IS NULL))",
                                (eff_reason, sid),
                            )
                        else:
                            void_rows = await tx.fetchall(
                                f"SELECT item_id FROM {self.q} WHERE session_id=? AND status='pending'",
                                (sid,),
                            )
                            voided = await tx.execute(
                                f"UPDATE {self.q} SET status='void',terminal_status='cancelled',"
                                "cancel_by='removed',error_code='session_closed',error_detail=?,"
                                f"completed_at={now_sql} WHERE session_id=? AND status='pending'",
                                (eff_reason, sid),
                            )
                        for vr in void_rows:
                            completed_refs.append(InputRef(target.store_id, sid, vr["item_id"]))

                    post_exec = (
                        "idle" if locked["execution_state"] == "queued"
                        else str(locked["execution_state"])
                    )
                    status_obj = Status(
                        None, target, post_exec, None, None,
                        int(locked["send_index"]) or None, False, None,
                        bool(sid in stopped_map and post_exec == "running"),
                        eff_reason if sid in stopped_map else locked["cancel_reason"],
                        None, None, None, None,
                        lifecycle=lifecycle,
                        need_wake=False,
                        stopped_descendant_ids=tuple(descendants),
                        voided_pending_inputs=int(voided),
                    )
                break
            except _SubtreeChangedRetry:
                continue
        else:
            raise RuntimeRequestError("request_conflict", "concurrent subtree mutation")
        return (status_obj, stopped_map, completed_refs) if _with_effects else status_obj

    async def deactivate(
        self,
        target: TargetRef,
        *,
        actor,
        stop_subtree: bool = False,
        reason: str = "removed",
        _with_effects: bool = False,
    ):
        """Thin alias for ``set_lifecycle(..., 'closed', inflight='abort', subtree=...)``."""
        return await self.set_lifecycle(
            target,
            "closed",
            actor=actor,
            inflight="abort",
            subtree="all" if stop_subtree else "none",
            reason=reason,
            _with_effects=_with_effects,
        )

    async def rebind_children(
        self,
        from_parent: TargetRef,
        to_parent: TargetRef,
        *,
        child_targets=None,
        replay_undelivered: bool = True,
        rewrite_host_context=None,
        actor,
    ) -> RebindResult:
        """Atomic child subtree rebinding enforcing all 3 Design 217 §5.2 hard invariants.

        1. Rejects when ``from_parent`` has an active synchronous tool wait.
        2. Enforces same ``(tenant_id, principal_id)``, ``to_parent.lifecycle IN ('open','suspended')``,
           cycle immunity, and BFS-recursively updates ``root_session_id`` and relative ``depth``
           for ALL descendants (including ``idle`` descendants) + optional ``rewrite_host_context``.
        3. When ``replay_undelivered=True``, atomically replays ``result_delivery_error='session_closed'``
           orphan receipts into ``to_parent`` and clears the error flag.
        """
        import inspect

        from power_loop.runtime.history_sanitize import find_dangling_tool_calls

        self.check_store(from_parent)
        self.check_store(to_parent)
        fp_sid = from_parent.session_id
        tp_sid = to_parent.session_id
        if fp_sid == tp_sid:
            raise RuntimeRequestError("request_conflict", "cycle detected: from_parent == to_parent")

        filter_ids: list[str] | None = None
        if child_targets is not None:
            filter_ids = []
            for item in child_targets:
                if isinstance(item, TargetRef):
                    self.check_store(item)
                    filter_ids.append(item.session_id)
                else:
                    filter_ids.append(str(item))

        now_sql = self._now_sql()
        suffix = " FOR UPDATE" if self._dialect in ("postgres", "mysql") else ""

        for _attempt in range(8):
            need_wake = False
            try:
                async with self.db.transaction() as tx:
                    # Step 1 (Unlocked discovery): read parents + entire descendant subtree so we
                    # can lock ALL affected rows in strict global deep-first order (-depth, session_id),
                    # eliminating AB-BA deadlocks with concurrent Final / cancel / set_lifecycle.
                    disc_pair = await tx.fetchall(
                        f"SELECT * FROM {self.s} WHERE session_id IN (?,?) ORDER BY session_id ASC",
                        (fp_sid, tp_sid),
                    )
                    disc_by_p = {r["session_id"]: dict(r) for r in disc_pair}
                    if fp_sid not in disc_by_p or tp_sid not in disc_by_p:
                        raise RuntimeRequestError("not_found")
                    disc_fp, disc_tp = disc_by_p[fp_sid], disc_by_p[tp_sid]

                    if (disc_fp["tenant_id"], disc_fp["principal_id"]) != (actor.tenant_id, actor.principal_id):
                        raise RuntimeRequestError("forbidden")
                    if (disc_tp["tenant_id"], disc_tp["principal_id"]) != (disc_fp["tenant_id"], disc_fp["principal_id"]):
                        raise RuntimeRequestError("forbidden", "cross-tenant/principal rebind forbidden")
                    if disc_tp["lifecycle"] not in ("open", "suspended"):
                        raise RuntimeRequestError("session_closed")

                    if filter_ids is not None:
                        if not filter_ids:
                            return RebindResult(
                                from_parent=from_parent,
                                to_parent=to_parent,
                                rebound_children=(),
                                updated_descendants=(),
                                replayed_results=0,
                                new_root_session_id=str(disc_tp["root_session_id"]),
                                depth_delta=0,
                                need_wake=False,
                            )
                        ph = ",".join("?" for _ in filter_ids)
                        disc_direct = [
                            dict(r) for r in await tx.fetchall(
                                f"SELECT * FROM {self.s} WHERE parent_session_id=? AND session_id IN ({ph}) "
                                "ORDER BY session_id ASC",
                                (fp_sid, *filter_ids),
                            )
                        ]
                    else:
                        disc_direct = [
                            dict(r) for r in await tx.fetchall(
                                f"SELECT * FROM {self.s} WHERE parent_session_id=? ORDER BY session_id ASC",
                                (fp_sid,),
                            )
                        ]

                    # BFS collect ENTIRE descendant subtree (INCLUDING idle descendants!)
                    disc_all: list[dict[str, Any]] = list(disc_direct)
                    frontier = list(disc_direct)
                    seen_ids: set[str] = {r["session_id"] for r in disc_direct}
                    while frontier:
                        p_ids = [r["session_id"] for r in frontier]
                        p_ph = ",".join("?" for _ in p_ids)
                        nxt = [
                            dict(r) for r in await tx.fetchall(
                                f"SELECT * FROM {self.s} WHERE parent_session_id IN ({p_ph}) ORDER BY session_id ASC",
                                tuple(p_ids),
                            )
                        ]
                        frontier = [r for r in nxt if r["session_id"] not in seen_ids]
                        for r in frontier:
                            seen_ids.add(r["session_id"])
                            disc_all.append(r)

                    if tp_sid in seen_ids:
                        raise RuntimeRequestError(
                            "request_conflict", f"cycle detected: to_parent {tp_sid} is a descendant"
                        )

                    # Step 2 (Unified deep-first locking): lock all descendants + (from_parent, to_parent)
                    # strictly by depth DESC, session_id ASC FOR UPDATE!
                    lock_by_depth: dict[int, list[str]] = {}
                    expected_depth: dict[str, int] = {}
                    expected_parent: dict[str, str | None] = {}
                    for r in (*disc_all, disc_fp, disc_tp):
                        rsid = str(r["session_id"])
                        if rsid in expected_depth:
                            continue
                        d_val = int(r["depth"])
                        expected_depth[rsid] = d_val
                        expected_parent[rsid] = r["parent_session_id"]
                        lock_by_depth.setdefault(d_val, []).append(rsid)

                    locked_by_id: dict[str, dict[str, Any]] = {}
                    for d_val in sorted(lock_by_depth, reverse=True):
                        lvl_ids = sorted(lock_by_depth[d_val])
                        l_ph = ",".join("?" for _ in lvl_ids)
                        lrows = await tx.fetchall(
                            f"SELECT * FROM {self.s} WHERE session_id IN ({l_ph}) ORDER BY session_id ASC{suffix}",
                            tuple(lvl_ids),
                        )
                        for lr in lrows:
                            locked_by_id[str(lr["session_id"])] = dict(lr)

                    if fp_sid not in locked_by_id or tp_sid not in locked_by_id:
                        raise RuntimeRequestError("not_found")
                    if len(locked_by_id) != len(expected_depth):
                        raise _RebindChangedRetry()
                    for rsid, exp_d in expected_depth.items():
                        cur_r = locked_by_id[rsid]
                        if int(cur_r["depth"]) != exp_d or cur_r["parent_session_id"] != expected_parent[rsid]:
                            raise _RebindChangedRetry()

                    fp, tp = locked_by_id[fp_sid], locked_by_id[tp_sid]

                    # Re-check under locks that no concurrent child/descendant was added in the discovery gap
                    if filter_ids is not None:
                        ph = ",".join("?" for _ in filter_ids)
                        re_direct = await tx.fetchall(
                            f"SELECT session_id FROM {self.s} WHERE parent_session_id=? AND session_id IN ({ph})",
                            (fp_sid, *filter_ids),
                        )
                    else:
                        re_direct = await tx.fetchall(
                            f"SELECT session_id FROM {self.s} WHERE parent_session_id=?",
                            (fp_sid,),
                        )
                    re_direct_ids = {str(r["session_id"]) for r in re_direct}
                    if re_direct_ids != {str(r["session_id"]) for r in disc_direct}:
                        raise _RebindChangedRetry()
                    if seen_ids:
                        check_ids = list(seen_ids)
                        c_ph = ",".join("?" for _ in check_ids)
                        re_sub = await tx.fetchall(
                            f"SELECT session_id FROM {self.s} WHERE parent_session_id IN ({c_ph})",
                            tuple(check_ids),
                        )
                        if any(str(r["session_id"]) not in seen_ids for r in re_sub):
                            raise _RebindChangedRetry()

                    # Invariant 2a (re-verified under locks): tenant/principal + actor auth + to_parent lifecycle
                    if (fp["tenant_id"], fp["principal_id"]) != (actor.tenant_id, actor.principal_id):
                        raise RuntimeRequestError("forbidden")
                    if (tp["tenant_id"], tp["principal_id"]) != (fp["tenant_id"], fp["principal_id"]):
                        raise RuntimeRequestError("forbidden", "cross-tenant/principal rebind forbidden")
                    if actor.session_id is not None:
                        await self._authorize_known(tx, from_parent, actor, fp)
                        await self._authorize_known(tx, to_parent, actor, tp)
                    if tp["lifecycle"] not in ("open", "suspended"):
                        raise RuntimeRequestError("session_closed")

                    # Invariant 1 (under locked from_parent): forbid rebind while from_parent has sync tool wait
                    if fp["execution_state"] == "running":
                        if fp["pending_tool_seq"] is not None:
                            raise RuntimeRequestError(
                                "request_conflict", "from_parent has active synchronous tool wait"
                            )
                        fp_msgs = await tx.fetchall(
                            f"SELECT seq,round_index,role,tool_calls_json,tool_call_id "
                            f"FROM {self.store.t.messages} WHERE session_id=? AND send_index=? ORDER BY seq ASC",
                            (fp_sid, int(fp["send_index"])),
                        )
                        if fp_msgs:
                            norm_fp = [
                                {
                                    "seq": int(m["seq"]),
                                    "round_index": int(m["round_index"]) if m["round_index"] is not None else 0,
                                    "role": m["role"],
                                    "tool_calls": json.loads(m["tool_calls_json"]) if m["tool_calls_json"] else None,
                                    "tool_call_id": m["tool_call_id"],
                                }
                                for m in fp_msgs
                            ]
                            if find_dangling_tool_calls(norm_fp):
                                raise RuntimeRequestError(
                                    "request_conflict", "from_parent has active synchronous tool wait"
                                )

                    direct_children = [locked_by_id[str(r["session_id"])] for r in disc_direct]
                    all_moved = [locked_by_id[str(r["session_id"])] for r in disc_all]

                    if not direct_children:
                        return RebindResult(
                            from_parent=from_parent,
                            to_parent=to_parent,
                            rebound_children=(),
                            updated_descendants=(),
                            replayed_results=0,
                            new_root_session_id=str(tp["root_session_id"]),
                            depth_delta=0,
                            need_wake=False,
                        )

                    new_root = str(tp["root_session_id"])
                    tp_depth = int(tp["depth"])
                    old_child_depth = int(direct_children[0]["depth"])
                    depth_delta = (tp_depth + 1) - old_child_depth

                    direct_ids = [r["session_id"] for r in direct_children]
                    d_ph = ",".join("?" for _ in direct_ids)
                    all_ids = [r["session_id"] for r in all_moved]
                    a_ph = ",".join("?" for _ in all_ids)

                    # Update direct children's parent_session_id and result_route_json
                    for child in direct_children:
                        new_route_json = child["result_route_json"]
                        if new_route_json:
                            route_obj = json.loads(new_route_json)
                            if (
                                isinstance(route_obj, dict)
                                and isinstance(route_obj.get("target"), dict)
                                and route_obj["target"].get("session_id") == fp_sid
                            ):
                                route_obj["target"] = {**route_obj["target"], "session_id": tp_sid}
                                new_route_json = canonical(route_obj)
                                child["result_route_json"] = new_route_json
                        child["parent_session_id"] = tp_sid
                        await tx.execute(
                            f"UPDATE {self.s} SET parent_session_id=?,result_route_json=?,updated_at={now_sql} "
                            "WHERE session_id=?",
                            (tp_sid, new_route_json, child["session_id"]),
                        )

                    # Recursively update root_session_id & relative depth for ALL descendants (incl. idle)
                    await tx.execute(
                        f"UPDATE {self.s} SET root_session_id=?,depth=depth+?,updated_at={now_sql} "
                        f"WHERE session_id IN ({a_ph})",
                        (new_root, depth_delta, *all_ids),
                    )
                    for node in all_moved:
                        node["root_session_id"] = new_root
                        node["depth"] = int(node["depth"]) + depth_delta

                    # 1) Always retarget direct children's active per-input routes pointing to
                    #    from_parent -> to_parent (preserves direct-parent rebind even on same-tree
                    #    rebinds where from_parent remains an ancestor of to_parent).
                    await tx.execute(
                        f"UPDATE {self.q} SET reply_to_session_id=? "
                        f"WHERE session_id IN ({d_ph}) AND reply_to_session_id=? "
                        "AND status IN ('pending','delivered') AND terminal_status IS NULL",
                        (tp_sid, *direct_ids, fp_sid),
                    )

                    # 2) Retarget active per-input reply routes across the entire moved subtree
                    #    that pointed to any stale ancestor above from_parent no longer in
                    #    to_parent's ancestor chain (preserves explicit routes to shared ancestors).
                    async def _ancestors_of(start_row: dict[str, Any]) -> set[str]:
                        anc: set[str] = {str(start_row["session_id"])}
                        cur = start_row.get("parent_session_id")
                        while cur is not None and str(cur) not in anc:
                            cur_s = str(cur)
                            anc.add(cur_s)
                            prow = await tx.fetchone(
                                f"SELECT parent_session_id FROM {self.s} WHERE session_id=?",
                                (cur_s,),
                            )
                            cur = prow["parent_session_id"] if prow is not None else None
                        return anc

                    stale_ancestors = sorted((await _ancestors_of(fp)) - (await _ancestors_of(tp)))
                    if stale_ancestors:
                        st_ph = ",".join("?" for _ in stale_ancestors)
                        await tx.execute(
                            f"UPDATE {self.q} SET reply_to_session_id=? "
                            f"WHERE session_id IN ({a_ph}) AND reply_to_session_id IN ({st_ph}) "
                            "AND status IN ('pending','delivered') AND terminal_status IS NULL",
                            (tp_sid, *all_ids, *stale_ancestors),
                        )

                    # Optional host context rewrite callback across every moved node
                    if rewrite_host_context is not None:
                        tp_desc = json.loads(tp["descriptor_json"])
                        parent_info = {
                            **tp,
                            "descriptor": tp_desc,
                            "host_context": dict(tp_desc.get("host_context") or {}),
                        }
                        for node in all_moved:
                            cur_desc = json.loads(node["descriptor_json"])
                            child_info = {
                                **node,
                                "descriptor": cur_desc,
                                "host_context": dict(cur_desc.get("host_context") or {}),
                            }
                            rewritten = rewrite_host_context(child_info, parent_info)
                            if inspect.isawaitable(rewritten):
                                rewritten = await rewritten
                            if rewritten is None:
                                continue
                            if isinstance(rewritten, RuntimeDescriptor):
                                new_desc_obj = rewritten
                            elif isinstance(rewritten, dict) and {
                                "schema_version", "definition_ref",
                            } <= rewritten.keys():
                                new_desc_obj = RuntimeDescriptor(**rewritten)
                            elif isinstance(rewritten, dict):
                                new_desc_obj = RuntimeDescriptor(**{**cur_desc, "host_context": rewritten})
                            else:
                                raise RuntimeRequestError("invalid_options", "rewrite_host_context return type")
                            nd_json = canonical(new_desc_obj)
                            nd_hash = sha256_canonical(new_desc_obj)
                            if node.get("next_descriptor_json"):
                                staged = json.loads(node["next_descriptor_json"])
                                staged_obj = RuntimeDescriptor(
                                    **{**staged, "host_context": plain(new_desc_obj.host_context)}
                                )
                                await tx.execute(
                                    f"UPDATE {self.s} SET descriptor_json=?,descriptor_hash=?,"
                                    "next_descriptor_json=?,next_descriptor_hash=?,"
                                    f"updated_at={now_sql} WHERE session_id=?",
                                    (nd_json, nd_hash, canonical(staged_obj), sha256_canonical(staged_obj),
                                     node["session_id"]),
                                )
                            else:
                                await tx.execute(
                                    f"UPDATE {self.s} SET descriptor_json=?,descriptor_hash=?,"
                                    f"updated_at={now_sql} WHERE session_id=?",
                                    (nd_json, nd_hash, node["session_id"]),
                                )

                    # Invariant 3: atomically replay orphan 'session_closed' receipts to to_parent
                    # enforcing River's exact item-count and byte capacity limits (rolls back on overflow!).
                    replayed = 0
                    if replay_undelivered:
                        child_by_id = {c["session_id"]: c for c in direct_children}
                        undeliv = await tx.fetchall(
                            f"SELECT id,session_id,item_id,delivered_send_index,terminal_status,"
                            "cancel_by,result_json,result_send_index,result_seq,error_code,error_detail,"
                            "reply_to_session_id,reply_result_kind "
                            f"FROM {self.q} WHERE session_id IN ({d_ph}) "
                            "AND result_delivery_error='session_closed' ORDER BY id ASC",
                            tuple(direct_ids),
                        )
                        # Group undelivered rows by settled send (c_sid, c_si) so a merged multi-input
                        # batch replays as ONE receipt matching the batch's authoritative result_kind
                        # regardless of input row order or session_id length; pending-cancelled rows
                        # (c_si is None) stay keyed per item_id.
                        grouped_undeliv: dict[tuple[str, int | None, str | None], list[dict[str, Any]]] = {}
                        for row_u in undeliv:
                            g_sid = str(row_u["session_id"])
                            g_si_raw = row_u["result_send_index"] or row_u["delivered_send_index"]
                            g_si = int(g_si_raw) if g_si_raw is not None else None
                            g_key = (g_sid, g_si, None if g_si is not None else str(row_u["item_id"]))
                            grouped_undeliv.setdefault(g_key, []).append(dict(row_u))

                        for (c_sid, c_si, _pending_item), group_rows in grouped_undeliv.items():
                            c_row = child_by_id[c_sid]
                            c_route = json.loads(c_row["result_route_json"]) if c_row.get("result_route_json") else None
                            wf_row = next(
                                (
                                    r for r in group_rows
                                    if r.get("reply_to_session_id") is not None
                                    and r.get("reply_result_kind") == "workflow_node_result"
                                ),
                                None,
                            )
                            routed_row = next(
                                (r for r in group_rows if r.get("reply_to_session_id") is not None),
                                None,
                            )
                            if isinstance(c_route, dict) and c_route.get("result_kind") == "workflow_node_result":
                                if wf_row is not None:
                                    urow = wf_row
                                    c_result_kind = "workflow_node_result"
                                else:
                                    urow = routed_row or group_rows[0]
                                    urow_rk = urow.get("reply_result_kind")
                                    c_result_kind = (
                                        urow_rk if urow_rk in _VALID_RESULT_KINDS else "agent_result"
                                    )
                            elif isinstance(c_route, dict) and c_route.get("result_kind") in _VALID_RESULT_KINDS:
                                urow = wf_row or routed_row or group_rows[0]
                                c_result_kind = c_route["result_kind"]
                            else:
                                urow = routed_row or group_rows[0]
                                urow_rk = urow.get("reply_result_kind")
                                c_result_kind = (
                                    urow_rk if urow_rk in _VALID_RESULT_KINDS else "agent_result"
                                )
                            if c_si is not None:
                                raw_replay_id = f"res_rebind_{c_sid}_s{c_si}"
                                replay_id = (
                                    raw_replay_id if len(raw_replay_id) <= 64
                                    else "res_rebind_" + sha256_canonical([c_sid, c_si])[:48]
                                )
                            else:
                                replay_id = (
                                    "res_rebind_"
                                    + sha256_canonical([c_sid, str(urow["item_id"]), None])[:48]
                                )
                            group_row_ids = [int(r["id"]) for r in group_rows]
                            g_id_ph = ",".join("?" for _ in group_row_ids)
                            orig_txt = ""
                            for r_txt in (urow, *group_rows):
                                if r_txt["result_json"]:
                                    try:
                                        parsed_rj = json.loads(r_txt["result_json"])
                                        if isinstance(parsed_rj, dict):
                                            orig_txt = str(parsed_rj.get("final_text") or parsed_rj.get("text") or "")
                                            if orig_txt:
                                                break
                                    except ValueError:
                                        pass
                            if not orig_txt and c_si is not None:
                                mrow = await tx.fetchone(
                                    f"SELECT content FROM {self.store.t.messages} "
                                    "WHERE session_id=? AND send_index=? AND role='assistant' "
                                    "ORDER BY seq DESC LIMIT 1",
                                    (c_sid, c_si),
                                )
                                if mrow is not None and mrow["content"]:
                                    orig_txt = str(mrow["content"])
                            header = f"[接管自前任会话 {fp_sid} 的子任务结果]"
                            combined_txt = f"{header}\n{orig_txt}".strip() if orig_txt else header
                            body = {
                                "child_target": plain(TargetRef(self.store.store_id, c_sid)),
                                "input_ref": plain(InputRef(self.store.store_id, c_sid, urow["item_id"])),
                                "source_session_id": c_sid,
                                "rebound_from_parent": fp_sid,
                                "status": urow["terminal_status"] or "completed",
                                "cancel_by": urow["cancel_by"],
                                "final_text": combined_txt,
                                "text": combined_txt,
                                "send_index": c_si,
                                "result_seq": urow["result_seq"],
                                "error": urow["error_detail"],
                                "error_code": urow["error_code"],
                            }
                            b_json = canonical(body)
                            b_meta = canonical({
                                "accepted_protocol": PROTOCOL_VERSION,
                                "rebound_from_parent": fp_sid,
                                "source_session_id": c_sid,
                            })
                            b_bytes = len(b_json.encode()) + len(b_meta.encode())
                            b_hash = self.request_hash(to_parent, body, "normal")
                            if self._dialect == "mysql":
                                rc_rp, rp_packed = await tx.execute_packed(
                                    f"UPDATE {self.s} SET "
                                    "inbox_version=(LAST_INSERT_ID("
                                    "((inbox_version+1)<<1)|"
                                    "(CASE WHEN lifecycle='open' AND execution_state='idle' THEN 1 ELSE 0 END)"
                                    ")>>1),"
                                    "wake_inbox_seq=wake_inbox_seq+1,"
                                    "wake_pending_count=wake_pending_count+1,"
                                    "pending_count=pending_count+1,"
                                    "pending_bytes=pending_bytes+?,"
                                    "execution_state=CASE WHEN lifecycle='open' AND execution_state='idle' "
                                    "THEN 'queued' ELSE execution_state END,"
                                    f"updated_at={now_sql} "
                                    "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                                    "AND tenant_id=? AND principal_id=? "
                                    "AND pending_count<? AND (pending_bytes+?)<=? "
                                    f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?)",
                                    (
                                        b_bytes, tp_sid, tp["tenant_id"], tp["principal_id"],
                                        self.hard_pending_limit, b_bytes, self.inbox_byte_limit,
                                        tp_sid, replay_id,
                                    ),
                                )
                                ok_rp = rc_rp == 1
                                rp_seq = (rp_packed >> 1) if ok_rp else 0
                                if ok_rp and bool(rp_packed & 1):
                                    need_wake = True
                            else:
                                rprow = await tx.fetchone(
                                    f"UPDATE {self.s} SET "
                                    "inbox_version=inbox_version+1,"
                                    "wake_inbox_seq=wake_inbox_seq+1,"
                                    "wake_pending_count=wake_pending_count+1,"
                                    "pending_count=pending_count+1,"
                                    "pending_bytes=pending_bytes+?,"
                                    "execution_state=CASE WHEN lifecycle='open' AND execution_state='idle' "
                                    "THEN 'queued' ELSE execution_state END,"
                                    f"updated_at={now_sql} "
                                    "WHERE session_id=? AND lifecycle IN ('open','suspended') "
                                    "AND tenant_id=? AND principal_id=? "
                                    "AND pending_count<? AND (pending_bytes+?)<=? "
                                    f"AND NOT EXISTS (SELECT 1 FROM {self.q} WHERE session_id=? AND item_id=?) "
                                    "RETURNING inbox_version,lifecycle,execution_state,wake_pending_count",
                                    (
                                        b_bytes, tp_sid, tp["tenant_id"], tp["principal_id"],
                                        self.hard_pending_limit, b_bytes, self.inbox_byte_limit,
                                        tp_sid, replay_id,
                                    ),
                                )
                                ok_rp = rprow is not None
                                rp_seq = int(rprow["inbox_version"]) if ok_rp else 0
                                if (
                                    ok_rp
                                    and rprow["lifecycle"] == "open"
                                    and rprow["execution_state"] == "queued"
                                    and int(rprow["wake_pending_count"]) == 1
                                ):
                                    need_wake = True
                            if ok_rp:
                                await tx.execute(
                                    f"INSERT INTO {self.q} ("
                                    "session_id,item_id,inbox_version,kind,priority,wake,status,"
                                    "content_json,content_bytes,meta_json,request_hash,result_kind,created_at) "
                                    f"VALUES (?,?,?,?,'normal',1,'pending',?,?,?,?,?,{now_sql})",
                                    (
                                        tp_sid, replay_id, rp_seq, c_result_kind,
                                        b_json, b_bytes, b_meta, b_hash, c_result_kind,
                                    ),
                                )
                                await tx.execute(
                                    f"UPDATE {self.q} SET result_delivery_error=NULL WHERE id IN ({g_id_ph})",
                                    tuple(group_row_ids),
                                )
                                replayed += 1
                            else:
                                dup = await tx.fetchone(
                                    f"SELECT id FROM {self.q} WHERE session_id=? AND item_id=?",
                                    (tp_sid, replay_id),
                                )
                                if dup is not None:
                                    await tx.execute(
                                        f"UPDATE {self.q} SET result_delivery_error=NULL WHERE id IN ({g_id_ph})",
                                        tuple(group_row_ids),
                                    )
                                else:
                                    raise CapacityError()
                break
            except _RebindChangedRetry:
                continue
        else:
            raise RuntimeRequestError("request_conflict", "concurrent rebind mutation")

        return RebindResult(
            from_parent=from_parent,
            to_parent=to_parent,
            rebound_children=tuple(direct_ids),
            updated_descendants=tuple(all_ids),
            replayed_results=replayed,
            new_root_session_id=new_root,
            depth_delta=depth_delta,
            need_wake=need_wake,
        )

    # ── Context Primitives: context_budget / seed_context / repair (Design 217 §4.3–§4.4) ──

    async def context_budget(
        self,
        target_or_descriptor: TargetRef | RuntimeDescriptor,
        *,
        actor=None,
    ) -> ContextBudget:
        """Compute context token budget from a ``RuntimeDescriptor`` (0 SQL) or ``TargetRef`` (1 SQL)."""
        from power_loop.runtime.budget import estimate_text_tokens
        from power_loop.runtime.store.projection_fold import parse_state

        history_tokens = 0
        sends_count = 0
        compact_to_send = None
        if isinstance(target_or_descriptor, RuntimeDescriptor):
            descriptor = target_or_descriptor
        else:
            self.check_store(target_or_descriptor)
            row = await self.db.fetchone(
                f"SELECT session_id,tenant_id,principal_id,parent_session_id,"
                f"descriptor_json,fold_stats_json FROM {self.s} WHERE session_id=?",
                (target_or_descriptor.session_id,),
            )
            if row is None:
                raise RuntimeRequestError("not_found")
            if actor is not None:
                await self._authorize_known(self.db, target_or_descriptor, actor, row)
            descriptor = RuntimeDescriptor(**json.loads(row["descriptor_json"]))
            stats = parse_state(row["fold_stats_json"])
            if stats is not None:
                history_tokens = max(0, int(stats.get("tokens") or 0))
                sends_count = max(0, int(stats.get("sends") or 0))
                cts = stats.get("compact_to_send")
                compact_to_send = int(cts) if cts is not None else None

        from collections.abc import Mapping

        cfg = descriptor.config if isinstance(descriptor.config, Mapping) else {}
        opts = cfg.get("child_options", cfg) if isinstance(cfg.get("child_options"), Mapping) else cfg
        model = str(opts.get("model") or cfg.get("model") or "") or None
        total_tokens = int(
            opts.get("context_budget_tokens")
            or cfg.get("context_budget_tokens")
            or opts.get("max_context_tokens")
            or cfg.get("max_context_tokens")
            or 128_000
        )
        raw_max_tokens = opts.get("max_tokens") if opts.get("max_tokens") is not None else cfg.get("max_tokens")
        max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else None
        raw_max_rows = (
            opts.get("max_context_rows")
            if opts.get("max_context_rows") is not None
            else cfg.get("max_context_rows")
        )
        max_context_rows = int(raw_max_rows) if raw_max_rows is not None else None
        system_prompt = opts.get("system_prompt") or cfg.get("system_prompt") or ""
        sys_tokens = estimate_text_tokens(str(system_prompt)) if system_prompt else 0
        reserve_output = max_tokens if max_tokens is not None else 0
        used_tokens = sys_tokens + history_tokens
        remaining_tokens = max(0, total_tokens - reserve_output - used_tokens)
        return ContextBudget(
            model=model,
            context_budget_tokens=total_tokens,
            max_tokens=max_tokens,
            max_context_rows=max_context_rows,
            used_tokens=used_tokens,
            remaining_tokens=remaining_tokens,
            sends=sends_count,
            compact_to_send=compact_to_send,
        )

    async def seed_context(
        self,
        target: TargetRef,
        messages: list[dict[str, Any]],
        *,
        actor,
        overflow: str = "truncate_oldest",
        max_tokens: int | None = None,
        summary_fn=None,
    ) -> dict[str, Any]:
        """Seed initial/injected history into an idle session (Design 217 §4.3 / Probe 5).

        - Empty session (``send_index=0`` and no prior ``(0, 'seed')``): writes ``send_index=0,
          kind='seed'`` and bumps ``inbox_version`` so Turn 1 sees the seed while unseeded cold
          Turn 1 still pays 0 extra DB reads.
        - Second seed on an empty session OR any seed on a non-empty session: increments
          ``send_index = cur_send_index + 1`` so ``uq_project_send_kind`` never collides and
          earlier turns are never overwritten.
        - Supports ``overflow`` in ``{'truncate_oldest', 'compact', 'reject'}``.
        """
        import inspect

        from power_loop.contracts.errors import SessionBusy
        from power_loop.runtime.budget import estimate_tokens
        from power_loop.runtime.history_sanitize import sanitize_request_messages

        self.check_store(target)
        if overflow not in {"truncate_oldest", "compact", "reject"}:
            raise RuntimeRequestError("invalid_options", "overflow")
        if not isinstance(messages, list) or not messages:
            raise RuntimeRequestError("invalid_options", "messages must be a non-empty list")

        cleaned = sanitize_request_messages([dict(m) for m in messages])
        if not cleaned:
            raise RuntimeRequestError("invalid_options", "messages produced no valid entries")

        now_sql = self._now_sql()
        sid = target.session_id
        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, sid)
            await self._authorize_known(tx, target, actor, state)
            if state["lifecycle"] != "open":
                if state["lifecycle"] == "suspended":
                    raise RuntimeRequestError("session_suspended")
                raise RuntimeRequestError("session_closed")
            if (
                state["execution_state"] != "idle"
                or int(state["pending_count"]) > 0
                or int(state["wake_pending_count"]) > 0
            ):
                raise SessionBusy(sid)

            from collections.abc import Mapping

            descriptor = RuntimeDescriptor(**json.loads(state["descriptor_json"]))
            cfg = descriptor.config if isinstance(descriptor.config, Mapping) else {}
            budget = int(
                max_tokens
                if max_tokens is not None
                else (cfg.get("context_budget_tokens") or cfg.get("max_context_tokens") or 128_000)
            )
            kept = list(cleaned)
            est = estimate_tokens(kept)
            truncated = False
            if est > budget:
                if overflow == "reject":
                    raise RuntimeRequestError(
                        "context_budget_exceeded",
                        f"seed messages ({est} tokens) exceed budget ({budget})",
                    )
                if overflow == "truncate_oldest":
                    truncated = True
                    while len(kept) > 1 and estimate_tokens(kept) > budget:
                        dropped = kept.pop(0)
                        if dropped.get("role") == "assistant" and dropped.get("tool_calls"):
                            while len(kept) > 1 and kept[0].get("role") == "tool":
                                kept.pop(0)
                    kept = sanitize_request_messages(kept) or [cleaned[-1]]
                    est = estimate_tokens(kept)
                else:
                    # overflow == "compact"
                    truncated = True
                    if summary_fn is not None:
                        res = summary_fn(kept)
                        summary_text = str(await res if inspect.isawaitable(res) else res)
                    else:
                        lines = [
                            f"{m.get('role', 'user')}: {str(m.get('content') or '')[:200]}"
                            for m in kept
                        ]
                        summary_text = "[seed_compact]\n" + "\n".join(lines)
                        max_chars = max(64, budget * 3)
                        if len(summary_text) > max_chars:
                            summary_text = summary_text[:max_chars]
                    kept = [{"role": "user", "content": summary_text}]
                    est = estimate_tokens(kept)

            cur_si = int(state["send_index"])
            existing_zero = None
            if cur_si == 0:
                existing_zero = await tx.fetchone(
                    f"SELECT 1 AS present FROM {self.store.t.project_messages} "
                    "WHERE session_id=? AND send_index=0 AND kind='seed' LIMIT 1",
                    (sid,),
                )
            if cur_si == 0 and existing_zero is None:
                seed_si = 0
                new_send_index = 0
            else:
                seed_si = cur_si + 1
                new_send_index = seed_si

            max_row = await tx.fetchone(
                f"SELECT COALESCE(MAX(seq),0) AS max_seq FROM {self.store.t.messages} WHERE session_id=?",
                (sid,),
            )
            seq = max(int(state["next_seq"]), int(max_row["max_seq"]) + 1 if max_row else 1)
            seq_lo = seq
            database_now = await self._db_now(tx)
            for msg in kept:
                role = str(msg.get("role") or "user")
                content_val = msg.get("content")
                if content_val is not None and not isinstance(content_val, str):
                    content_str = canonical(content_val)
                    meta_obj: dict[str, Any] = {"seeded": True, "content_encoding": "json"}
                else:
                    content_str = content_val
                    meta_obj = {"seeded": True}
                tc_json = canonical(msg["tool_calls"]) if msg.get("tool_calls") else None
                await tx.execute(
                    f"INSERT INTO {self.store.t.messages} ("
                    "session_id,seq,role,name,content,tool_calls_json,tool_call_id,"
                    "round_index,state,meta_json,send_index,created_at) "
                    "VALUES (?,?,?,?,?,?,?,0,'active',?,?,?)",
                    (
                        sid, seq, role, msg.get("name"), content_str, tc_json,
                        msg.get("tool_call_id"), canonical(meta_obj), seed_si, database_now,
                    ),
                )
                seq += 1
            seq_hi = seq - 1

            rendered_text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in kept
            )
            await tx.execute(
                f"INSERT INTO {self.store.t.project_messages} ("
                "session_id,send_index,kind,content_json,rendered_text,"
                "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                "projector_version,token_estimate,created_at,completed_through_send_index) "
                "VALUES (?,?,'seed',?,?,?,?,NULL,NULL,0,?,?,?)",
                (
                    sid, seed_si,
                    canonical({"messages": kept, "overflow": overflow, "truncated": truncated}),
                    rendered_text, seq_lo, seq_hi, est, database_now, seed_si,
                ),
            )
            await tx.execute(
                f"UPDATE {self.s} SET send_index=?,next_seq=?,inbox_version=inbox_version+1,"
                f"updated_at={now_sql} WHERE session_id=?",
                (new_send_index, seq, sid),
            )
        return {
            "session_id": sid,
            "send_index": seed_si,
            "seeded_count": len(kept),
            "truncated": truncated,
            "token_estimate": est,
            "overflow": overflow,
        }

    async def repair(
        self,
        target: TargetRef,
        *,
        mode: str = "heal_broken",
        transform=None,
        actor,
    ) -> RepairResult:
        """Explicit session repair primitive (Design 217 §4.4).

        Modes:
        - ``heal_broken``: stitch dangling assistant ``tool_calls`` across sends (honouring
          ``self.on_repair_interrupted_turn`` / M1c ``TURN_REPAIR`` seam), clear
          ``pending_tool_seq``, and write fresh ``kind='repaired'`` projections per affected send.
        - ``fold_tools_to_text``: fold tool call/result rounds into plain text attributed to
          each turn's own ``send_index`` and persist fresh ``kind='repaired'`` projections.
        - ``compact``: load all active projections via bounded ``load_projections_paged``,
          write a ``kind='compact'`` summary up to current ``send_index``, and clear
          ``sessions.fold_stats_json.dedup``.
        - Optional ``transform`` callback overrides the message list directly.
        """
        import inspect

        from power_loop.contracts.errors import SessionBusy
        from power_loop.runtime.budget import estimate_text_tokens
        from power_loop.runtime.history_sanitize import (
            build_interrupted_turn_repairs,
            fold_tool_rounds_by_send_index,
        )
        from power_loop.runtime.store.projection_fold import _encode_state, parse_state, update_fold_stats

        self.check_store(target)
        if mode not in {"heal_broken", "fold_tools_to_text", "compact"}:
            raise RuntimeRequestError("invalid_options", "mode")

        sid = target.session_id
        now_sql = self._now_sql()

        if mode == "compact" and transform is None:
            # Verify ownership/idle state before paging projections
            srow = await self.db.fetchone(f"SELECT * FROM {self.s} WHERE session_id=?", (sid,))
            if srow is None:
                raise RuntimeRequestError("not_found")
            await self._authorize_known(self.db, target, actor, srow)
            if srow["execution_state"] == "running":
                raise SessionBusy(sid)
            paged_rows = await self.store.load_projections_paged(
                sid, before_send_index=int(srow["send_index"]) + 1,
            )
            async with self.db.transaction() as tx:
                state = await self.db.dialect.lock_state(tx, self.s, sid)
                if state["execution_state"] == "running":
                    raise SessionBusy(sid)
                cur_si = int(state["send_index"])
                database_now = await self._db_now(tx)
                non_compact = [r for r in paged_rows if r.kind != "compact"]
                if not non_compact:
                    return RepairResult(
                        target=target, mode="compact", repaired_messages=0,
                        repaired_projections=0, send_index=cur_si,
                        next_seq=int(state["next_seq"]), summary=None,
                    )
                sends = sorted({int(r.send_index) for r in non_compact})
                from_send = sends[0]
                to_send = max(cur_si, sends[-1])
                lines: list[str] = []
                for r in non_compact:
                    txt = r.rendered_text or (
                        r.content if isinstance(r.content, str) else canonical(r.content)
                    )
                    if txt:
                        lines.append(f"[send {r.send_index} {r.kind}] {txt[:300]}")
                summary = "\n".join(lines) or "[compacted]"
                tok_est = estimate_text_tokens(summary)
                await tx.execute(
                    f"DELETE FROM {self.store.t.project_messages} "
                    "WHERE session_id=? AND send_index=? AND kind='compact'",
                    (sid, to_send),
                )
                await tx.execute(
                    f"INSERT INTO {self.store.t.project_messages} ("
                    "session_id,send_index,kind,content_json,rendered_text,"
                    "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                    "projector_version,token_estimate,created_at,completed_through_send_index) "
                    "VALUES (?,?,'compact',?,?,NULL,NULL,?,?,0,?,?,?)",
                    (
                        sid, to_send,
                        canonical({"summary": summary, "from_send": from_send, "to_send": to_send}),
                        summary, from_send, to_send, tok_est, database_now, to_send,
                    ),
                )
                existing_stats = parse_state(state["fold_stats_json"])
                reset_stats = update_fold_stats(
                    existing_stats, send_index=to_send, token_delta=tok_est, compact_reset=True,
                )
                await tx.execute(
                    f"UPDATE {self.s} SET fold_stats_json=?,pending_tool_seq=NULL,"
                    f"updated_at={now_sql} WHERE session_id=?",
                    (_encode_state(reset_stats), sid),
                )
            return RepairResult(
                target=target,
                mode="compact",
                repaired_messages=0,
                repaired_projections=1,
                send_index=to_send,
                next_seq=int(state["next_seq"]),
                summary=summary,
            )

        async with self.db.transaction() as tx:
            state = await self.db.dialect.lock_state(tx, self.s, sid)
            await self._authorize_known(tx, target, actor, state)
            if state["execution_state"] == "running":
                raise SessionBusy(sid)
            database_now = await self._db_now(tx)
            msgs = await tx.fetchall(
                f"SELECT id,seq,send_index,round_index,role,name,content,"
                f"tool_calls_json,tool_call_id,meta_json,state "
                f"FROM {self.store.t.messages} WHERE session_id=? AND state='active' ORDER BY seq ASC",
                (sid,),
            )
            cur_si = int(state["send_index"])
            max_seq = max((int(m["seq"]) for m in msgs), default=0)
            next_seq = max(int(state["next_seq"]), max_seq + 1)

            if transform is not None:
                raw_dicts = [dict(m) for m in msgs]
                res = transform(raw_dicts)
                new_msgs = list(await res if inspect.isawaitable(res) else res)
                await tx.execute(
                    f"DELETE FROM {self.store.t.messages} WHERE session_id=?",
                    (sid,),
                )
                new_seq = 1
                by_si: dict[int, list[dict[str, Any]]] = {}
                for m in new_msgs:
                    si = int(m.get("send_index") or max(1, cur_si))
                    tc_raw = m.get("tool_calls_json")
                    if tc_raw is None and m.get("tool_calls") is not None:
                        tc_raw = canonical(m["tool_calls"])
                    meta_obj = (
                        json.loads(m["meta_json"]) if isinstance(m.get("meta_json"), str)
                        else dict(m.get("meta") or {})
                    )
                    meta_obj["repaired"] = True
                    await tx.execute(
                        f"INSERT INTO {self.store.t.messages} ("
                        "session_id,seq,send_index,round_index,role,name,content,"
                        "tool_calls_json,tool_call_id,meta_json,state,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,'active',?)",
                        (
                            sid, new_seq, si, int(m.get("round_index") or 0),
                            str(m["role"]), m.get("name"), m.get("content"),
                            tc_raw, m.get("tool_call_id"), canonical(meta_obj), database_now,
                        ),
                    )
                    entry_m: dict[str, Any] = {"role": str(m["role"]), "content": m.get("content")}
                    if tc_raw:
                        entry_m["tool_calls"] = json.loads(tc_raw)
                    if m.get("tool_call_id"):
                        entry_m["tool_call_id"] = m["tool_call_id"]
                    by_si.setdefault(si, []).append(entry_m)
                    new_seq += 1
                proj_written = 0
                for si, si_msgs in sorted(by_si.items()):
                    await tx.execute(
                        f"DELETE FROM {self.store.t.project_messages} "
                        "WHERE session_id=? AND send_index=? AND kind='repaired'",
                        (sid, si),
                    )
                    rtxt = "\n".join(f"{x['role']}: {x.get('content') or ''}" for x in si_msgs)
                    await tx.execute(
                        f"INSERT INTO {self.store.t.project_messages} ("
                        "session_id,send_index,kind,content_json,rendered_text,"
                        "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                        "projector_version,token_estimate,created_at,completed_through_send_index) "
                        "VALUES (?,?,'repaired',?,?,1,?,NULL,NULL,0,?,?,?)",
                        (
                            sid, si, canonical({"messages": si_msgs, "repaired": True}),
                            rtxt, new_seq - 1, estimate_text_tokens(rtxt), database_now, si,
                        ),
                    )
                    proj_written += 1
                await tx.execute(
                    f"UPDATE {self.s} SET next_seq=?,pending_tool_seq=NULL,"
                    f"updated_at={now_sql} WHERE session_id=?",
                    (new_seq, sid),
                )
                return RepairResult(
                    target=target,
                    mode=mode,  # type: ignore[arg-type]
                    repaired_messages=new_seq - 1,
                    repaired_projections=proj_written,
                    send_index=cur_si,
                    next_seq=new_seq,
                )

            if mode == "heal_broken":
                by_send_rows: dict[int, list[dict[str, Any]]] = {}
                for m in msgs:
                    si = int(m["send_index"]) if m["send_index"] is not None else max(1, cur_si)
                    by_send_rows.setdefault(si, []).append({
                        "seq": int(m["seq"]),
                        "round_index": int(m["round_index"]) if m["round_index"] is not None else 0,
                        "role": m["role"],
                        "name": m["name"],
                        "content": m["content"],
                        "tool_calls": json.loads(m["tool_calls_json"]) if m["tool_calls_json"] else None,
                        "tool_call_id": m["tool_call_id"],
                        "send_index": si,
                    })
                repaired_sends: list[int] = []
                inserted = 0
                proj_written = 0
                for si, turn_list in sorted(by_send_rows.items()):
                    repairs = await build_interrupted_turn_repairs(
                        sid, turn_list,
                        on_repair_interrupted_turn=self._turn_repair_callback(),
                    )
                    if not repairs:
                        continue
                    repaired_sends.append(si)
                    stitched: list[dict[str, Any]] = []
                    for nm in turn_list:
                        em: dict[str, Any] = {"role": nm["role"], "content": nm["content"]}
                        if nm.get("name"):
                            em["name"] = nm["name"]
                        if nm.get("tool_calls"):
                            em["tool_calls"] = nm["tool_calls"]
                        if nm.get("tool_call_id"):
                            em["tool_call_id"] = nm["tool_call_id"]
                        stitched.append(em)
                    for rep in repairs:
                        meta = dict(rep.get("meta") or {})
                        meta["repaired"] = True
                        await tx.execute(
                            f"INSERT INTO {self.store.t.messages} ("
                            "session_id,seq,role,name,content,tool_call_id,round_index,"
                            "state,meta_json,send_index,created_at) "
                            "VALUES (?,?,'tool',?,?,?,?,'active',?,?,?)",
                            (
                                sid, next_seq, rep.get("name"), rep["content"], rep["tool_call_id"],
                                int(rep.get("round_index") or 0), canonical(meta), si, database_now,
                            ),
                        )
                        stitched.append({
                            "role": "tool", "tool_call_id": rep["tool_call_id"],
                            "name": rep.get("name"), "content": rep["content"],
                        })
                        next_seq += 1
                        inserted += 1
                    await tx.execute(
                        f"DELETE FROM {self.store.t.project_messages} "
                        "WHERE session_id=? AND send_index=? AND kind='repaired'",
                        (sid, si),
                    )
                    rep_text = "\n".join(
                        f"[repaired:{r['tool_call_id']}] {r['content']}" for r in repairs
                    )
                    await tx.execute(
                        f"INSERT INTO {self.store.t.project_messages} ("
                        "session_id,send_index,kind,content_json,rendered_text,"
                        "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                        "projector_version,token_estimate,created_at,completed_through_send_index) "
                        "VALUES (?,?,'repaired',?,?,?,?,NULL,NULL,0,?,?,?)",
                        (
                            sid, si,
                            canonical({"messages": stitched, "repaired_tools": repairs}),
                            rep_text, turn_list[0]["seq"], next_seq - 1,
                            estimate_text_tokens(rep_text), database_now, si,
                        ),
                    )
                    proj_written += 1
                await tx.execute(
                    f"UPDATE {self.s} SET next_seq=?,pending_tool_seq=NULL,"
                    f"updated_at={now_sql} WHERE session_id=?",
                    (next_seq, sid),
                )
                return RepairResult(
                    target=target,
                    mode="heal_broken",
                    repaired_messages=inserted,
                    repaired_projections=proj_written,
                    send_index=cur_si,
                    next_seq=next_seq,
                )

            # mode == "fold_tools_to_text"
            tagged_msgs: list[dict[str, Any]] = []
            for m in msgs:
                si = int(m["send_index"]) if m["send_index"] is not None else max(1, cur_si)
                em = {"role": m["role"], "content": m["content"], "_source_send_index": si}
                if m["name"]:
                    em["name"] = m["name"]
                if m["tool_calls_json"]:
                    em["tool_calls"] = json.loads(m["tool_calls_json"])
                if m["tool_call_id"]:
                    em["tool_call_id"] = m["tool_call_id"]
                tagged_msgs.append(em)
            _folded_all, per_send_folded, _cur_txt = fold_tool_rounds_by_send_index(
                tagged_msgs,
                current_send_index=cur_si + 1,
                historical_end_idx=len(tagged_msgs),
            )
            proj_written = 0
            for si, folded_text in sorted(per_send_folded.items()):
                await tx.execute(
                    f"DELETE FROM {self.store.t.project_messages} "
                    "WHERE session_id=? AND send_index=? AND kind='repaired'",
                    (sid, si),
                )
                await tx.execute(
                    f"INSERT INTO {self.store.t.project_messages} ("
                    "session_id,send_index,kind,content_json,rendered_text,"
                    "source_seq_lo,source_seq_hi,compact_from_send,compact_to_send,"
                    "projector_version,token_estimate,created_at,completed_through_send_index) "
                    "VALUES (?,?,'repaired',?,?,NULL,NULL,NULL,NULL,0,?,?,?)",
                    (
                        sid, si,
                        canonical({"folded_tools": True, "rendered_text": folded_text}),
                        folded_text, estimate_text_tokens(folded_text), database_now, si,
                    ),
                )
                proj_written += 1
            await tx.execute(
                f"UPDATE {self.s} SET pending_tool_seq=NULL,updated_at={now_sql} WHERE session_id=?",
                (sid,),
            )
            return RepairResult(
                target=target,
                mode="fold_tools_to_text",
                repaired_messages=0,
                repaired_projections=proj_written,
                send_index=cur_si,
                next_seq=int(state["next_seq"]),
            )

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
        if is_driver and out.get("kind") in {"user", "workflow_continue"}:
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
                    lifecycle=state["lifecycle"],
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
                lifecycle=state["lifecycle"],
            )

    async def _continuation_child(self, reader, record):
        row = await reader.fetchone(
            f"SELECT session_id,item_id,kind,status,terminal_status,result_seq,"
            "result_send_index,error_code,error_detail,result_delivery_error,"
            "delivered_send_index,continuation_of,result_json,cancel_by,meta_json "
            f"FROM {self.q} WHERE session_id=? AND continuation_of=? "
            "ORDER BY continuation_index DESC LIMIT 1",
            (record.ref.session_id, record.ref.message_id),
        )
        if row is None:
            return None
        return ResultInput(
            InputRef(self.store.store_id, row["session_id"], row["item_id"]),
            {name: row[name] for name in _RESULT_INPUT_COLUMNS},
            result_usage=_extract_result_usage(row["meta_json"]),
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
            "q.meta_json AS input_meta_json,"
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
        record = ResultInput(
            ref, input_dict, result_usage=_extract_result_usage(row["input_meta_json"]),
        )
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
            resolved = ResultInput(ref, derived, result_usage=record.result_usage)
            joined_msg = (
                {"content": row["message_content"], "meta_json": row["message_meta_json"]}
                if row["message_seq"] is not None else None
            )
            result = await self._terminal_result(reader, ref, target, resolved, message_row=joined_msg)
            return ResultRead(ref, target, session, record, resolved, result), False
        resolved = await self._latest_continuation(reader, record)
        derived = await self._derive_input_terminal(reader, session, resolved.row)
        resolved = ResultInput(resolved.ref, derived, result_usage=resolved.result_usage)
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
        if not usage and resolved.result_usage:
            usage = dict(resolved.result_usage)
        if structured is None and resolved.row.get("result_json"):
            structured = json.loads(resolved.row["result_json"])
        return Result(
            ref, target, terminal, result_si, result_seq,
            final_text, usage, structured, resolved.row["error_detail"],
            resolved.row["error_code"], resolved.row["result_delivery_error"],
            cancel_by=resolved.row.get("cancel_by"),
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
