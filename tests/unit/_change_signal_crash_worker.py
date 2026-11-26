"""Die inside a signal/payload transaction, without interpreter cleanup."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
from contextlib import asynccontextmanager

from power_loop.agent.runtime_store import ClaimedBatch, RuntimeInputStore
from power_loop.agent.runtime_types import InputRef, TargetRef, TrustedActor
from power_loop.runtime.store.factory import open_store
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR


async def prepare_background_batch(store, parent_sid: str) -> ClaimedBatch:
    """Create and claim a real schema-18 background child session routed to parent_sid."""
    port = RuntimeInputStore(store)
    seq = await store.append_message(parent_sid, role="user", content="launch background")
    call_ref = await store.message_ref(parent_sid, seq)
    parent_actor = TrustedActor(
        STORE_ACTOR.tenant_id, STORE_ACTOR.principal_id, parent_sid, call_ref, "bg-call-1",
    )
    frozen = {
        "tool": "echo",
        "arguments": {"text": "once"},
        "definition_hash": "fixture-bg-hash",
        "call_ref": call_ref.to_dict(),
        "call_id": "bg-call-1",
    }
    bg_descriptor = dataclasses.replace(
        STORE_DESCRIPTOR,
        kind="background",
        config={**dict(STORE_DESCRIPTOR.config), "background": frozen},
    )
    child_target, _start, _receipt = await port.create_child(
        bg_descriptor,
        parent=TargetRef(store.store_id, parent_sid),
        actor=parent_actor,
        call_ref=call_ref,
        tool_call_id="bg-call-1",
        body=frozen,
        result_kind="background_result",
        require_assistant_call=False,
    )
    batch = await port.claim(child_target, node_id="bg-node", boot_id="bg-boot")
    assert batch is not None
    return batch


def serialize_batch(batch: ClaimedBatch) -> str:
    return json.dumps({
        "store_id": batch.target.store_id,
        "session_id": batch.target.session_id,
        "send_index": batch.send_index,
        "owner_epoch": batch.owner_epoch,
        "owner_node_id": batch.owner_node_id,
        "owner_boot_id": batch.owner_boot_id,
        "message_seq": batch.message_seq,
        "cancel_seq": batch.cancel_seq,
        "session_updated_at": batch.session_updated_at,
        "input_refs": [
            {"store_id": r.store_id, "session_id": r.session_id, "message_id": r.message_id}
            for r in batch.input_refs
        ],
        "tenant_id": batch.tenant_id,
        "principal_id": batch.principal_id,
        "result_route": batch.result_route,
        "start_input_message_id": batch.start_input_message_id,
        "parent_session_id": batch.parent_session_id,
        "root_session_id": batch.root_session_id,
        "depth": batch.depth,
        "reply_to_session_id": batch.reply_to_session_id,
        "reply_result_kind": batch.reply_result_kind,
        "reply_to_item_id": batch.reply_to_item_id,
    })


def deserialize_batch(raw: str) -> ClaimedBatch:
    from power_loop.runtime.store.types import MessageRow, MessageState

    data = json.loads(raw)
    target = TargetRef(data["store_id"], data["session_id"])
    user_row = MessageRow(
        session_id=data["session_id"],
        seq=data["message_seq"],
        role="user",
        name=None,
        content="",
        tool_calls=None,
        tool_call_id=None,
        round_index=0,
        state=MessageState.ACTIVE,
        meta={},
        created_at=0,
        send_index=data["send_index"],
    )
    return ClaimedBatch(
        target=target,
        send_index=data["send_index"],
        message_seq=data["message_seq"],
        input_refs=tuple(InputRef(**r) for r in data["input_refs"]),
        owner_epoch=data["owner_epoch"],
        owner_node_id=data["owner_node_id"],
        owner_boot_id=data["owner_boot_id"],
        descriptor=dataclasses.replace(STORE_DESCRIPTOR, kind="background"),
        tenant_id=data["tenant_id"],
        principal_id=data["principal_id"],
        parent_session_id=data["parent_session_id"],
        user_row=user_row,
        result_route=data["result_route"],
        session_updated_at=data["session_updated_at"],
        cancel_seq=data["cancel_seq"],
        start_input_message_id=data["start_input_message_id"],
        root_session_id=data["root_session_id"],
        depth=data["depth"],
        reply_to_session_id=data["reply_to_session_id"],
        reply_result_kind=data["reply_result_kind"],
        reply_to_item_id=data["reply_to_item_id"],
    )


async def write(store, sid, kind, *, batch: ClaimedBatch | None = None):
    if kind == "inbox":
        port = RuntimeInputStore(store)
        await port.accept(TargetRef(store.store_id, sid), "once", message_id="late-input", actor=STORE_ACTOR)
    elif kind == "background":
        assert batch is not None, "background write requires a claimed background child batch"
        port = RuntimeInputStore(store)
        return await port.final(batch, status="completed", final_text="once")
    else:
        await store.set_runtime_state(sid, "todo", {"rendered": "once"})


async def main(dsn, prefix, sid, kind, stage, batch_json=""):
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    batch = deserialize_batch(batch_json) if batch_json else None
    if stage == "after_signal":
        if kind in ("inbox", "background"):
            orig_tx = store._db.transaction
            @asynccontextmanager
            async def stop_after_signal():
                async with orig_tx() as tx:
                    orig_exec = tx.execute
                    orig_fetchone = tx.fetchone
                    orig_packed = getattr(tx, "execute_packed", None)
                    async def wrapped_exec(sql, params=()):
                        res = await orig_exec(sql, params)
                        if "inbox_version" in sql and "UPDATE" in sql.upper():
                            os._exit(91)
                        return res
                    async def wrapped_fetchone(sql, params=()):
                        res = await orig_fetchone(sql, params)
                        if "inbox_version" in sql and "UPDATE" in sql.upper():
                            os._exit(91)
                        return res
                    tx.execute = wrapped_exec
                    tx.fetchone = wrapped_fetchone
                    if orig_packed is not None:
                        async def wrapped_packed(sql, params=(), *args, **kwargs):
                            res = await orig_packed(sql, params, *args, **kwargs)
                            if "inbox_version" in sql and "UPDATE" in sql.upper():
                                os._exit(91)
                            return res
                        tx.execute_packed = wrapped_packed
                    yield tx
            store._db.transaction = stop_after_signal
        else:
            signal = store.signal_changes
            async def stop(*args):
                await signal(*args)
                os._exit(91)
            store.signal_changes = stop
    else:
        transaction = store._db.transaction
        @asynccontextmanager
        async def stop_before_commit():
            async with transaction() as tx:
                yield tx
                os._exit(92)
        store._db.transaction = stop_before_commit
    await write(store, sid, kind, batch=batch)
    raise AssertionError("crash point not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
