"""Real-process ownership worker; DSN is an argument and is never printed."""
from __future__ import annotations

import asyncio
import json
import sys

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import TargetRef
from power_loop.contracts.errors import SessionLeaseLost
from power_loop.runtime.store.factory import open_store
from tests.unit.session_native_fixture import ACTOR


def emit(**payload):
    print(json.dumps(payload), flush=True)


async def command():
    return (await asyncio.wait_for(asyncio.to_thread(sys.stdin.readline), 20)).strip()


def pause_after_lock(store, event):
    original = store._db.dialect.lock_state

    async def gated(tx, state_table, session_id):
        row = await original(tx, state_table, session_id)
        emit(event=event)
        await command()
        return row

    store._db.dialect.lock_state = gated


def batch_json(batch):
    return None if batch is None else {
        "send_index": batch.send_index,
        "input_ids": [ref.message_id for ref in batch.input_refs],
    }


async def main():
    dsn, prefix, session_id, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    port = RuntimeInputStore(store)
    target = TargetRef(store.store_id, session_id)
    try:
        if role == "old":
            batch = await port.claim(target, node_id="old-process", boot_id="old-boot")
            emit(event="claimed", send_index=batch.send_index)
            await command()
            try:
                await port.final(batch, final_text="stale")
            except SessionLeaseLost:
                emit(event="stale")
            else:
                raise AssertionError("stale owner finalized")
        elif role == "new":
            await port.accept(target, "new", message_id="new", actor=ACTOR)
            assert (await port.inspect(target, actor=ACTOR)).execution_state == "recovering"
            recovered = await port.recover(target, node_id="new-process", boot_id="new-boot")
            assert recovered.execution_state == "queued"
            batch = await port.claim(target, node_id="new-process", boot_id="new-boot")
            await port.final(batch, final_text="new process")
            emit(event="completed", send_index=batch.send_index)
        elif role == "owner":
            batch = await port.claim(target, node_id="owner", boot_id="owner")
            emit(event="claimed", send_index=batch.send_index)
            action = await command()
            if action == "final_hold":
                pause_after_lock(store, "final_locked")
            elif action != "final":
                raise AssertionError(action)
            final = await port.final(batch, final_text="owner result")
            emit(event="final", execution_state=final.execution_state)
        elif role in {"accept", "accept_hold"}:
            if role == "accept_hold":
                pause_after_lock(store, "accept_locked")
            else:
                emit(event="accept_started")
            receipt = await port.accept(target, "new", message_id="new", actor=ACTOR)
            emit(event="accepted", state=receipt.state)
        elif role in {"claim", "claim_hold"}:
            if role == "claim_hold":
                pause_after_lock(store, "claim_locked")
            else:
                emit(event="claim_started")
            batch = await port.claim(target, node_id=role, boot_id=role)
            emit(event="claim", batch=batch_json(batch))
        elif role == "claim_twice":
            first = await port.claim(target, node_id="claim-twice", boot_id="claim-twice")
            emit(event="first_claim", batch=batch_json(first))
            await command()
            second = await port.claim(target, node_id="claim-twice", boot_id="claim-twice")
            emit(event="second_claim", batch=batch_json(second))
        else:
            raise AssertionError(role)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
