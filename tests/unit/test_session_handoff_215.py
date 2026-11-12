"""Deterministic ownership, handoff, and commit-order tests for schema 17."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.contracts.errors import SessionLeaseLost

from .session_native_fixture import ACTOR, fail_next_commit, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_claim_claim_has_one_owner_and_one_merged_message(registry_store):
    store, connect = registry_store
    port_a = RuntimeInputStore(store)
    port_b = RuntimeInputStore(await connect(schema="verify"))
    root = await target(port_a)
    await port_a.accept(root, "one", message_id="one", actor=ACTOR)
    first, second = await asyncio.gather(
        port_a.claim(root, node_id="a", boot_id="a"),
        port_b.claim(root, node_id="b", boot_id="b"),
    )
    assert (first is None) != (second is None)
    batch = first or second
    state = await store._db.fetchone(
        f"SELECT execution_state,owner_epoch,send_index FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"execution_state": "running", "owner_epoch": 1, "send_index": 1}
    assert len((await port_a.messages(root, actor=ACTOR)).items) == 1
    assert batch is not None


async def test_accept_then_final_and_final_then_accept_have_no_sleep_window(
    registry_store, monkeypatch,
):
    for order in ("accept-first", "final-first"):
        store, connect = registry_store
        other = await connect(schema="verify")
        port_a, port_b = RuntimeInputStore(store), RuntimeInputStore(other)
        root = await target(port_a, session_id="root_" + order.replace("-", "_"))
        await port_a.accept(root, "old", message_id="old_" + order, actor=ACTOR)
        batch = await port_a.claim(root, node_id="a", boot_id="a")
        if order == "accept-first":
            entered, release = asyncio.Event(), asyncio.Event()
            original = port_b._accept_tx

            async def paused_accept(*args, **kwargs):
                receipt = await original(*args, **kwargs)
                entered.set()
                await release.wait()
                return receipt

            monkeypatch.setattr(port_b, "_accept_tx", paused_accept)
            accepting = asyncio.create_task(port_b.accept(
                root, "new", message_id="new_" + order, actor=ACTOR
            ))
            await entered.wait()
            finishing = asyncio.create_task(port_a.final(batch, final_text="done"))
            await asyncio.sleep(0)
            assert not finishing.done()
            release.set()
            await accepting
            final = await finishing
        else:
            entered, release = asyncio.Event(), asyncio.Event()
            original = port_a._final_tx

            async def paused_final(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(port_a, "_final_tx", paused_final)
            finishing = asyncio.create_task(port_a.final(batch, final_text="done"))
            await entered.wait()
            accepting = asyncio.create_task(port_b.accept(
                root, "new", message_id="new_" + order, actor=ACTOR
            ))
            await asyncio.sleep(0)
            assert not accepting.done()
            release.set()
            final = await finishing
            await accepting
        assert final.execution_state == ("queued" if order == "accept-first" else "idle")
        old_ref, new_ref = batch.input_refs[0], type(batch.input_refs[0])(
            store.store_id, root.session_id, "new_" + order
        )
        assert (await port_b.inspect(old_ref, actor=ACTOR)).terminal_status == "completed"
        assert (await port_b.inspect(new_ref, actor=ACTOR)).input_state == "pending"
        monkeypatch.undo()


async def test_claim_ack_loss_reuses_committed_batch_without_second_message(
    registry_store, monkeypatch,
):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    fail_next_commit(monkeypatch, store)
    batch = await port.claim(root, node_id="a", boot_id="a")
    assert batch.send_index == 1
    monkeypatch.undo()
    assert len((await port.messages(root, actor=ACTOR)).items) == 1
    await port.final(batch, final_text="done")
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).status == "completed"


async def test_final_ack_loss_reads_original_result(registry_store, monkeypatch):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    fail_next_commit(monkeypatch, store)
    final = await port.final(batch, final_text="committed")
    monkeypatch.undo()
    assert final.results[0].status == "completed"
    assert final.results[0].final_text == "committed"
    reopened = RuntimeInputStore(store)
    result = await reopened.get_result(batch.input_refs[0], actor=ACTOR)
    assert result.final_text == "committed" and result.send_index == batch.send_index


async def test_expired_accept_blocks_stale_final_then_recovers_before_new_claim(registry_store):
    store, connect = registry_store
    old = RuntimeInputStore(store)
    new = RuntimeInputStore(await connect(schema="verify"))
    root = await target(old)
    await old.accept(root, "old", message_id="old", actor=ACTOR)
    batch = await old.claim(root, node_id="old", boot_id="old")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (root.session_id,),
    )
    follow_up = await new.accept(root, "new", message_id="new", actor=ACTOR)
    assert (await new.inspect(root, actor=ACTOR)).execution_state == "recovering"
    assert await new.claim(root, node_id="new", boot_id="new") is None
    with pytest.raises(SessionLeaseLost):
        await old.final(batch, final_text="stale")
    with pytest.raises(SessionLeaseLost):
        await old.cancel_active(batch)
    assert not await old.release_owner(batch)
    recovered = await new.recover(root, node_id="new", boot_id="new")
    assert recovered.execution_state == "queued"
    assert recovered.results[0].status == "interrupted"
    next_batch = await new.claim(root, node_id="new", boot_id="new")
    assert [ref.message_id for ref in next_batch.input_refs] == ["new"]
    await new.final(next_batch, final_text="new owner")
    assert (await new.get_result(follow_up.input_ref, actor=ACTOR)).final_text == "new owner"


async def test_expired_running_without_new_input_is_discovered_and_finished(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    batch = await port.claim(root, node_id="old", boot_id="old")
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (root.session_id,),
    )
    assert root in await port.discover_unfinished()
    recovered = await port.recover(root, node_id="new", boot_id="new")
    assert recovered.execution_state == "idle"
    result = await port.get_result(batch.input_refs[0], actor=ACTOR)
    assert result.status == "interrupted" and result.send_index == batch.send_index


@pytest.mark.parametrize("race", ("accept-claim", "claim-accept"))
async def test_accept_claim_commit_order_is_deterministic(registry_store, race):
    store, connect = registry_store
    first = RuntimeInputStore(store)
    second = RuntimeInputStore(await connect(schema="verify"))
    root = await target(first)
    if race == "accept-claim":
        receipt = await first.accept(root, "one", message_id="one", actor=ACTOR)
        batch = await second.claim(root, node_id="b", boot_id="b")
        assert batch and batch.input_refs == (receipt.input_ref,)
    else:
        assert await second.claim(root, node_id="b", boot_id="b") is None
        await first.accept(root, "one", message_id="one", actor=ACTOR)
        batch = await second.claim(root, node_id="b", boot_id="b")
        assert batch is not None


async def test_two_real_processes_hand_off_on_shared_server_db(registry_store, tmp_path):
    store, _ = registry_store
    backend = store._db.dialect.name
    if backend == "sqlite":
        pytest.skip("SQLite logic is covered above; real handoff uses a shared server DB")
    port = RuntimeInputStore(store)
    root = await target(port, session_id="process-handoff")
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    repo = Path(__file__).resolve().parents[2]
    common = [
        sys.executable, "-m", "tests.unit._session_handoff_worker_215",
        dsn, store.table_prefix, root.session_id,
    ]
    processes = []

    async def launch(role):
        process = await asyncio.create_subprocess_exec(
            *common, role, cwd=repo,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    async def event(process):
        line = await asyncio.wait_for(process.stdout.readline(), 20)
        assert line, (await process.stderr.read()).decode()[-3000:]
        return json.loads(line)

    old = await launch("old")
    try:
        assert (await event(old))["event"] == "claimed"
        await store._db.execute(
            f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
            (root.session_id,),
        )
        new = await launch("new")
        completed = await event(new)
        assert completed["event"] == "completed" and completed["send_index"] == 2
        old.stdin.write(b"final\n")
        await old.stdin.drain()
        assert (await event(old))["event"] == "stale"
        for process in processes:
            await asyncio.wait_for(process.wait(), 20)
            assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
        statuses = await store._db.fetchall(
            f"SELECT item_id,terminal_status FROM {store.t.follow_up_queue} ORDER BY item_id"
        )
        assert statuses == [
            {"item_id": "new", "terminal_status": "completed"},
            {"item_id": "old", "terminal_status": "interrupted"},
        ]
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
