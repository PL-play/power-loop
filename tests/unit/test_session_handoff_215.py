"""Deterministic ownership, handoff, and commit-order tests for schema 17."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import InputRef
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


async def test_recover_honours_matching_active_cancellation(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    receipt = await port.accept(root, "old", message_id="old", actor=ACTOR)
    await port.claim(root, node_id="old", boot_id="old")
    requested = await port.cancel(receipt.input_ref, actor=ACTOR, reason="stop old send")
    assert requested.state == "cancel_requested"
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (root.session_id,),
    )
    new = RuntimeInputStore(await connect(schema="verify"))
    recovered = await new.recover(root, node_id="new", boot_id="new")
    assert recovered.execution_state == "idle"
    result = await new.get_result(receipt.input_ref, actor=ACTOR)
    assert result.status == "cancelled" and result.error == "stop old send"


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


def process_args(store, root, role):
    backend = store._db.dialect.name
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    return [
        sys.executable, "-m", "tests.unit._session_handoff_worker_215",
        dsn, store.table_prefix, root.session_id, role,
    ]


async def process_event(process):
    line = await asyncio.wait_for(process.stdout.readline(), 20)
    assert line, (await process.stderr.read()).decode()[-3000:]
    return json.loads(line)


async def process_command(process, value):
    process.stdin.write((value + "\n").encode())
    await process.stdin.drain()


async def launch_process(store, root, role):
    return await asyncio.create_subprocess_exec(
        *process_args(store, root, role), cwd=Path(__file__).resolve().parents[2],
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def stop_processes(processes):
    for process in processes:
        if process.returncode is None:
            process.kill()
        await process.wait()


@pytest.mark.parametrize("order", ("accept-first", "final-first"))
async def test_two_process_accept_final_linearization(registry_store, order):
    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("requires a shared server DB")
    port = RuntimeInputStore(store)
    root = await target(port, "process-accept-final-" + order)
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    owner, acceptor = await launch_process(store, root, "owner"), None
    processes = [owner]
    try:
        assert (await process_event(owner))["event"] == "claimed"
        if order == "accept-first":
            acceptor = await launch_process(store, root, "accept_hold")
            processes.append(acceptor)
            assert (await process_event(acceptor))["event"] == "accept_locked"
            await process_command(owner, "final")
            await process_command(acceptor, "release")
            assert (await process_event(acceptor))["event"] == "accepted"
            expected = "queued"
        else:
            await process_command(owner, "final_hold")
            assert (await process_event(owner))["event"] == "final_locked"
            acceptor = await launch_process(store, root, "accept")
            processes.append(acceptor)
            assert (await process_event(acceptor))["event"] == "accept_started"
            await process_command(owner, "release")
            expected = "idle"
        assert await process_event(owner) == {"event": "final", "execution_state": expected}
        if order == "final-first":
            assert (await process_event(acceptor))["event"] == "accepted"
        old_status = await port.inspect(
            InputRef(store.store_id, root.session_id, "old"), actor=ACTOR,
        )
        new_status = await port.inspect(
            InputRef(store.store_id, root.session_id, "new"), actor=ACTOR,
        )
        assert old_status.terminal_status == "completed"
        assert new_status.input_state == "pending" and new_status.terminal_status is None
        assert (await port.inspect(root, actor=ACTOR)).execution_state == "queued"
        for process in processes:
            await asyncio.wait_for(process.wait(), 20)
            assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
    finally:
        await stop_processes(processes)


async def test_two_process_claim_claim_has_single_winner(registry_store):
    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("requires a shared server DB")
    port = RuntimeInputStore(store)
    root = await target(port, "process-claim-claim")
    await port.accept(root, "one", message_id="one", actor=ACTOR)
    first = await launch_process(store, root, "claim_hold")
    second = None
    processes = [first]
    try:
        assert (await process_event(first))["event"] == "claim_locked"
        second = await launch_process(store, root, "claim")
        processes.append(second)
        assert (await process_event(second))["event"] == "claim_started"
        await process_command(first, "release")
        winner = await process_event(first)
        loser = await process_event(second)
        assert winner["event"] == "claim" and winner["batch"]["send_index"] == 1
        assert loser == {"event": "claim", "batch": None}
        assert len((await port.messages(root, actor=ACTOR)).items) == 1
        for process in processes:
            await asyncio.wait_for(process.wait(), 20)
            assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
    finally:
        await stop_processes(processes)


async def test_two_process_accept_claim_linearization(registry_store):
    store, _ = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("requires a shared server DB")
    port = RuntimeInputStore(store)
    root = await target(port, "process-accept-claim")
    acceptor = await launch_process(store, root, "accept_hold")
    claimant = None
    processes = [acceptor]
    try:
        assert (await process_event(acceptor))["event"] == "accept_locked"
        await process_command(acceptor, "release")
        assert (await process_event(acceptor))["event"] == "accepted"
        claimant = await launch_process(store, root, "claim")
        processes.append(claimant)
        assert (await process_event(claimant))["event"] == "claim_started"
        claimed = await process_event(claimant)
        assert claimed["event"] == "claim" and claimed["batch"]["input_ids"] == ["new"]
        for process in processes:
            await asyncio.wait_for(process.wait(), 20)
            assert process.returncode == 0, (await process.stderr.read()).decode()[-3000:]
    finally:
        await stop_processes(processes)
