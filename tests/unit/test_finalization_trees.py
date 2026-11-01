"""Closure evidence is prepared without holding another task tree's writer."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from power_loop.runtime import finalization, session_obligations
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import issue_nudge
from .test_processing_decisions_192 import processing_case
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import finish, managed
from .test_wait_handoff_192 import handoff, waiting
from .test_wait_terminal_192 import stop_owner

__all__ = ["registry_store", "runtime", "processing_case", "ready", "managed", "handoff", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def closing(processing_case, runtime):
    c = processing_case
    target, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    decisions = await c.commit(snapshot, response)
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
        "last_assistant_ref": response.to_dict(), "processing_decisions": decisions})
    await c.manager.close_model_execution(c.actor, runtime[4], evidence={"exit": "completed"})
    return SimpleNamespace(**vars(c), root=runtime[4].job_id, target=target)


async def test_closure_writer_never_reads_the_other_trees_actual_handler(closing, monkeypatch):
    c = closing
    roots = {row["job_id"]: row["root_job_id"] for row in await c.manager.db.fetchall(
        f"SELECT job_id,root_job_id FROM {c.manager.jobs}")}
    write, one, many = c.manager._write, ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault("_test_read_jobs", set()).update(
            row["job_id"] for row in rows if row is not None and "job_id" in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        seen = tx.touched_jobs | tx.__dict__.get("_test_read_jobs", set())
        footprint = {roots[job] for job in seen}
        assert footprint <= {c.root}, "the other handler's original evidence belongs outside the tree writer"
        footprints.append(footprint)

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, "fetchone", fetchone)
        patch.setattr(ControlTransaction, "fetchall", fetchall)
        patch.setattr(c.manager, "_write", scoped)
        receipt = await c.manager.finalize_supervision(c.actor, c.root, generation=1)
    assert receipt["job_id"] == c.root and footprints == [{c.root}]
    assert (await c.manager.get_job(c.actor, c.target)).supervision_status == "closed"
    assert (await c.manager.get_job(c.actor, c.root)).supervision_status == "closed"
    await session_obligations.verify(c.manager, c.actor)


async def test_paused_closure_discovery_allows_a_different_tree_to_stop(closing, monkeypatch):
    c = closing
    peer = JobManager(await c.connect(), now_ms=c.manager.now)
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = finalization._prepare
    pauses = []

    async def paused(owner, reader, actor, row, generation, state, depth=0):
        if row["job_id"] == c.root and not pauses:
            assert owner._control_intents.get() is None
            pauses.append(row["job_id"])
            entered.set()
            await release.wait()
        return await prepare(owner, reader, actor, row, generation, state, depth)

    with monkeypatch.context() as patch:
        patch.setattr(finalization, "_prepare", paused)
        task = asyncio.create_task(c.manager.finalize_supervision(c.actor, c.root, generation=1))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            stopped = await asyncio.wait_for(peer.request_cancel(c.actor, c.caller.job_id,
                system_key="stop-during-other-root-closure", reason="stop"), 2)
            assert stopped.result["cancel_requested"]
        finally:
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
    assert len(pauses) == 1 and not isinstance(result[0], BaseException), result
    assert (await peer.get_job(c.actor, c.root)).supervision_status == "closed"
    assert (await peer.get_job(c.actor, c.caller.job_id)).cancel_requested_at is not None


async def test_an_already_closed_descendant_resuming_after_preparation_cannot_be_hidden_by_its_unchanged_parent(ready, monkeypatch):
    c = ready
    c.runtime.lease_ms = 30000
    actor = c.runtime.actor(c.parent)
    root = c.old.root_job_id
    original = c.manager.finalize_supervision

    async def keep_parent_open(current, job_id, *, generation):
        if job_id == root:
            raise JobError("test_parent_closure_delayed")
        return await original(current, job_id, generation=generation)

    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "finalize_supervision", keep_parent_open)
        await issue_nudge(c.fixture, monkeypatch)
        await finish(c.fixture)
        await c.loop.flush_follow_ups(c.parent)
    before = await c.manager.get_job(actor, root)
    child = await c.manager.get_job(actor, c.old.job_id)
    assert before.supervision_status == "open" and child.generation == 2 and child.supervision_status == "closed"
    old_closures = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='supervision_closed' ORDER BY event_id", (root,))
    c.model.entered.clear()
    c.model.release.clear()
    calls = c.model.child_calls
    publish = finalization._publish_plan
    resumed = []

    async def resume_before_commit(owner, plans, tree):
        assert owner._control_intents.get() is None
        if tree.root_id == root and not resumed:
            resumed.append(tree)
            # The child already had a closure, so it is absent from `plans`.
            assert {row["job_id"] for row, _, _, _ in plans} == {root}
            await issue_nudge(c.fixture, monkeypatch)
            await asyncio.wait_for(c.model.entered.wait(), 3)
            assert (await owner.get_job(actor, root)).version == before.version
        return await publish(owner, plans, tree)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(finalization, "_publish_plan", resume_before_commit)
            with pytest.raises(JobError, match="finalization_execution_unsettled"):
                await c.manager.finalize_supervision(actor, root, generation=1)
        assert len(resumed) == 1
        current = await c.manager.get_job(actor, c.old.job_id)
        assert current.generation == 3 and current.resource_state == "running"
        assert (await c.manager.get_job(actor, root)).supervision_status == "open"
        assert c.model.child_calls == calls+1
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='supervision_closed' ORDER BY event_id", (root,)) == old_closures
    finally:
        await finish(c.fixture)


async def test_peer_closure_after_preparation_returns_its_original_receipt_without_a_second_publication(closing, monkeypatch):
    c = closing
    peer = JobManager(await c.connect(), now_ms=c.manager.now)
    publish = finalization._publish_plan
    receipts = []

    async def close_in_peer(owner, plans, tree):
        if owner is c.manager and not receipts:
            receipts.append(await peer.finalize_supervision(c.actor, c.root, generation=1))
        return await publish(owner, plans, tree)

    with monkeypatch.context() as patch:
        patch.setattr(finalization, "_publish_plan", close_in_peer)
        receipt = await c.manager.finalize_supervision(c.actor, c.root, generation=1)
    assert receipts == [receipt]
    [saved] = await peer.db.fetchall(f"SELECT event_id FROM {peer.events} WHERE job_id=? AND event_type='supervision_closed'", (c.root,))
    assert saved["event_id"] == receipt["event_id"]
    await session_obligations.verify(peer, c.actor)


@pytest.mark.parametrize("phase", ["prepared", "original", "projection", "committed"])
async def test_real_process_death_preserves_an_atomic_tree_closure_without_replaying_any_execution(closing, tmp_path, phase):
    import os
    import sys
    from pathlib import Path

    c = closing
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, output = tmp_path / "closure.ready", tmp_path / "closure.json"
    context = {"actor": asdict(c.actor), "root": c.root, "generation": 1, "now": c.manager.now()}
    before_jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id")
    before_events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    messages = await c.store.load_all_messages(c.actor.session_id)
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._finalization_tree_worker",
        dsn, c.manager.prefix, json.dumps(context), phase, str(ready), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 79, stdout.decode()
        assert ready.read_text() == phase and not output.exists()
        if phase != "committed":
            assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id") == before_jobs
            assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == before_events
        else:
            assert (await c.manager.get_job(c.actor, c.root)).supervision_status == "closed"
            assert (await c.manager.get_job(c.actor, c.target)).supervision_status == "closed"
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        receipt = await cold.finalize_supervision(c.actor, c.root, generation=1)
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        originals = {row["event_id"]: row for row in after}
        assert all(originals[row["event_id"]] == row for row in before_events)
        assert len(after) == len(before_events)+2
        assert await c.manager.finalize_supervision(c.actor, c.root, generation=1) == receipt
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == after
        assert await c.store.load_all_messages(c.actor.session_id) == messages
        assert (await cold.get_job(c.actor, c.target)).generation == 1
        await session_obligations.verify(cold, c.actor)
    finally:
        await stop_worker(process)


@pytest.mark.parametrize("committed", [False, True])
async def test_closure_rollback_or_unknown_commit_ack_never_leaves_a_partial_tree(closing, monkeypatch, committed):
    c = closing
    transaction, failed = c.manager.db.transaction, []

    @asynccontextmanager
    async def fault():
        async with transaction() as tx:
            yield tx
            row = await tx.fetchone(f"SELECT supervision_status FROM {c.manager.jobs} WHERE job_id=?", (c.root,))
            hit = not failed and row["supervision_status"] == "closed"
            if hit:
                failed.append(committed)
                if not committed:
                    raise OSError("closure rollback before commit")
        if hit:
            raise OSError("closure committed but ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", fault)
        with pytest.raises(OSError, match="rollback|ACK lost"):
            await c.manager.finalize_supervision(c.actor, c.root, generation=1)
    assert failed == [committed]
    for job in (c.root, c.target):
        assert (await c.manager.get_job(c.actor, job)).supervision_status == ("closed" if committed else "open")
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    receipt = await cold.finalize_supervision(c.actor, c.root, generation=1)
    assert await c.manager.finalize_supervision(c.actor, c.root, generation=1) == receipt
    assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='supervision_closed'")) == 2
    await session_obligations.verify(cold, c.actor)


@pytest.mark.parametrize("phase", ["bookkeeping", "projection", "committed"])
async def test_real_process_death_keeps_negative_wait_termination_and_its_closure_atomic(handoff, tmp_path, phase):
    import os
    import sys
    from pathlib import Path

    h, c = handoff, handoff.c
    await stop_owner(h)
    timer = h.registration.job.job_id
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, output = tmp_path / "negative-closure.ready", tmp_path / "negative-closure.json"
    context = {"actor": asdict(c.main), "root": timer, "generation": 0, "now": c.manager.now()}
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id")
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._finalization_tree_worker",
        dsn, c.manager.prefix, json.dumps(context), phase, str(ready), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 79, stdout.decode()
        assert ready.read_text() == phase and not output.exists()
        if phase != "committed":
            assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id") == jobs
            assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == events
        else:
            row = await c.manager.get_job(c.main, timer)
            assert row.outcome == "cancelled" and row.supervision_status == "closed"
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        receipt = await cold.finalize_supervision(c.main, timer, generation=0)
        row = await cold.get_job(c.main, timer)
        assert row.generation == 0 and row.outcome == "cancelled" and row.resource_state == "stopped"
        assert row.supervision_status == "closed"
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await c.manager.finalize_supervision(c.main, timer, generation=0) == receipt
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == after
        await session_obligations.verify(cold, c.main)
    finally:
        await stop_worker(process)


async def test_inbox_delivery_after_preparation_keeps_its_real_message_and_cannot_publish_a_stale_withdrawal(handoff, monkeypatch):
    from .test_input_dispositions_192 import _applied, exited

    h, c = handoff, handoff.c
    await _applied(h)
    await exited(h)
    timer = h.registration.job.job_id
    publish = finalization._publish_plan
    prepared, delivered = [], []
    calls = h.model.calls

    async def deliver_before_commit(owner, plans, tree):
        assert owner._control_intents.get() is None
        prepared.append(tree)
        if not delivered:
            items = await c.store.inbox_claim(c.sid)
            assert len(items) == 1
            seq = await c.store.append_message(c.sid, role="user", name="follow_up", content=items[0]["content"], send_index=2,
                inbox={"claim_token": items[0]["claim_token"], "ids": [item["id"] for item in items]},
                meta={"inbox": {"item_ids": [item["item_id"] for item in items], "kinds": [item["kind"] for item in items]}})
            delivered.append(await c.store.message_ref(c.sid, seq))
        return await publish(owner, plans, tree)

    with monkeypatch.context() as patch:
        patch.setattr(finalization, "_publish_plan", deliver_before_commit)
        await c.manager.finalize_supervision(c.main, timer, generation=0)
    assert len(prepared) == 2, "only the actual inbox delivery race should trigger rediscovery"
    [proof] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='input_disposition'")
    data = json.loads(proof["payload_json"])
    assert data["disposition"] == "delivered_unprocessed" and data["delivered_message"]["ref"] == delivered[0].to_dict()
    [item] = await c.store.inbox_items(c.sid)
    assert item["status"] == "delivered" and item["delivered_seq"] == delivered[0].seq
    assert (await c.manager.get_job(c.main, timer)).supervision_status == "closed"
    assert h.model.calls == calls
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    await session_obligations.verify(cold, c.main)
