"""Context cancellation commits its barrier and one repeatable tree at a time."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import root_context, root_control
from power_loop.runtime.jobs import JobError, JobManager, _hash

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store
from .test_root_context_192 import NAMESPACE, root_case
from .test_root_control_192 import PAYLOAD, send, stop

__all__ = ["registry_store", "root_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def stops(manager):
    return await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='stop' ORDER BY event_id")


async def test_unknown_commit_after_first_root_retains_barrier_and_resumes_exact_scope(root_case, monkeypatch):
    c = root_case
    for _ in range(2):
        await send(c)
    roots = set(c.model.calls)
    transaction = c.manager.db.transaction
    lost = []

    @asynccontextmanager
    async def lose_first_tree_ack():
        contains_stop = False
        async with transaction() as tx:
            yield tx
            if not lost and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'"):
                contains_stop = True
        if contains_stop:
            lost.append(True)
            raise OSError("first root committed; ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", lose_first_tree_ack)
        with pytest.raises(OSError, match="first root committed"):
            await stop(c)
    assert lost == [True]
    original_stops = await stops(c.manager)
    assert len(original_stops) == 1
    cancelled = [await c.manager.get_job(c.actor, root) for root in roots]
    assert sum(job.cancel_requested_at is not None for job in cancelled) == 1
    with pytest.raises(JobError, match="root_context_cancelled"):
        await send(c)
    cold = JobManager(await c.connect())
    receipt = await stop(c, manager=cold)
    assert {effect["source"]["root_job_id"] for effect in receipt["roots"]} == roots
    assert len(await stops(cold)) == 2
    assert original_stops[0] in await stops(cold)
    assert not receipt["resource_stop_confirmed"] and not receipt["billing_settled"]
    assert await stop(c, manager=cold) == receipt and len(c.model.calls) == 2


async def test_tree_stop_rollback_leaves_committed_admission_barrier_and_frozen_scope(root_case, monkeypatch):
    c = root_case
    for _ in range(2):
        await send(c)
    roots = set(c.model.calls)
    transaction = c.manager.db.transaction
    failed = []

    @asynccontextmanager
    async def reject_first_tree_commit():
        async with transaction() as tx:
            yield tx
            if not failed and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'"):
                failed.append(True)
                raise OSError("tree stop commit rolled back")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", reject_first_tree_commit)
        with pytest.raises(OSError, match="tree stop commit rolled back"):
            await stop(c)
    assert failed == [True] and await stops(c.manager) == []
    for root in roots:
        assert (await c.manager.get_job(c.actor, root)).cancel_requested_at is None
    with pytest.raises(JobError, match="root_context_cancelled"):
        await send(c)
    cold = JobManager(await c.connect())
    receipt = await stop(c, manager=cold)
    assert {effect["source"]["root_job_id"] for effect in receipt["roots"]} == roots
    assert len(await stops(cold)) == 2 and len(c.model.calls) == 2


async def test_begin_rollback_does_not_publish_barrier_or_lose_request_retry(root_case, monkeypatch):
    c = root_case
    await send(c)
    transaction = c.manager.db.transaction

    @asynccontextmanager
    async def reject_begin_commit():
        async with transaction() as tx:
            yield tx
            raise OSError("admission barrier rolled back")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", reject_begin_commit)
        with pytest.raises(OSError, match="admission barrier rolled back"):
            await stop(c)
    assert await stops(c.manager) == []
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}task_context_events WHERE state_key LIKE ?",
                                         ("power-loop.root-stop%",))
    await send(c)
    receipt = await stop(c)
    assert len(receipt["roots"]) == len(c.model.calls) == 2


async def test_partial_request_identity_cannot_publish_a_barrier_for_a_different_context(root_case, monkeypatch):
    c = root_case
    await send(c)
    freeze = root_control._freeze_scope

    async def lose_scope_ack(*args, **kwargs):
        await freeze(*args, **kwargs)
        raise OSError("scope committed before first stop")

    with monkeypatch.context() as patch:
        patch.setattr(root_control, "_freeze_scope", lose_scope_ack)
        with pytest.raises(OSError, match="scope committed"):
            await stop(c)
    with pytest.raises(JobError, match="operation_conflict"):
        await stop(c, payload={"origin_run_id": 202})
    assert await stops(c.manager) == []
    await send(c, {"origin_run_id": 202})
    assert len(c.model.calls) == 2
    receipt = await stop(c)
    assert [effect["source"]["root_job_id"] for effect in receipt["roots"]] == [c.model.calls[0]]
    assert (await c.manager.get_job(c.actor, c.model.calls[1])).cancel_requested_at is None


async def test_context_intent_cancels_a_queued_roots_current_generation_after_scope_freeze(root_case, monkeypatch):
    c = root_case
    registered = await c.manager.register_job(c.actor, kind="send", session_id=c.sid, system_key="send:99",
        request={"host_context": root_context._document(NAMESPACE, PAYLOAD), "send_index": 99, "model_attempt_protocol": 1})
    tree = root_control._cancel_tree
    leases = []

    async def claim_before_stop(*args, **kwargs):
        leases.append(await c.manager.claim_job(c.actor, registered.job.job_id,
            executor_id="new-owner", boot_id="new-boot", expected_version=1))
        return await tree(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(root_control, "_cancel_tree", claim_before_stop)
        receipt = await stop(c)
    assert leases[0].generation == 1 and receipt["roots"][0]["source"]["root_generation"] == 0
    row = await c.manager.get_job(c.actor, registered.job.job_id)
    assert row.generation == 1 and row.cancel_requested_at is not None and row.status == "cancelling"
    with pytest.raises(JobError, match="cancel_requested"):
        await c.manager.check_execution(c.actor, leases[0])
    assert not c.model.calls and not receipt["resource_stop_confirmed"]


async def test_two_connections_reuse_one_frozen_scope_and_one_stop_per_root(root_case):
    import asyncio

    c = root_case
    for _ in range(2):
        await send(c)
    versions = {job: (await c.manager.get_job(c.actor, job)).version for job in c.model.calls}
    other = JobManager(await c.connect())
    first, second = await asyncio.gather(stop(c), stop(c, manager=other))
    assert first == second and len(first["roots"]) == 2 and len(await stops(c.manager)) == 2
    for job, version in versions.items():
        assert (await c.manager.get_job(c.actor, job)).version == version + 1


@pytest.mark.parametrize("phase", ["begin", "scope", "request"])
async def test_lost_metadata_ack_reuses_every_committed_original_and_never_replays_model(root_case, monkeypatch, phase):
    c = root_case
    for _ in range(2):
        await send(c)
    transaction = c.manager.db.transaction
    lost = []

    @asynccontextmanager
    async def lose_metadata_ack():
        ready = False
        async with transaction() as tx:
            yield tx
            if not lost:
                ready = await tx.fetchone(f"SELECT state_key FROM {c.manager.prefix}task_context_events WHERE state_key LIKE ?",
                                          (f"power-loop.root-stop-{phase}.v1:%",)) is not None
        if ready:
            lost.append(True)
            raise OSError("original metadata committed; ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", lose_metadata_ack)
        with pytest.raises(OSError, match="original metadata committed"):
            await stop(c)
    assert lost == [True]
    originals = await c.manager.db.fetchall(
        f"SELECT * FROM {c.manager.prefix}task_context_events WHERE state_key LIKE ? ORDER BY state_key",
        ("power-loop.root-stop%",))
    original_stops = await stops(c.manager)
    assert len(original_stops) == (2 if phase == "request" else 0)
    cold = JobManager(await c.connect())
    receipt = await stop(c, manager=cold)
    after = await cold.db.fetchall(
        f"SELECT * FROM {c.manager.prefix}task_context_events WHERE state_key LIKE ? ORDER BY state_key",
        ("power-loop.root-stop%",))
    final_stops = await stops(cold)
    assert all(original in after for original in originals)
    assert len(final_stops) == len(receipt["roots"]) == len(c.model.calls) == 2
    assert all(row in final_stops for row in original_stops)
    assert await stop(c, manager=cold) == receipt
    assert not receipt["resource_stop_confirmed"] and not receipt["billing_settled"]


@pytest.mark.parametrize("damage", ["generation", "missing-generation", "context"])
async def test_cold_scope_retry_refuses_invalid_original_target_without_repair_or_stop(root_case, monkeypatch, damage):
    import json

    c = root_case
    await send(c)
    freeze = root_control._freeze_scope

    async def lose_scope_ack(*args, **kwargs):
        await freeze(*args, **kwargs)
        raise OSError("scope committed before process exit")

    with monkeypatch.context() as patch:
        patch.setattr(root_control, "_freeze_scope", lose_scope_ack)
        with pytest.raises(OSError, match="scope committed"):
            await stop(c)
    original = await c.manager.db.fetchone(
        f"SELECT * FROM {c.manager.prefix}task_context_events WHERE state_key LIKE ?", ("power-loop.root-stop-scope.v1:%",))
    assert original is not None
    value = json.loads(original["payload_json"])
    source = value["response"]["sources"][0]
    if damage == "generation":
        source["root_generation"] = 999
    elif damage == "missing-generation":
        del source["root_generation"]
    else:
        source["payload"] = {"origin_run_id": 202}
    # Exercise the cold source comparison, beyond the envelope checksum alone.
    value["evidence_hash"] = _hash({key: item for key, item in value.items() if key != "evidence_hash"})
    encoded = json.dumps(value)
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_context_events SET payload_json=? WHERE state_key=?",
                              (encoded, original["state_key"]))
    try:
        jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id")
        events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        cold = JobManager(await c.connect())
        with pytest.raises(JobError, match="root_control_evidence_corrupt"):
            await stop(c, manager=cold)
        assert await cold.db.fetchall(f"SELECT * FROM {cold.jobs} ORDER BY job_id") == jobs
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == events
        assert (await cold.db.fetchone(
            f"SELECT payload_json FROM {c.manager.prefix}task_context_events WHERE state_key=?",
            (original["state_key"],)))["payload_json"] == encoded
        assert len(c.model.calls) == 1 and await stops(cold) == []
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_context_events SET payload_json=? WHERE state_key=?",
                                  (original["payload_json"], original["state_key"]))


async def spawn_worker(c, tmp_path, mode):
    import asyncio
    import json
    import os
    import sys
    from pathlib import Path

    engine = c.manager.db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, release, output = (tmp_path / (mode + suffix) for suffix in (".ready", ".release", ".json"))
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._root_cancel_worker",
        dsn, c.manager.prefix, c.actor.tenant_id, c.actor.principal_id, c.sid,
        NAMESPACE, json.dumps(PAYLOAD), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    return process, ready, release, output


async def test_real_process_death_after_first_root_preserves_scope_and_exact_stop_id(root_case, tmp_path):
    import asyncio
    import json

    c = root_case
    for _ in range(2):
        await send(c)
    process, ready, release, output = await spawn_worker(c, tmp_path, "crash-after-first")
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 74, stdout.decode()
        original = json.loads(ready.read_text())
        assert not output.exists() and len(await stops(c.manager)) == 1
        with pytest.raises(JobError, match="root_context_cancelled"):
            await send(c)
        cold = JobManager(await c.connect())
        receipt = await stop(c, key="process-context-stop", manager=cold)
        assert {effect["source"]["root_job_id"] for effect in receipt["roots"]} == set(c.model.calls)
        assert original in receipt["roots"] and len(await stops(cold)) == 2
        assert await stop(c, key="process-context-stop", manager=cold) == receipt
        assert len(c.model.calls) == 2 and not receipt["resource_stop_confirmed"] and not receipt["billing_settled"]
    finally:
        release.touch()
        await stop_worker(process)


async def test_real_new_send_is_refused_while_cancel_waits_outside_first_root_transaction(root_case, tmp_path):
    import asyncio
    import json

    c = root_case
    for _ in range(2):
        await send(c)
    drainer, ready, release, output = await spawn_worker(c, tmp_path, "hold-before-first")
    sender = None
    try:
        await wait_ready(drainer, ready)
        assert json.loads(ready.read_text()) == {"state": "barrier-and-scope-committed"}
        assert await stops(c.manager) == []
        sender, _send_ready, _send_release, send_output = await spawn_worker(c, tmp_path, "new-send")
        stdout, _ = await asyncio.wait_for(sender.communicate(), 30)
        assert sender.returncode == 0, stdout.decode()
        assert json.loads(send_output.read_text()) == {"error": "root_context_cancelled", "calls": 0}
        assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")) == 2
        release.touch()
        stdout, _ = await asyncio.wait_for(drainer.communicate(), 30)
        assert drainer.returncode == 0, stdout.decode()
        receipt = json.loads(output.read_text())
        assert {effect["source"]["root_job_id"] for effect in receipt["roots"]} == set(c.model.calls)
        assert len(await stops(c.manager)) == 2 and len(c.model.calls) == 2
        assert not receipt["resource_stop_confirmed"] and not receipt["billing_settled"]
    finally:
        release.touch()
        await stop_worker(drainer)
        if sender is not None:
            await stop_worker(sender)
