"""Historical inspection uses snapshots and never claims a control writer."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

import pytest

from power_loop.runtime import execution_membership, root_context, task_sessions
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import call, registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_root_context_192 import root_case
from .test_send_sql_profile import Profiled
from .test_workflow_controller_192 import unbound
from .test_workflow_restore_plan_192 import finished, preview
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "registry_runtime", "root_case", "unbound", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@asynccontextmanager
async def forbidden_writer():
    raise AssertionError("inspection must not open a control writer")
    yield


async def originals(manager):
    return {table: await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        for table in (manager.jobs, manager.events, manager.store.t.messages,
                      manager.prefix+"session_obligations", manager.prefix+"session_controls")}


async def test_membership_usage_and_call_lookup_have_no_control_writer(registry_runtime, monkeypatch):
    store, manager, actor, root, parent, _, _ = registry_runtime
    origin = await call(store, actor)
    child = await manager.register_job(actor, kind="subagent", parent=parent, call_ref=origin,
        tool_call_id="call-1", independent=True)
    before = await originals(manager)
    database = manager.db
    manager.db = measured = Profiled(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_write", forbidden_writer)
            membership = await manager.get_execution_membership(actor, root.job.job_id, 1)
            assert membership["scope"]["lease"] == asdict(parent) and not membership["sealed"]
            assert await manager.execution_usage(actor, root.job.job_id, generation=1) == {}
            assert (await manager.operation_for_call(actor, origin, "call-1")).operation_id == child.operation.operation_id
    finally:
        manager.db = database
    assert measured.transactions == 3 and measured.records
    assert all(record["sql"].startswith("SELECT ") and "FOR UPDATE" not in record["sql"]
        and manager.prefix+"task_store" not in record["sql"] for record in measured.records)
    assert await originals(manager) == before


async def test_workflow_profile_has_no_control_writer(registry_runtime, monkeypatch):
    from power_loop.runtime import workflow_profile

    from .test_workflow_profile_192 import spec

    _, manager, actor, _, parent, _, _ = registry_runtime
    request = workflow_profile.creation_request(spec(), None)
    child = await manager.register_job(actor, kind="workflow", parent=parent, independent=True,
        system_key="read-profile", request=request)
    before = await originals(manager)
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", forbidden_writer)
        profile = await manager.get_workflow_profile(actor, child.job.job_id)
        assert profile == request["execution_profile"]
    assert await originals(manager) == before


async def test_frozen_workflow_controller_has_no_control_writer(unbound, monkeypatch):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    before = await originals(c.manager)
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", forbidden_writer)
        data = await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1)
        assert data["workflow"] == asdict(c.proof) and data["session_id"] == c.sid
    assert await originals(c.manager) == before


async def test_full_restore_preview_uses_a_read_port_and_does_not_dispatch(managed_workflow, monkeypatch):
    _, manager, _, _, model, _, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    before, calls = await originals(manager), model.counts.copy()
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", forbidden_writer)
        plan = await preview(managed_workflow)
        assert plan["workflow"]["job_id"] == handle.job_id
        assert plan["dependency"]["invalidated"] == ["BUILD", "REVIEW"] and plan["replay"] == []
    assert await originals(manager) == before and model.counts == calls


async def test_rejected_delivery_verification_cannot_create_an_original_even_if_its_caller_commits(managed_workflow):
    from power_loop.runtime import workflow_results

    _, manager, runtime, _, model, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    removed = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
        "AND event_type IN ('workflow_delivery','workflow_projected')", (handle.job_id,))
    assert len(removed) == 2
    await manager.db.execute(f"DELETE FROM {manager.events} WHERE job_id=? "
        "AND event_type IN ('workflow_delivery','workflow_projected')", (handle.job_id,))
    before, calls = await originals(manager), model.counts.copy()
    try:
        # Catch the rejection inside a real caller transaction, so an accidental
        # preparation write could commit rather than being hidden by rollback.
        async with manager.joined_write() as tx:
            with pytest.raises(JobError, match="workflow_projection_unsettled"):
                await workflow_results.require_closed(manager, tx, runtime.actor(sid), handle.job_id, 1)
        assert await originals(manager) == before and model.counts == calls
    finally:
        for event in removed:
            columns = tuple(event)
            await manager.db.execute(f"INSERT INTO {manager.events} ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})", tuple(event[key] for key in columns))


async def test_root_provenance_and_owned_transcript_bindings_have_no_control_writer(root_case, monkeypatch):
    c = root_case
    with c.runtime.root_context(c.sid, namespace="test.read", payload={"request_id": "read-only"}):
        assert (await c.loop.send("one actual root", c.sid)).status == "completed"
    [root] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE parent_job_id IS NULL")
    before, calls = await originals(c.manager), list(c.model.calls)
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", forbidden_writer)
        context = await root_context.read(c.manager, c.actor, root["job_id"])
        assert context["payload"] == {"request_id": "read-only"} and not context["execution_authority"]
        bindings = await task_sessions.list_sessions(c.manager, c.actor, root["job_id"])
        assert [binding["session_id"] for binding in bindings["bindings"]] == [c.sid]
        assert not bindings["execution_authority"]
    assert await originals(c.manager) == before and c.model.calls == calls


@pytest.mark.parametrize("port", ["membership", "usage"])
async def test_paused_execution_inspection_allows_a_stop_to_commit(registry_runtime, monkeypatch, port):
    _, manager, actor, root, _, connect, _ = registry_runtime
    other = JobManager(await connect(), now_ms=manager.now)
    entered, release = asyncio.Event(), asyncio.Event()
    read = execution_membership.read

    async def paused(owner, reader, row, generation):
        if owner is manager:
            assert manager._control_intents.get() is None
            entered.set()
            await release.wait()
        return await read(owner, reader, row, generation)

    with monkeypatch.context() as patch:
        patch.setattr(execution_membership, "read", paused)
        inspection = asyncio.create_task(manager.get_execution_membership(actor, root.job.job_id, 1) if port == "membership"
            else manager.execution_usage(actor, root.job.job_id, generation=1))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            stop = await asyncio.wait_for(other.request_cancel(actor, root.job.job_id,
                system_key="stop-during-read", reason="concurrent stop"), 5)
            assert stop.result["cancel_requested"] and not inspection.done()
            release.set()
            result = await asyncio.wait_for(inspection, 5)
            assert result["scope"]["lease"]["job_id"] == root.job.job_id if port == "membership" else result == {}
        finally:
            release.set()
            await asyncio.gather(inspection, return_exceptions=True)
    assert (await other.get_job(actor, root.job.job_id)).cancel_requested_at is not None


async def test_read_inside_an_explicit_joined_write_uses_its_existing_connection(registry_runtime, monkeypatch):
    _, manager, actor, root, parent, _, _ = registry_runtime
    before = await originals(manager)

    @asynccontextmanager
    async def forbidden_snapshot():
        raise AssertionError("joined read must reuse its existing transaction")
        yield

    with pytest.raises(OSError, match="caller transaction rolled back"):
        async with manager.joined_write():
            child = await manager.register_job(actor, kind="subagent", parent=parent, system_key="uncommitted-child")
            lease = await manager.claim_job(actor, child.job.job_id, executor_id="joined-child", boot_id="boot", expected_version=1)
            with monkeypatch.context() as patch:
                patch.setattr(manager.db, "read_snapshot", forbidden_snapshot)
                membership = await manager.get_execution_membership(actor, root.job.job_id, 1)
                assert membership["scope"]["lease"] == asdict(parent)
                assert [member["payload"]["member"] for member in membership["members"]] == [asdict(lease)]
                assert await manager.execution_usage(actor, root.job.job_id, generation=1) == {}
            raise OSError("caller transaction rolled back")
    assert await originals(manager) == before
