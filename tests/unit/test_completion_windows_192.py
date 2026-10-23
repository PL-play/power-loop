"""Original deadlines survive recall, clock changes and control failures."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime import completion_windows as windows
from power_loop.runtime import node_inputs
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.completion import CompletionWindow
from power_loop.runtime.jobs import JobActor, JobError, JobLease, JobManager, _hash, _json
from power_loop.runtime.task_runtime import _OWNER, Execution, TaskRuntime

from .test_job_registry_192 import registry_store, runtime
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def registered(runtime, protocol=1):
    store, manager, main, root, root_lease, connect, now = runtime
    request = {} if protocol == "legacy" else {windows.PROTOCOL: protocol}
    child = await manager.register_job(main, kind="subagent", parent=root_lease, independent=True,
        system_key="window-owner", request=request)
    sid = await store.create_session(parent_session_id=main.session_id,
        metadata={"execution_role": "subagent", "standalone_subagent": True})
    child = await manager.bind_session(main, child.job.job_id, sid)
    actor = replace(main, session_id=sid)
    return SimpleNamespace(store=store, manager=manager, actor=actor, main=main, root=root, child=child,
        sid=sid, connect=connect, now=now)


async def claim(c):
    return await c.manager.claim_job(c.actor, c.child.job_id, executor_id="child", boot_id="boot",
        expected_version=c.child.version)


@pytest.fixture
async def original(runtime):
    c = await registered(runtime)
    c.owner = await claim(c)
    c.saved = await windows.read(c.manager, c.actor, c.owner.job_id)
    return c


async def test_original_claim_and_reopened_manager_recall_one_unchanged_utc_deadline(original):
    c = original
    assert c.saved["deadline_at_ms"] == c.now[0] + windows.DEFAULT_TIMEOUT_MS
    c.now[0] += 1000
    restored = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    assert await windows.read(restored, c.main, c.owner.job_id) == c.saved
    assert len(await restored.db.fetchall(f"SELECT event_id FROM {restored.events} WHERE event_type='completion_window'")) == 1
    assert (await restored.get_job(c.main, c.owner.job_id)).generation == 1


async def test_expired_window_remains_recallable_but_cannot_authorize_execution_or_extension(original):
    c = original
    c.now[0] = c.saved["deadline_at_ms"]
    # Isolate deadline from the independent short renewable execution lease.
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=? WHERE job_id=?",
        (c.now[0] + 10000, c.owner.job_id))
    assert await windows.read(c.manager, c.actor, c.owner.job_id) == c.saved
    with pytest.raises(JobError, match="owner_completion_deadline_exceeded"):
        await c.manager.check_execution(c.actor, c.owner)
    stop = await c.manager.request_cancel(c.main, c.owner.job_id, system_key="expired-window-stop", reason="stop")
    assert stop.operation_id
    assert (await c.manager.get_job(c.main, c.owner.job_id)).cancel_requested_at is not None
    assert await windows.read(c.manager, c.actor, c.owner.job_id) == c.saved


async def test_local_collector_rebind_and_new_collector_never_refresh_original_window(original):
    c = original
    task_runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
        executor_id="child", boot_id="boot")
    loop = SimpleNamespace(task_runtime=task_runtime)
    token = _OWNER.set(Execution(task_runtime, c.actor, c.owner))
    try:
        first = CompletionWindow()
        await windows.bind_window(loop, first, c.sid)
        c.now[0] += 1000
        restored = CompletionWindow()
        await windows.bind_window(loop, restored, c.sid)
        assert restored.utc_deadline_at_ms == first.utc_deadline_at_ms == c.saved["deadline_at_ms"]
        assert restored.deadline - asyncio.get_running_loop().time() <= 1799
        previous = first.deadline
        c.now[0] -= 2000  # Clock correction must not extend an existing collector.
        await windows.bind_window(loop, first, c.sid)
        assert first.deadline <= previous
        c.now[0] = c.saved["deadline_at_ms"]
        expired = CompletionWindow()
        await windows.bind_window(loop, expired, c.sid)
        with pytest.raises(TimeoutError):
            expired.require_time()
    finally:
        _OWNER.reset(token)


@pytest.mark.parametrize("field,value", [("deadline_at_ms", True), ("deadline_at_ms", 0),
    ("deadline_at_ms", 2**63), ("now_ms", True), ("now_ms", None)])
async def test_collector_rejects_ambiguous_deadline_types(field, value):
    args = {"deadline_at_ms": 2000, "now_ms": 1000, field: value}
    with pytest.raises(ValueError, match="invalid_completion_deadline"):
        CompletionWindow().bind("same-sid", **args)


async def test_bound_collector_cannot_switch_to_another_utc_deadline():
    window = CompletionWindow()
    window.bind("same-sid", deadline_at_ms=2000, now_ms=1000)
    with pytest.raises(ValueError, match="completion_deadline_changed"):
        window.bind("same-sid", deadline_at_ms=3000, now_ms=1000)


@pytest.mark.parametrize("protocol", [True, "1", 2, 0, None])
async def test_bad_protocol_cannot_claim_an_owner_or_leave_partial_membership(runtime, protocol):
    c = await registered(runtime, protocol)
    with pytest.raises(JobError):
        await claim(c)
    row = await c.manager.get_job(c.main, c.child.job_id)
    assert row.generation == 0 and row.resource_state == "not_started"
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='execution_scope'", (row.job_id,))


async def test_legacy_owner_has_no_invented_original_window(runtime):
    c = await registered(runtime, "legacy")
    c.owner = await claim(c)
    assert await windows.read(c.manager, c.actor, c.owner.job_id) is None


@pytest.mark.parametrize("point", ["insert_window", "read_window"])
async def test_window_and_original_owner_claim_are_one_atomic_transaction(runtime, monkeypatch, point):
    c = await registered(runtime)
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    execute = transaction_type.execute
    read_row = windows.read_row

    async def fail_write(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "completion_window":
            raise RuntimeError("window commit failed")
        return result

    async def fail_read(*args):
        await read_row(*args)
        raise RuntimeError("window commit failed")

    if point == "insert_window":
        monkeypatch.setattr(transaction_type, "execute", fail_write)
    else:
        monkeypatch.setattr(windows, "read_row", fail_read)
    with pytest.raises(RuntimeError, match="window commit failed"):
        await claim(c)
    row = await c.manager.get_job(c.main, c.child.job_id)
    assert row.generation == 0 and row.resource_state == "not_started"
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('completion_window','execution_scope')", (row.job_id,))


@pytest.mark.parametrize("fault", ["missing", "deadline", "time_bool", "origin", "hash", "delivery", "marker"])
async def test_original_window_cannot_be_missing_retimed_or_rebound(original, fault):
    c = original
    key = windows.event_id(c.manager, c.owner.job_id)
    if fault == "missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
    elif fault == "marker":
        [creation] = await c.manager.db.fetchall(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND action='create'", (c.owner.job_id,))
        data = json.loads(creation["payload_json"])
        del data["request"][windows.PROTOCOL]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (_json(data), creation["event_id"]))
    elif fault == "delivery":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='handled' WHERE event_id=?", (key,))
    else:
        data = json.loads(_json(c.saved))
        if fault == "deadline":
            data["deadline_at_ms"] += 1
        elif fault == "time_bool":
            data["admitted_at_ms"] = True
        elif fault == "origin":
            data["origin"]["executor_id"] = "other-executor"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (_json(data), "bad" if fault == "hash" else _hash(data), key))
    with pytest.raises(JobError):
        await windows.read(c.manager, c.actor, c.owner.job_id)
    with pytest.raises(JobError):
        await c.manager.check_execution(c.actor, c.owner)


async def test_wait_policy_cannot_reset_remaining_owner_deadline(original):
    c = original
    c.now[0] += 1000
    source = await c.manager.register_job(c.actor, kind="send", parent=c.owner, session_id=c.sid,
        system_key="window-source", request={"send_index": 1})
    caller = await c.manager.claim_job(c.actor, source.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    seq = await c.store.append_message(c.sid, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "arm-window", "type": "function", "function": {"name": "wait", "arguments": "{}"}}])
    kwargs = dict(caller=caller, call_ref=await c.store.message_ref(c.sid, seq), tool_call_id="arm-window",
        request=waits.WaitRequest(c.now[0] + 100, "Original reason", "Continue original task"))
    with pytest.raises(JobError, match="wait_deadline_exceeds_owner_window"):
        await waits.register_wait(c.manager, c.actor, **kwargs,
            policy=waits.WaitPolicy(c.now[0] + windows.DEFAULT_TIMEOUT_MS, max_wait_ms=3600000))
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='timer'")
    registration = await waits.register_wait(c.manager, c.actor, **kwargs,
        policy=waits.WaitPolicy(c.saved["deadline_at_ms"]))
    saved = await waits.read_wait(c.manager, c.main, registration.job.job_id)
    assert saved["request"]["policy"]["deadline_at_ms"] == c.saved["deadline_at_ms"]


async def test_later_processing_error_preserves_already_paid_round_usage(registry_store, monkeypatch):
    store, _ = registry_store
    manager = JobManager(store)
    task_runtime = TaskRuntime(manager, tenant_id="original-usage", principal_id="host", executor_id="executor", boot_id="boot")
    calls = []

    class Provider(LLMService):
        async def complete(self, request, **kwargs):
            calls.append(request)
            assert len(calls) == 1
            return LLMResponse(raw_text="42", token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))

        def stream(self, request):
            async def empty():
                if False:
                    yield None
            return empty()

        async def close(self):
            pass

    loop = StatefulAgentLoop(llm=Provider(), store=store, task_runtime=task_runtime,
        config=AgentLoopConfig(max_rounds=3, delegation_policy="single_level"))
    sid = await loop.new_session()
    original = loop._run_with_tool_context

    async def unavailable(*args, **kwargs):
        result = await original(*args, **kwargs)
        assert result.usage["total_tokens"] == 10
        raise RuntimeError("test: post-model processing unavailable")

    monkeypatch.setattr(loop, "_run_with_tool_context", unavailable)
    with pytest.raises(RuntimeError, match="post-model processing unavailable"):
        await loop.send("Complete original request", sid)
    assert len(calls) == 1
    [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert (await manager.execution_usage(task_runtime.actor(sid), job["job_id"]))["total_tokens"] == 10
    [closed] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='model_closed'", (job["job_id"],))
    assert json.loads(closed["payload_json"])["evidence"]["usage"]["total_tokens"] == 10


@pytest.mark.parametrize("fault", ["none", "missing_window", "corrupt_window", "missing_lease", "bool_lease", "other_generation", "unknown_resource", "expired_lease", "write_failure"])
async def test_expired_original_node_closes_input_only_with_exact_retained_authority(managed_workflow, monkeypatch, fault):
    store, manager, task_runtime, loop, model, parent, state, _ = managed_workflow
    # Isolate the mechanical port from the separately tested live heartbeat
    # while this fixture temporarily advances its shared runtime clock.
    task_runtime.lease_ms = 300000
    await loop.send("Run original workflow", parent)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [node] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")
    [bound] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='node_session'", (node["job_id"],))
    sid = json.loads(bound["payload_json"])["session_id"]
    actor = JobActor(node["tenant_id"], node["principal_id"], node["created_by_session_id"])
    binding = Execution(task_runtime, actor, JobLease(node["job_id"], node["executor_id"], node["boot_id"], node["generation"]))
    original = await windows.read(manager, actor, node["job_id"])
    window_row = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (windows.event_id(manager, node["job_id"]),))
    clock = manager.now
    try:
        manager.now = lambda: original["deadline_at_ms"]
        await manager.db.execute(f"UPDATE {manager.jobs} SET lease_until_ms=? WHERE resource_state='running'", (manager.now() + 30000,))
        if fault in ("missing_window", "corrupt_window"):
            key = windows.event_id(manager, node["job_id"])
            if fault == "missing_window":
                await manager.db.execute(f"DELETE FROM {manager.events} WHERE event_id=?", (key,))
            else:
                await manager.db.execute(f"UPDATE {manager.events} SET payload_json='{{}}' WHERE event_id=?", (key,))
        elif fault == "missing_lease":
            binding = replace(binding, lease=None)
        elif fault in ("bool_lease", "other_generation"):
            binding = replace(binding, lease=replace(binding.lease, generation=True if fault == "bool_lease" else 2))
        elif fault == "unknown_resource":
            await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='unknown' WHERE job_id=?", (node["job_id"],))
        elif fault == "expired_lease":
            await manager.db.execute(f"UPDATE {manager.jobs} SET lease_until_ms=? WHERE job_id=?", (manager.now(), node["job_id"]))
        before = await manager._row(manager.db, node["job_id"])
        if fault == "write_failure":
            async with manager._write() as tx:
                transaction_type = type(tx)
            execute = transaction_type.execute
            async def fail(tx, sql, params=()):
                result = await execute(tx, sql, params)
                if sql.startswith("INSERT INTO " + manager.events) and params[2] == "node_input_closed":
                    raise RuntimeError("original input close write failed")
                return result
            with monkeypatch.context() as patch:
                patch.setattr(transaction_type, "execute", fail)
                with pytest.raises(RuntimeError, match="input close write failed"):
                    await node_inputs.close(task_runtime, binding, sid)
            assert not await node_inputs.is_closed(manager, manager.db, before, sid)
            assert (await manager._row(manager.db, node["job_id"]))["version"] == before["version"]
        elif fault != "none":
            with pytest.raises(JobError):
                await node_inputs.close(task_runtime, binding, sid)
            if fault in ("missing_window", "corrupt_window"):
                with pytest.raises(JobError, match="completion_window_evidence_(missing|corrupt)"):
                    await manager.bind_node_session(actor, binding.lease, sid)
            assert not await node_inputs.is_closed(manager, manager.db, before, sid)
        else:
            await node_inputs.close(task_runtime, binding, sid)
            await node_inputs.close(task_runtime, binding, sid)
            current = await manager._row(manager.db, node["job_id"])
            assert await node_inputs.is_closed(manager, manager.db, current, sid)
            [count] = await manager.db.fetchall(f"SELECT COUNT(*) AS count FROM {manager.events} WHERE job_id=? AND event_type='node_input_closed'", (node["job_id"],))
            assert count["count"] == 1
            assert current["resource_state"] == "running" and current["outcome"] is None
        assert model.counts["BUILD"] == 1 and model.counts["REVIEW"] == 0
    finally:
        # Restore only fault-injected test state so the original fixture's live
        # controller can exit; these repairs do not count as runtime recovery.
        manager.now = clock
        if fault in ("missing_window", "corrupt_window"):
            key = windows.event_id(manager, node["job_id"])
            if fault == "missing_window":
                await manager.db.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (key, node["job_id"], "completion_window", window_row["job_version"], _hash(original), _json(original), "not_applicable", original["admitted_at_ms"]))
            else:
                await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (_json(original), key))
        if fault == "unknown_resource":
            await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='running' WHERE job_id=?", (node["job_id"],))
        if fault == "expired_lease":
            await manager.db.execute(f"UPDATE {manager.jobs} SET lease_until_ms=? WHERE job_id=?", (manager.now() + 30000, node["job_id"]))
        model.release["BUILD"].set()
