"""Original deadlines survive recall, clock changes and control failures."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage
from power_loop.runtime import completion_windows as windows
from power_loop.runtime.completion import CompletionWindow
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import _OWNER, Execution, TaskRuntime

from .test_job_registry_192 import registry_store, runtime

__all__ = ['registry_store', 'runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def registered(runtime, protocol=1):
    store, manager, main, root, root_lease, connect, now = runtime
    request = {} if protocol == "legacy" else {windows.PROTOCOL: protocol}
    sid = await store.create_session(parent_session_id=main.session_id,
        metadata={"execution_role": "subagent", "standalone_subagent": True})
    registration = await manager.register_job(main, kind="subagent", session_id=sid,
        parent=root_lease, independent=True, system_key="window-owner", request=request)
    child = registration.job
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
    execute = transaction_type._execute
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
        monkeypatch.setattr(transaction_type, "_execute", fail_write)
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




async def test_later_processing_error_preserves_already_paid_round_usage(registry_store, monkeypatch, runtime_agent):
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

    from power_loop.agent.stateful_loop import _ExecutionLoop
    agent = await runtime_agent(model=Provider(), store=store, task_runtime=task_runtime,
        config=AgentLoopConfig(max_rounds=3, delegation_policy="single_level"))
    original = _ExecutionLoop._run_with_tool_context

    async def unavailable(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        assert result.usage["total_tokens"] == 10
        raise RuntimeError("test: post-model processing unavailable")

    monkeypatch.setattr(_ExecutionLoop, "_run_with_tool_context", unavailable)
    result = await agent.send("Complete original request")
    assert result.status == 'failed' and result.error == 'RuntimeError'
    assert len(calls) == 1
    [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='send'")
    assert (await manager.execution_usage(task_runtime.actor(agent.target.session_id), job["job_id"]))["total_tokens"] == 10
    assert result.usage['total_tokens'] == 10
