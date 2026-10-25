"""Original host stage registration is atomic and cannot admit another send."""
import asyncio
import json

import pytest

from power_loop import StatefulAgentLoop
from power_loop.runtime import host_send_steps as steps
from power_loop.runtime import root_context
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_root_context_192 import NAMESPACE, root_case

__all__ = ["registry_store", "root_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
STEP = {"namespace": "test.host-processing", "plan_hash": "a" * 64, "step_id": "initial:0"}
SOURCE = {"namespace": NAMESPACE, "payload": {"run_id": 101}}


async def lookup(c, **kwargs):
    return await steps.lookup(c.manager, c.actor, **{**STEP, **kwargs}, expected_host_context=SOURCE)


async def send(c, *, step=STEP):
    with c.runtime.root_context(c.sid, **SOURCE):
        with c.runtime.host_send_step(c.sid, **step):
            return await c.loop.send("original input for this host stage", c.sid)


async def test_binding_exists_before_provider_and_survives_reopen(root_case):
    c = root_case
    assert await lookup(c) is None
    observed = []
    async def before(binding):
        value = await lookup(c)
        assert value["job_id"] == binding.lease.job_id
        assert not value["execution_authority"] and not value["billing_settled"]
        observed.append(value)
    c.model.before = before
    assert (await send(c)).status == "completed"
    assert len(observed) == len(c.model.calls) == 1
    first = await lookup(c)
    reopened = await c.connect()
    try:
        assert await steps.lookup(JobManager(reopened), c.actor, **STEP, expected_host_context=SOURCE) == first
    finally:
        await reopened.close()
    assert first["creation_event_id"] and first["creation_request_hash"] and first["binding_event_id"]


async def test_same_host_step_cannot_dispatch_a_second_paid_send(root_case):
    c = root_case
    await send(c)
    first = await lookup(c)
    with pytest.raises(JobError, match="host_send_step_already_registered"):
        await send(c)
    assert len(c.model.calls) == 1
    assert await lookup(c) == first
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")) == 1


async def test_different_stage_has_its_own_original_send_and_root_provenance(root_case):
    c = root_case
    await send(c)
    first = await lookup(c)
    second_step = {**STEP, "step_id": "follow-up:0"}
    await send(c, step=second_step)
    second = await lookup(c, step_id="follow-up:0")
    assert first["job_id"] != second["job_id"] and len(c.model.calls) == 2
    assert (await root_context.read(c.manager, c.actor, second["job_id"]))["payload"] == SOURCE["payload"]


async def test_failed_atomic_binding_rolls_back_the_job_and_operation_before_model(root_case, monkeypatch):
    c = root_case
    original = steps.record
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("storage acknowledgement failed before commit")
    monkeypatch.setattr(steps, "record", fail)
    with pytest.raises(RuntimeError, match="before commit"):
        await send(c)
    assert not c.model.calls
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='create' OR event_type='host_send_step'")
    assert await lookup(c) is None


async def test_exact_create_replay_recalls_binding_but_missing_evidence_is_not_recreated(root_case):
    c = root_case
    await send(c)
    original = await lookup(c)
    [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE action='create'")
    request = json.loads(event["payload_json"])["request"]
    replay = await c.manager.register_job(c.actor, kind="send", session_id=c.sid,
        system_key=f"send:{request['send_index']}", request=request)
    assert replay.replayed and replay.job.job_id == original["job_id"]
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='host_send_step'")
    with pytest.raises(JobError, match="evidence_missing"):
        await c.manager.register_job(c.actor, kind="send", session_id=c.sid,
            system_key=f"send:{request['send_index']}", request=request)
    assert len(c.model.calls) == 1
    with pytest.raises(JobError, match="evidence_missing"):
        await lookup(c)


async def test_two_connections_cannot_register_two_jobs_for_one_host_step(root_case):
    c = root_case
    reopened = await c.connect()
    source = root_context._document(**SOURCE)
    async def register(manager, index):
        return await manager.register_job(c.actor, kind="send", session_id=c.sid,
            system_key=f"send:{index}", request={"send_index": index, "model_attempt_protocol": 1,
                "host_context": source, steps.FIELD: {"version": 1, **STEP}})
    try:
        results = await asyncio.gather(register(c.manager, 1), register(JobManager(reopened), 2), return_exceptions=True)
        assert len([r for r in results if isinstance(r, JobError) and r.code == "host_send_step_already_registered"]) == 1
        assert len([r for r in results if not isinstance(r, BaseException)]) == 1
        assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")) == 1
        assert not c.model.calls
    finally:
        await reopened.close()


async def test_model_cannot_bind_another_host_stage_and_context_does_not_leak(root_case):
    c = root_case
    with pytest.raises(JobError, match="root_required"):
        c.runtime.host_send_step(c.sid, **STEP)
    async def before(binding):
        with pytest.raises(JobError, match="requires_host"):
            with c.runtime.host_send_step(c.sid, **{**STEP, "step_id": "forged"}):
                pytest.fail("a model execution cannot assign its host stage")
    c.model.before = before
    await send(c)
    assert steps.for_send(c.runtime, c.sid) is None


async def test_missing_binding_index_is_not_proof_that_an_old_step_never_ran(root_case):
    c = root_case
    await send(c)
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='host_send_step'")
    with pytest.raises(JobError, match="host_send_step_evidence_missing"):
        await lookup(c)
    with pytest.raises(JobError, match="host_send_step_evidence_missing"):
        await send(c)
    assert len(c.model.calls) == 1
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")) == 1


async def test_boolean_binding_version_cannot_match_integer_version(root_case):
    c = root_case
    await send(c)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='host_send_step'")
    payload = json.loads(event['payload_json'])
    payload['version'] = True
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload),event['event_id']))
    with pytest.raises(JobError, match="host_send_step_evidence_corrupt"):
        await lookup(c)


@pytest.mark.parametrize("fault", ["binding_hash", "job_id", "root_hash", "creation_hash", "step", "oversized"])
async def test_corrupt_original_binding_refuses_lookup_and_redispatch(root_case, fault):
    c = root_case
    await send(c)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='host_send_step'")
    payload = json.loads(event["payload_json"])
    if fault == "binding_hash":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET request_hash=? WHERE event_id=?", ("0" * 64, event["event_id"]))
    else:
        column = {"job_id": "job_id", "root_hash": "root_context_hash", "creation_hash": "creation_request_hash", "step": "step_id", "oversized": "extra"}[fault]
        if fault == "step":
            payload["step"][column] = "changed"
        else:
            payload[column] = "x" * 40000 if fault == "oversized" else "changed"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), event["event_id"]))
    with pytest.raises(JobError, match="host_send_step_evidence_"):
        await lookup(c)
    with pytest.raises(JobError, match="host_send_step_evidence_"):
        await send(c)
    assert len(c.model.calls) == 1
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")) == 1


async def test_create_commit_lost_ack_keeps_original_unstarted_job_without_retry(root_case, monkeypatch):
    c = root_case
    original = c.manager.register_job
    async def lose_ack(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("request", {}).get(steps.FIELD):
            raise ConnectionError("original create committed; acknowledgement lost")
        return result
    monkeypatch.setattr(c.manager, "register_job", lose_ack)
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await send(c)
    first = await lookup(c)
    assert first is not None and not c.model.calls
    [job] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs}")
    assert job["status"] == "queued" and job["generation"] == 0
    monkeypatch.setattr(c.manager, "register_job", original)
    with pytest.raises(JobError, match="already_registered"):
        await send(c)
    assert await lookup(c) == first and not c.model.calls


async def test_cancelled_original_model_does_not_erase_its_step_or_allow_another_send(root_case):
    c = root_case
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def before(binding):
        calls.append(binding.lease.job_id)
        entered.set()
        await release.wait()
    c.model.before = before
    sending = asyncio.create_task(send(c))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        original = await lookup(c)
        assert original["job_id"] == calls[0]
        with pytest.raises(JobError, match="already_registered"):
            await send(c)
        assert len(calls) == 1
        assert not original["resource_stop_confirmed"] and not original["billing_settled"]
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)


async def test_host_can_require_steps_and_cannot_downgrade_that_requirement_mid_run(root_case):
    c = root_case
    runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
        executor_id="strict-host", boot_id="boot", require_host_send_steps=True)
    loop = StatefulAgentLoop(llm=c.model, store=c.store, task_runtime=runtime, config=c.loop.config)
    try:
        with runtime.root_context(c.sid, **SOURCE):
            with pytest.raises(JobError, match="host_send_step_required"):
                await loop.send("cannot silently use an unplanned paid root", c.sid)
            assert not c.model.calls
            with runtime.host_send_step(c.sid, **STEP):
                assert (await loop.send("proper original planned stage", c.sid)).status == "completed"
        runtime.require_host_send_steps = False
        with pytest.raises(JobError, match="configuration_changed"):
            await loop.send("a runtime cannot disable its original host requirement", c.sid)
        assert len(c.model.calls) == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("value", [None, 1, "true"])
async def test_step_requirement_is_a_strict_host_boolean(value):
    with pytest.raises(JobError, match="invalid_host_send_step_requirement"):
        TaskRuntime(object(), tenant_id="test", principal_id="host", executor_id="e", boot_id="b", require_host_send_steps=value)


async def test_boolean_root_protocol_is_rejected_before_any_registration(root_case):
    c = root_case
    source = root_context._document(**SOURCE)
    source["version"] = True
    with pytest.raises(JobError, match="main_root_required|root_context_evidence_corrupt"):
        await c.manager.register_job(c.actor, kind="send", session_id=c.sid, system_key="send:1", request={
            "send_index": 1, "model_attempt_protocol": 1, "host_context": source,
            steps.FIELD: {"version": 1, **STEP}})
    assert not c.model.calls
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")


async def test_test_provider_counter_excludes_requests_refused_before_transport():
    from types import SimpleNamespace

    from tests.real.test_real_async_boundaries_192 import BoundedProvider
    class FakeTransport:
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            return SimpleNamespace(token_usage=SimpleNamespace(total_tokens=10))
    # Avoid constructing any real client in this unit regression.
    provider = object.__new__(BoundedProvider)
    provider.inner, provider.limit, provider.calls = FakeTransport(), 1, 0
    provider.refused_calls, provider.responses, provider.hold_children = 0, [], False
    await provider.complete(object())
    with pytest.raises(AssertionError, match="exceeded"):
        await provider.complete(object())
    values = {}
    provider.report(values.__setitem__)
    assert values == {"real_provider_requests": 1, "local_provider_calls_refused": 1, "reported_tokens": 10}
    assert provider.inner.calls == provider.calls == 1


@pytest.mark.parametrize("field,value", [("namespace", ""), ("plan_hash", "x" * 64), ("step_id", " whitespace "), ("step_id", "a" * 129)])
async def test_invalid_descriptor_is_refused_before_execution(field, value):
    with pytest.raises(JobError, match="invalid_host_send_step"):
        steps.descriptor(**{**STEP, field: value})
