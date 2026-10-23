"""Frozen runtime capabilities, original-call fingerprints and mutable callers."""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import workflow_profile as profile
from power_loop.runtime.jobs import JobActor, JobError, JobManager, _json
from power_loop.workflow.engine import InProcessExecutor
from power_loop.workflow.result import SharedBudget
from power_loop.workflow.spec import WorkflowSpec

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "registry_runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def spec(*, budget=None):
    data = {"name": "frozen-profile", "root": {"type": "agent", "id": "build", "input": "42",
        "spec": {"name": "build", "tools": [], "system_prompt": "answer"}}}
    if budget:
        data["budget"] = budget
    return WorkflowSpec.from_json(data)


@pytest.mark.parametrize("executor", [None, InProcessExecutor()])
@pytest.mark.parametrize("with_budget", [False, True])
async def test_native_profile_is_strict_serializable_detached_evidence(executor, with_budget):
    s = spec(budget={"max_tokens": 200, "stop_at_remaining_pct": 10} if with_budget else None)
    data = profile.capture(s, executor=executor)
    assert profile.require_native(data) == data
    assert data["executor"] == {"kind": "native_in_process"}
    if with_budget:
        assert data["budget"] == {"kind": "spec_soft", "max_tokens": 200, "stop_at_remaining_pct": 10.0,
            "continuation_policy": "logical_run_cumulative"}
    else:
        assert data["budget"] == {"kind": "unbounded"}
    assert json.loads(_json(data)) == data
    request = profile.creation_request(s, {"z", "a"}, executor=executor)
    request["execution_profile"]["executor"]["kind"] = "mutated"
    request["spec"]["name"] = "mutated"
    assert s.name == "frozen-profile" and data["executor"]["kind"] == "native_in_process"
    assert request["allowed_tools"] == ["a", "z"]


@pytest.mark.parametrize("adapter", ["executor", "budget", "file_io", "notification"])
async def test_external_ports_are_recorded_without_serializing_host_secrets_and_not_reconstructed(adapter):
    class HostPort:
        api_key = "SENSITIVE-HOST-CREDENTIAL"
        def __repr__(self):
            raise AssertionError("must not serialize host object")
    class NativeSubclass(InProcessExecutor):
        pass
    kwargs = {adapter: HostPort()} if adapter != "notification" else {"completion_callback": True}
    if adapter == "executor":
        kwargs[adapter] = NativeSubclass()
    data = profile.capture(spec(), **kwargs)
    assert "SENSITIVE" not in _json(data)
    assert data[adapter]["kind"] == ("opaque_callback" if adapter == "notification" else "external")
    with pytest.raises(JobError, match="adapter_required:" + adapter):
        profile.require_native(data)


async def test_runtime_soft_budget_keeps_original_spend_floor_and_host_responsibility():
    pool = SharedBudget(100, stop_at_remaining_pct=20)
    pool.commit({"total_tokens": 7})
    data = profile.capture(spec(budget={"max_tokens": 999}), budget=pool)
    assert data["budget"] == {"kind": "runtime_soft", "max_tokens": 100, "floor_tokens": 20.0,
        "spent_at_admission": 7, "continuation_policy": "host_adapter_required"}
    pool.commit({"total_tokens": 11})
    assert data["budget"]["spent_at_admission"] == 7 and pool.spent == 18
    with pytest.raises(JobError, match="adapter_required:budget"):
        profile.require_native(data)


@pytest.mark.parametrize("mutation", [
    "version_bool", "protocol", "missing", "extra", "bad_executor", "empty_class", "bad_file",
    "budget_list", "budget_kind", "budget_widen", "callback", "callback_type", "nonfinite", "absent",
])
async def test_invalid_or_incomplete_profile_never_defaults_to_native(mutation):
    s, callback = spec(), False
    data = profile.capture(s)
    if mutation == "version_bool":
        data["version"] = True
    elif mutation == "protocol":
        data["protocol"] = "future"
    elif mutation == "missing":
        data.pop("executor")
    elif mutation == "extra":
        data["arbitrary"] = 1
    elif mutation == "bad_executor":
        data["executor"] = {"kind": "native_in_process", "extra": 1}
    elif mutation == "empty_class":
        data["executor"] = {"kind": "external", "class": ""}
    elif mutation == "bad_file":
        data["file_io"] = None
    elif mutation == "budget_list":
        data["budget"] = []
    elif mutation == "budget_kind":
        data["budget"] = {"kind": "unknown"}
    elif mutation == "budget_widen":
        s = spec(budget={"max_tokens": 20})
    elif mutation == "callback":
        callback = True
    elif mutation == "callback_type":
        callback = 0
    elif mutation == "nonfinite":
        data["executor"] = {"kind": "external", "class": float("nan")}
    elif mutation == "absent":
        data = None
    with pytest.raises(JobError, match="workflow_profile_(mismatch|unknown)"):
        profile.validate(data, spec=s, completion_callback=callback)


@pytest.mark.parametrize("field,value", [
    ("max_tokens", True), ("max_tokens", -1), ("spent_at_admission", True), ("spent_at_admission", -1),
    ("floor_tokens", True), ("floor_tokens", float("nan")), ("floor_tokens", float("inf")),
    ("floor_tokens", -1), ("floor_tokens", 101), ("continuation_policy", "fresh_unlimited"),
])
async def test_bad_runtime_budget_does_not_erase_original_limit(field, value):
    s = spec()
    data = profile.capture(s, budget=SharedBudget(100))
    data["budget"][field] = value
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.validate(data, spec=s, completion_callback=False)


async def test_original_admission_survives_object_replacement_before_background_dispatch(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, connect = managed_workflow
    state["on_complete"] = None
    original = manager.register_job
    async def mutate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("kind") == "workflow":
            workflow = state["workflows"][0]
            workflow.spec = spec()
            workflow._executor = SimpleNamespace(run_agent=None)
            workflow._budget = SharedBudget(0)
            workflow._file_io = object()
            workflow._allowed_tools = frozenset({"unexpected"})
        return result
    monkeypatch.setattr(manager, "register_job", mutate)
    assert (await loop.send("launch", sid)).status == "completed"
    await settle(managed_workflow)
    [handle] = state["handles"]
    actor = runtime.actor(sid)
    data = await manager.get_workflow_profile(actor, handle.job_id)
    assert profile.require_native(data) == data
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 1
    saved = await manager.get_job(actor, handle.job_id)
    assert saved.outcome == "succeeded" and saved.title == "registered-workflow"
    assert await JobManager(await connect()).get_workflow_profile(actor, handle.job_id) == data
    assert saved.generation == 1  # No profile reader grants a new execution.


@pytest.mark.parametrize("mutation", ["legacy", "malformed", "tamper", "callback", "spec"])
async def test_historical_request_unknown_or_corrupt_cannot_supply_recovery_defaults(registry_runtime, mutation):
    _, manager, actor, _, caller, _, _ = registry_runtime
    s = spec()
    request = profile.creation_request(s, None)
    if mutation == "legacy":
        request.pop("execution_profile")
    elif mutation == "malformed":
        request["execution_profile"]["version"] = True
    elif mutation == "callback":
        request["completion_callback"] = True
    elif mutation == "spec":
        request["spec"] = {"root": "bad"}
    reg = await manager.register_job(actor, kind="workflow", parent=caller, independent=True,
        system_key="profile", title=s.name, request=request)
    if mutation == "tamper":
        event = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (reg.operation.operation_id,))
        data = json.loads(event["payload_json"])
        data["request"]["execution_profile"]["file_io"] = {"kind": "external", "class": "other.Port"}
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (_json(data), reg.operation.operation_id))
    with pytest.raises(JobError, match="workflow_profile_(unknown|mismatch|creation_mismatch)"):
        await manager.get_workflow_profile(actor, reg.job.job_id)


async def test_profile_reader_has_normal_tree_authorization_and_does_not_require_driver(registry_runtime):
    _, manager, actor, root, caller, _, _ = registry_runtime
    s = spec()
    reg = await manager.register_job(actor, kind="workflow", parent=caller, independent=True,
        system_key="profile", title=s.name, request=profile.creation_request(s, []))
    assert (await manager.get_workflow_profile(actor, reg.job.job_id))["executor"] == {"kind": "native_in_process"}
    for stranger in (replace(actor, tenant_id="other"), replace(actor, principal_id="other"), JobActor(actor.tenant_id, actor.principal_id, "other")):
        with pytest.raises(JobError):
            await manager.get_workflow_profile(stranger, reg.job.job_id)
    with pytest.raises(JobError, match="profile_target_mismatch"):
        await manager.get_workflow_profile(actor, root.job.job_id)
    assert (await manager.get_job(actor, reg.job.job_id)).generation == 0


async def test_strict_callback_parameter_and_nonfinite_pool_fail_before_creating_request():
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.capture(spec(), completion_callback="yes")
    pool = SharedBudget(10, stop_at_remaining_pct=float("nan"))
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.creation_request(spec(), [], budget=pool)
    old = profile.capture(spec())
    copied = copy.deepcopy(old)
    profile.require_native(copied)
    assert copied == old


@pytest.mark.parametrize("fault", ["rollback", "cancel", "commit_ack"])
async def test_profile_and_original_operation_share_admission_transaction(registry_runtime, monkeypatch, fault):
    _, manager, actor, _, caller, connect, _ = registry_runtime
    s = spec(budget={"max_tokens": 200})
    request = profile.creation_request(s, [])
    original = manager._write
    @asynccontextmanager
    async def broken():
        async with original() as tx:
            yield tx
            if fault == "rollback":
                raise RuntimeError("precommit fault")
            if fault == "cancel":
                raise asyncio.CancelledError()
        if fault == "commit_ack":
            raise RuntimeError("lost commit acknowledgement")
    kwargs = dict(kind="workflow", parent=caller, independent=True, system_key="profile", title=s.name, request=request)
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", broken)
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else RuntimeError):
            await manager.register_job(actor, **kwargs)
    other = JobManager(await connect(), now_ms=manager.now)
    rows = await other.db.fetchall(f"SELECT job_id FROM {other.jobs} WHERE kind='workflow'")
    assert len(rows) == int(fault == "commit_ack")
    replay = await other.register_job(actor, **kwargs)
    assert replay.replayed == (fault == "commit_ack")
    assert await other.get_workflow_profile(actor, replay.job.job_id) == request["execution_profile"]
    assert (await other.get_job(actor, replay.job.job_id)).generation == 0


async def test_changed_runtime_budget_on_same_original_call_is_conflict_not_new_work(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state.update(repeat=True, on_complete=None)
    original = manager.register_job
    async def mutate(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("kind") == "workflow":
            state["workflows"][0]._budget = SharedBudget(100)
        return result
    monkeypatch.setattr(manager, "register_job", mutate)
    await loop.send("launch", sid)
    await settle(managed_workflow)
    assert len(state["handles"]) == 1
    rows = await store.load_all_messages(sid)
    assert any(r.role == "tool" and r.name == "launch" and "operation_conflict" in r.content for r in rows)
    assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 1
    data = await manager.get_workflow_profile(runtime.actor(sid), state["handles"][0].job_id)
    assert data["budget"] == {"kind": "unbounded"}
