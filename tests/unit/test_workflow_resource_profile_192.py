"""Bound resources participate in original admission, not mutable metadata."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from power_loop import ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import workflow_profile as profile
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    execution_contract,
    require_managed_creation,
)
from power_loop.workflow import InProcessExecutor, SharedBudget, WorkflowSpec, create_workflow
from power_loop.workflow.facade import WorkflowToolInputError, WorkflowToolPorts, prepare_workflow

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "registry_runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def spec():
    return WorkflowSpec.from_json({"name": "bound", "root": {
        "type": "agent", "id": "BUILD", "input": "work",
        "spec": {"name": "BUILD", "system_prompt": "ROLE-BUILD", "tools": []}}})


def binding(s=None, *, tenant="t", principal="p", sid="s", tools=None, **options):
    return WorkflowResourceBinding("host", "1.0.0", "original-resource", 1, tenant, principal, sid,
        "a" * 64, execution_contract(s or spec(), tools), **options)


async def test_v2_captures_original_binding_without_changing_legacy_profile():
    s = spec()
    legacy = profile.creation_request(s, ["read"])
    original = binding(s, tools=["read"])
    request = profile.creation_request(s, ["read"], resource_binding=original)
    assert legacy["execution_profile"]["version"] == 1 and "resource_binding" not in legacy["execution_profile"]
    assert request["execution_profile"]["version"] == 2
    assert request["execution_profile"]["resource_binding"] == original.to_dict()
    assert profile.validate(request["execution_profile"], spec=s, completion_callback=False,
        allowed_tools=["read"]) == request["execution_profile"]
    with pytest.raises(JobError, match="adapter_required:resource_binding"):
        profile.require_native(request["execution_profile"])


@pytest.mark.parametrize("fault", ["contract", "budget", "files", "notify", "extra", "version", "protocol", "missing"])
async def test_v2_rejects_incomplete_or_mismatched_capability_evidence(fault):
    s = spec()
    data = profile.capture(s, resource_binding=binding(s))
    if fault == "contract":
        data["resource_binding"]["contract_sha256"] = "b" * 64
    elif fault in ("budget", "files"):
        data["resource_binding"]["budget_scope_id" if fault == "budget" else "workspace_id"] = "unbound"
    elif fault == "notify":
        data["resource_binding"]["notification_mode"] = "host_receipt"
    elif fault == "extra":
        data["secret"] = "not-accepted"
    elif fault == "version":
        data["version"] = True
    elif fault == "protocol":
        data["protocol"] = "untrusted"
    else:
        data.pop("resource_binding")
    with pytest.raises(JobError, match="workflow_profile_mismatch"):
        profile.validate(data, spec=s, completion_callback=False)


async def test_v2_cannot_expand_original_effective_tools():
    data = profile.capture(spec(), resource_binding=binding(tools=[]), allowed_tools=[])
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.validate(data, spec=spec(), completion_callback=False, allowed_tools=None)


async def test_profile_does_not_accept_untyped_binding_or_malformed_external_budget():
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.capture(spec(), resource_binding=binding().to_dict())
    data = profile.capture(spec(), budget=object())
    data["budget"]["class"] = ""
    with pytest.raises(JobError, match="profile_mismatch"):
        profile.validate(data, spec=spec(), completion_callback=False)


@pytest.mark.parametrize("port", ["executor", "budget", "file_io", "notification"])
async def test_prepared_identity_is_not_external_execution_control(port):
    options = {"binding": binding(), "executor": InProcessExecutor()}
    if port == "executor":
        options["executor"] = SimpleNamespace(run_agent=lambda: None)
    elif port == "budget":
        options.update(binding=binding(budget_scope_id="pool"), budget=object())
    elif port == "file_io":
        options.update(binding=binding(workspace_id="workspace"), file_io=object())
    else:
        options.update(binding=binding(notification_mode="host_receipt"), on_complete=lambda *_: None)
    with pytest.raises(JobError, match="workflow_resource_adapter_required:" + port):
        require_managed_creation(WorkflowResources(**options))


async def test_untyped_prepared_object_does_not_grant_execution():
    with pytest.raises(JobError, match="resources_invalid"):
        require_managed_creation(None)


@pytest.mark.parametrize("foreign", [False, True])
async def test_profile_read_requires_original_request_hash_and_actual_owner_scope(registry_runtime, foreign):
    _, manager, actor, _, caller, connect, _ = registry_runtime
    original = binding(tenant=actor.tenant_id, principal=actor.principal_id,
        sid="foreign" if foreign else actor.session_id, tools=[])
    request = profile.creation_request(spec(), [], resource_binding=original)
    registration = await manager.register_job(actor, kind="workflow", title=spec().name, parent=caller,
        independent=True, system_key="bound-profile", request=request)
    reader = JobManager(await connect(), now_ms=manager.now)
    if foreign:
        with pytest.raises(JobError, match="scope_mismatch"):
            await reader.get_workflow_profile(actor, registration.job.job_id)
    else:
        assert await reader.get_workflow_profile(actor, registration.job.job_id) == request["execution_profile"]
        event = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?",
            (registration.operation.operation_id,))
        payload = json.loads(event["payload_json"])
        payload["request"]["execution_profile"]["resource_binding"]["resource_id"] = "tampered"
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?",
            (json.dumps(payload), registration.operation.operation_id))
        with pytest.raises(JobError, match="creation_mismatch"):
            await reader.get_workflow_profile(actor, registration.job.job_id)


def install(fixture, *, fault=None, repeat=False):
    store, manager, runtime, loop, model, sid, state, _ = fixture
    s = WorkflowSpec.from_json(copy.deepcopy(state["spec"]))
    original = binding(s, tenant=runtime.tenant_id, principal=runtime.principal_id, sid=sid,
        tools=[], budget_scope_id="original-pool")
    pool = SharedBudget(10000)
    pool.commit({"total_tokens": 7})
    registry = WorkflowResourceRegistry()
    runtime.workflow_resources = registry
    track = {"resolve": 0, "authorize": 0, "allow": True}
    resources = WorkflowResources(original,
        SimpleNamespace(run_agent=lambda: None) if fault == "external" else InProcessExecutor(), budget=pool)
    async def prepare(_scope, _contract):
        return resources
    async def resolve(saved):
        track["resolve"] += 1
        assert saved == original
        if fault == "mutable" and track["resolve"] == 1:
            state["workflows"][0].spec.root.steps[0].spec.metadata["untrusted_change"] = "after-snapshot"
        if track["resolve"] == 2:
            if fault == "revoked":
                registry.unregister("host", "1.0.0")
            if fault == "capacity":
                pool.max_tokens = 20000
            if fault == "rewound":
                pool._spent = 0
            if fault == "denied":
                track["allow"] = False
            if fault == "floor":
                pool._floor = 5000
            if fault == "wait":
                track["entered"].set()
                await track["release"].wait()
        return resources
    async def authorize(*_args):
        track["authorize"] += 1
        return track["allow"]
    registry.register("host", "1.0.0", WorkflowResourceAdapter(prepare, resolve, authorize))
    async def launch():
        workflow = create_workflow(s, parent_loop=get_current_loop(), parent_session_id=get_session_id(),
            allowed_tools=[], resource_binding=original)
        state["workflows"].append(workflow)
        handle = await workflow.start(detached=True)
        state["handles"].append(handle)
        if repeat:
            pool.commit({"total_tokens": 13})
            runtime.workflow_resources = None
            assert await workflow.start(detached=True) is handle
        return json.dumps({"run_id": handle.run_id, "status": "accepted"})
    loop.tool_registry.register(ToolDefinition(name="launch", description="Start bound workflow",
        input_schema={"type": "object"}), launch, overwrite=True)
    return original, pool, registry, track


async def test_actual_create_and_dispatch_reauthorize_original_resources_and_persist_v2(managed_workflow):
    original, pool, _, track = install(managed_workflow)
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    await settle(managed_workflow)
    [handle] = state["handles"]
    saved = await manager.get_workflow_profile(runtime.actor(sid), handle.job_id)
    assert saved["version"] == 2 and saved["resource_binding"] == original.to_dict()
    assert saved["budget"]["spent_at_admission"] == 7
    assert track == {"resolve": 2, "authorize": 2, "allow": True}
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 1
    assert pool.spent == 27
    job = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert job.result["status"] == "completed"
    receipt = await manager.get_operation(runtime.actor(sid), handle.operation_id)
    assert receipt.call_ref["id"] != receipt.receipt_ref["id"]


async def test_external_executor_cannot_be_admitted_without_stop_evidence(managed_workflow):
    install(managed_workflow, fault="external")
    store, manager, _, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    assert not state["handles"] and model.counts["BUILD"] == 0
    assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")
    assert any("workflow_resource_adapter_required:executor" in m.content for m in await store.load_all_messages(sid)
        if m.role == "tool" and m.name == "launch")


@pytest.mark.parametrize("fault", ["denied", "revoked", "capacity", "rewound", "floor"])
async def test_changed_resources_after_admission_block_all_child_provider_dispatch(managed_workflow, fault):
    _, _, _, track = install(managed_workflow, fault=fault)
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    await settle(managed_workflow)
    [handle] = state["handles"]
    job = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert job.result["status"] == "failed"
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    assert track["resolve"] == 2
    assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1


async def test_sdk_mutation_during_resource_resolution_cannot_change_frozen_execution(managed_workflow):
    original, _, _, _ = install(managed_workflow, fault="mutable")
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    await settle(managed_workflow)
    assert state["workflows"][0].spec.root.steps[0].spec.metadata["untrusted_change"] == "after-snapshot"
    saved = await manager.get_workflow_profile(runtime.actor(sid), state["handles"][0].job_id)
    assert saved["resource_binding"] == original.to_dict()
    assert model.counts["BUILD"] == 1 and model.counts["OTHER"] == 0
    job = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    child = await manager.store.get_session(job.result["results"]["BUILD"]["session_id"])
    assert "untrusted_change" not in child.metadata


async def test_cancel_during_dispatch_resolution_cannot_start_a_child(managed_workflow):
    _, _, _, track = install(managed_workflow, fault="wait")
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    track.update(entered=asyncio.Event(), release=asyncio.Event())
    try:
        await loop.send("launch", sid)
        await asyncio.wait_for(track["entered"].wait(), 2)
        state["handles"][0].cancel("stop before resource dispatch")
        track["release"].set()
        await settle(managed_workflow)
        job = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
        assert job.result["status"] == "cancelled"
        assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    finally:
        track["release"].set()


async def test_exact_call_replay_uses_saved_receipt_without_resolver_or_budget_refill(managed_workflow):
    _, pool, _, track = install(managed_workflow, repeat=True)
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("launch", sid)
    await settle(managed_workflow)
    assert len(state["handles"]) == 1 and pool.spent == 20
    assert track["resolve"] == 1
    rows = await store.load_all_messages(sid)
    assert not any(r.role == "tool" and r.name == "launch" and "Error" in r.content for r in rows)
    # Registry removal prevents the first execution; replay never dispatches a substitute.
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1


@pytest.mark.parametrize("key", ["resource_binding", "resource_adapter", "workflow_resources"])
async def test_model_cannot_choose_host_resource_identity(key):
    with pytest.raises(WorkflowToolInputError, match="invalid_workflow_parameters"):
        await prepare_workflow({"spec": spec().to_dict(), key: "host"}, WorkflowToolPorts())


async def test_bound_and_legacy_factories_cannot_be_mixed():
    with pytest.raises(WorkflowToolInputError, match="ports_conflict"):
        WorkflowToolPorts(resource_adapter=("host", "1.0.0"), executor_factory=lambda *_: InProcessExecutor())
    with pytest.raises(JobError, match="ports_conflict"):
        create_workflow(spec(), parent_loop=object(), executor=InProcessExecutor(), resource_binding=binding())


async def test_resource_binding_cannot_be_ignored_by_legacy_synchronous_sdk_execution():
    wf = create_workflow(spec(), parent_loop=object(), resource_binding=binding())
    with pytest.raises(JobError, match="requires_detached_task_runtime"):
        await wf.run()


async def test_bound_callback_profile_does_not_claim_business_delivery_acceptance():
    data = profile.capture(spec(), completion_callback=True,
        resource_binding=binding(notification_mode="host_receipt"))
    assert data["notification"] == {"kind": "opaque_callback", "settlement": "library_exit_only"}
    with pytest.raises(JobError, match="adapter_required:resource_binding"):
        profile.require_native(data)


async def test_wrong_scope_or_contract_binding_cannot_authorize_actual_workflow(managed_workflow):
    install(managed_workflow)
    _, _, _, loop, model, sid, state, _ = managed_workflow
    async def launch():
        wf = create_workflow(state["spec"], parent_loop=loop, parent_session_id=sid, resource_binding=binding())
        return str(await wf.start(detached=True))
    loop.tool_registry.register(ToolDefinition(name="launch", description="Wrong scope",
        input_schema={"type": "object"}), launch, overwrite=True)
    await loop.send("launch", sid)
    assert not state["handles"] and model.counts["BUILD"] == 0
