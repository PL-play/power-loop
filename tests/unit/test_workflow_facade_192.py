"""Submission ports, actual async admission, and bounded owned inspection."""
from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, SessionStore, StatefulAgentLoop
from power_loop.core.agent_context import (
    get_session_id,
    reset_current_loop,
    reset_effective_tools,
    reset_session_id,
    set_current_loop,
    set_effective_tools,
    set_session_id,
)
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkflowSpecError, journal
from power_loop.workflow.facade import (
    WorkflowToolPorts,
    prepare_workflow,
    register_workflow_tool,
    run_workflow_tool,
    submit_workflow,
    workflow_tool_definition,
)

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
SPEC = {"name": "facade", "root": {"type": "agent", "id": "calculate", "input": "19+23", "spec": {
    "name": "calculate", "system_prompt": "Return only 42.", "tools": [], "max_rounds": 2}}}


class Model(LLMService):
    def __init__(self, arguments):
        self.root = None
        self.arguments = arguments
        self.calls = Counter()
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def complete(self, request, **kwargs):
        sid = get_session_id()
        self.calls[sid] += 1
        if sid == self.root:
            if self.calls[sid] <= len(self.arguments):
                return LLMResponse(raw_text="", tool_calls=[{"id": "call-"+str(self.calls[sid]), "type": "function",
                    "function": {"name": "workflow", "arguments": json.dumps(self.arguments[self.calls[sid]-1])}}])
            return LLMResponse(raw_text="WAITING" if not self.release.is_set() else "42")
        self.entered.set()
        await self.release.wait()
        return LLMResponse(raw_text="42")

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.mark.parametrize("observer_fault", [None, "compiled", "accepted", "timeout"])
async def test_native_model_receives_admission_before_actual_result_and_observer_fault_cannot_duplicate(registry_store, observer_fault):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="facade", principal_id="host", executor_id="executor", boot_id="boot")
    model = Model([{"action": "validate", "spec": SPEC}, {"action": "create", "spec": SPEC}])
    accepted, completed, compiled = [], [], []

    async def compiled_view(payload):
        compiled.append(copy.deepcopy(payload))
        payload["name"] = "malicious-view-mutation"
        if observer_fault == "compiled":
            raise RuntimeError("presentation secret")

    async def accepted_view(admission):
        accepted.append(admission)
        admission.receipt["run_id"] = "view-only-mutation"
        admission.spec.root.spec.metadata["view_mutation"] = True
        if observer_fault == "accepted":
            raise RuntimeError("presentation secret")
        if observer_fault == "timeout":
            await asyncio.Event().wait()

    async def completed_view(completion):
        completed.append(completion)

    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_compiled=compiled_view, on_accepted=accepted_view,
        on_complete=completed_view, observer_timeout_s=.02))
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5))
    sid = await loop.new_session()
    model.root = sid
    try:
        result = await asyncio.wait_for(loop.send("Create once", sid), 8)
        await asyncio.wait_for(model.entered.wait(), 3)
        assert result.final_text == "WAITING" and not completed
        rows = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
        assert len(rows) == 2
        validation, receipt = [json.loads(m.content) for m in rows]
        assert validation["status"] == "valid" and receipt["status"] == "accepted", rows
        assert receipt["workflow"] == "facade" and receipt["run_id"] != "view-only-mutation"
        assert receipt["completion_delivery"] == "owner_inbox"
        assert len(receipt["view_warnings"]) == bool(observer_fault)
        assert len(accepted) == len(compiled) == 1
        handle = accepted[0].handle
        assert not handle.task.done() and handle.job_id == receipt["job_id"]
        actor = runtime.actor(sid)
        operation = await manager.get_operation(actor, receipt["operation_id"])
        assert operation.call_ref["id"] != operation.receipt_ref["id"]
        assert operation.receipt_ref["id"] == str(rows[-1].id)
        assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow'")) == 1
        saved = await journal.read(store, sid, handle.run_id)
        assert saved["spec"]["name"] == "facade"
        assert "view_mutation" not in saved["spec"]["root"]["spec"].get("metadata", {})
        lt, st = set_current_loop(loop), set_session_id(sid)
        try:
            listed = json.loads(await run_workflow_tool({"action": "list"}))
            detail = json.loads(await run_workflow_tool({"action": "status", "run_id": handle.run_id}))
            assert listed["matched_count"] == 1 and detail["node"]["job_id"] == handle.job_id
            other = await loop.new_session()
            token = set_session_id(other)
            try:
                assert "error" in json.loads(await run_workflow_tool({"action": "status", "run_id": handle.run_id}))
            finally:
                reset_session_id(token)
        finally:
            reset_session_id(st)
            reset_current_loop(lt)
        model.release.set()
        await asyncio.wait_for(handle.task, 5)
        assert len(completed) == 1 and completed[0].parent_session_id == sid
        assert (await store.inbox_counts(sid))["pending"] == 1
        assert (await manager.get_job(actor, handle.job_id)).outcome == "succeeded"
        await loop.flush_follow_ups(sid, max_rounds=2)
        assert (await store.inbox_counts(sid))["pending"] == 0
    finally:
        model.release.set()
        if accepted:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 5)


async def test_prepare_freezes_arguments_before_async_loader_and_captures_empty_tool_clamp():
    entered, release = asyncio.Event(), asyncio.Event()
    options = {"file": "skill:sample/main.yaml", "args": {"name": "original"}, "input": ""}
    seen = []
    async def load(ref, args):
        entered.set()
        await release.wait()
        seen.append((ref, args))
        return SPEC
    token = set_effective_tools(frozenset())
    try:
        task = asyncio.create_task(prepare_workflow(options, WorkflowToolPorts(load_spec=load,
            allowed_tools_factory=lambda *_: ["bash"])))
        await entered.wait()
        options["args"]["name"] = "mutated"
        options["input"] = "mutated"
        release.set()
        prepared = await task
        assert seen == [("skill:sample/main.yaml", {"name": "original"})]
        assert prepared.allowed_tools == frozenset() and prepared.spec.input == ""
    finally:
        reset_effective_tools(token)


@pytest.mark.parametrize(("options", "code"), [({}, "exactly_one"),
    ({"spec": SPEC, "file": "x.yaml"}, "exactly_one"), ({"spec": SPEC, "args": {}}, "args_require_file"),
    ({"file": "x.yaml", "args": []}, "args_must_be_object"), ({"spec": SPEC, "input": None}, "input_must_be_string")])
async def test_prepare_rejects_ambiguous_or_invalid_input(options, code):
    with pytest.raises(ValueError, match=code):
        await prepare_workflow(options, WorkflowToolPorts())


async def test_file_loader_applies_workspace_boundary_size_limit_and_same_policy(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.yaml").write_text("name: original\njobs:\n  a:\n    prompt: Return 42.\n    tools: []\n")
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret")
    (workspace / "escape.yaml").symlink_to(outside)
    (workspace / "large.yaml").write_bytes(b"a" * (1024*1024+1))
    ports = WorkflowToolPorts(spec_transform=lambda spec: replace(spec, name="clamped"))
    with runtime_env_context(RuntimeEnv(workspace_dir=workspace)):
        prepared = await prepare_workflow({"file": "main.yaml"}, ports)
        assert prepared.spec.name == "clamped" and prepared.spec.root.spec.tools == []
        for ref in ("../outside.yaml", "escape.yaml", str(outside)):
            with pytest.raises(ValueError, match="escapes"):
                await prepare_workflow({"file": ref}, ports)
        with pytest.raises(ValueError, match="too_large"):
            await prepare_workflow({"file": "large.yaml"}, ports)


@pytest.mark.parametrize("mode", [{"detached": False}, {"detached": 1}, {"wait": False}, {"sync": True}])
async def test_sync_inputs_rejected_before_loading_creating_or_charging(tmp_path, mode):
    store = await SessionStore.open(str(tmp_path / "legacy.db"))
    model = Model([])
    loop = StatefulAgentLoop(llm=model, store=store, config=AgentLoopConfig(delegation_policy="single_level"))
    sid = await loop.new_session()
    lt, st = set_current_loop(loop), set_session_id(sid)
    seen = []
    try:
        with pytest.raises(ValueError, match="async_only"):
            await submit_workflow({"file": "main.yaml", **mode}, WorkflowToolPorts(load_spec=lambda *_: seen.append(1)))
        assert not seen and not model.calls
        assert await store.get_runtime_state(sid, journal.INDEX_KEY) is None
    finally:
        reset_session_id(st)
        reset_current_loop(lt)
        await store.close()


async def test_schema_ports_and_registration_do_not_grant_model_selected_host_objects():
    definition = workflow_tool_definition(allow_inline_spec=False)
    assert "spec" not in definition.input_schema["properties"]
    assert not {"detached", "wait", "sync", "ports", "executor"} & definition.input_schema["properties"].keys()
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(allow_inline_spec=False))
    assert registry.names() == ["workflow"]
    result = json.loads(await registry.get("workflow").handler(action="create", ports={"allow_inline_spec": True}))
    assert result["error"]["code"] == "invalid_workflow_parameters"
    with pytest.raises(ValueError, match="file_required"):
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(allow_inline_spec=False))
    with pytest.raises(TypeError, match="spec_transform"):
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(spec_transform=lambda spec: {}))
    with pytest.raises(WorkflowSpecError, match="denied"):
        def reject(spec):
            raise WorkflowSpecError(["denied"])
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(spec_transform=reject))


@pytest.fixture
async def legacy_context(tmp_path):
    store = await SessionStore.open(str(tmp_path / "legacy-context.db"))
    model = Model([])
    loop = StatefulAgentLoop(llm=model, store=store, config=AgentLoopConfig(delegation_policy="single_level"))
    sid = await loop.new_session()
    lt, st = set_current_loop(loop), set_session_id(sid)
    try:
        yield loop, sid, store, model
    finally:
        model.release.set()
        reset_session_id(st)
        reset_current_loop(lt)
        await store.close()


@pytest.mark.parametrize("role", ["workflow_node", "subagent"])
@pytest.mark.parametrize("action", ["create", "validate", "status"])
async def test_child_roles_cannot_use_workflow_even_when_tool_remains_installed(legacy_context, role, action):
    _, sid, store, model = legacy_context
    child = await store.create_session(parent_session_id=sid, metadata={"execution_role": role})
    token = set_session_id(child)
    called = []
    try:
        options = {"action": action, **({"run_id": "a"*16} if action == "status" else {"file": "main.yaml"})}
        result = json.loads(await run_workflow_tool(options, ports=WorkflowToolPorts(load_spec=lambda *_: called.append(True))))
        assert result["error"]["code"] == "execution_role_forbids_workflow"
        assert not called and not model.calls
        assert await store.get_runtime_state(child, journal.INDEX_KEY) is None
    finally:
        reset_session_id(token)


@pytest.mark.parametrize("fault", ["host_value", "host_runtime", "compile", "missing"])
async def test_model_errors_do_not_leak_arbitrary_host_secrets_or_accept_work(legacy_context, fault):
    _, sid, store, model = legacy_context
    def load(*_):
        if fault == "host_value":
            raise ValueError("sensitive_host_credential")
        if fault == "host_runtime":
            raise RuntimeError("sensitive_host_credential")
        if fault == "compile":
            raise WorkflowSpecError(["node a needs spec.name"])
        raise FileNotFoundError("sensitive_host_credential")
    response = await run_workflow_tool({"action": "create", "file": "main.yaml"}, ports=WorkflowToolPorts(load_spec=load))
    assert "sensitive_host_credential" not in response
    assert "error" in json.loads(response)
    assert not model.calls and await store.get_runtime_state(sid, journal.INDEX_KEY) is None


@pytest.mark.parametrize("arguments", [{"action": "status", "run_id": "a"*16, "detail": 1},
    {"action": "list", "limit": True}, {"action": "list", "max_chars": True},
    {"action": "list", "cursor": "fake"}, {"action": "create", "spec": SPEC, "force": True}])
async def test_invalid_query_and_submission_options_do_not_start_work(legacy_context, arguments):
    _, sid, store, model = legacy_context
    response = json.loads(await run_workflow_tool(arguments))
    assert "error" in response and not model.calls
    assert await store.get_runtime_state(sid, journal.INDEX_KEY) is None


async def test_alias_ports_and_async_only_contract_are_shared_with_unified_entry(legacy_context):
    from power_loop.workflow import register_workflow_tools
    loop, sid, store, model = legacy_context
    seen = []
    ports = WorkflowToolPorts(load_spec=lambda *_: SPEC, allow_inline_spec=False,
        spec_transform=lambda spec: replace(spec, name="same-policy"), on_accepted=lambda a: seen.append(a))
    registry = ToolRegistry()
    register_workflow_tools(registry, ports=ports)
    validation = await registry.get("validate_workflow").handler(file="main.yaml")
    assert "VALID." in validation and "same-policy" in validation and not model.calls
    for mode in ({"detached": False}, {"wait": True}, {"sync": False}):
        response = json.loads(await registry.get("create_workflow").handler(file="main.yaml", **mode))
        assert response["error"]["code"] == "workflow_async_only"
    receipt = json.loads(await registry.get("create_workflow").handler(file="main.yaml", action="help"))
    assert receipt["status"] == "accepted" and receipt["workflow"] == "same-policy"
    assert len(seen) == 1 and not seen[0].handle.task.done()
    model.release.set()
    await seen[0].handle.task
    assert (await journal.read(store, sid, receipt["run_id"]))["status"] == "completed"
    with pytest.raises(ValueError, match="mutually_exclusive"):
        register_workflow_tools(ToolRegistry(), ports=ports, budget_factory=lambda *_: None)
    assert loop.tool_registry is None
