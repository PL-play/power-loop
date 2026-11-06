"""Host policy is applied before a frozen workflow is admitted."""
import asyncio
from dataclasses import replace

import pytest

from power_loop.core.agent_context import reset_effective_tools, set_effective_tools
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkflowSpecError
from power_loop.workflow.facade import (
    WorkflowToolPorts,
    prepare_workflow,
    register_workflow_tool,
    workflow_tool_definition,
)

SPEC = {"name":"facade","root":{"type":"agent","id":"calculate","input":"19+23","spec":{"name":"calculate","system_prompt":"Return only 42.","tools":[],"max_rounds":2}}}
pytestmark = pytest.mark.asyncio


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


async def test_schema_ports_and_registration_do_not_grant_model_selected_host_objects():
    definition = workflow_tool_definition(allow_inline_spec=False)
    assert "spec" not in definition.input_schema["properties"]
    assert not {"detached", "wait", "sync", "ports", "executor"} & definition.input_schema["properties"].keys()
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(allow_inline_spec=False))
    assert registry.names() == ["workflow"]
    with pytest.raises(ValueError, match="invalid_workflow_parameters"):
        await prepare_workflow({"spec": SPEC, "ports": {}}, WorkflowToolPorts())
    with pytest.raises(ValueError, match="file_required"):
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(allow_inline_spec=False))
    with pytest.raises(TypeError, match="spec_transform"):
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(spec_transform=lambda spec: {}))
    with pytest.raises(WorkflowSpecError, match="denied"):
        def reject(spec):
            raise WorkflowSpecError(["denied"])
        await prepare_workflow({"spec": SPEC}, WorkflowToolPorts(spec_transform=reject))
