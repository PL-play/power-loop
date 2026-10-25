"""Generic host tools use the same admission, role, receipt and wrapper seams."""

import asyncio
from dataclasses import replace
from functools import wraps

import pytest

from power_loop import ToolDefinition
from power_loop.contracts.errors import ToolValidationError
from power_loop.runtime.env import RuntimeEnv, get_runtime_env
from power_loop.runtime.execution_roles import role_tools
from power_loop.runtime.invocation import invocation_context
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import _SEND, Execution, TaskRuntime, check_tool_dispatch
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool,
    RuntimeTool,
    ToolContractError,
    ToolDispatchPolicy,
    ToolRuntimeContract,
    capability_tools,
    format_result,
    policy,
    resolve,
    validate_registration,
)
from power_loop.tools import _bind_handler
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_task_control_192 import control

__all__ = ["registry_store", "control"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Probe(RuntimeTool):
    def __init__(self, *, roles=("main",), control_only=False):
        self.contract = ToolRuntimeContract("example.probe", allowed_roles=roles, capabilities=("example.read",))
        self.control_only = control_only

    @property
    def runtime_contract(self):
        return self.contract

    def dispatch_policy(self, arguments, *, runtime=None):
        return ToolDispatchPolicy(native_admitted=True, control_only=self.control_only)

    def format_result(self, result, max_chars):
        return "VIEW:" + result[: max(0, max_chars - 5)]

    async def __call__(self, **arguments):
        await asyncio.sleep(0)
        return "ok"


@pytest.mark.parametrize("name", ["probe", "jobs", "workflow", "send_message"])
async def test_plain_handler_names_and_lookalike_attributes_grant_no_capabilities(name):
    async def plain(**arguments):
        return "unused"

    plain.runtime_contract = ToolRuntimeContract("example.fake", capabilities=("example.read",))
    plain.dispatch_policy = lambda arguments: ToolDispatchPolicy(True, True)
    registry = ToolRegistry()
    registry.register(ToolDefinition(name, ""), plain)
    assert resolve(registry.get(name)) is None
    assert policy(registry.get(name), {}) == ToolDispatchPolicy()
    assert capability_tools(registry, "example.read") == []


async def test_inherited_implementation_roles_and_result_views_follow_registration_not_names():
    registry = ToolRegistry()
    for name, roles in [("launch_research", ("main",)), ("read_notes", ("main", "subagent"))]:
        registry.register(ToolDefinition(name, ""), Probe(roles=roles))
    assert role_tools(registry.names(), "subagent", registry=registry) == ["read_notes"]
    assert role_tools(registry.names(), "workflow_node", registry=registry) == []
    assert [tool.definition.name for tool in capability_tools(registry, "example.read", role="subagent")] == [
        "read_notes"
    ]
    assert format_result(registry.get("launch_research"), "abcdefgh", 9) == "VIEW:abcd"


@pytest.mark.parametrize("change", ["contract", "schema", "interrupt", "required"])
async def test_changed_registration_is_not_reauthorized_by_subset_or_wrapper(change):
    registry, implementation = ToolRegistry(), Probe()
    definition = ToolDefinition("custom", "", input_schema={"type": "object"})
    registry.register(definition, implementation)
    if change == "contract":
        implementation.contract = replace(implementation.contract, allowed_roles=("main", "subagent"))
    elif change == "schema":
        definition.input_schema["required"] = ["new_field"]
    elif change == "interrupt":
        object.__setattr__(definition, "interrupt", "background")
    else:
        object.__setattr__(definition, "required_params", ("new_field",))
    for selected in (registry, registry.subset(["custom"])):
        with pytest.raises(ToolContractError, match="binding_changed"):
            resolve(selected.get("custom"))


async def test_explicit_wrapping_preserves_real_base_and_async_environment(tmp_path):
    seen = []

    async def original(**arguments):
        await asyncio.sleep(0)
        seen.append(get_runtime_env().workspace_dir)
        return "original"

    implementation = FunctionRuntimeTool(
        original, ToolRuntimeContract("example.async"), policy=lambda arguments: ToolDispatchPolicy(True)
    )
    bound = _bind_handler(implementation, RuntimeEnv(workspace_dir=tmp_path))

    @wraps(bound)
    async def outer(**arguments):
        return await bound(**arguments)

    registry = ToolRegistry()
    registry.register(ToolDefinition("renamed_async", ""), outer)
    assert registry.get("renamed_async").is_async
    assert resolve(registry.get("renamed_async")) is implementation
    assert await registry.invoke_async("renamed_async", {}) == "original"
    assert seen == [tmp_path]


@pytest.mark.parametrize("name", ["generate_image", "bash", "renamed_image"])
async def test_validation_uses_actual_schema_and_required_parameters_not_name(name):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name, "", input_schema={"type": "object", "required": ["value"]}), lambda **kw: kw["value"]
    )
    assert registry.validate(name, {"value": 42}) is None
    assert await registry.invoke_async(name, {"value": 42}) == 42
    with pytest.raises(ToolValidationError):
        await registry.invoke_async(name, {})


async def test_generic_control_tool_can_query_expired_original_without_new_execution(control):
    c = control
    runtime = TaskRuntime(
        c.manager,
        tenant_id=c.actor.tenant_id,
        principal_id=c.actor.principal_id,
        executor_id="control-query",
        boot_id="query-boot",
    )
    registry = ToolRegistry()
    registry.register(ToolDefinition("original_query", ""), Probe(control_only=True))
    registry.register(ToolDefinition("original_execute", ""), Probe())
    c.now[0] += 30001
    token = _SEND.set(Execution(runtime, c.actor, c.caller))
    try:
        with invocation_context(c.actor.session_id, c.ref.seq, "stop-child"):
            await check_tool_dispatch(c.actor.session_id, "original_query", registry, {})
            with pytest.raises(JobError, match="expired"):
                await check_tool_dispatch(c.actor.session_id, "original_execute", registry, {})
        with pytest.raises(JobError, match="persisted_tool_call_required"):
            await check_tool_dispatch(c.actor.session_id, "original_query", registry, {})
    finally:
        _SEND.reset(token)


async def test_abstract_and_invalid_policy_declarations_fail_closed():
    with pytest.raises(TypeError):
        RuntimeTool()
    for args in [(False, True), (1, False), (True, 1)]:
        with pytest.raises(ToolContractError):
            ToolDispatchPolicy(*args)
    implementation = FunctionRuntimeTool(
        lambda **kw: "unused", ToolRuntimeContract("example.invalid"), policy=lambda args: True
    )
    registry = ToolRegistry()
    registry.register(ToolDefinition("invalid", ""), implementation)
    with pytest.raises(ToolContractError, match="invalid_tool_dispatch_policy"):
        policy(registry.get("invalid"), {})


async def test_replaced_wrapper_target_with_equal_contract_is_not_the_registered_implementation():
    original, replacement = Probe(), Probe()

    @wraps(original)
    async def wrapper(**arguments):
        return await original(**arguments)

    registry = ToolRegistry()
    registry.register(ToolDefinition("probe", ""), wrapper)
    wrapper.__wrapped__ = replacement
    with pytest.raises(ToolContractError, match="binding_changed"):
        resolve(registry.get("probe"))


@pytest.mark.parametrize("change", ["replace", "revoke", "mutate"])
async def test_registration_change_across_admission_await_cannot_execute(change):
    registry, implementation = ToolRegistry(), Probe()
    definition = ToolDefinition("probe", "")
    registry.register(definition, implementation)
    admitted = registry.get("probe")

    async def concurrent_change():
        await asyncio.sleep(0)
        if change == "replace":
            registry.register(definition, Probe(), overwrite=True)
        elif change == "revoke":
            registry.unregister("probe")
        else:
            implementation.contract = replace(implementation.contract, allowed_roles=("main", "subagent"))

    await concurrent_change()
    with pytest.raises(ToolContractError, match="binding_changed"):
        validate_registration(registry, "probe", admitted)


async def test_actual_pipeline_does_not_invoke_a_replacement_after_admission(tmp_path, monkeypatch):
    from power_loop import AgentLoopConfig, LLMResponse, SessionStore, StatefulAgentLoop
    from power_loop.runtime import task_runtime

    from .test_stateful_loop import _Scripted, _tool_resp

    called = []

    async def original(**arguments):
        called.append("original")
        return "original"

    async def replacement(**arguments):
        called.append("replacement")
        return "replacement"

    registry = ToolRegistry()
    definition = ToolDefinition("probe", "")
    registry.register(definition, FunctionRuntimeTool(original, ToolRuntimeContract("example.probe")))
    admit = task_runtime.check_tool_dispatch

    async def replace_during_admission(*args, **kwargs):
        registered = await admit(*args, **kwargs)
        registry.register(
            definition, FunctionRuntimeTool(replacement, ToolRuntimeContract("example.probe")), overwrite=True
        )
        return registered

    monkeypatch.setattr(task_runtime, "check_tool_dispatch", replace_during_admission)
    store = await SessionStore.open(str(tmp_path / "binding-race.db"))
    model = _Scripted([_tool_resp("probe-call", "probe"), LLMResponse(raw_text="DONE")])
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=registry, config=AgentLoopConfig(max_rounds=3))
    try:
        sid = await loop.new_session()
        await loop.send("Read the probe", sid)
        assert called == []
        messages = await store.load_all_messages(sid)
        assert any(row.role == "tool" and "tool_runtime_binding_changed" in row.content for row in messages)
    finally:
        await loop.aclose(drain_timeout_s=0.1)
        await store.close()


async def test_original_protocol_five_requires_explicit_upgrade_without_new_identity(registry_store, monkeypatch):
    import power_loop.runtime.jobs as jobs
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.store import task_schema
    from power_loop.runtime.store.schema import StoreSchemaError

    store, _ = registry_store
    manager = JobManager(store)
    sid = await store.create_session()
    seq = await store.append_message(sid, role="user", content="Keep the original identity")
    original = await store.message_ref(sid, seq)
    await manager.db.execute(f"UPDATE {manager.prefix}task_store SET protocol_version=5 WHERE id=1")
    with pytest.raises(StoreSchemaError):
        await task_schema.verify_task_schema(manager.db, manager.prefix)
    with pytest.raises(StoreSchemaError, match="maintenance_required"):
        await task_schema.migrate_task_schema(manager.db, manager.prefix)
    await task_schema.migrate_task_schema(manager.db, manager.prefix, maintenance=True)
    assert await store.message_ref(sid, seq) == original
    assert (
        await manager.db.fetchone(f"SELECT store_id,protocol_version FROM {manager.prefix}task_store WHERE id=1")
    ) == {"store_id": store.store_id, "protocol_version": 6}
    with monkeypatch.context() as patch:
        patch.setattr(jobs, "TASK_PROTOCOL_VERSION", 5)
        with pytest.raises(JobError, match="store_identity_changed"):
            async with manager._write():
                pass
