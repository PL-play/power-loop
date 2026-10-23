"""Resource preparation never substitutes identity or grants execution authority."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.workflow_resources import (
    WorkflowResourceAdapter,
    WorkflowResourceBinding,
    WorkflowResourceRegistry,
    WorkflowResources,
    WorkflowResourceScope,
)

pytestmark = pytest.mark.asyncio
SCOPE = WorkflowResourceScope("tenant", "principal", "sess_original")
CONTRACT = "c" * 64


def binding(**changes):
    return replace(WorkflowResourceBinding("host", "1.0.0", "original-resource", 1,
        SCOPE.tenant_id, SCOPE.principal_id, SCOPE.main_session_id, "a" * 64, CONTRACT), **changes)


class Host:
    def __init__(self):
        self.binding = binding()
        self.executor = SimpleNamespace(run_agent=lambda: None)
        self.resources = WorkflowResources(self.binding, self.executor)
        self.calls = []
        self.access = True

    async def prepare(self, scope, contract):
        self.calls.append("prepare")
        return self.resources

    async def resolve(self, original):
        self.calls.append("resolve")
        assert original == self.binding
        return self.resources

    async def authorize(self, original, resources, scope):
        self.calls.append("authorize")
        assert resources is self.resources and original is resources.binding and scope == SCOPE
        return self.access

    def register(self, registry):
        registry.register("host", "1.0.0", WorkflowResourceAdapter(self.prepare, self.resolve, self.authorize))


async def prepare(registry):
    return await registry.prepare("host", "1.0.0", scope=SCOPE, contract_sha256=CONTRACT)


async def resolve(registry, original=None, **options):
    return await registry.resolve(original or binding().to_dict(), scope=options.get("scope", SCOPE),
        contract_sha256=options.get("contract", CONTRACT))


async def test_roundtrip_uses_original_identity_and_objects_across_registry_recreation():
    host = Host()
    registry = WorkflowResourceRegistry()
    host.register(registry)
    original = await prepare(registry)
    saved = json.loads(json.dumps(original.binding.to_dict()))
    second = WorkflowResourceRegistry()
    host.register(second)
    restored = await resolve(second, saved)
    assert restored is original and restored.executor is host.executor
    assert restored.binding.digest == WorkflowResourceBinding.from_dict(saved).digest
    assert host.calls == ["prepare", "authorize", "resolve", "authorize"]
    saved["resource_id"] = "tampered"
    assert restored.binding.resource_id == "original-resource"


@pytest.mark.parametrize("changes", [
    {"tenant_id": "other"}, {"principal_id": "other"}, {"main_session_id": "other"},
])
async def test_cross_scope_is_rejected_before_any_host_callback(changes):
    registry = WorkflowResourceRegistry()
    host = Host()
    host.register(registry)
    with pytest.raises(JobError, match="workflow_resource_scope_mismatch"):
        await resolve(registry, scope=replace(SCOPE, **changes))
    assert not host.calls


async def test_different_contract_and_missing_version_do_not_fallback_to_current_factory():
    registry = WorkflowResourceRegistry()
    host = Host()
    host.register(registry)
    with pytest.raises(JobError, match="contract_mismatch"):
        await resolve(registry, contract="d" * 64)
    with pytest.raises(JobError, match="adapter_unavailable"):
        await resolve(registry, binding(adapter_version="2.0.0"))
    assert not host.calls


@pytest.mark.parametrize("changes", [
    {"resource_id": "new"}, {"resource_revision": 2}, {"config_sha256": "b" * 64},
    {"adapter_key": "replacement"}, {"adapter_version": "2.0.0"},
    {"workspace_id": "new-workspace"}, {"budget_scope_id": "new-pool"},
    {"notification_mode": "host_receipt"},
])
async def test_same_class_with_changed_binding_cannot_replace_original(changes):
    host = Host()
    changed = binding(**changes)
    resources = WorkflowResources(changed, host.executor,
        budget=object() if changed.budget_scope_id else None,
        file_io=object() if changed.workspace_id else None,
        on_complete=(lambda _: None) if changed.notification_mode == "host_receipt" else None)
    async def altered(_original):
        host.calls.append("resolve")
        return resources
    registry = WorkflowResourceRegistry()
    registry.register("host", "1.0.0", WorkflowResourceAdapter(host.prepare, altered, host.authorize))
    with pytest.raises(JobError, match="binding_mismatch"):
        await resolve(registry)
    assert host.calls == ["resolve"]


@pytest.mark.parametrize("access", [False, None, 1, "true"])
async def test_current_authorization_requires_exact_acceptance(access):
    host = Host()
    host.access = access
    registry = WorkflowResourceRegistry()
    host.register(registry)
    with pytest.raises(JobError, match="access_denied"):
        await resolve(registry)
    assert host.calls == ["resolve", "authorize"]


@pytest.mark.parametrize("phase", ["prepare", "resolve", "authorize"])
@pytest.mark.parametrize("exception", [RuntimeError, JobError, ValueError])
async def test_host_errors_do_not_leak_private_details(phase, exception):
    host = Host()
    async def failed(*_args):
        raise exception("credential-and-private-path")
    setattr(host, phase, failed)
    registry = WorkflowResourceRegistry()
    host.register(registry)
    with pytest.raises(JobError) as error:
        await (prepare(registry) if phase == "prepare" else resolve(registry))
    assert str(error.value) == "workflow_resource_resolution_failed"


@pytest.mark.parametrize("phase", ["prepare", "resolve", "authorize"])
@pytest.mark.parametrize("reregister", [False, True])
async def test_revocation_during_host_await_cannot_return_old_resources(phase, reregister):
    host = Host()
    registry = WorkflowResourceRegistry()
    real = getattr(host, phase)
    async def revoked(*args):
        result = await real(*args)
        registry.unregister("host", "1.0.0")
        if reregister:
            host.register(registry)
        return result
    setattr(host, phase, revoked)
    host.register(registry)
    with pytest.raises(JobError, match="adapter_changed"):
        await (prepare(registry) if phase == "prepare" else resolve(registry))


async def test_mutating_serialized_input_after_resolve_started_does_not_change_identity():
    host = Host()
    entered, release = asyncio.Event(), asyncio.Event()
    original_resolve = host.resolve
    async def held(original):
        entered.set()
        await release.wait()
        return await original_resolve(original)
    host.resolve = held
    registry = WorkflowResourceRegistry()
    host.register(registry)
    saved = binding().to_dict()
    task = asyncio.create_task(resolve(registry, saved))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        saved.update(tenant_id="foreign", resource_id="new")
        release.set()
        result = await task
        assert result.binding == binding()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancel_release_host_wait_without_fallback(cancel):
    host = Host()
    entered, exited = asyncio.Event(), asyncio.Event()
    async def held(_original):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()
    host.resolve = held
    registry = WorkflowResourceRegistry(timeout_s=0.02 if not cancel else 10)
    host.register(registry)
    task = asyncio.create_task(resolve(registry))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else JobError, match=None if cancel else "timeout"):
            await task
        assert exited.is_set() and not host.calls
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 2), ("protocol", "unknown"), ("resource_revision", True),
    ("resource_revision", 0), ("resource_revision", 1.0), ("resource_revision", 2**63), ("config_sha256", "secret"),
    ("contract_sha256", "C" * 64), ("workspace_id", ""), ("budget_scope_id", 3),
    ("adapter_version", " v1"), ("main_session_id", "sid\n"),
    ("notification_mode", "opaque_callback"), ("api_key", "secret"),
])
async def test_binding_strict_shape_rejects_ambiguous_history_and_arbitrary_configuration(field, value):
    saved = binding().to_dict()
    saved[field] = value
    with pytest.raises(JobError, match="binding_invalid"):
        WorkflowResourceBinding.from_dict(saved)


@pytest.mark.parametrize("value", [None, [], {"kind": "external", "class": "host.Executor"}, {},
    {k: v for k, v in binding().to_dict().items() if k != "resource_id"}])
async def test_unknown_or_class_only_history_never_creates_a_binding(value):
    with pytest.raises(JobError, match="binding_invalid"):
        await WorkflowResourceRegistry().resolve(value, scope=SCOPE, contract_sha256=CONTRACT)


async def test_registration_is_explicit_and_objects_do_not_expose_credentials_in_repr():
    host = Host()
    registry = WorkflowResourceRegistry()
    host.register(registry)
    with pytest.raises(JobError, match="duplicate"):
        host.register(registry)
    with pytest.raises(JobError, match="adapter_invalid"):
        registry.register("host", "2", object())
    with pytest.raises(TypeError, match="async functions"):
        WorkflowResourceAdapter(lambda *_: None, host.resolve, host.authorize)
    host.executor.secret = "private-token"
    assert "private-token" not in repr(host.resources)
    registry.unregister("host", "1.0.0")
    with pytest.raises(JobError, match="unavailable"):
        await prepare(registry)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 31, "10"])
async def test_invalid_timeout(value):
    with pytest.raises(ValueError):
        WorkflowResourceRegistry(timeout_s=value)


@pytest.mark.parametrize("changes", [
    {"file_io": object()}, {"budget": object()}, {"on_complete": lambda *_: None},
    {"executor": object()}, {"binding": {}},
])
async def test_resource_shape_cannot_silently_enable_unbound_host_capabilities(changes):
    values = {"binding": binding(), "executor": Host().executor, **changes}
    with pytest.raises(JobError, match="resources_invalid"):
        WorkflowResources(**values)


@pytest.mark.parametrize("value", [None, "", True, "session\n", "x" * 513])
async def test_scope_must_have_complete_original_identity(value):
    with pytest.raises(JobError, match="scope_invalid"):
        WorkflowResourceScope("tenant", "principal", value)


@pytest.mark.parametrize("scope,contract", [(None, CONTRACT), (SCOPE, None), (SCOPE, True), (SCOPE, "invalid")])
async def test_invalid_scope_or_contract_never_enters_host(scope, contract):
    registry = WorkflowResourceRegistry()
    host = Host()
    host.register(registry)
    for operation in (registry.prepare("host", "1.0.0", scope=scope, contract_sha256=contract),
            registry.resolve(binding(), scope=scope, contract_sha256=contract)):
        with pytest.raises(JobError, match="scope_invalid"):
            await operation
    assert not host.calls


async def test_invalid_host_result_and_foreign_prepared_identity_cannot_escape_authorization():
    host = Host()
    async def malformed(*_args):
        return {"executor": host.executor, "binding": binding().to_dict()}
    host.prepare = malformed
    registry = WorkflowResourceRegistry()
    host.register(registry)
    with pytest.raises(JobError, match="resources_invalid"):
        await prepare(registry)
    assert not host.calls
    host.resources = WorkflowResources(binding(tenant_id="foreign"), host.executor)
    with pytest.raises(JobError, match="scope_mismatch"):
        await resolve(registry)
    assert host.calls == ["resolve"]


async def test_bad_selector_cannot_hash_an_untrusted_object_or_register_a_port():
    registry = WorkflowResourceRegistry()
    for key in (None, [], {}, True):
        with pytest.raises(JobError, match="adapter_invalid"):
            registry.unregister(key, "1.0.0")


async def test_registered_pool_is_not_reconstructed_or_refilled_on_resolution():
    from power_loop.workflow.result import SharedBudget
    host = Host()
    host.binding = binding(budget_scope_id="shared-original")
    pool = SharedBudget(100)
    pool.commit({"total_tokens": 70})
    host.resources = WorkflowResources(host.binding, host.executor, budget=pool)
    registry = WorkflowResourceRegistry()
    host.register(registry)
    original = await prepare(registry)
    pool.commit({"total_tokens": 31})  # another scope's spend also belongs to this pool
    resolved = await resolve(registry, original.binding.to_dict())
    assert resolved.budget is pool and resolved.budget.spent == 101
    assert not resolved.budget.can_spawn() and resolved.budget.remaining == 0
