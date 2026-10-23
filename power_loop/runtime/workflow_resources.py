"""Trusted host resource identity, independent of model/YAML tool parameters.

This preparation port grants no task admission, dispatch, stop, or billing
authority. Adapters must only resolve resources and check access here; they may
not run models, create business artifacts, or reserve/charge budgets. Their
registration is host code, never an import path taken from saved/model data.
Profile-v1 class names cannot be converted to a binding by this module.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .jobs import JobError, _hash


def _identifier(value):
    return (isinstance(value, str) and 0 < len(value) <= 512 and
        value == value.strip() and not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


async def _host_call(function, *args):
    try:
        return await function(*args)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Even a host JobError may contain arbitrary private text in its code.
        raise JobError("workflow_resource_resolution_failed") from exc


@dataclass(frozen=True)
class WorkflowResourceScope:
    tenant_id: str
    principal_id: str
    main_session_id: str

    def __post_init__(self):
        if not all(_identifier(value) for value in asdict(self).values()):
            raise JobError("workflow_resource_scope_invalid")


@dataclass(frozen=True)
class WorkflowResourceBinding:
    """Private, credential-free identity retained with the original contract.

    Resource IDs refer to host-owned persistent records, not paths or Python
    classes. A budget scope must identify the real pool, including shared pools;
    a resolver must never infer its remaining balance from one workflow's usage.
    """

    adapter_key: str
    adapter_version: str
    resource_id: str
    resource_revision: int
    tenant_id: str
    principal_id: str
    main_session_id: str
    config_sha256: str
    contract_sha256: str
    workspace_id: str | None = None
    budget_scope_id: str | None = None
    notification_mode: str = "durable_inbox"

    def __post_init__(self):
        for name in ("adapter_key", "adapter_version", "resource_id", "tenant_id", "principal_id", "main_session_id"):
            if not _identifier(getattr(self, name)):
                raise JobError("workflow_resource_binding_invalid")
        if (type(self.resource_revision) is not int or not 1 <= self.resource_revision < 2**63 or
                not _digest(self.config_sha256) or not _digest(self.contract_sha256) or
                self.notification_mode not in ("durable_inbox", "host_receipt")):
            raise JobError("workflow_resource_binding_invalid")
        if any(value is not None and not _identifier(value) for value in (self.workspace_id, self.budget_scope_id)):
            raise JobError("workflow_resource_binding_invalid")

    def to_dict(self):
        return {"version": 1, "protocol": "workflow-resource-binding-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        try:
            fields = set(cls.__dataclass_fields__)
            if (not isinstance(data, dict) or set(data) != fields | {"version", "protocol"} or
                    type(data["version"]) is not int or data["version"] != 1 or
                    data["protocol"] != "workflow-resource-binding-v1"):
                raise JobError("workflow_resource_binding_invalid")
            return cls(**{key: data[key] for key in fields})
        except (TypeError, ValueError, KeyError) as exc:
            raise JobError("workflow_resource_binding_invalid") from exc

    @property
    def digest(self):
        return _hash(self.to_dict())

    def require(self, scope: WorkflowResourceScope, contract_sha256: str):
        if type(scope) is not WorkflowResourceScope or not _digest(contract_sha256):
            raise JobError("workflow_resource_scope_invalid")
        if (self.tenant_id, self.principal_id, self.main_session_id) != (
                scope.tenant_id, scope.principal_id, scope.main_session_id):
            raise JobError("workflow_resource_scope_mismatch")
        if self.contract_sha256 != contract_sha256:
            raise JobError("workflow_resource_contract_mismatch")


@dataclass(frozen=True)
class WorkflowResources:
    binding: WorkflowResourceBinding
    executor: Any = field(repr=False)
    budget: Any = field(default=None, repr=False)
    file_io: Any = field(default=None, repr=False)
    on_complete: Any = field(default=None, repr=False)

    def __post_init__(self):
        if (type(self.binding) is not WorkflowResourceBinding or
                not callable(getattr(self.executor, "run_agent", None))):
            raise JobError("workflow_resources_invalid")
        if ((self.file_io is not None) != (self.binding.workspace_id is not None) or
                (self.budget is not None) != (self.binding.budget_scope_id is not None) or
                (self.on_complete is not None) != (self.binding.notification_mode == "host_receipt") or
                self.on_complete is not None and not callable(self.on_complete)):
            raise JobError("workflow_resources_invalid")


@dataclass(frozen=True)
class WorkflowResourceAdapter:
    """Async, side-effect-free preparation and authorization by trusted code.

    authorize(binding, resources, scope) must recheck original workspace/pool/
    configuration identity and current access, returning exactly True. Factory
    exceptions are private. A normal callback return never proves notification
    acceptance; that later protocol is separate from this preparation port.
    """

    prepare: Callable[[WorkflowResourceScope, str], Awaitable[WorkflowResources]]
    resolve: Callable[[WorkflowResourceBinding], Awaitable[WorkflowResources]]
    authorize: Callable[[WorkflowResourceBinding, WorkflowResources, WorkflowResourceScope], Awaitable[bool]]

    def __post_init__(self):
        if not all(inspect.iscoroutinefunction(fn) for fn in (self.prepare, self.resolve, self.authorize)):
            raise TypeError("workflow resource adapter ports must be async functions")


class WorkflowResourceRegistry:
    """Exact key/version registration; no fallback and revocation across awaits.

    Resolved resources are only prepared capabilities. Callers must re-resolve /
    reauthorize and compare the original binding at admission and dispatch. This
    port does not make an arbitrary returned object durable or restart-safe.
    """

    def __init__(self, *, timeout_s=10.0):
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 30):
            raise ValueError("workflow resource timeout must be in (0, 30]")
        self._timeout_s = timeout_s
        self._entries = {}

    def register(self, key: str, version: str, adapter: WorkflowResourceAdapter):
        if not _identifier(key) or not _identifier(version) or type(adapter) is not WorkflowResourceAdapter:
            raise JobError("workflow_resource_adapter_invalid")
        identity = (key, version)
        if identity in self._entries:
            raise JobError("workflow_resource_adapter_duplicate")
        # New registration always receives a fresh identity, even for the same
        # object after revoke/re-register (an ABA across await is not accepted).
        self._entries[identity] = (object(), adapter)

    def unregister(self, key: str, version: str):
        self._entry(key, version)
        del self._entries[(key, version)]

    def _entry(self, key, version):
        if not _identifier(key) or not _identifier(version):
            raise JobError("workflow_resource_adapter_invalid")
        try:
            return self._entries[(key, version)]
        except KeyError as exc:
            raise JobError("workflow_resource_adapter_unavailable") from exc

    async def prepare(self, key, version, *, scope, contract_sha256):
        if type(scope) is not WorkflowResourceScope or not _digest(contract_sha256):
            raise JobError("workflow_resource_scope_invalid")
        return await self._prepare(key, version, scope, contract_sha256, None)

    async def resolve(self, binding, *, scope, contract_sha256):
        # Snapshot a serialized input before any host code can yield. Do not
        # silently accept an old class-only profile as a resource identity.
        if isinstance(binding, Mapping):
            binding = WorkflowResourceBinding.from_dict(dict(binding))
        if type(binding) is not WorkflowResourceBinding:
            raise JobError("workflow_resource_binding_invalid")
        binding.require(scope, contract_sha256)
        return await self._prepare(binding.adapter_key, binding.adapter_version, scope, contract_sha256, binding)

    async def _prepare(self, key, version, scope, contract_sha256, original):
        entry = self._entry(key, version)
        _, adapter = entry
        try:
            async with asyncio.timeout(self._timeout_s):
                resources = (await _host_call(adapter.resolve, original) if original is not None else
                    await _host_call(adapter.prepare, scope, contract_sha256))
                if self._entries.get((key, version)) is not entry:
                    raise JobError("workflow_resource_adapter_changed")
                if type(resources) is not WorkflowResources:
                    raise JobError("workflow_resources_invalid")
                binding = resources.binding
                binding.require(scope, contract_sha256)
                if ((binding.adapter_key, binding.adapter_version) != (key, version) or
                        original is not None and binding != original):
                    raise JobError("workflow_resource_binding_mismatch")
                # Authorization receives the exact immutable binding and actual
                # objects used by the caller, not a profile/class-name proxy.
                if await _host_call(adapter.authorize, binding, resources, scope) is not True:
                    raise JobError("workflow_resource_access_denied")
                if self._entries.get((key, version)) is not entry:
                    raise JobError("workflow_resource_adapter_changed")
                return resources
        except TimeoutError as exc:
            raise JobError("workflow_resource_timeout") from exc
