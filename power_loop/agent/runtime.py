"""The default single-process composition uses the same client/worker ports."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator

from .runtime_client import RuntimeClient
from .runtime_transport import MemoryMessageTransport
from .runtime_types import RuntimeRequestError, TargetRef, identifier, plain

_PROCESS_BOOT_ID = uuid.uuid4().hex


@dataclass
class ExecutionEnvironment:
    llm: object
    tool_registry: object
    config: AgentLoopConfig
    hooks: object = None
    resources: object = None
    task_runtime: object = None

    async def aclose(self):
        if self.llm is not None:
            await self.llm.close()
        if self.resources is not None:
            await self.resources.aclose()


class RuntimeFactory(Protocol):
    """Resolve fixed profiles and execution_context.options.representation_ref.

    Builders receive the frozen request options; a missing referenced profile
    must raise definition_unavailable rather than silently use another profile.
    """
    async def build(self, descriptor, execution_context) -> ExecutionEnvironment: ...


class RegistryRuntimeFactory:
    def __init__(self):
        self._builders = {}

    def register(self, name, builder):
        if not name or "@" not in name:
            raise ValueError("a versioned definition name is required")
        if name in self._builders:
            raise ValueError("definition already registered: " + name)
        self._builders[name] = builder

    async def build(self, descriptor, execution_context):
        builder = self._builders.get(descriptor.definition_ref)
        if builder is None:
            raise RuntimeRequestError("definition_unavailable", descriptor.definition_ref)
        return await builder(descriptor, execution_context)


class Runtime:
    def __init__(self, *, store, factory, node_id, boot_id=None, coordinator=None, transport=None,
                 namespace="power-loop", capacity=8, session_lease_ttl_s=90.0,
                 recovery_interval_s=30.0, claim_limit=100, shared_execution=False, host_context_fields=None, hooks=None, hook_timeout_s=1.0):
        from .runtime_worker import RuntimeWorker
        if store.store_id is None:
            raise RuntimeRequestError("invalid_options", "current task store required")
        if not 1 <= claim_limit <= 100 or capacity < 1 or session_lease_ttl_s <= 0:
            raise ValueError("invalid runtime limits")
        identifier(node_id)
        boot_id = boot_id or _PROCESS_BOOT_ID
        identifier(boot_id)
        if hook_timeout_s <= 0:
            raise ValueError("hook_timeout_s must be positive")
        if recovery_interval_s is not None and recovery_interval_s <= 0:
            raise ValueError("recovery_interval_s must be positive or None")
        self.store, self.factory = store, factory
        self.hooks, self.hook_timeout_s = hooks, hook_timeout_s
        self.node_id, self.boot_id, self.namespace = node_id, boot_id, namespace
        self.coordinator = coordinator or MemorySessionOwnershipCoordinator()
        self.transport = transport or MemoryMessageTransport()
        if shared_execution and (getattr(self.coordinator, "scope", "process") != "shared"
                                 or self.transport.capabilities.scope != "shared"):
            raise RuntimeRequestError("invalid_options", "shared_execution requires shared adapters")
        self.host_context_fields = {k: frozenset(v) for k, v in (host_context_fields or {}).items()}
        self.capacity, self.claim_limit = capacity, claim_limit
        self.session_lease_ttl_s, self.recovery_interval_s = session_lease_ttl_s, recovery_interval_s
        self.client = RuntimeClient(store=store, transport=self.transport, namespace=namespace, hooks=hooks, hook_timeout_s=hook_timeout_s)
        self.client._accepting = False
        self.worker = RuntimeWorker(self)
        self._closed = False

    async def open_session(self, descriptor, *, actor):
        self.client._require_open()
        if descriptor.kind != "agent":
            raise RuntimeRequestError("feature_unavailable", descriptor.kind + " is not enabled")
        if actor.session_id is not None:
            raise RuntimeRequestError("forbidden", "host identity required to open a root session")
        sid = await self.store.create_session(metadata={
            "tenant_id": actor.tenant_id, "principal_id": actor.principal_id,
            "runtime_descriptor": plain(descriptor), "execution_role": "main"})
        return TargetRef(self.store.store_id, sid)

    async def aclose(self, *, timeout_s=30.0, cancel_running=False):
        if self._closed:
            return self.worker.shutdown_result
        result = await self.worker.shutdown(timeout_s=timeout_s, cancel_running=cancel_running)
        await self.client.aclose()
        await self.transport.aclose()
        await self.store.close()
        self._closed = True
        return result
