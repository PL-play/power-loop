"""The default single-process composition uses the same client/worker ports."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from power_loop.agent.types import AgentLoopConfig

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
    # A trusted factory may resolve an explicit host command instead of an LLM
    # turn. The registered name/arguments come from the frozen descriptor.
    command: tuple[str, dict] | None = None
    runtime_env: object = None
    workflow_file_io: object = None
    process_control: object = None
    resource_profile_ref: str | None = None
    resource_scope: str = "process"
    # Optional host event bus, resolved by the factory on this worker.
    event_bus: object = None

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
    async def preflight(self, required_profiles: frozenset[str]) -> None: ...
    async def build(self, descriptor, execution_context) -> ExecutionEnvironment: ...


class RegistryRuntimeFactory:
    def __init__(self):
        self._builders = {}
        self.profiles = {"builtin.none@1", "builtin.empty@1"}
        self._preflights = {}

    def register_profile(self, name, *, preflight=None):
        """Declare a versioned port/resource and its async startup check.

        Checks resolve deployment configuration and resource references only;
        they must not invoke models or chargeable tools. No check runs per send.
        """
        if not name or "@" not in name:
            raise ValueError("a versioned profile name is required")
        self.profiles.add(name)
        if preflight is not None:
            self._preflights[name] = preflight

    def register(self, name, builder, *, profiles=()):
        if not name or "@" not in name:
            raise ValueError("a versioned definition name is required")
        if name in self._builders:
            raise ValueError("definition already registered: " + name)
        self._builders[name] = builder
        self.profiles.update((name, *profiles))

    async def preflight(self, required_profiles):
        missing = set(required_profiles) - self.profiles
        if missing:
            raise RuntimeRequestError("capability_unavailable", "missing profiles: " + ", ".join(sorted(missing)))
        for check in self._preflights.values():
            await check()

    async def build(self, descriptor, execution_context):
        builder = self._builders.get(descriptor.definition_ref)
        if builder is None:
            raise RuntimeRequestError("definition_unavailable", descriptor.definition_ref)
        return await builder(descriptor, execution_context)


class Runtime:
    def __init__(self, *, store, factory, node_id, boot_id=None, coordinator=None, transport=None,
                 namespace="power-loop", capacity=8, session_lease_ttl_s=90.0,
                 recovery_interval_s=30.0, claim_limit=100, shared_execution=False, host_context_fields=None, hooks=None, hook_timeout_s=1.0,
                 timer_backend=None, timer_driver=None, timers_enabled=True, required_profiles=()):
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
        self.shared_execution = shared_execution
        self.required_profiles = frozenset(required_profiles)
        self.hooks, self.hook_timeout_s = hooks, hook_timeout_s
        self.node_id, self.boot_id, self.namespace = node_id, boot_id, namespace
        # The database lock is the default. Pass coordinator= for another service.
        if coordinator is None:
            from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
            coordinator = StoreSessionOwnershipCoordinator(store)
        self.coordinator = coordinator
        self.transport = transport or MemoryMessageTransport()
        shared_lock = coordinator is None or getattr(coordinator, "scope", "process") == "shared"
        if shared_execution and not (shared_lock and self.transport.capabilities.scope == "shared"):
            raise RuntimeRequestError("invalid_options", "shared_execution requires shared adapters")
        self.host_context_fields = {k: frozenset(v) for k, v in (host_context_fields or {}).items()}
        self.capacity, self.claim_limit = capacity, claim_limit
        self.session_lease_ttl_s, self.recovery_interval_s = session_lease_ttl_s, recovery_interval_s
        self.client = RuntimeClient(store=store, transport=self.transport, namespace=namespace, hooks=hooks, hook_timeout_s=hook_timeout_s)
        self.client._accepting = False
        from power_loop.runtime.timer_backend import StoreTimerBackend
        from power_loop.runtime.timers import TimerRunner
        if timer_backend is not None and timer_driver is None:
            raise RuntimeRequestError("invalid_options", "host timer_backend requires timer_driver")
        if not timers_enabled and (timer_backend is not None or timer_driver is not None):
            raise RuntimeRequestError("invalid_options", "disabled timers cannot have a backend/driver")
        self.timer_backend = (timer_backend or StoreTimerBackend(store)) if timers_enabled else None
        self.timer_driver = (timer_driver or TimerRunner(store=store, client=self.client, hooks=hooks)) if timers_enabled else None
        from power_loop.runtime.store.process_guard import ProcessStoreGuard
        # Create process guard when using default (non-shared) runtime configuration.
        # The guard prevents multiple independent processes from using the same store,
        # independent of whether the session-level coordinator scope is shared or process.
        self.process_guard = (ProcessStoreGuard(store, self.coordinator, self.transport)
            if not shared_execution and self.transport.capabilities.scope != "shared" else None)
        self.worker = RuntimeWorker(self)
        self._closed = False

    def status(self):
        from dataclasses import asdict
        return {"accepting": self.client._accepting, "required_profiles": sorted(self.required_profiles),
            "timer": asdict(self.timer_driver.status()) if self.timer_driver else None,
            "last_error": self.worker.last_error, "pending_sessions": len(self.worker._pending),
            "active_sessions": len(self.worker._active),
            "oldest_pending_ms": min(self.worker._pending.values(), default=None),
            "pending_reason": "capacity" if self.worker._pending else None,
            "ports": {"timer_backend": type(self.timer_backend).__name__ if self.timer_backend else None,
                      "coordination": self.coordinator.scope, "transport": self.transport.capabilities.scope}}

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
