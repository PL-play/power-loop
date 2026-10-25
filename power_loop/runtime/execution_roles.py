"""Host-selected finite delegation policy; roles survive continuation/restart.

The default library policy remains legacy. DeepTalk opts into single_level.
An AgentSpec/prompt cannot widen the host policy or choose a child's role.
"""
from __future__ import annotations

from power_loop.core.agent_context import get_session_id


def finite_delegation(loop) -> bool:
    policy = getattr(getattr(loop, "config", None), "delegation_policy", "legacy")
    if policy not in ("legacy", "single_level"):
        raise ValueError("invalid_delegation_policy")
    return policy == "single_level"


async def session_role(store, sid: str) -> str:
    row = await store.get_session(sid)
    if row is None:
        raise ValueError("execution_role_unavailable")
    meta = row.metadata or {}
    role = meta.get("execution_role")
    if meta.get("kind") == "wf_driver":
        if role not in (None, "workflow"):
            raise ValueError("execution_role_conflict")
        return "workflow"
    # Workflow driver cleanup reparents retained leaves to NULL. Their host
    # role remains authoritative; a removed parent must never promote a leaf.
    if role in ("workflow_node", "subagent"):
        return role
    if row.parent_session_id is None:
        if meta.get("workflow_run_id") or meta.get("standalone_subagent"):
            raise ValueError("execution_role_unavailable")
        if role not in (None, "main"):
            raise ValueError("execution_role_conflict")
        return "main"
    raise ValueError("execution_role_unavailable")


async def require_orchestration(loop, capability: str, *, sid: str | None = None) -> None:
    allowed_roles = {"workflow": ("main",), "subagent": ("main", "workflow_node")}
    if capability not in allowed_roles:
        raise ValueError("unknown_orchestration_capability")
    if not finite_delegation(loop):
        return
    sid = get_session_id() or sid
    if not sid:
        raise ValueError("execution_role_unavailable")
    role = await session_role(await loop._ensure_store(), sid)
    if role not in allowed_roles[capability]:
        raise ValueError("execution_role_forbids_" + capability)


def role_tools(names, role: str, *, registry) -> list[str]:
    from .tool_contracts import ROLES, resolve

    if role not in ROLES:
        raise ValueError("execution_role_unavailable")
    result = []
    for name in names:
        registered = registry.get(name) if registry is not None else None
        implementation = resolve(registered)
        if implementation is None or role in implementation.runtime_contract.allowed_roles:
            result.append(name)
    return result


async def require_timer_creation(loop, *, sid: str | None = None) -> None:
    """Reject finite child timers until managed waiting participates in closure.

    Check both the caller and an explicit SDK target, so a child using a host
    wrapper cannot arm a main-session reminder instead. Existing timers remain
    readable/cancellable; legacy delegation preserves its original behavior.
    """
    if loop is None or not finite_delegation(loop):
        return
    sessions = {value for value in (get_session_id(), sid) if value}
    if not sessions:
        raise ValueError("execution_role_unavailable")
    store = await loop._ensure_store()
    for session in sessions:
        if await session_role(store, session) != "main":
            raise ValueError("execution_role_forbids_timer")
