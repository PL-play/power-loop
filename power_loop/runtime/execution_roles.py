"""Host-selected finite delegation policy; roles survive continuation/restart.

The default library policy remains legacy. DeepTalk opts into single_level.
An AgentSpec/prompt cannot widen the host policy or choose a child's role.
"""
from __future__ import annotations

from power_loop.core.agent_context import get_session_id


def finite_delegation(loop) -> bool:
    policy = getattr(getattr(loop, "config", None), "delegation_policy", "single_level")
    if policy != "single_level":
        raise ValueError("invalid_delegation_policy")
    return True


def role_from_facts(parent_session_id: str | None, session_kind: str, config) -> str:
    if session_kind == "workflow_driver":
        return "workflow"
    if parent_session_id is None:
        return "main"
    cfg = config or {}
    if "workflow_node" in cfg:
        return "workflow_node"
    if "child_spec" in cfg or session_kind == "background":
        return "subagent"
    raise ValueError("execution_role_unavailable")


async def session_role(store, sid: str) -> str:
    row = await store._db.fetchone(
        f"SELECT parent_session_id,session_kind,descriptor_json FROM {store.t.sessions} "
        "WHERE session_id=?", (sid,),
    )
    if row is None:
        raise ValueError("execution_role_unavailable")
    import json
    config = json.loads(row["descriptor_json"]).get("config") or {}
    return role_from_facts(row["parent_session_id"], row["session_kind"], config)


async def require_orchestration(loop, capability: str, *, sid: str | None = None) -> None:
    allowed_roles = {"workflow": ("main",), "subagent": ("main",)}
    if capability not in allowed_roles:
        raise ValueError("unknown_orchestration_capability")
    if not finite_delegation(loop):
        return
    sid = get_session_id() or sid
    if not sid:
        raise ValueError("execution_role_unavailable")
    ctx = getattr(loop, "_runtime_worker_context", None)
    if (
        ctx is not None
        and getattr(ctx, "session_id", None) == sid
        and getattr(ctx, "descriptor", None) is not None
    ):
        role = role_from_facts(ctx.parent_session_id, ctx.descriptor.kind, ctx.descriptor.config)
    else:
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
