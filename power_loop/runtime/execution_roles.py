"""Host-selected finite delegation policy; roles survive continuation/restart.

The default library policy remains legacy. DeepTalk opts into single_level.
An AgentSpec/prompt cannot widen the host policy or choose a child's role.
"""
from __future__ import annotations

from power_loop.core.agent_context import get_session_id

WORKFLOW_TOOLS = frozenset({"workflow", "create_workflow", "validate_workflow", "workflow_status",
                            "workflow_cancel", "workflow_delete", "run_agent"})


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


async def require_orchestration(loop, tool: str, *, sid: str | None = None) -> None:
    if not finite_delegation(loop):
        return
    sid = get_session_id() or sid
    if not sid:
        raise ValueError("execution_role_unavailable")
    role = await session_role(await loop._ensure_store(), sid)
    allowed = role == "main" if tool == "workflow" else role in ("main", "workflow_node")
    if not allowed:
        raise ValueError("execution_role_forbids_" + tool)


def role_tools(names, role: str) -> list[str]:
    blocked = set(WORKFLOW_TOOLS) if role != "main" else set()
    if role == "subagent":
        blocked.add("spawn_agent")
    return [name for name in names if name not in blocked]
