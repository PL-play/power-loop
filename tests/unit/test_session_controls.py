"""Committed notice routing invalidates its recipient, including cross-session handoffs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from power_loop.runtime import session_controls
from power_loop.runtime.jobs import JobActor, JobManager

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("rollback", [False, True])
async def test_notice_recipient_version_changes_only_with_its_transaction(registry_store, rollback):
    store, _ = registry_store
    manager = JobManager(store)
    sender, recipient = await store.create_session(), await store.create_session()
    registered = await manager.register_job(JobActor("tenant", "host", sender), kind="send",
                                            session_id=sender, system_key="source")
    before = await session_controls.version(manager, manager.db, recipient)

    async def route_notice():
        async with manager._write() as tx:
            await tx.execute(
                f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                ("notice", registered.job.job_id, "controller_review_handoff", 1, "{}", "pending", manager.now()))
            await tx.set_event_recipient("notice", recipient)
            if rollback:
                raise OSError("route rolled back")

    if rollback:
        with pytest.raises(OSError, match="route rolled back"):
            await route_notice()
    else:
        await route_notice()
    after = await session_controls.version(manager, manager.db, recipient)
    assert after == (before[0] + (0 if rollback else 1), before[1])


async def test_all_notice_route_writers_use_explicit_recipient_invalidation():
    runtime = Path(session_controls.__file__).parent
    expected = {
        ("abort_outcomes.py", "_reconcile"),
        ("managed_delivery.py", "enqueue_runtime"),
        ("pending_native.py", "prepare"),
        ("pending_owner.py", "transfer"),
        ("controller_handoff.py", "transfer_review_runtime"),
        ("controller_recovery.py", "reconcile"),
        ("controller_recovery.py", "enqueue"),
    }
    routed, direct_writes = set(), []

    def inspect(node, filename, function=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_event_recipient":
                routed.add((filename, function))
            elif node.func.attr in {"execute", "fetchone", "fetchall"} and node.args:
                statement = node.args[0]
                parts = statement.values if isinstance(statement, ast.JoinedStr) else [statement]
                sql = " ".join(part.value for part in parts if isinstance(part, ast.Constant) and isinstance(part.value, str))
                if re.search(r"\bUPDATE\b[\s\S]*\bSET\b[\s\S]*\bcaller_session_id\s*=", sql, re.I):
                    direct_writes.append((filename, function))
        for child in ast.iter_child_nodes(node):
            inspect(child, filename, function)

    for path in runtime.rglob("*.py"):
        inspect(ast.parse(path.read_text()), path.name)
    assert routed == expected
    assert direct_writes == [("session_controls.py", "set_event_recipient")]
