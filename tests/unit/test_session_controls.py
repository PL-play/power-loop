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
            await tx.append_event(
                ("event_id", "job_id", "event_type", "job_version", "payload_json", "delivery_status", "created_at"),
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
        ("controller_recovery.py", "_publish_recovery"),
        ("controller_recovery.py", "enqueue"),
    }
    routed, initialized, direct_writes = set(), set(), []

    def inspect(node, filename, function=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Dict):
            fields = {key.value: value for key, value in zip(node.keys, node.values, strict=True)
                      if isinstance(key, ast.Constant)}
            kind = fields.get("event_type")
            if (isinstance(kind, ast.Constant) and kind.value in session_controls.OBLIGATION_EVENTS
                    and "caller_session_id" in fields):
                initialized.add((filename, function))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_event_recipient":
                routed.add((filename, function))
            elif node.func.attr in {"execute", "_execute", "fetchone", "fetchall"} and node.args:
                statement = node.args[0]
                parts = statement.values if isinstance(statement, ast.JoinedStr) else [statement]
                sql = " ".join(part.value for part in parts if isinstance(part, ast.Constant) and isinstance(part.value, str))
                if re.search(r"\bUPDATE\b[\s\S]*\bSET\b[\s\S]*\bcaller_session_id\s*=", sql, re.I):
                    direct_writes.append((filename, function))
        for child in ast.iter_child_nodes(node):
            inspect(child, filename, function)

    for path in runtime.rglob("*.py"):
        inspect(ast.parse(path.read_text()), path.name)
    assert routed | initialized == expected
    assert initialized == {("controller_recovery.py", "_publish_recovery")}
    assert not routed & initialized
    assert direct_writes == [("session_controls.py", "set_event_recipient")]


@pytest.mark.parametrize("target", ["jobs", "job_events", "task_executors", "task_context_events", "task_executor_admissions"])
async def test_undeclared_control_mutations_fail_before_writing(registry_store, target):
    store, _ = registry_store
    manager = JobManager(store)
    table = manager.prefix + target
    before = await manager.db.fetchall(f"SELECT * FROM {table}")
    with pytest.raises(AssertionError, match="must declare an explicit intent"):
        async with manager._write() as tx:
            await tx.execute(f"DELETE FROM {table}")
    assert await manager.db.fetchall(f"SELECT * FROM {table}") == before


async def test_runtime_event_and_job_writers_declare_their_intents():
    runtime = Path(session_controls.__file__).parent
    direct = []
    for path in runtime.glob("*.py"):
        if path.name == "session_controls.py":
            continue
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                    or node.func.attr not in {"execute", "fetchone", "fetchall"} or not node.args):
                continue
            statement = node.args[0]
            tables = {part.attr for part in ast.walk(statement) if isinstance(part, ast.Attribute)}
            if not tables.intersection({"jobs", "events"}):
                continue
            parts = statement.values if isinstance(statement, ast.JoinedStr) else [statement]
            words = " ".join(part.value for part in parts
                             if isinstance(part, ast.Constant) and isinstance(part.value, str))
            if re.match(r"\s*(?:INSERT|UPDATE|DELETE|REPLACE)\b", words, re.I):
                direct.append((path.name, node.lineno))
            elif isinstance(statement, ast.Call) and isinstance(statement.func, ast.Attribute):
                if statement.func.attr in {"insert_ignore", "upsert"}:
                    direct.append((path.name, node.lineno))
    assert direct == []
    source = Path(session_controls.__file__).read_text()
    assert "_observe" not in source and "_value_rows" not in source
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   (getattr(node, "module", None) == "re" or
                    any(name.name == "re" for name in node.names))
                   for node in ast.walk(ast.parse(source)))
