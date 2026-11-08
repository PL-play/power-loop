"""Committed notice routing invalidates its recipient, including cross-session handoffs."""
from __future__ import annotations

import pytest

from power_loop.runtime import session_controls
from power_loop.runtime.jobs import JobActor, JobManager

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("rollback", [False, True])
async def test_notice_recipient_version_changes_only_with_its_transaction(registry_store, rollback):
    store, connect = registry_store
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
    observer = JobManager(await connect(schema="verify"))
    after = await session_controls.version(observer, observer.db, recipient)
    assert after == (before[0] + (0 if rollback else 1), before[1])
