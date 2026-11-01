"""Lock metrics cover commit, rollback and a joined control transaction."""
from __future__ import annotations

import asyncio
import logging

import pytest

from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("rollback", [False, True])
async def test_joined_transaction_reports_one_actual_lock_lifetime(registry_store, caplog, rollback):
    store, _ = registry_store
    manager = JobManager(store)

    async def run():
        async with manager.joined_write() as outer:
            async with manager._write() as inner:
                assert inner is outer
                await asyncio.sleep(.01)
            if rollback:
                raise OSError("original transaction rollback")

    with caplog.at_level(logging.DEBUG, logger="power_loop.control.transactions"):
        if rollback:
            with pytest.raises(OSError, match="original transaction rollback"):
                await run()
        else:
            await run()
    [record] = [r for r in caplog.records if r.name == "power_loop.control.transactions"]
    assert record.wait_ms >= 0
    assert record.hold_ms >= 8
    assert record.outcome == ("rolled_back" if rollback else "committed")
    assert record.dialect == store._db.dialect.name
