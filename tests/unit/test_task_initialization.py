"""Fresh current stores converge under concurrent provisioning on all three engines."""
from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, TASK_SCHEMA_VERSION

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_empty_concurrent_initialization_and_current_reopen(registry_store):
    _, connect = registry_store
    stores = await asyncio.gather(*(connect(suffix="fresh_") for _ in range(4)))
    assert len({store.store_id for store in stores}) == 1
    for store in stores:
        assert store.schema_version == TASK_SCHEMA_VERSION == 16
        row = await store._db.fetchone(f"SELECT protocol_version FROM {store.table_prefix}task_store WHERE id=1")
        assert row["protocol_version"] == TASK_PROTOCOL_VERSION == 42
    reopened = await connect(schema="verify", suffix="fresh_")
    assert reopened.store_id == stores[0].store_id
    assert reopened.store_id != registry_store[0].store_id
