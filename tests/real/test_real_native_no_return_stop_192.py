import pytest

from tests.unit._native_no_return_stop_192 import exercise
from tests.unit.test_job_registry_192 import registry_store

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_no_return_child_stops_without_provider_replay(registry_store, record_property):
    provider = BoundedProvider(limit=2)
    try:
        await exercise(registry_store, provider, record_property, target='root', real=True)
    finally:
        provider.report(record_property)
        await provider.close()
