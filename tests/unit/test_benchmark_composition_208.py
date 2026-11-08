"""Timing harness workers can execute any session in their declared group."""
from contextlib import AsyncExitStack

import pytest

from power_loop import MemoryMessageTransport
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from tests.design203_benchmark import batch
from tests.run_design203_runtime_benchmark import SCENARIOS, composition

from .test_job_registry_192 import registry_store

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('scenario',SCENARIOS)
async def test_current_timing_composition_has_matching_session_models(registry_store,scenario):
    store,_=registry_store
    coordinator,transport=MemorySessionOwnershipCoordinator(),MemoryMessageTransport()
    models={}
    try:
        async with AsyncExitStack() as stack:
            operations=[await stack.enter_async_context(composition(store,lane,scenario,
                coordinator=coordinator,transport=transport,models=models)) for lane in range(1 if scenario=='same_session' else 2)]
            if scenario=='same_session':
                operations*=2
            await batch(operations,4)
        assert not transport._subscriptions
    finally:
        await transport.aclose()
