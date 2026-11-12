import pytest

from ._native_no_return_stop_192 import exercise
from .test_task_runtime_192 import AdapterModel

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('target', ['root', 'child'])
async def test_original_no_return_child_stops_from_another_process(registry_store, record_property, target):
    await exercise(registry_store, AdapterModel(), record_property, target=target)
