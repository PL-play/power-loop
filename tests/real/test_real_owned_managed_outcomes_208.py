"""Real child actions retain their original result, parent and budget."""
import pytest

from power_loop import SessionStore
from tests.unit.test_owned_managed_outcomes_208 import (
    test_owned_effect_original_result_is_visible_without_replaying_action as run_owned,
)

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('kind',['subagent','workflow_node'])
@pytest.mark.parametrize('outcome',['known','unknown','missing_receipt','partial','exhausted'])
async def test_real_owned_effect_uses_original_task_and_budget(tmp_path,record_property,kind,outcome):
    store=await SessionStore.open(str(tmp_path/'owned-effect.db'),schema='tasks')
    provider=BoundedProvider(limit=3)
    try:
        await run_owned((store,None),kind,outcome,provider=provider)
    finally:
        provider.report(record_property)
        await provider.close()
        await store.close()
