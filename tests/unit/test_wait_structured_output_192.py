"""A child resolves input and accepted nudges before publishing structured output."""

import pytest

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
SCHEMA = {'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer']}


@pytest.mark.parametrize('status',['failed','cancelled','hit_round_limit','degraded','timed_out'])
async def test_schema_shaped_negative_result_does_not_become_success(status):
    from types import SimpleNamespace

    from power_loop import RequestOptions
    from power_loop.agent.runtime_worker import _InputStart
    context=SimpleNamespace(options=RequestOptions(response_format={'type':'json_schema','json_schema':{'name':'Answer','schema':SCHEMA}}),
        descriptor=SimpleNamespace(config={'child_spec':{}}))
    start=_InputStart(None,context,None,1)
    result=SimpleNamespace(status=status,final_text='{"answer":42}')
    start.completed(result)
    assert result.status==status and start.structured_output is None
