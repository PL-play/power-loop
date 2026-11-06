"""Business context is scoped and cannot mutate the SDK's execution identity."""
import asyncio
from dataclasses import FrozenInstanceError

import pytest

from power_loop import ExecutionContext, bind_host_context, get_execution_context
from power_loop.core.execution_context import _bind_execution_context, independent_execution_scope

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_host_fields_are_whitelisted_frozen_and_reset_across_tasks():
    context = ExecutionContext('store', 'session', 'node', 'boot', owner_epoch=7, owner_token='secret')
    assert get_execution_context(required=False) is None
    with _bind_execution_context(lambda: context, {'app': {'trace', 'values'}}):
        with pytest.raises(ValueError):
            with bind_host_context('owner', {'owner_epoch': 9}):
                pass
        with pytest.raises(ValueError):
            with bind_host_context('app', {'job_ref': 'forged'}):
                pass
        with pytest.raises(FrozenInstanceError):
            get_execution_context().owner_token = 'forged'
        source = {'values': [1]}
        with bind_host_context('app', source):
            source['values'].append(2)
            assert get_execution_context().host_context['app']['values'] == (1,)
            with pytest.raises(TypeError):
                get_execution_context().host_context['app']['trace'] = 'forged'
            async def child():
                with independent_execution_scope():
                    assert get_execution_context(required=False) is None
                assert get_execution_context().owner_epoch == 7
            await asyncio.create_task(child())
        assert not get_execution_context().host_context
    assert get_execution_context(required=False) is None
