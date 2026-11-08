"""A completed original input cannot authorize another tool dispatch."""
from types import SimpleNamespace

import pytest

from power_loop import ToolDefinition
from power_loop.runtime.invocation import invocation_context
from power_loop.runtime.jobs import JobError
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND
from power_loop.runtime.tool_call_plans import dispatch
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_closed_original_source_cannot_dispatch_even_a_control_call(registry_store):
    bindings, calls = [], []
    async def capture(context, arguments):
        bindings.append(_SEND.get())
    async def control(**arguments):
        calls.append('unexpected')
    async def configure(runner, state):
        runner.registry.register(ToolDefinition('inspect_original', 'Inspect the original task',
            input_schema={'type':'object','properties':{'text':{'type':'string'}}}, pricing=Free('fixture')),
            FunctionRuntimeTool(control, ToolRuntimeContract('example.inspect-original'),
                policy=lambda args: ToolDispatchPolicy(True, control_only=True)))
    c = await scenario(registry_store, inspect_prepare=capture, configure=configure,
        raw_receipt_failure=True, call_names=['post_once','inspect_original'])
    binding = bindings[0]
    pending = (await c.store.get_state(c.sid)).pending
    token = _SEND.set(binding)
    try:
        with invocation_context(c.sid, pending['assistant_seq'], 'original-call-2'):
            with pytest.raises(JobError, match='execution_control_closed'):
                await dispatch(SimpleNamespace(session_id=c.sid), 'inspect_original', {'text':'hello'},
                               c.registry.get('inspect_original'))
    finally:
        _SEND.reset(token)
    assert not calls and c.state.dispatched == 1
