"""Actual model-exit snapshots distinguish an unstarted native subset."""


import pytest

from power_loop.runtime.jobs import JobError

from ._owned_pending_case_192 import owned_pending_case
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]








async def test_closed_original_source_cannot_dispatch_even_a_control_call(registry_store):
    from types import SimpleNamespace

    from power_loop.runtime.invocation import invocation_context
    from power_loop.runtime.task_runtime import _SEND
    from power_loop.runtime.tool_call_plans import dispatch

    c = await owned_pending_case(registry_store, 'subagent', native_tail='renamed_control')
    try:
        binding = c.state.source_binding
        pending = (await c.store.get_state(binding.actor.session_id)).pending
        token = _SEND.set(binding)
        try:
            with invocation_context(binding.actor.session_id, pending['assistant_seq'], 'original-native'):
                with pytest.raises(JobError, match='execution_control_closed'):
                    await dispatch(SimpleNamespace(session_id=binding.actor.session_id), 'renamed_control', {},
                                   c.loop.tool_registry.get('renamed_control'))
        finally:
            _SEND.reset(token)
        assert c.state.native_calls == 0
    finally:
        await cleanup(c)
