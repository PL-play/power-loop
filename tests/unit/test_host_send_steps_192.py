"""Host input validation and provider request accounting."""
import pytest

from power_loop.runtime import host_send_steps as steps
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import TaskRuntime

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
STEP = {"namespace":"test.host-processing", "plan_hash":"a"*64, "step_id":"initial:0"}

@pytest.mark.parametrize("value", [None, 1, "true"])
async def test_step_requirement_is_a_strict_host_boolean(value):
    with pytest.raises(JobError, match="invalid_host_send_step_requirement"):
        TaskRuntime(object(), tenant_id="test", principal_id="host", executor_id="e", boot_id="b", require_host_send_steps=value)



async def test_test_provider_counter_excludes_requests_refused_before_transport():
    from types import SimpleNamespace

    from tests.real._bounded import BoundedProvider
    class FakeTransport:
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            return SimpleNamespace(token_usage=SimpleNamespace(total_tokens=10))
    # Avoid constructing any real client in this unit regression.
    provider = object.__new__(BoundedProvider)
    provider.inner, provider.limit, provider.calls = FakeTransport(), 1, 0
    provider.refused_calls, provider.responses, provider.hold_children = 0, [], False
    await provider.complete(object())
    with pytest.raises(AssertionError, match="exceeded"):
        await provider.complete(object())
    values = {}
    provider.report(values.__setitem__)
    assert values == {"real_provider_requests": 1, "local_provider_calls_refused": 1, "reported_tokens": 10}
    assert provider.inner.calls == provider.calls == 1



@pytest.mark.parametrize("field,value", [("namespace", ""), ("plan_hash", "x" * 64), ("step_id", " whitespace "), ("step_id", "a" * 129)])
async def test_invalid_descriptor_is_refused_before_execution(field, value):
    with pytest.raises(JobError, match="invalid_host_send_step"):
        steps.descriptor(**{**STEP, field: value})
