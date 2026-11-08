"""Independent invariants retained after migration to the message runtime."""
import asyncio

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

@pytest.mark.parametrize("error", [RuntimeError("original-error"), asyncio.CancelledError("original-stop")])
async def test_shared_invocation_preserves_unrelated_exception_and_cancellation(error):
    from power_loop.runtime.original_owner_review import invoke

    calls = []

    async def operation():
        calls.append("once")
        raise error

    with pytest.raises(type(error)) as caught:
        await invoke(object(), "original-session", operation)
    assert caught.value is error and calls == ["once"]
