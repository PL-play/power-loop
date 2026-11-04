"""Provider facts survive callback checkpoint faults without repeating tools."""

import pytest

from power_loop.runtime.pricing import BillingReceipt

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]






async def test_receipt_outside_original_tool_is_rejected():
    from power_loop import record_billing_receipt
    from power_loop.runtime.jobs import JobError
    with pytest.raises(JobError, match='checkpoint_unavailable'):
        await record_billing_receipt(BillingReceipt())
