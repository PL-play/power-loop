"""Provider facts survive callback checkpoint faults without repeating tools."""
import asyncio
import json
from decimal import Decimal

import pytest

from power_loop.contracts.tools import ToolResult
from power_loop.runtime.jobs import _hash, _json
from power_loop.runtime.native_returns import flush_local
from power_loop.runtime.pending_calls import recover_runtime
from power_loop.runtime.pricing import BillingReceipt, Money
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

from ._native_return_faults_192 import fail_return
from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_jobs_tool_192 import invoke
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', [None, 'write', 'ack', 'cancel', 'tamper'])
async def test_receipt_survives_checkpoint_and_recovery(registry_store, stage):
    receipt = BillingReceipt(provider_amount=Money(Decimal('0.123456789012'), 'USD'),
                             provider_request_id='private-provider-request',
                             units={'images': 1}, raw={'private': 'billing metadata'})

    def configure(c):
        old = c.loop.tool_registry.get('native_data')

        async def native():
            c.state.native_calls += 1
            return ToolResult({'answer': 'image ready'}, receipt)

        c.loop.tool_registry.register(old.definition, FunctionRuntimeTool(native,
            ToolRuntimeContract('example.receipt'), policy=lambda args: ToolDispatchPolicy(True)), overwrite=True)
        if stage is not None:
            c.return_fault = fail_return(c, 'write' if stage == 'tamper' else stage)

    c = await owned_pending_case(registry_store, 'subagent', native_tail='native_data', native_first=True,
                                configure=configure, flush_checkpoints=False, raw_fail_at=999)
    loop = runtime = None
    try:
        if stage is not None:
            assert c.return_fault.fired == 1
            assert c.runtime._native_returns.pending_count() == 1
            await asyncio.gather(*(flush_local(c.runtime) for _ in range(3)))
        rows = await c.manager.db.fetchall(
            f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'",
            (c.state.source,))
        assert len(rows) == 1
        payload = json.loads(rows[0]['payload_json'])
        assert payload['version'] == 2
        assert BillingReceipt.from_dict(payload['billing_receipt']) == receipt
        assert json.loads(payload['return_text']) == {'answer': 'image ready'}
        assert 'private' not in payload['return_text']
        public = await invoke(c.loop, c.state.root, action='get', job_id=c.state.source,
                              view='returns', event_id=rows[0]['event_id'], max_chars=12000)
        visible = json.loads(public['text'])
        assert visible['return_text'] == payload['return_text']
        assert 'billing_receipt' not in visible
        assert 'private' not in public['text']
        assert visible['billing_settled'] is False
        if stage == 'tamper':
            payload['billing_receipt']['provider_amount']['amount'] = '999'
            await c.manager.db.execute(
                f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?',
                (_json(payload), _hash(payload), rows[0]['event_id']))
        if stage is not None:
            loop, runtime = await new_main(c, registry_store)
            recovered = await recover_runtime(runtime, runtime.actor(c.state.root), c.state.source)
            assert recovered['state'] == ('unconfirmed' if stage == 'tamper' else 'queued_for_review')
        assert c.state.native_calls == 1
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('stage', [None, 'write', 'ack', 'cancel', 'tamper'])
async def test_receipt_is_preserved_when_callback_does_not_return(registry_store, stage):
    from power_loop import record_billing_receipt
    from power_loop.runtime.jobs import JobError
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_billing import read_receipt
    from power_loop.runtime.tool_call_plans import read as read_plan
    receipt = BillingReceipt(provider_amount=Money(Decimal('0.25'), 'USD'), provider_request_id='paid-failure')

    def configure(c):
        old = c.loop.tool_registry.get('native_data')

        async def native():
            c.state.native_calls += 1
            with pytest.raises(JobError, match='original_required'):
                await asyncio.create_task(record_billing_receipt(receipt))
            await record_billing_receipt(receipt)
            await record_billing_receipt(receipt)
            # Provider charged successfully; later response processing fails.
            raise ValueError('cannot decode tool response')

        c.loop.tool_registry.register(old.definition, FunctionRuntimeTool(native,
            ToolRuntimeContract('example.paid-failure'), policy=lambda args: ToolDispatchPolicy(True)), overwrite=True)
        if stage not in (None, 'tamper'):
            c.return_fault = fail_return(c, stage, event_type='tool_billing_receipt')

    c = await owned_pending_case(registry_store, 'subagent', native_tail='native_data', native_first=True,
                                configure=configure, flush_checkpoints=False, raw_fail_at=999)
    try:
        if stage not in (None, 'tamper'):
            assert c.return_fault.fired == 1
            assert c.runtime._native_returns.pending_count() == 1
            assert not (await c.runtime.executor_status())['store_safe_to_close']
            await asyncio.gather(*(flush_local(c.runtime) for _ in range(3)))
        assert c.runtime._native_returns.pending_count() == 0
        rows = await c.manager.db.fetchall(
            f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='tool_billing_receipt'",
            (c.state.source,))
        assert len(rows) == 1
        payload = json.loads(rows[0]['payload_json'])
        assert BillingReceipt.from_dict(payload['billing_receipt']) == receipt
        assert not await c.manager.db.fetchall(
            f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'",
            (c.state.source,))
        async with c.manager._write() as tx:
            source = await c.manager._authorize(tx, c.runtime.actor(c.state.root), c.state.source)
            ref = _ref(payload['call_ref'])
            plan = await read_plan(c.manager, tx, source, ref)
            entry = next(e for e in plan['entries'] if e['call_id'] == payload['call_id'])
            _, original = await read_receipt(c.manager, tx, source, ref, entry, plan)
            assert original == payload
            if stage == 'tamper':
                payload['billing_receipt']['provider_amount']['amount'] = '999'
                await tx.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?',
                                 (_json(payload), _hash(payload), rows[0]['event_id']))
                with pytest.raises(JobError, match='anchor_changed'):
                    await read_receipt(c.manager, tx, source, ref, entry, plan)
        assert c.state.native_calls == 1
    finally:
        await cleanup(c, None, None)


async def test_receipt_outside_original_tool_is_rejected():
    from power_loop import record_billing_receipt
    from power_loop.runtime.jobs import JobError
    with pytest.raises(JobError, match='checkpoint_unavailable'):
        await record_billing_receipt(BillingReceipt())
