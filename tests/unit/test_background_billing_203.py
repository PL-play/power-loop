"""The actual background owner settles its provider receipt once."""
import json
from decimal import Decimal

import pytest

from power_loop import AgentLoopConfig, record_billing_receipt
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import BillingReceipt, Money
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

from .test_background_runtime_203 import BackgroundParent
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host
from .test_nested_tool_billing_196 import Paid

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize("automatic", [False, True])
async def test_paid_background_receipt_uses_original_child_and_root(registry_store,tmp_path,automatic):
    store,_=registry_store
    model,host=BackgroundParent(),Host()
    if automatic:
        complete = model.complete
        async def direct(request, **kwargs):
            first = not model.sent
            response = await complete(request, **kwargs)
            if first:
                response.tool_calls[0]['function'] = {'name':'slow', 'arguments':'{}'}
            return response
        model.complete = direct
    host.service=model
    prepared,observed=[],[]
    async def prepare_tool(context,call,*,pricing):
        if call.name!='slow':
            return None
        prepared.append(context)
        return BudgetPlan(0,{'attempt_id':context['attempt_id'],'origin':context})
    host.prepare_tool=prepare_tool
    prepare_usage=host.prepare_usage
    async def usage(context,grant,observation):
        if context['attempt_id'].startswith('toolattempt_'):
            observed.append(observation)
        return await prepare_usage(context,grant,observation)
    host.prepare_usage=usage
    runtime=TaskRuntime(JobManager(store),tenant_id='test',principal_id='host',executor_id='fixture',boot_id='boot',
        boot_contract_hash='a'*64,budget_config=BudgetConfig('p3-paid',
            (LocalBudgetScope('root',max_calls=50,binding='root'),),host))
    registry=create_default_tool_registry(workspace_dir=str(tmp_path))
    async def slow():
        await record_billing_receipt(BillingReceipt(provider_amount=Money(Decimal('1'),'CNY')))
        return 'paid background result'
    registry.register(ToolDefinition('slow','paid operation',async_capable=True,interrupt='background' if automatic else 'finish',pricing=Paid()),slow)
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,max_tokens=20),task_runtime=runtime).start()
    try:
        parent=await runner.send('start')
        assert parent.status=='completed',parent
        rows=await store.load_all_messages(runner.target.session_id)
        if automatic:
            row = await store._db.fetchone(f"SELECT job_id FROM {store.table_prefix}jobs WHERE kind='background_tool'")
            receipt = {'job_id':row['job_id'], 'job_ref':{'store_id':store.store_id,'job_id':row['job_id'],'generation':0}}
        else:
            receipt=json.loads(next(m.content for m in rows if m.tool_call_id=='background'))
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed'
        await runner.drain()
        tools=[context for context in prepared if context['attempt_id'].startswith('toolattempt_')]
        assert len(tools)==1 and tools[0]['job_id']==receipt['job_id']
        assert tools[0]['root_job_id']==parent.job_ref.job_id
        tool_applied={k:v for k,v in host.applied.items() if k.startswith('toolattempt_')}
        assert len(tool_applied)==1
        assert len(observed)==1 and observed[0]['charge']['amount']=='1'
        assert observed[0]['charge']['currency']=='CNY'
        for name in ('execution_checkpoints','execution_allocations','task_budget_attempts'):
            assert (await store._db.fetchone(f'SELECT COUNT(*) AS n FROM {store.table_prefix}{name}'))['n']==0
    finally:
        await runner.aclose()
        await runtime.shutdown(timeout_s=2)
