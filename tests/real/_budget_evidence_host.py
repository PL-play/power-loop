"""Budget host used by current response-capture and evidence tests."""
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from tests.unit.test_budget_initial_evidence_192 import InitialHost


class RecoveryHost(InitialHost):
    async def prepare(self, context, logical_request, *, service):
        assert getattr(service, "provider", service) is self.service
        return BudgetPlan(16384, {'attempt_id': context['attempt_id'], 'origin': context})


def config(host):
    return BudgetConfig('initial-evidence-host', (
        LocalBudgetScope('shared', max_calls=1, max_tokens=16384, max_inflight=1),), host)


