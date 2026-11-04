"""Shared original accounting fixture for staged-close process tests."""
from types import SimpleNamespace

from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_budget_initial_evidence_192 import InitialHost


def case(store, *, cold=False):
    host = InitialHost()
    scopes = tuple(LocalBudgetScope(key, max_calls=2, max_tokens=200, max_inflight=1, binding=key)
                   for key in ("authority", "root", "owner"))
    config = BudgetConfig("host-budget-v1", scopes, host, max_pending=1)
    runtime = TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler" if cold else "executor", boot_id="new-boot" if cold else "boot",
        boot_contract_hash="a" * 64, budget_config=config)
    return SimpleNamespace(store=store, host=host, config=config, runtime=runtime, manager=runtime.manager)
