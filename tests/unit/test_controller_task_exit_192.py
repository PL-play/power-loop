"""Independent invariants retained after migration to the message runtime."""
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_new_controller_event_identifiers_fit_every_declared_task_store_dialect():
    from types import SimpleNamespace

    from power_loop.runtime.controller_handoff import identity as handoff_id
    from power_loop.runtime.controller_recovery import identity as stop_id
    from power_loop.runtime.controller_recovery import notice_id
    from power_loop.runtime.controller_retirement import identity as retirement_id
    from power_loop.runtime.controller_task_exit import identity as exit_id
    from power_loop.runtime.jobs import JobLease

    manager = SimpleNamespace(store=SimpleNamespace(store_id='original-store'))
    lease = JobLease('job_' + 'a'*32, 'original-executor', 'original-boot', 1)
    source = {'job_id': lease.job_id, 'generation': lease.generation}
    for identity in (handoff_id(manager, source), retirement_id(manager, lease), exit_id(manager, lease),
                     stop_id(manager, source), notice_id(manager, source)):
        assert len(identity) <= 64
