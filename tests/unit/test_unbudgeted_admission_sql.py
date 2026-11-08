"""After principal initialization, free sends do not read billing state."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled, _counts
from .test_stateful_loop import _Scripted

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_unbudgeted_send_reuses_control_admission_without_any_budget_read(registry_store):
    store, _ = registry_store
    original = store._db
    store._db = profiled = Profiled(original)
    model = _Scripted()
    runtime = TaskRuntime(JobManager(store), tenant_id='test', principal_id='host', executor_id='worker', boot_id='boot')
    loop = await RunningAgent(store, model, config=AgentLoopConfig(compactor=None), task_runtime=runtime).start()
    try:
        assert (await loop.send('warm')).status == 'completed'
        cold_records = list(profiled.records)
        profiled.records.clear()
        profiled.tx_sites.clear()
        assert (await loop.send('measured')).status == 'completed'
        records = list(profiled.records)
        counts = _counts(records, profiled.tx_sites)
        assert len(model.calls) == 2
        forbidden = [record for record in records if not record['sql'].lstrip().upper().startswith('DELETE') and re.search(
            r'\b'+re.escape(store.table_prefix)+r'(?:task_budget_\w+|execution_allocations)\b', record['sql'])]
        assert not forbidden, forbidden
        # Startup reads an existing principal policy once; warm sends reuse the
        # admission row. Terminal allocation cleanup may still issue DELETE.
        initial_policies = [r for r in cold_records if 'SELECT manifest_json' in r['sql']
            and 'task_budget_policies' in r['sql']]
        assert len(initial_policies) == 1
        assert counts['billing'] == 0, [r for r in records if any(c.startswith('runtime.budget_') for c in r['chain'])]
        if target := os.environ.get('POWER_LOOP_SQL_PROFILE'):
            folder = Path(target)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/f'unbudgeted-{profiled.dialect.name}.json').write_text(json.dumps({
                'cold_records':cold_records, 'records':records,
                'tx_sites':profiled.tx_sites, 'counts':counts}, indent=2)+'\n')
    finally:
        await loop.aclose()
        store._db = original
