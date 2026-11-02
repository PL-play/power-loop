"""Unbudgeted send admission must not query budget storage or its originals."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from ._budget_policy_admission_worker import make_loop
from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled, _counts

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('registered', [False, True])
async def test_unbudgeted_send_reuses_control_admission_without_any_budget_read(registry_store, registered):
    store, _ = registry_store
    original = store._db
    store._db = profiled = Profiled(original)
    loop, runtime, model, host = make_loop(store, executor='free', budgeted=False, registered=registered)
    sid = await loop.new_session()
    try:
        assert (await loop.send('warm', sid)).status == 'completed'
        cold_records = list(profiled.records)
        profiled.records.clear()
        profiled.tx_sites.clear()
        assert (await loop.send('measured', sid)).status == 'completed'
        records = list(profiled.records)
        counts = _counts(records, profiled.tx_sites)
        assert model.calls == 2 and host.reserve_calls == host.apply_calls == 0
        forbidden = [record for record in [*cold_records, *records] if re.search(
            r'\b'+re.escape(store.table_prefix)+r'(?:task_budget_\w+|execution_allocations)\b', record['sql'])]
        assert not forbidden, forbidden
        assert counts['billing'] == 0, counts
        if target := os.environ.get('POWER_LOOP_SQL_PROFILE'):
            folder = Path(target)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/f'unbudgeted-{profiled.dialect.name}-registered{int(registered)}.json').write_text(json.dumps({
                'registered':registered, 'cold_records':cold_records, 'records':records,
                'tx_sites':profiled.tx_sites, 'counts':counts}, indent=2)+'\n')
    finally:
        if registered:
            await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
        store._db = original
