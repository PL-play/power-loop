"""Test-only full comparison; never run as part of an execution transaction."""
from __future__ import annotations

import json
import os
from pathlib import Path

from power_loop.runtime.jobs import JobManager
from power_loop.runtime.session_obligations import COLUMNS, _desired


async def assert_obligation_projection(store, prefix):
    manager = JobManager(store)
    assert manager.prefix == prefix
    async with store._db.transaction() as tx:
        jobs = await tx.fetchall(f"SELECT * FROM {manager.jobs} ORDER BY job_id")
        expected = set()
        for job in jobs:
            expected.update(await _desired(manager, tx, job))
        rows = await tx.fetchall(f"SELECT {','.join(COLUMNS)} FROM {prefix}session_obligations")
        actual = {tuple(row[key] for key in COLUMNS) for row in rows}
        assert actual == expected, {
            "missing_obligations": sorted(expected - actual),
            "stale_obligations": sorted(actual - expected),
        }
        target = os.environ.get("POWER_LOOP_OBLIGATION_AUDIT")
        if target:
            with Path(target).open("a") as log:
                log.write(json.dumps({"case": os.environ.get("PYTEST_CURRENT_TEST"),
                    "prefix": prefix, "engine": store._db.dialect.name,
                    "jobs": len(jobs), "expected": len(expected), "actual": len(actual)}) + "\n")
