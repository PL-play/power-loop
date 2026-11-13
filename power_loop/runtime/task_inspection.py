"""Legacy inspection declarations; session results are read through RuntimeClient."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .jobs import JobError


@dataclass(frozen=True)
class InspectionPolicy:
    grace_ms: int = 120000
    runtime_ms: int = 1800000
    result_ms: int = 300000
    escalation_ms: int = 1800000

    def __post_init__(self):
        if any(type(n) is not int or not 1000 <= n <= 30 * 86400000 for n in asdict(self).values()):
            raise JobError('invalid_inspection_policy')
        if self.escalation_ms < self.grace_ms:
            raise JobError('invalid_inspection_policy')


async def list_open_roots(manager, *, after_job_id='', limit=100):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise JobError('invalid_inspection_limit')
    if type(after_job_id) is not str or len(after_job_id) > 64:
        raise JobError('invalid_inspection_cursor')
    rows = await manager.db.fetchall(
        f"SELECT * FROM {manager.jobs} WHERE supervision_status='open' "
        "AND parent_job_id IS NULL AND job_id>? ORDER BY job_id LIMIT ?",
        (after_job_id, limit + 1),
    )
    return {
        'roots': rows[:limit],
        'next_cursor': rows[limit - 1]['job_id'] if len(rows) > limit else None,
    }


async def inspect_tasks(*_args, **_kwargs):
    raise JobError('capability_unavailable', 'job inspection is removed')


async def read_alert(*_args, **_kwargs):
    raise JobError('capability_unavailable', 'job alert inspection is removed')


__all__ = ['InspectionPolicy', 'inspect_tasks', 'list_open_roots', 'read_alert']
