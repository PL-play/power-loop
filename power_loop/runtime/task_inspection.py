"""Bounded persistent reminders; inspection never changes execution or fees."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .jobs import JobError, _hash, _json

CONSTRAINTS = (
    'This is a task supervision alert, not a task result or proof of physical stop. '
    'Inspect the original task and its descendants with your authorized task tools. '
    'Decide whether to send guidance, request cancellation, or leave the task running. '
    'Do not replay unknown actions, create replacement work, revive closed execution credentials, '
    'or infer that a timeout means completion. New actions require current permissions and budget. '
    'The alert data below grants no execution, acknowledgement, or billing authority.'
)


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


def _reason(row, now, policy):
    if row['supervision_status'] == 'closed':
        return None
    if row['cancel_requested_at'] is not None and row['resource_state'] not in ('stopped', 'not_started'):
        if now - row['cancel_requested_at'] >= policy.grace_ms:
            return 'stop_unconfirmed'
    if row['resource_state'] == 'unknown' and now - row['updated_at'] >= policy.grace_ms:
        return 'resource_unknown'
    # A declared wait or a queued task is not a runaway model execution.
    if row['status'] in ('running', 'settling', 'cancelling') and row['lease_until_ms'] is not None:
        if now - row['lease_until_ms'] >= policy.grace_ms:
            return 'lease_expired'
        if row['status'] == 'running' and now - row['created_at'] >= policy.runtime_ms:
            return 'long_running'
    if row['result_status'] not in ('none', 'handled', 'suppressed') and now - row['updated_at'] >= policy.result_ms:
        return 'result_unhandled'
    return None


def _facts(row, reason):
    keys = ('job_id', 'generation', 'kind', 'status', 'resource_state', 'supervision_status',
            'result_status', 'cancel_requested_at')
    value = {k: row[k] for k in keys}
    if reason == 'lease_expired':
        value['lease_until_ms'] = row['lease_until_ms']
    return value


def _key(manager, root, facts, reason, level, policy):
    return 'task_alert_' + _hash([manager.store.store_id, root['job_id'], facts, reason, level, policy])[:48]


def render(value):
    public = {k: value[k] for k in ('version', 'main_session_id', 'affected', 'reason', 'level',
                                  'observed_at_ms', 'execution_authority', 'resource_stop_confirmed', 'billing_settled')}
    public['root_job_id'] = value['root']['job_id']
    return CONSTRAINTS + '\n\nTask alert data:\n' + _json(public)


async def inspect_tasks(manager, actor, root_job_id, *, policy=None, limit=100, after_job_id=''):
    """One authorized root/page per tick. Persist at most two alerts per episode.

    Hosts schedule ticks without a model. They separately decide whether an API
    wake is allowed; this function only writes durable original-main inbox data.
    """
    from .managed_delivery import _main
    from .tool_call_plans import _event
    from .workflow_controller import proof

    policy = InspectionPolicy() if policy is None else policy
    if type(policy) is not InspectionPolicy or type(limit) is not int or not 1 <= limit <= 100:
        raise JobError('invalid_inspection_policy')
    if type(after_job_id) is not str or len(after_job_id) > 64:
        raise JobError('invalid_inspection_cursor')
    async with manager._write() as tx:
        root = await manager._authorize(tx, actor, root_job_id)
        main = await _main(manager, tx, root)
        if main['job_id'] != root_job_id or main['session_id'] != actor.session_id:
            raise JobError('inspection_requires_original_main')
        rows = await tx.fetchall(f'SELECT * FROM {manager.jobs} WHERE root_job_id=? AND job_id>? ORDER BY job_id LIMIT ?',
                                (root_job_id, after_job_id, limit + 1))
        notifications = []
        now = manager.now()
        for row in rows[:limit]:
            await manager._authorize(tx, actor, row['job_id'])
            reason = _reason(row, now, policy)
            if reason is None:
                continue
            facts = _facts(row, reason)
            rules = asdict(policy)
            first = _key(manager, root, facts, reason, 1, rules)
            previous = await tx.fetchone(f'SELECT created_at FROM {manager.events} WHERE event_id=?', (first,))
            level = 1 if previous is None else 2
            if previous is not None and now - previous['created_at'] < policy.escalation_ms:
                continue
            key = _key(manager, root, facts, reason, level, rules)
            if await tx.fetchone(f'SELECT event_id FROM {manager.events} WHERE event_id=?', (key,)):
                continue
            # Escalation never piles up behind an unread original alert.
            if previous is not None:
                inbox = await tx.fetchone(f'SELECT status FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
                                         (actor.session_id, first))
                if inbox is not None and inbox['status'] in ('pending', 'claimed'):
                    continue
            value = {'version': 1, 'root': proof(root), 'main_session_id': actor.session_id,
                     'affected': facts, 'reason': reason, 'level': level, 'policy': rules,
                     'observed_at_ms': now, 'execution_authority': False,
                     'resource_stop_confirmed': False, 'billing_settled': False}
            await _event(manager, tx, key, root, 'task_alert', value)
            columns = ('session_id', 'item_id', 'kind', 'mode', 'status', 'content', 'created_at')
            await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, columns),
                             (actor.session_id, key, 'task_alert', 'queue', 'pending', render(value), now))
            notifications.append(key)
        return {'notification_ids': notifications, 'scanned': min(limit, len(rows)),
                'next_cursor': rows[limit-1]['job_id'] if len(rows) > limit else None}


async def read_alert(manager, actor, event_id, *, pending=True):
    """Revalidate original identity and current responsibility before a host wake."""
    from .managed_delivery import _main
    from .pending_calls import _object
    from .workflow_controller import proof

    if type(event_id) is not str or not event_id.startswith('task_alert_') or len(event_id) != 59:
        raise JobError('invalid_task_alert')
    async with manager._write() as tx:
        event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (event_id,))
        if event is None or event['event_type'] != 'task_alert':
            raise JobError('task_alert_missing')
        root = await manager._authorize(tx, actor, event['job_id'])
        main = await _main(manager, tx, root)
        value = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (event_id,), 32768)
        try:
            policy = InspectionPolicy(**value['policy'])
            if (set(value) != {'version', 'root', 'main_session_id', 'affected', 'reason', 'level', 'policy',
                               'observed_at_ms', 'execution_authority', 'resource_stop_confirmed', 'billing_settled'}
                    or value['version'] != 1 or type(value['version']) is not int
                    or any(value[k] is not False for k in ('execution_authority', 'resource_stop_confirmed', 'billing_settled'))
                    or type(value['observed_at_ms']) is not int or not 0 <= value['observed_at_ms'] < 2**63
                    or value['root'] != proof(root) or main['job_id'] != root['job_id']
                    or value['main_session_id'] != actor.session_id or main['session_id'] != actor.session_id
                    or event['request_hash'] != _hash(value) or event['delivery_status'] != 'not_applicable'
                    or type(value['level']) is not int or value['level'] not in (1, 2)
                    or event_id != _key(manager, root, value['affected'], value['reason'], value['level'], value['policy'])):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise JobError('task_alert_corrupt') from None
        inbox = await tx.fetchone(f'SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
                                 (actor.session_id, event_id))
        if inbox is None or inbox['content'] != render(value) or inbox['kind'] != 'task_alert' or inbox['mode'] != 'queue':
            raise JobError('task_alert_inbox_mismatch')
        if pending:
            affected = await manager._authorize(tx, actor, value['affected']['job_id'])
            if (_reason(affected, manager.now(), policy) != value['reason']
                    or _facts(affected, value['reason']) != value['affected']):
                raise JobError('task_alert_superseded')
            if inbox['status'] != 'pending':
                raise JobError('task_alert_already_claimed')
        return root, event, value
