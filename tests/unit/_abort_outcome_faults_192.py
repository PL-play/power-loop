"""Fault the aggregate negative outcome's one control transaction."""
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_outcome_once(manager, source_id, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)
    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags
        def __getattr__(self, name):
            return getattr(self.target, name)
        async def execute(self, statement, params=()):
            event = params[2] if statement.startswith(f'INSERT INTO {manager.events} ') and len(params) > 2 else None
            selected = (stage == 'child' and event == 'result' and params[1] != source_id
                or stage == 'result' and event == 'result' and params[1] == source_id
                or stage == 'notice' and event == 'source_abort_delivery'
                or stage == 'inbox' and statement.startswith('INSERT') and manager.store.t.follow_up_queue in statement
                or stage in ('disposition', 'commit', 'commit_ack') and event == 'pending_call_disposition'
                or stage == 'pending' and statement.startswith(f'UPDATE {manager.store.t.session_state} SET pending_json=NULL'))
            if selected and not state.fired:
                self.flags['target'] = True
                if stage not in ('commit', 'commit_ack'):
                    state.fired = True
                    raise OSError('test_outcome_'+stage)
            return await self.target.execute(statement, params)
    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
            if flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_outcome_'+stage)
        if flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_outcome_'+stage)
    manager._write = intercepted
    return state
