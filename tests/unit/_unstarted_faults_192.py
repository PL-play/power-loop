"""Faults at the actual automatic generation-zero negative commit boundary."""
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_unstarted_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)
    selected = {'result': 'result', 'negative': 'managed_effect_unstarted',
                'closure': 'supervision_closed', 'commit': 'supervision_closed',
                'commit_ack': 'supervision_closed'}[stage]

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, key):
            return getattr(self.target, key)

        async def execute(self, statement, params=()):
            if (not state.fired and statement.startswith(f'INSERT INTO {manager.events} ')
                    and len(params) > 2 and params[2] == selected):
                job = await self.target.fetchone(f'SELECT kind,generation FROM {manager.jobs} WHERE job_id=?', (params[1],))
                if job is not None and job['kind'] == 'background_tool' and job['generation'] == 0:
                    self.flags['target'] = True
                    if stage not in ('commit', 'commit_ack'):
                        state.fired = True
                        raise OSError('test_unstarted_' + stage)
            return await self.target.execute(statement, params)

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
            if flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_unstarted_' + stage)
        if flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_unstarted_' + stage)

    manager._write = intercepted
    return state
