"""Faults at the actual automatic generation-zero negative commit boundary."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

from tests.control_faults import control_mutations


def fail_unstarted_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)
    selected = {'result': 'result', 'negative': 'managed_effect_unstarted',
                'closure': 'supervision_closed', 'commit': 'supervision_closed',
                'commit_ack': 'supervision_closed'}[stage]

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        joined = manager._joined_write.get() is not None

        async def intercept(execute, statement, params):
            if (not state.fired and statement.startswith(f'INSERT INTO {manager.events} ')
                    and len(params) > 2 and params[2] == selected):
                job = await tx.fetchone(f'SELECT kind,generation FROM {manager.jobs} WHERE job_id=?', (params[1],))
                if job is not None and job['kind'] == 'background_tool' and job['generation'] == 0:
                    flags['target'] = True
                    if stage not in ('commit', 'commit_ack'):
                        state.fired = True
                        raise OSError('test_unstarted_' + stage)
            return await execute(statement, params)

        async with original() as tx:
            async with control_mutations(tx, intercept):
                yield tx
            if not joined and flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_unstarted_' + stage)
        if not joined and flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_unstarted_' + stage)

    manager._write = intercepted
    return state
