"""Native return persistence faults at the actual source transaction."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_return(c, stage, *, persistent=False, event_type='native_callback_returned'):
    from power_loop.runtime.execution_state import CheckpointPolicy
    # Exercise an actual boundary during the callback, with a small explicit
    # buffer. Ordinary calls otherwise legitimately remain in memory.
    c.runtime.checkpoint_policy = CheckpointPolicy(bytes=1024)
    original = c.manager.db.transaction
    state = SimpleNamespace(fired=0, enabled=True)

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, name):
            return getattr(self.target, name)

        async def execute(self, sql, params=()):
            matches = False
            if sql.startswith(f'INSERT INTO {c.manager.events} '):
                columns = sql.split('(', 1)[1].split(')', 1)[0].split(',')
                columns = [column.strip() for column in columns]
                if 'event_type' in columns and 'job_id' in columns:
                    matches = any(params[i + columns.index('job_id')] == c.state.source and
                                  params[i + columns.index('event_type')] == event_type
                                  for i in range(0, len(params), len(columns)))
            if (state.enabled and (persistent or not state.fired)
                    and matches):
                self.flags['target'] = True
                if stage != 'ack':
                    state.fired += 1
                    if stage == 'cancel':
                        raise asyncio.CancelledError('original return checkpoint cancelled')
                    raise OSError('original return checkpoint failed')
            return await self.target.execute(sql, params)

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
        if flags['target'] and stage == 'ack':
            state.fired += 1
            raise OSError('original return checkpoint commit lost ACK')

    c.manager.db.transaction = intercepted
    return state
