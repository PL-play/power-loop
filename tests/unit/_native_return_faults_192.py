"""Native return persistence faults at the actual source transaction."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_return(c, stage, *, persistent=False, event_type='native_callback_returned'):
    original = c.manager._write
    state = SimpleNamespace(fired=0, enabled=True)

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, name):
            return getattr(self.target, name)

        async def execute(self, sql, params=()):
            if (state.enabled and (persistent or not state.fired)
                    and sql.startswith(f'INSERT INTO {c.manager.events} ') and len(params) > 2
                    and params[1] == c.state.source and params[2] == event_type):
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

    c.manager._write = intercepted
    return state
