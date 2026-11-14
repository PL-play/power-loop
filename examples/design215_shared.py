"""Two workers, independent connections, one session store. No Redis required.

This example uses a temporary SQLite database and database polling transport.
It verifies shared composition; the real three-node deployment drill is separate.
"""
import asyncio
import tempfile
from pathlib import Path

from design215_single_process import build_demo

from power_loop import RegistryRuntimeFactory, Runtime, RuntimeDescriptor, SessionStore, TrustedActor
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_transport import DatabasePollingTransport


async def main():
    with tempfile.TemporaryDirectory(prefix='power-loop-215-') as directory:
        runtimes = []
        for node in ('a', 'b'):
            store = await SessionStore.open(str(Path(directory) / 'shared.db'), schema='tasks')
            factory = RegistryRuntimeFactory()
            factory.register('demo.agent@215', build_demo)
            runtime = Runtime(store=store, factory=factory, node_id=node, shared_execution=True,
                transport=DatabasePollingTransport(RuntimeInputStore(store), interval_s=.02),
                recovery_interval_s=.1)
            runtimes.append(runtime)
            await runtime.worker.start()
        try:
            actor = TrustedActor('demo', 'host', None)
            target = await runtimes[0].client.create_session(RuntimeDescriptor(
                2, 'agent', 'demo.agent@215', None, 'builtin.empty@215', 'builtin.none@215',
                'builtin.none@215', {'max_rounds':2}, {}), actor=actor)
            receipt = await runtimes[1].client.send(target, 'hello', message_id='hello', actor=actor)
            result = await runtimes[0].client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+10000)
            assert result.status == 'completed' and result.final_text == 'done'
            print('shared:', result.final_text)
        finally:
            for runtime in runtimes:
                await runtime.aclose(timeout_s=5)


if __name__ == '__main__':
    asyncio.run(main())
