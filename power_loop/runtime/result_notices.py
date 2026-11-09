"""Known host outcomes use the original main descriptor and current input format."""
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import Command, InputRef, RuntimeDescriptor, TargetRef, TrustedActor


async def accept(manager, tx, actor, root_job_id, key, kind, body):
    inputs = RuntimeInputStore(manager.store)
    original = await inputs.execution_input(tx, actor.session_id, root_job_id)
    await inputs.accept_input(tx, Command(key, TargetRef(manager.store.store_id, actor.session_id),
        'send', {'body': body}, options=original.meta['options']),
        TrustedActor(actor.tenant_id, actor.principal_id, None),
        descriptor=RuntimeDescriptor(**original.meta['descriptor']))
    await tx.execute(f'UPDATE {inputs.q} SET kind=? WHERE session_id=? AND item_id=?',
        (kind, actor.session_id, key))


async def notify(client, actor, key):
    """Only signal accepted work; never await its model execution."""
    target = TargetRef(client.store.store_id, actor.session_id)
    return await client._notify(target, key, InputRef(target.store_id, actor.session_id, key), ['work', 'activation'])
