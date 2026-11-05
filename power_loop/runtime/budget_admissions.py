"""Principal admission shared by sends, exclusive for first policy activation.

The hash is derived from the immutable policy original. Its row lock bridges
absence checking and tree publication; it grants no model or host authority.
"""
from __future__ import annotations

from .jobs import JobError, _hash


def key(manager, tenant_id, principal_id):
    return 'bp_'+_hash([manager.store.store_id, tenant_id, principal_id])[:48]


async def initialize(manager, tx, identity):
    from .execution_membership import _remember_after_commit, _trusted

    policy_key = key(manager, identity.tenant_id, identity.principal_id)
    if _trusted(manager, '_principal_gate_presence', policy_key):
        return
    table = manager.prefix+'task_principal_admissions'
    if await tx.fetchone(f'SELECT policy_key FROM {table} WHERE policy_key=?', (policy_key,)):
        _remember_after_commit(manager, tx, '_principal_gate_presence', policy_key, True)
        return
    marker = await tx.fetchone(f'SELECT manifest_json AS payload_json,request_hash FROM {manager.prefix}task_budget_policies WHERE policy_key=?',
        (policy_key,))
    if marker is not None:
        from .model_budgets import _object
        _object(marker['payload_json'], marker['request_hash'])
    await tx.write_principal_admission(policy_key,
        manager.db.dialect.insert_ignore(table, ('policy_key', 'policy_hash')),
        (policy_key, marker['request_hash'] if marker is not None else None))
    _remember_after_commit(manager, tx, '_principal_gate_presence', policy_key, True)


async def enter(runtime, tx):
    """Called before the first tree read/write in a send's entry transaction."""
    from .budget_policy_sources import PolicySourceChanged, current

    manager = runtime.manager
    policy_key = key(manager, runtime.tenant_id, runtime.principal_id)
    if policy_key in tx._budget_policy_admissions:
        return
    if runtime.budget_config is None:
        suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR SHARE'
        sql = f'SELECT policy_hash FROM {manager.prefix}task_principal_admissions WHERE policy_key=?'+suffix
        row = await tx.fetchone(sql, (policy_key,))
        if row is None:
            await initialize(manager, tx, runtime)
            row = await tx.fetchone(sql, (policy_key,))
        if row['policy_hash'] is not None:
            raise JobError('budget_provider_required')
        tx._budget_policy_admissions[policy_key] = None
        return
    source = current(runtime)
    exclusive = source.marker is None
    # Only the non-null admission side is locked. PostgreSQL does not permit
    # locking the nullable side of an outer join. Both server backends support
    # OF on the named admission alias; SQLite serializes its writers itself.
    suffix = '' if manager.db.dialect.name == 'sqlite' else (' FOR UPDATE OF admission' if exclusive else ' FOR SHARE OF admission')
    sql = (f'SELECT admission.policy_hash AS admission_hash,policy.* '
        f'FROM {manager.prefix}task_principal_admissions admission '
        f'LEFT JOIN {manager.prefix}task_budget_policies policy ON policy.policy_key=admission.policy_key '
        'WHERE admission.policy_key=?'+suffix)
    row = await tx.fetchone(sql, (policy_key,))
    if row is None:
        await initialize(manager, tx, runtime)
        row = await tx.fetchone(sql, (policy_key,))
    required = row.pop('admission_hash')
    if required is not None:
        if runtime.budget_config is None:
            raise JobError('budget_provider_required')
        if row['policy_key'] is None and source.marker is None:
            # The locking statement can precede a wait for the first policy's
            # commit. Reprepare outside the writer, then retry tree admission.
            raise PolicySourceChanged(required)
    tx._budget_policy_admissions[policy_key] = row if row['policy_key'] is not None else None


async def activated(runtime, tx, digest):
    policy_key = key(runtime.manager, runtime.tenant_id, runtime.principal_id)
    await tx.write_principal_admission(policy_key,
        f'UPDATE {runtime.manager.prefix}task_principal_admissions SET policy_hash=? WHERE policy_key=?',
        (digest, policy_key))


async def rebuild(tx, prefix, store_id):
    """Explicit offline repair, exclusively from existing policy originals."""
    from types import SimpleNamespace

    from .model_budgets import _object
    from .store.schema import StoreSchemaError

    manager = SimpleNamespace(store=SimpleNamespace(store_id=store_id))
    expected = {}
    originals = await tx.fetchall(f'SELECT policy_key,manifest_json AS payload_json,request_hash,tenant_id,principal_id '
        f'FROM {prefix}task_budget_policies')
    for original in originals:
        policy_key = key(manager, original['tenant_id'], original['principal_id'])
        if original['policy_key'] != policy_key or policy_key in expected:
            raise StoreSchemaError('principal admission policy original invalid')
        try:
            _object(original['payload_json'], original['request_hash'])
        except JobError as exc:
            raise StoreSchemaError('principal admission policy original invalid') from exc
        expected[policy_key] = original['request_hash']
    table = prefix+'task_principal_admissions'
    actual = {row['policy_key']: row['policy_hash'] for row in await tx.fetchall(f'SELECT * FROM {table}')}
    if any(digest is not None and policy_key not in expected for policy_key, digest in actual.items()):
        raise StoreSchemaError('principal admission policy original missing')
    for policy_key, digest in sorted(expected.items()):
        if policy_key not in actual:
            await tx.execute(f'INSERT INTO {table} (policy_key,policy_hash) VALUES (?,?)', (policy_key, digest))
        elif actual[policy_key] != digest:
            await tx.execute(f'UPDATE {table} SET policy_hash=? WHERE policy_key=?', (digest, policy_key))
