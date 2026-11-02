"""Conditional ordinary writes use the current send's frozen ancestor proofs.

Root locking is part of the DML permission,not a separate permission read. The
root subquery is deliberately materialized before locking descendant proofs;
locking reads then see a takeover committed while this statement waited.
"""
from __future__ import annotations


def permission(store, session_id):
    from ..task_runtime import _SEND

    binding = _SEND.get()
    if (binding is None or binding.runtime.manager.store is not store
            or binding.actor.session_id != session_id):
        return None
    evidence = binding.resource_evidence
    cached = evidence.get('_message_write_permission')
    if cached is not None:
        return cached
    backend = store._db.dialect.name
    jobs = store.table_prefix+'jobs'
    proofs = tuple(reversed(evidence['_message_write_proofs']))
    equal = {'sqlite':'IS', 'postgres':'IS NOT DISTINCT FROM', 'mysql':'<=>'}[backend]

    def point(alias):
        return (f'{alias}.job_id=? AND {alias}.generation=? '
            f'AND {alias}.executor_id {equal} ? AND {alias}.boot_id {equal} ?')

    lock = '' if backend == 'sqlite' else ' FOR UPDATE'
    root = f'SELECT message_root.job_id FROM {jobs} message_root WHERE '+point('message_root')+' LIMIT 1'+lock
    arguments = list(proofs[0])
    if len(proofs) == 1:
        query = f'SELECT 1 AS permitted FROM {jobs} message_root WHERE '+point('message_root')+' LIMIT 1'+lock
    else:
        query = 'SELECT 1 AS permitted FROM ('+root+') message_root'
        aliases = []
        join = ' STRAIGHT_JOIN ' if backend == 'mysql' else ' JOIN '
        for offset, proof in enumerate(proofs[1:], 1):
            alias = 'message_owner_'+str(offset)
            aliases.append(alias)
            query += join+jobs+' '+alias+' ON '+alias+'.root_job_id=message_root.job_id AND '+point(alias)
            arguments.extend(proof)
        query += ' LIMIT 1'+(lock+' OF '+','.join(aliases) if backend == 'postgres' else lock)
    evidence['_message_write_permission'] = result = query, tuple(arguments)
    return result
