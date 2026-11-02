"""One conditional lease update, retaining the registry transaction's lock order."""
from __future__ import annotations


def statement(manager):
    jobs, events, executors = manager.jobs, manager.events, manager.prefix + "task_executors"
    backend = manager.db.dialect.name
    deadline = {
        "sqlite": "CAST(json_extract(e.payload_json,'$.deadline_at_ms') AS INTEGER)",
        "postgres": "CAST(CAST(e.payload_json AS jsonb)->>'deadline_at_ms' AS BIGINT)",
        "mysql": "CAST(JSON_UNQUOTE(JSON_EXTRACT(e.payload_json,'$.deadline_at_ms')) AS SIGNED)",
    }[backend]
    cte = f"""WITH RECURSIVE lineage AS (
        SELECT j.*,0 AS depth,0 AS ignore_unknown FROM {jobs} j WHERE job_id=?
        UNION ALL
        SELECT p.*,c.depth+1,CASE WHEN p.kind='send' AND p.execution_role='main'
            AND p.status='model_closed' AND c.kind IN ('workflow','subagent')
            AND c.control_owner_job_id IS NULL AND c.blocks_completion=0
            AND c.completion_scope_job_id=c.job_id THEN 1 ELSE 0 END
        FROM {jobs} p JOIN lineage c ON p.job_id=c.parent_job_id WHERE c.depth<63
    ), owners AS (
        SELECT j.*,0 AS depth,j.generation AS expected_generation,
            j.executor_id AS expected_executor,j.boot_id AS expected_boot FROM {jobs} j WHERE job_id=?
        UNION ALL
        SELECT p.*,c.depth+1,c.control_owner_generation,c.control_owner_executor_id,c.control_owner_boot_id
        FROM {jobs} p JOIN owners c ON p.job_id=c.control_owner_job_id WHERE c.depth<63
    ) """
    cancelled = "(EXISTS (SELECT 1 FROM lineage WHERE cancel_requested_at IS NOT NULL) OR " \
                "EXISTS (SELECT 1 FROM owners WHERE cancel_requested_at IS NOT NULL))"
    # Every mutable permission participates in this UPDATE. The surrounding
    # registry lock still orders this statement with stops, takeovers and drain.
    valid = f"""intake='open' AND status IN ('running','waiting','settling') AND lease_until_ms>?
        AND NOT EXISTS (SELECT 1 FROM lineage WHERE resource_state='unknown' AND ignore_unknown=0)
        AND NOT EXISTS (SELECT 1 FROM owners WHERE depth>0 AND
            (generation<>expected_generation OR executor_id<>expected_executor OR boot_id<>expected_boot
             OR intake<>'open' OR status NOT IN ('running','waiting','settling') OR lease_until_ms<=?))
        AND NOT EXISTS (SELECT 1 FROM owners o JOIN {events} e ON e.job_id=o.job_id
            AND e.event_type='completion_window' WHERE {deadline}<=?)
        AND NOT EXISTS (SELECT 1 FROM owners o LEFT JOIN {executors} x
            ON x.tenant_id=o.tenant_id AND x.principal_id=o.principal_id
            AND x.executor_id=o.executor_id AND x.boot_id=o.boot_id WHERE
            (o.executor_key IS NULL AND x.executor_key IS NOT NULL) OR
            (o.executor_key IS NOT NULL AND (x.executor_key IS NULL OR x.executor_key<>o.executor_key
              OR x.state<>'accepting' OR x.lease_until_ms<=?)))"""
    flag = f"LAST_INSERT_ID({cancelled})" if backend == "mysql" else cancelled
    update = cte + f"""UPDATE {jobs} SET lease_until_ms=CASE WHEN {flag} THEN lease_until_ms ELSE ? END,updated_at=?
        WHERE job_id=? AND tenant_id=? AND principal_id=? AND executor_id=? AND boot_id=? AND generation=?
        AND EXISTS (SELECT 1 FROM lineage WHERE session_id=? OR created_by_session_id=?)
        AND ({cancelled} OR ({valid}))"""
    if backend != "mysql":
        update += f" RETURNING {cancelled} AS cancelled"
    return update


async def renew(manager, tx, actor, lease, lease_ms):
    now = manager.now()
    return await tx.renew_job(lease.job_id, statement(manager), (
        lease.job_id, lease.job_id, now+lease_ms, now,
        lease.job_id, actor.tenant_id, actor.principal_id, lease.executor_id, lease.boot_id, lease.generation,
        actor.session_id, actor.session_id, now, now, now, now))
