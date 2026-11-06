"""Explicit continuation of the original interrupted child generation."""
from power_loop.agent.runtime_types import RuntimeRequestError


async def resume_claim(runtime, tx, record, row):
    """Start one explicit continuation; original inputs and results stay fixed."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .execution_membership import start
    from .jobs import JobLease
    from .task_runtime import Execution
    manager = runtime.manager
    RuntimeInputStore.require_resumable(row)
    if row["generation"] != record.row["expected_job_generation"]:
        raise RuntimeRequestError("stale_generation")
    # Ancestors retain cancellation authority, independently of their old owner.
    parent = row["parent_job_id"]
    if parent:
        await manager._supervision_barriers(tx, {**row, "resource_state": "stopped"})
    lease = JobLease(row["job_id"], runtime.executor_id, runtime.boot_id, row["generation"] + 1)
    from .executor_registry import ExecutorIdentity, require_accepting
    executor = await require_accepting(manager, tx,
        ExecutorIdentity(row["tenant_id"], row["principal_id"], runtime.executor_id, runtime.boot_id),
        required=row["executor_key"] is not None)
    changed = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,executor_key=?,"
        "status='running',outcome=NULL,result_json=NULL,result_status='none',resource_state='running',intake='open',"
        "supervision_status='open',version=version+1,updated_at=? WHERE job_id=? AND generation=? AND version=?",
        (lease.generation, lease.executor_id, lease.boot_id, executor["executor_key"] if executor else None,
         manager.now(), row["job_id"], row["generation"], row["version"]),
        obligations=True)
    if changed != 1:
        raise RuntimeRequestError("stale_generation")
    await start(manager, tx, await manager._row(tx, row["job_id"]))
    return Execution(runtime, runtime.actor(row["session_id"]), lease)
