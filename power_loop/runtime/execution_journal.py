"""Project control writes and advance one summary in the same transaction."""
from __future__ import annotations

import json

from .jobs import JobError, _json


def control_tables_ddl(name, prefix):
    short = "VARCHAR(255)" if name == "mysql" else "TEXT"
    body = "LONGTEXT" if name == "mysql" else "TEXT"
    integer = "INTEGER" if name == "sqlite" else "BIGINT"
    options = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin" if name == "mysql" else ""
    return [
        f"""CREATE TABLE IF NOT EXISTS {prefix}execution_checkpoints (
            job_id {short} NOT NULL, generation {integer} NOT NULL,
            batch_no {integer} NOT NULL, summary_json {body} NOT NULL,
            PRIMARY KEY (job_id, generation)){options}""",
        f"""CREATE TABLE IF NOT EXISTS {prefix}execution_allocations (
            allocation_id {short} PRIMARY KEY, execution_id {body}, job_id {short} NOT NULL,
            generation {integer} NOT NULL, state_json {body} NOT NULL,
            state_hash {short} NOT NULL){options}""",
    ]


class ExecutionJournal:
    def __init__(self, store, *, projector=None, writer=None):
        self.store = store
        self.db = store._db
        self.heads = store.table_prefix + "execution_checkpoints"
        self.projector = projector
        self.transaction = writer or self.db.transaction

    async def create(self, identity, summary, *, transaction=None):
        if transaction is not None:
            return await self._create(transaction, identity, summary)
        async with self.transaction() as tx:
            await self._create(tx, identity, summary)

    async def _create(self, tx, identity, summary):
        statement = self.db.dialect.insert_ignore(
            self.heads, ["job_id", "generation", "batch_no", "summary_json"])
        if await tx.execute(statement, (identity.job_id, identity.generation, 0, _json(summary))) != 1:
            raise JobError("execution_checkpoint_conflict")

    async def append(self, identity, number, records, summary, *, transaction=None):
        if transaction is not None:
            return await self._append(transaction, identity, number, records, summary)
        async with self.transaction() as tx:
            await self._append(tx, identity, number, records, summary)

    async def _append(self, tx, identity, number, records, summary):
        changed = await tx.execute(
            f"UPDATE {self.heads} SET batch_no=?,summary_json=? "
            "WHERE job_id=? AND generation=? AND batch_no=?",
            (number, _json(summary), identity.job_id, identity.generation, number - 1))
        if changed != 1:
            raise JobError("execution_checkpoint_conflict")
        if self.projector is not None:
            await self.projector(tx, identity, records)

    async def inspect(self, identity):
        row = await self.db.fetchone(
            f"SELECT batch_no,summary_json FROM {self.heads} WHERE job_id=? AND generation=?",
            (identity.job_id, identity.generation))
        if row is None:
            raise JobError("execution_checkpoint_missing")
        return {"batch_no": row["batch_no"], "summary": json.loads(row["summary_json"])}
