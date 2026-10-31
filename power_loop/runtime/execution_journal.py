"""Incremental, owner-bound control checkpoints and explicit repair.

Raw result bodies and financial receipts keep their own authoritative stores.
This journal contains new control facts and a bounded summary. Derived summaries
are repairable; a broken original chain is detected, never silently guessed.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import asdict

from .execution_state import ExecutionIdentity
from .jobs import JobError, _hash, _json


def control_tables_ddl(name, prefix):
    short = "VARCHAR(255)" if name == "mysql" else "TEXT"
    body = "LONGTEXT" if name == "mysql" else "TEXT"
    integer = "INTEGER" if name == "sqlite" else "BIGINT"
    options = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin" if name == "mysql" else ""
    return [
        f"""CREATE TABLE IF NOT EXISTS {prefix}execution_checkpoints (
            job_id {short} NOT NULL, generation {integer} NOT NULL,
            identity_json {body} NOT NULL, identity_hash {short} NOT NULL,
            admission_nonce {short} NOT NULL,
            batch_no {integer} NOT NULL, chain_hash {short} NOT NULL,
            summary_json {body} NOT NULL, summary_hash {short} NOT NULL,
            PRIMARY KEY (job_id, generation)){options}""",
        f"""CREATE TABLE IF NOT EXISTS {prefix}execution_batches (
            job_id {short} NOT NULL, generation {integer} NOT NULL,
            batch_no {integer} NOT NULL, previous_hash {short} NOT NULL,
            payload_json {body} NOT NULL, payload_hash {short} NOT NULL,
            PRIMARY KEY (job_id, generation, batch_no)){options}""",
        f"""CREATE TABLE IF NOT EXISTS {prefix}execution_allocations (
            allocation_id {short} PRIMARY KEY, job_id {short} NOT NULL,
            generation {integer} NOT NULL, state_json {body} NOT NULL,
            state_hash {short} NOT NULL){options}""",
    ]


class ExecutionJournal:
    def __init__(self, store, *, projector=None):
        self.store = store
        self.db = store._db
        self.heads = store.table_prefix + "execution_checkpoints"
        self.batches = store.table_prefix + "execution_batches"
        self.projector = projector
        self.admission_nonce = secrets.token_hex(16)

    def _identity(self, identity):
        if type(identity) is not ExecutionIdentity or identity.store_id != self.store.store_id:
            raise JobError("execution_store_identity_mismatch")

    @staticmethod
    def _head(row, identity, *, check_summary=True):
        if row is None:
            raise JobError("execution_checkpoint_missing")
        try:
            values = json.loads(row["identity_json"])
            if values != asdict(identity) or row["identity_hash"] != identity.digest:
                raise ValueError()
            if type(row["batch_no"]) is not int or row["batch_no"] < 0:
                raise ValueError()
            if check_summary:
                summary = json.loads(row["summary_json"])
                if type(summary) is not dict or _hash(summary) != row["summary_hash"]:
                    raise ValueError()
        except (ValueError, TypeError, KeyError) as exc:
            raise JobError("execution_checkpoint_corrupt") from exc

    async def create(self, identity, summary, *, transaction=None):
        self._identity(identity)
        if transaction is not None:
            return await self._create(transaction, identity, summary)
        async with self.db.transaction() as tx:
            await self._create(tx, identity, summary)

    async def _create(self, tx, identity, summary):
        key = (identity.job_id, identity.generation)
        encoded_identity, encoded_summary = _json(asdict(identity)), _json(summary)
        # Unlike the old registry transaction, unrelated jobs never share
        # a singleton FOR UPDATE lock here. The primary key owns the race.
        statement = self.db.dialect.insert_ignore(
            self.heads,
            ["job_id", "generation", "identity_json", "identity_hash", "batch_no",
             "chain_hash", "summary_json", "summary_hash", "admission_nonce"],
        )
        if await tx.execute(statement, (*key, encoded_identity, identity.digest, 0,
                                        identity.digest, encoded_summary, _hash(summary), self.admission_nonce)):
            return
        row = await tx.fetchone(f"SELECT * FROM {self.heads} WHERE job_id=? AND generation=?", key)
        self._head(row, identity)
        if (row["batch_no"] != 0 or row["summary_json"] != encoded_summary
                or row["admission_nonce"] != self.admission_nonce):
            raise JobError("execution_requires_reconciliation")

    async def append(self, identity, number, previous, records, summary, *, transaction=None):
        self._identity(identity)
        if type(number) is not int or number < 1 or type(records) is not list or not records:
            raise JobError("invalid_execution_batch")
        if type(summary) is not dict or type(previous) is not str or len(previous) != 64:
            raise JobError("invalid_execution_batch")
        for record in records:
            if (type(record) is not dict or set(record) != {"kind", "key", "value"}
                    or type(record["kind"]) is not str or not record["kind"]
                    or type(record["key"]) is not str or not record["key"]
                    or type(record["value"]) is not dict):
                raise JobError("invalid_execution_record")
        value = {"version": 1, "identity": identity.digest, "batch_no": number,
                 "previous_hash": previous, "records": records, "summary": summary}
        encoded, digest = _json(value), _hash(value)
        if transaction is not None:
            return await self._append_payload(transaction, identity, number, previous, records, summary, encoded, digest)
        async with self.db.transaction() as tx:
            return await self._append_payload(tx, identity, number, previous, records, summary, encoded, digest)

    async def _append_payload(self, tx, identity, number, previous, records, summary, encoded, digest):
        key = (identity.job_id, identity.generation)
        # The owner knows its head; the conditional UPDATE is the fence. Only a
        # miss (retry, conflict, foreign writer) reads the head to classify it.
        if await tx.execute(
                f"UPDATE {self.heads} SET batch_no=?,chain_hash=?,summary_json=?,summary_hash=? "
                "WHERE job_id=? AND generation=? AND batch_no=? AND chain_hash=?",
                (number, digest, _json(summary), _hash(summary), *key, number - 1, previous)) == 1:
            await tx.execute(
                f"INSERT INTO {self.batches} "
                "(job_id,generation,batch_no,previous_hash,payload_json,payload_hash) VALUES (?,?,?,?,?,?)",
                (*key, number, previous, encoded, digest))
            if self.projector is not None:
                await self.projector(tx, identity, records)
            return digest
        suffix = "" if self.db.dialect.name == "sqlite" else " FOR UPDATE"
        head = await tx.fetchone(
            f"SELECT * FROM {self.heads} WHERE job_id=? AND generation=?" + suffix, key)
        self._head(head, identity)
        if head["batch_no"] >= number:
            old = await tx.fetchone(
                f"SELECT previous_hash,payload_json,payload_hash FROM {self.batches} "
                "WHERE job_id=? AND generation=? AND batch_no=?", (*key, number))
            if (old is None or old["previous_hash"] != previous or old["payload_json"] != encoded
                    or old["payload_hash"] != digest):
                raise JobError("execution_checkpoint_conflict")
            return digest
        if head["batch_no"] != number - 1 or head["chain_hash"] != previous:
            raise JobError("execution_checkpoint_conflict")
        if json.loads(head["summary_json"]).get("closed") and not summary.get("closed"):
            raise JobError("execution_already_closed")
        await tx.execute(
            f"INSERT INTO {self.batches} "
            "(job_id,generation,batch_no,previous_hash,payload_json,payload_hash) VALUES (?,?,?,?,?,?)",
            (*key, number, previous, encoded, digest))
        if self.projector is not None:
            await self.projector(tx, identity, records)
        changed = await tx.execute(
            f"UPDATE {self.heads} SET batch_no=?,chain_hash=?,summary_json=?,summary_hash=? "
            "WHERE job_id=? AND generation=? AND batch_no=? AND chain_hash=?",
            (number, digest, _json(summary), _hash(summary), *key, number - 1, previous))
        if changed != 1:
            raise JobError("execution_checkpoint_conflict")
        return digest

    async def inspect(self, identity):
        self._identity(identity)
        row = await self.db.fetchone(
            f"SELECT * FROM {self.heads} WHERE job_id=? AND generation=?",
            (identity.job_id, identity.generation))
        self._head(row, identity)
        return {"batch_no": row["batch_no"], "chain_hash": row["chain_hash"],
                "summary": json.loads(row["summary_json"])}

    async def recover(self, identity, *, repair=False):
        """Validate originals before rebuilding a derived summary.

        Open executions always require reconciliation: the committed prefix
        cannot prove that the unflushed suffix did not reach an external system.
        This port grants neither a lease nor a fresh quota allocation.
        """
        self._identity(identity)
        key = (identity.job_id, identity.generation)
        async with self.db.transaction() as tx:
            suffix = "" if self.db.dialect.name == "sqlite" else " FOR UPDATE"
            head = await tx.fetchone(
                f"SELECT * FROM {self.heads} WHERE job_id=? AND generation=?" + suffix, key)
            self._head(head, identity, check_summary=False)
            rows = await tx.fetchall(
                f"SELECT * FROM {self.batches} WHERE job_id=? AND generation=? ORDER BY batch_no", key)
            previous, number, records, summary = identity.digest, 0, {}, None
            for row in rows:
                try:
                    value = json.loads(row["payload_json"])
                    if (value.get("version") != 1 or value.get("identity") != identity.digest
                            or value.get("batch_no") != number + 1 or row["batch_no"] != number + 1
                            or value.get("previous_hash") != previous or row["previous_hash"] != previous
                            or _hash(value) != row["payload_hash"] or type(value.get("records")) is not list
                            or type(value.get("summary")) is not dict):
                        raise ValueError()
                    for record in value["records"]:
                        if (set(record) != {"kind", "key", "value"}
                                or type(record["kind"]) is not str or type(record["key"]) is not str
                                or type(record["value"]) is not dict):
                            raise ValueError()
                        record_key = (record["kind"], record["key"])
                        if (record["kind"].endswith("_immutable") and record_key in records
                                and records[record_key] != record["value"]):
                            raise ValueError()
                        records[record_key] = record["value"]
                    summary = value["summary"]
                    previous, number = row["payload_hash"], row["batch_no"]
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise JobError("execution_journal_corrupt") from exc
            # A truncated tail is not a damaged projection: don't silently
            # manufacture a shorter history or release its unknown obligations.
            if number != head["batch_no"] or previous != head["chain_hash"]:
                raise JobError("execution_journal_incomplete")
            if summary is None:
                self._head(head, identity)
                summary = json.loads(head["summary_json"])
            mismatch = head["summary_hash"] != _hash(summary) or head["summary_json"] != _json(summary)
            if mismatch and not repair:
                raise JobError("execution_checkpoint_corrupt")
            if mismatch:
                await tx.execute(
                    f"UPDATE {self.heads} SET summary_json=?,summary_hash=? WHERE job_id=? AND generation=?",
                    (_json(summary), _hash(summary), *key))
            if repair and self.projector is not None:
                # Retained completion proofs may reference an older budget
                # version. Rebuild each original version, then the latest view;
                # cumulative allocation versions prevent repeated refunds.
                for row in rows:
                    await self.projector(tx, identity, json.loads(row["payload_json"])["records"])
            return {"batch_no": number, "chain_hash": previous, "summary": summary,
                    "records": records,
                    "needs_reconciliation": not summary.get("closed")
                    or bool(summary.get("needs_reconciliation") or summary.get("inflight") or summary.get("unknown")
                            or summary.get("pending_obligations"))}
