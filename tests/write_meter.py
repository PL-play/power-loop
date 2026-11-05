"""Opt-in, test-only row-image accounting. Never use in a timing run.

AFTER triggers count actual I/U/D rows (including equal-value UPDATEs), not
net growth or driver upsert rowcount. Bytes are UTF-8 text row images; NULL
is empty. DELETE bytes are reported separately from inserted/updated bytes.
The counter updates roll back with the operation. No production DDL changes.
"""
from __future__ import annotations

from contextlib import asynccontextmanager


class WriteMeter:
    def __init__(self, db, prefix):
        self.db, self.prefix = db, prefix
        self.table = prefix + "p0_meter"
        self.triggers = []
        self.max_tx_rows = self.max_tx_bytes = 0

    @classmethod
    async def install(cls, db, prefix):
        self = cls(db, prefix)
        dialect = db.dialect.name
        if dialect == "sqlite":
            rows = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        else:
            schema = "current_schema()" if dialect == "postgres" else "DATABASE()"
            rows = await db.fetchall("SELECT table_name AS name FROM information_schema.tables "
                                    f"WHERE table_schema={schema} AND table_type='BASE TABLE'")
        tables = [r["name"] for r in rows if r["name"].startswith(prefix)]
        await db.execute(f"CREATE TABLE {self.table} (name VARCHAR(255), op VARCHAR(8), "
                         "n BIGINT NOT NULL, b BIGINT NOT NULL, PRIMARY KEY(name,op))")
        for index, table in enumerate(tables):
            if dialect == "sqlite":
                cols = [r["name"] for r in await db.fetchall(f"PRAGMA table_info({table})")]
            else:
                cols = [r["name"] for r in await db.fetchall(
                    "SELECT column_name AS name FROM information_schema.columns "
                    f"WHERE table_schema={schema} AND table_name=? ORDER BY ordinal_position", (table,))]
            for op in ("INSERT", "UPDATE", "DELETE"):
                await db.execute(f"INSERT INTO {self.table} VALUES (?,?,0,0)", (table, op))
                row = "OLD" if op == "DELETE" else "NEW"
                if dialect == "sqlite":
                    size = "+".join(f'COALESCE(LENGTH(CAST({row}."{c}" AS BLOB)),0)' for c in cols)
                elif dialect == "postgres":
                    size = "+".join(f'COALESCE(OCTET_LENGTH({row}."{c}"::text),0)' for c in cols)
                else:
                    size = "+".join(f'COALESCE(OCTET_LENGTH(CAST({row}.`{c}` AS CHAR)),0)' for c in cols)
                name = f"{prefix}meter_{index}_{op.lower()}"
                update = (f"UPDATE {self.table} SET n=n+1,b=b+({size}) "
                          f"WHERE name='{table}' AND op='{op}';")
                if dialect == "postgres":
                    await db.execute(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
                                     f"BEGIN {update} RETURN NULL; END $$")
                    await db.execute(f"CREATE TRIGGER {name} AFTER {op} ON {table} "
                                     f"FOR EACH ROW EXECUTE FUNCTION {name}()")
                else:
                    await db.execute(f"CREATE TRIGGER {name} AFTER {op} ON {table} "
                                     f"FOR EACH ROW BEGIN {update} END")
                self.triggers.append((name, table))
        return self

    async def totals(self, reader):
        row = await reader.fetchone(f"SELECT SUM(n) AS n,SUM(b) AS b FROM {self.table}")
        return int(row["n"] or 0), int(row["b"] or 0)

    @asynccontextmanager
    async def transaction(self, tx):
        before = await self.totals(tx)
        yield
        after = await self.totals(tx)
        self.max_tx_rows = max(self.max_tx_rows, after[0] - before[0])
        self.max_tx_bytes = max(self.max_tx_bytes, after[1] - before[1])

    async def reset(self):
        await self.db.execute(f"UPDATE {self.table} SET n=0,b=0")
        self.max_tx_rows = self.max_tx_bytes = 0

    async def report(self):
        rows = await self.db.fetchall(f"SELECT * FROM {self.table} ORDER BY name,op")
        by_table = {}
        total = {"insert_rows": 0, "update_rows": 0, "delete_rows": 0,
                 "write_bytes": 0, "delete_bytes": 0}
        for row in rows:
            name, op, n, b = row["name"][len(self.prefix):], row["op"].lower(), int(row["n"]), int(row["b"])
            by_table.setdefault(name, {})[op] = {"rows": n, "bytes": b}
            total[op + "_rows"] += n
            total["delete_bytes" if op == "delete" else "write_bytes"] += b
        total["affected_rows"] = sum(total[k] for k in ("insert_rows", "update_rows", "delete_rows"))
        return {**total, "max_tx_rows": self.max_tx_rows, "max_tx_bytes": self.max_tx_bytes,
                "tables": by_table}

    async def close(self):
        for name, table in reversed(self.triggers):
            if self.db.dialect.name == "postgres":
                await self.db.execute(f"DROP TRIGGER {name} ON {table}")
                await self.db.execute(f"DROP FUNCTION {name}()")
            else:
                await self.db.execute(f"DROP TRIGGER {name}")
        await self.db.execute(f"DROP TABLE {self.table}")
