"""Prove cumulative accounting cannot be replaced by before/after net rows."""
import pytest

from tests.unit.test_job_registry_192 import registry_store  # noqa: F401
from tests.write_meter import WriteMeter

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_bulk_same_value_update_delete_and_rollback(registry_store):  # noqa: F811
    store, _ = registry_store
    db, prefix = store._db, store.table_prefix
    table = prefix + "meter_probe"
    await db.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, value VARCHAR(50))")
    meter = await WriteMeter.install(db, prefix)
    try:
        async with db.transaction() as tx, meter.transaction(tx):
            await tx.execute(f"INSERT INTO {table} VALUES (1,'中'),(2,'ab')")
            await tx.execute(f"UPDATE {table} SET value=value")
            await tx.execute(f"DELETE FROM {table}")
        report = await meter.report()
        assert report["insert_rows"] == report["update_rows"] == report["delete_rows"] == 2
        assert report["write_bytes"] == 14 and report["delete_bytes"] == 7
        assert report["max_tx_rows"] == 6 and report["max_tx_bytes"] == 21
        assert (await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}"))["n"] == 0
        with pytest.raises(RuntimeError, match="rollback"):
            async with db.transaction() as tx, meter.transaction(tx):
                await tx.execute(f"INSERT INTO {table} VALUES (3,'discard')")
                raise RuntimeError("rollback")
        assert await meter.report() == report
        await meter.reset()
        assert (await meter.report())["affected_rows"] == 0
        async with db.transaction() as tx, meter.transaction(tx):
            insert = db.dialect.insert_ignore(table, ('id', 'value'))
            await tx.execute(insert, (1, '中'))
            await tx.execute(insert, (1, 'ignored'))
            await tx.execute(db.dialect.upsert(table, ('id',), ('value',)), (1, 'xyz'))
        actual = await meter.report()
        assert actual['insert_rows'] == actual['update_rows'] == 1
        assert actual['delete_rows'] == 0
        assert actual['write_bytes'] == 8
        assert actual['max_tx_rows'] == 2
    finally:
        await meter.close()
        await db.execute(f"DROP TABLE {table}")
