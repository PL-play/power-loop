"""Native backup safety checks; protocol-21 upgrade rehearsals live in tests/unit."""
import pytest

from scripts.acceptance._backup_restore import database_name, restore_copy

pytestmark = pytest.mark.asyncio




@pytest.mark.parametrize('dsn', ['postgresql://localhost/production', 'sqlite:///db_test',
    'postgresql://localhost/a_test/extra', 'postgresql://localhost/a-test', 'postgresql://localhost/'])
async def test_backup_refuses_non_disposable_databases(dsn):
    with pytest.raises(ValueError):
        database_name(dsn)


async def test_backup_rejects_source_overwrite_and_unowned_prefix(tmp_path):
    for prefix, destination in [('bridge_12345678_', 'original.db'), ('pl_', 'new.db')]:
        with pytest.raises(ValueError):
            await restore_copy('original.db', destination, prefix=prefix,
                archive=tmp_path / 'archive', backend='sqlite')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('collision', ['archive', 'target'])
async def test_sqlite_restore_never_overwrites_existing_files(tmp_path, collision):
    import sqlite3
    from contextlib import closing
    source, target, archive = (tmp_path / name for name in ('source.db', 'target.db', 'archive.db'))
    with closing(sqlite3.connect(source)) as db:
        db.execute('CREATE TABLE original (id INTEGER PRIMARY KEY, value TEXT)')
        db.execute("INSERT INTO original VALUES (1,'retain')")
        db.commit()
    before = source.read_bytes()
    occupied = archive if collision == 'archive' else target
    occupied.write_bytes(b'KEEP')
    with pytest.raises((ValueError, FileExistsError)):
        await restore_copy(str(source), str(target), prefix='bridge_12345678_',
            archive=archive, backend='sqlite')
    assert occupied.read_bytes() == b'KEEP' and source.read_bytes() == before
