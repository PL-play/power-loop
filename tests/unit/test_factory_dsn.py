"""G4/G5 regression: DSN scheme + sqlite-path parsing in the store factory.

G4 — an absolute ``sqlite://`` URL must stay absolute (it used to be mangled to a relative
path). G5 — a non-SQLite or driver-qualified scheme must route to the right backend (or
raise), never silently create a SQLite file named after the DSN.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from power_loop.runtime.store.factory import _split_scheme, _sqlite_path, open_store

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "dsn,expected",
    [
        ("sqlite:///rel.db", "rel.db"),           # 3-slash → relative (SQLAlchemy convention)
        ("sqlite:////abs/x.db", "/abs/x.db"),     # 4-slash → absolute (was "//abs/x.db" / relative)
        ("sqlite://", ":memory:"),
        ("sqlite://:memory:", ":memory:"),
        ("/abs/bare.db", "/abs/bare.db"),          # bare path verbatim
        ("./rel/bare.db", "./rel/bare.db"),
        (":memory:", ":memory:"),
    ],
)
def test_sqlite_path(dsn, expected) -> None:
    assert _sqlite_path(dsn) == expected


@pytest.mark.parametrize(
    "dsn,scheme,normalized",
    [
        ("postgresql+psycopg://u:p@h/db", "postgresql", "postgresql://u:p@h/db"),
        ("postgres://u:p@h/db", "postgres", "postgres://u:p@h/db"),
        ("mysql+aiomysql://u:p@h:3307/db", "mysql", "mysql://u:p@h:3307/db"),
        ("sqlite:///x.db", "sqlite", "sqlite:///x.db"),
        ("./bare.db", None, "./bare.db"),
        (":memory:", None, ":memory:"),
    ],
)
def test_split_scheme(dsn, scheme, normalized) -> None:
    assert _split_scheme(dsn) == (scheme, normalized)


async def test_absolute_sqlite_url_opens_at_absolute_path(tmp_path) -> None:
    target = tmp_path / "owned.db"
    store = await open_store(f"sqlite:///{target}")  # tmp_path is absolute → 4-slash form
    try:
        await store.create_session(system_prompt="x")
    finally:
        await store.close()
    assert target.exists(), "absolute sqlite:// path was mangled (G4)"


async def test_unknown_scheme_raises_instead_of_silent_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unsupported store DSN scheme"):
        await open_store("redis://localhost:6379/0")
    # and it did NOT create a bogus SQLite file named after the DSN
    assert not any(p.name.startswith("redis:") for p in tmp_path.iterdir())


async def test_driver_qualified_pg_scheme_does_not_become_sqlite(tmp_path, monkeypatch) -> None:
    """A SQLAlchemy-style postgresql+psycopg:// DSN must NOT silently open a SQLite file
    (G5). It routes to the PG backend; with no server it raises a connection error — never
    a stray file in cwd."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception) as ei:  # noqa: B017
        await open_store("postgresql+psycopg://u:p@127.0.0.1:1/db")
    assert not isinstance(ei.value, ValueError) or "unsupported" not in str(ei.value)
    assert not any("postgresql" in p.name for p in tmp_path.iterdir())


PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN", "postgresql://deeptalk:deeptalk@localhost:5433/power_loop_test"
)


def _reachable(dsn: str) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _reachable(PG_DSN), reason=f"Postgres not reachable at {PG_DSN}")
async def test_driver_qualified_pg_dsn_connects(tmp_path) -> None:
    pytest.importorskip("asyncpg")
    driver_dsn = PG_DSN.replace("postgresql://", "postgresql+psycopg://", 1)
    store = await open_store(driver_dsn, table_prefix="pldsn_")
    try:
        sid = await store.create_session(system_prompt="x")
        assert sid
    finally:
        await store.close()
