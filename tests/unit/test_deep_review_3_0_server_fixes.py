"""The explicit protocol-21 upgrade probes only the current PostgreSQL schema."""

from __future__ import annotations

import os
import socket
import urllib.parse
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.unit

PG_DSN = os.environ.get(
    "POWER_LOOP_TEST_PG_DSN", f"postgresql://deeptalk:{os.environ.get('POSTGRES_PASSWORD', 'deeptalk')}@localhost:5433/power_loop_test"
)


def _reachable(dsn: str, default_port: int) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=2):
            return True
    except OSError:
        return False


_pg = pytest.mark.skipif(not _reachable(PG_DSN, 5432), reason=f"Postgres not reachable at {PG_DSN}")


def _dsn(schema: str) -> str:
    return f"{PG_DSN}?options=" + urllib.parse.quote(f"-csearch_path={schema}")
