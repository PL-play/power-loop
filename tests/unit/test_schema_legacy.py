"""G6 regression: opening a pre-2.0 (unprefixed) SQLite DB with the 2.0 ``pl_`` schema must
not silently front an empty store.

1.x was SQLite-only and used no table prefix; 2.0 defaults to ``pl_``. On upgrade the old
sessions sit intact under the unprefixed tables but are invisible to the prefixed schema.
The prefix change is an accepted break, so we don't auto-migrate — but we must make it
OBSERVABLE: warn under AUTO_CREATE, raise a specific error under VERIFY. A genuinely fresh DB
must NOT false-warn.
"""

from __future__ import annotations

import logging

import pytest

from power_loop.runtime.session_store import SessionStore as LegacyStore
from power_loop.runtime.store.schema import SchemaPolicy, StoreSchemaError
from power_loop.runtime.store.store import SessionStore

pytestmark = pytest.mark.unit


def _make_legacy_db(path: str) -> str:
    """Create a pre-2.0 DB (unprefixed tables) with one session; return its id."""
    legacy = LegacyStore.open(path)
    try:
        return legacy.create_session(system_prompt="old")
    finally:
        legacy.close()


async def test_auto_create_warns_on_legacy_unprefixed_db(tmp_path, caplog) -> None:
    db = str(tmp_path / "legacy.db")
    old_sid = _make_legacy_db(db)

    with caplog.at_level(logging.WARNING, logger="power_loop.runtime.store.schema"):
        store = await SessionStore.open(db)  # default pl_ prefix, AUTO_CREATE
    try:
        assert "pre-2.0 power-loop database" in caplog.text
        # the 2.0 (pl_) store does not see the legacy session — but it is NOT deleted
        assert await store.get_session(old_sid) is None
    finally:
        await store.close()


async def test_verify_raises_specific_error_on_legacy_db(tmp_path) -> None:
    db = str(tmp_path / "legacy.db")
    _make_legacy_db(db)
    with pytest.raises(StoreSchemaError, match="pre-2.0 power-loop database"):
        await SessionStore.open(db, schema=SchemaPolicy.VERIFY)


async def test_fresh_db_does_not_false_warn(tmp_path, caplog) -> None:
    db = str(tmp_path / "fresh.db")
    with caplog.at_level(logging.WARNING, logger="power_loop.runtime.store.schema"):
        store = await SessionStore.open(db)
    try:
        assert "pre-2.0" not in caplog.text
    finally:
        await store.close()


async def test_legacy_data_readable_with_empty_prefix(tmp_path, caplog) -> None:
    """The escape hatch the warning points to: open with table_prefix='' (the 1.x layout) to
    read the old sessions — and that path must not warn."""
    db = str(tmp_path / "legacy.db")
    old_sid = _make_legacy_db(db)
    with caplog.at_level(logging.WARNING, logger="power_loop.runtime.store.schema"):
        store = await SessionStore.open(db, table_prefix="")
    try:
        assert "pre-2.0" not in caplog.text
        sess = await store.get_session(old_sid)
        assert sess is not None and sess.system_prompt == "old"
    finally:
        await store.close()
