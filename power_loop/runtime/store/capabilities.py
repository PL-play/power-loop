"""Optional backend capabilities — real on some backends, absent on others.

SQLite exposes in-process WAL checkpoint / VACUUM / online backup; on a server backend
those are operator concerns (autovacuum, ``pg_dump``/``mysqldump``) with no in-process
analogue, so they are NOT part of the required ``Database`` protocol. The store probes
for them and no-ops when unsupported, so callers can call ``store.checkpoint()`` etc.
uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Maintenance(Protocol):
    """In-process storage maintenance (SQLite). A server backend simply does not
    implement this, and the store's maintenance methods become no-ops."""

    async def checkpoint(self, *, mode: str = "TRUNCATE") -> None: ...

    async def vacuum(self, *, incremental: bool = True) -> None: ...

    async def backup(self, dest_path: str) -> None: ...


__all__ = ["Maintenance"]
