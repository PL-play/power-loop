"""Explicit offline projection conversion; no provider or external action is called.

Run after the store schema upgrade, with all session writers stopped. The default
is a read-only plan. Original messages and fees remain unchanged; an old in-place
summary seeds a compact covering no newly attributed send. Actual sends are
projected individually, leaving later compression to the configured live fold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.factory import _make_database
from power_loop.runtime.store.projection_fold import invalidate
from power_loop.runtime.store.schema import validate_table_prefix
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.task_schema import TASK_SCHEMA_VERSION, verify_task_schema


async def convert(store, sid, *, apply=False, writers_stopped=False, projector=None):
    if apply and not writers_stopped:
        raise ValueError('maintenance_required: stop writers before --apply')
    projector = projector or ProjectedRepresentation()
    session = await store.get_session(sid)
    if session is None:
        raise ValueError('session_not_found')
    originals = await store.load_active_messages(sid)
    projections = await store.load_project_messages(sid)
    current = {(r.send_index, r.kind): r for r in projections}
    compacts = [r for r in projections if r.kind == 'compact']
    cutoff = compacts[-1].compact_to_send if compacts else 0
    version = projector.version
    by_send = {}
    for row in originals:
        if row.name != 'compact_note' and row.send_index is not None and row.send_index > (cutoff or 0):
            by_send.setdefault(row.send_index, []).append(row)
    planned = []
    for send_index, rows in sorted(by_send.items()):
        projected = projector.project_send(rows, send_index=send_index, tool_registry=None)
        for row in projected.rows:
            saved = current.get((send_index, row.kind))
            if saved is None or saved.projector_version != version:
                planned.append((send_index, row, projected.source_seq_lo, projected.source_seq_hi))
    notes = [r for r in originals if r.name == 'compact_note']
    seed = notes[-1].content if notes and not compacts else None
    metadata = {**session.metadata, 'history_mode': 'projection', 'projector_version': version}
    metadata.pop('projection_migrated', None)
    report = {'session_id': sid, 'project_rows': len(planned), 'compact_seeded': seed is not None,
              'original_messages': len(originals), 'applied': apply}
    if not apply or (not planned and seed is None and metadata == session.metadata):
        return report
    now = time.time_ns() // 1_000_000
    async with store._db.transaction() as tx:
        await store._db.dialect.lock_state(tx, store.t.session_state, sid)
        await invalidate(store, tx, sid)
        for send_index, row, lo, hi in planned:
            await store._upsert_project_message_tx(tx, sid, send_index=send_index, kind=row.kind,
                content=row.content, rendered_text=row.rendered_text, source_seq_lo=lo, source_seq_hi=hi,
                projector_version=version, now=now)
        if seed is not None:
            await store._upsert_project_message_tx(tx, sid, send_index=0, kind='compact',
                content={'summary': seed}, compact_from_send=0, compact_to_send=0,
                projector_version=version, now=now)
        await tx.execute(f'UPDATE {store.t.sessions} SET metadata_json=?,updated_at=? WHERE session_id=?',
                         (json.dumps(metadata, ensure_ascii=False), now, sid))
    # Rebuild only derived fold counters offline, so the next live send uses its
    # normal incremental path. A crash here leaves complete projections and an
    # invalid counter; the existing cold repair can rebuild it from originals.
    await store.prepare_send_projection_fold(sid, send_index=max(by_send, default=0), rows=[],
        source_seq_lo=None, source_seq_hi=None, projector_version=version,
        projector=projector, keep_last_sends=1, threshold=0)
    return report


async def run(dsn, prefix, sid, *, apply=False, writers_stopped=False):
    validate_table_prefix(prefix)
    db = await _make_database(dsn)
    try:
        await verify_task_schema(db, prefix)
        store = SessionStore(db, table_prefix=prefix)
        await store._initialize_identity(TASK_SCHEMA_VERSION)
        return await convert(store, sid, apply=apply, writers_stopped=writers_stopped)
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--prefix', default='pl_')
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--writers-stopped', action='store_true')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.dsn, args.prefix, args.session_id,
        apply=args.apply, writers_stopped=args.writers_stopped)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
