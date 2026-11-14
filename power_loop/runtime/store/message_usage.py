"""Per-response usage originals stored with their response message.

Rounds without a response message use session_runtime_state. A lost append ACK can leave
both observations; the committed response original wins for the same round.
This reporting query is cold and does not run during an ordinary send.
"""
from __future__ import annotations

META_KEY = '_round_usage'
COLUMNS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_tokens', 'model')
DETAIL_COLUMNS = ('session_id', 'send_index', 'round_index', *COLUMNS, 'created_at')


def remove_sql(backend):
    if backend == 'postgres':
        return f"CAST(CAST(meta_json AS jsonb)-'{META_KEY}' AS text)"
    return f"JSON_REMOVE(meta_json,'$.{META_KEY}')"
