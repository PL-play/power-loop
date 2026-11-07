"""Per-response usage originals stored with their response message.

Rounds without a response message use usage_observations. A lost append ACK can leave
both observations; the committed response original wins for the same round.
This reporting query is cold and does not run during an ordinary send.
"""
from __future__ import annotations

META_KEY = '_round_usage'
COLUMNS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_tokens', 'model')
DETAIL_COLUMNS = ('session_id', 'send_index', 'round_index', *COLUMNS, 'created_at')


def field(backend, column):
    path = '$.'+META_KEY+'.'+column
    if backend == 'postgres':
        value = f"(CAST(m.meta_json AS jsonb)->'{META_KEY}'->>'{column}')"
        return value if column == 'model' else f'CAST({value} AS BIGINT)'
    if backend == 'mysql':
        raw = f"JSON_EXTRACT(m.meta_json,'{path}')"
        value = f"CASE WHEN JSON_TYPE({raw})='NULL' THEN NULL ELSE JSON_UNQUOTE({raw}) END"
        return value if column == 'model' else f'CAST({value} AS SIGNED)'
    return f"json_extract(m.meta_json,'{path}')"


def view_query(backend, prefix):
    present = (f"CAST(m.meta_json AS jsonb)->'{META_KEY}'" if backend == 'postgres' else
        f"JSON_EXTRACT(m.meta_json,'$.{META_KEY}')")
    values = ','.join(field(backend, column)+' AS '+column for column in COLUMNS)
    columns = ','.join(DETAIL_COLUMNS)
    return (f'WITH responses AS (SELECT m.session_id,COALESCE(m.send_index,0) AS send_index,'
        f'm.round_index,{values},m.created_at,ROW_NUMBER() OVER ('
        'PARTITION BY m.session_id,COALESCE(m.send_index,0),m.round_index ORDER BY m.seq DESC) AS n '
        f'FROM {prefix}messages m WHERE {present} IS NOT NULL) '
        f'SELECT {columns} FROM responses WHERE n=1 UNION ALL '
        f'SELECT '+','.join('u.'+column for column in DETAIL_COLUMNS)+
        f' FROM {prefix}usage_observations u WHERE NOT EXISTS ('
        'SELECT 1 FROM responses r WHERE r.session_id=u.session_id '
        'AND r.send_index=u.send_index AND r.round_index=u.round_index)')


def view_ddl(backend, prefix):
    create = 'CREATE VIEW IF NOT EXISTS' if backend == 'sqlite' else 'CREATE OR REPLACE VIEW'
    return [f'{create} {prefix}usage_rounds AS '+view_query(backend, prefix)]




def remove_sql(backend):
    if backend == 'postgres':
        return f"CAST(CAST(meta_json AS jsonb)-'{META_KEY}' AS text)"
    return f"JSON_REMOVE(meta_json,'$.{META_KEY}')"
