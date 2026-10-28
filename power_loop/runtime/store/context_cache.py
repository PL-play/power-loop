"""Tables of the retired background tool-summary feature.

Background tool summaries are retired: nothing enqueues, claims, or calls a model for
them anymore. The tables stay so existing databases keep their schema and history rows.
"""
from __future__ import annotations


def context_cache_ddl(name: str, prefix: str) -> list[str]:
    text = 'LONGTEXT' if name == 'mysql' else 'TEXT'
    statements = [
        f'''CREATE TABLE IF NOT EXISTS {prefix}context_cache (
            cache_key VARCHAR(64) PRIMARY KEY, executor_key VARCHAR(64) NOT NULL,
            owner_session_id VARCHAR(200) NOT NULL, status VARCHAR(20) NOT NULL,
            payload_json {text} NOT NULL, summary {text}, result_hash VARCHAR(64),
            lease_token VARCHAR(64), lease_until BIGINT NOT NULL DEFAULT 0,
            created_at BIGINT NOT NULL, accessed_at BIGINT NOT NULL, expires_at BIGINT NOT NULL,
            size_bytes BIGINT NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
            retry_at BIGINT NOT NULL DEFAULT 0)''',
        f'''CREATE TABLE IF NOT EXISTS {prefix}context_attempts (
            attempt_id VARCHAR(64) PRIMARY KEY, cache_key VARCHAR(64) NOT NULL,
            owner_session_id VARCHAR(200) NOT NULL, outcome VARCHAR(20) NOT NULL,
            record_json {text} NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
            accounted INTEGER NOT NULL DEFAULT 0, created_at BIGINT NOT NULL)''',
    ]
    indexes = (
        ('context_cache', 'context_due', 'executor_key,status,retry_at,created_at'),
        ('context_cache', 'context_running', 'status,lease_until'),
        ('context_cache', 'context_lru', 'status,accessed_at'),
        ('context_cache', 'idx_context_capacity', 'size_bytes'),
        ('context_attempts', 'context_outbox', 'accounted,delivered,created_at'),
        ('context_attempts', 'context_attempt_key', 'cache_key,outcome'),
        ('context_attempts', 'context_attempt_owner', 'owner_session_id,created_at'),
    )
    for table, index, columns in indexes:
        if name == 'mysql':
            i = 0 if table == 'context_cache' else 1
            statements[i] = statements[i].rstrip()[:-1] + f', INDEX {prefix}{index} ({columns}))'
        else:
            statements.append(f'CREATE INDEX IF NOT EXISTS {prefix}{index} ON {prefix}{table} ({columns})')
    return statements
