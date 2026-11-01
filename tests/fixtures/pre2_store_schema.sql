-- Historical SQLite schema version 2; fixture only.

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    created_at          INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL,
    system_prompt       TEXT,
    model               TEXT,
    config_json         TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    kind                TEXT NOT NULL DEFAULT 'root',
    parent_session_id   TEXT,
    spawn_tool_call_id  TEXT,
    spawn_depth         INTEGER NOT NULL DEFAULT 0,
    lifecycle           TEXT NOT NULL DEFAULT 'ephemeral',
    metadata_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_parent
    ON sessions(parent_session_id);

CREATE TABLE IF NOT EXISTS messages (
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    name            TEXT,
    content         TEXT,
    tool_calls_json TEXT,
    tool_call_id    TEXT,
    round_index     INTEGER,
    state           TEXT NOT NULL DEFAULT 'active',
    meta_json       TEXT,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session_state
    ON messages(session_id, state, seq);

CREATE TABLE IF NOT EXISTS compactions (
    session_id      TEXT NOT NULL,
    compact_seq     INTEGER NOT NULL,
    note_seq        INTEGER NOT NULL,
    from_seq        INTEGER NOT NULL,
    to_seq          INTEGER NOT NULL,
    before_tokens   INTEGER,
    after_tokens    INTEGER,
    round_index     INTEGER,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (session_id, compact_seq)
);

CREATE TABLE IF NOT EXISTS usage_rounds (
    session_id        TEXT NOT NULL,
    send_index        INTEGER NOT NULL DEFAULT 0,
    round_index       INTEGER NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    model             TEXT,
    created_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, send_index, round_index)
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id        TEXT PRIMARY KEY,
    next_seq          INTEGER NOT NULL DEFAULT 1,
    round_index       INTEGER NOT NULL DEFAULT 0,
    last_compact_seq  INTEGER NOT NULL DEFAULT 0,
    pending_json      TEXT
);

CREATE TABLE IF NOT EXISTS session_runtime_state (
    session_id        TEXT NOT NULL,
    key               TEXT NOT NULL,
    value_json        TEXT,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, key)
);

-- Keyed JSON owned by an arbitrary scope key (NOT a session): the blackboard
-- and other cross-session shared state. No session existence check / no cascade
-- with any session — the owner is opaque (e.g. a workflow run id, a board id).
CREATE TABLE IF NOT EXISTS shared_state (
    owner             TEXT NOT NULL,
    key               TEXT NOT NULL,
    value_json        TEXT,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (owner, key)
);

CREATE TABLE IF NOT EXISTS background_tasks (
    session_id        TEXT NOT NULL,
    task_id           TEXT NOT NULL,
    command           TEXT NOT NULL,
    status            TEXT NOT NULL,
    return_code       INTEGER,
    output_tail       TEXT,
    output_path       TEXT,
    last_seen_at      INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_background_tasks_session_status
    ON background_tasks(session_id, status, updated_at);

CREATE TABLE IF NOT EXISTS session_stats (
    session_id        TEXT PRIMARY KEY,
    sends             INTEGER NOT NULL DEFAULT 0,
    rounds            INTEGER NOT NULL DEFAULT 0,
    llm_calls         INTEGER NOT NULL DEFAULT 0,
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    first_send_at     INTEGER,
    last_send_at      INTEGER,
    updated_at        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS timers (
    session_id        TEXT NOT NULL,
    timer_id          INTEGER NOT NULL,
    due_at            INTEGER NOT NULL,
    note              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'armed',
    interval_s        INTEGER,
    fire_count        INTEGER NOT NULL DEFAULT 0,
    last_fired_at     INTEGER,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, timer_id)
);
CREATE INDEX IF NOT EXISTS idx_timers_due ON timers (status, due_at);

CREATE TABLE IF NOT EXISTS notes (
    session_id        TEXT NOT NULL,
    note_id           INTEGER NOT NULL,
    content           TEXT NOT NULL,
    pinned            INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY (session_id, note_id)
);

PRAGMA user_version=2;
