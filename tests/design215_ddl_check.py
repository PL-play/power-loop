"""Execute the P0-R1 contract DDL on three isolated database engines."""
from __future__ import annotations

import asyncio
import os
import secrets
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiomysql
import asyncpg

DDL = Path(__file__).resolve().parents[2] / "platform/docs/design/assets/215/P0-R1/ddl"
EXPECTED = {
    "schema_migrations", "sessions", "session_state", "follow_up_queue",
    "messages", "session_runtime_state", "project_messages",
}
FORBIDDEN = {"jobs", "job_events", "task_store", "session_runs", "executions", "run_results"}
BAD_OWNER = ("INSERT INTO session_state(session_id,execution_state,active_send_index) "
             "VALUES ('bad','running',1)")


def sqlite_check():
    db = sqlite3.connect(":memory:")
    db.executescript((DDL / "sqlite.sql").read_text())
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED <= tables and not FORBIDDEN & tables
    try:
        db.execute(BAD_OWNER)
        raise AssertionError("SQLite accepted running without owner")
    except sqlite3.IntegrityError:
        pass
    print("sqlite: 7 contract tables; running without owner rejected")


async def postgres_check():
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN"]
    name = "design215_r1_" + secrets.token_hex(4)
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f'CREATE SCHEMA "{name}"')
        await conn.execute(f'SET search_path TO "{name}"')
        await conn.execute((DDL / "postgres.sql").read_text())
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname=$1", name)
        tables = {row["tablename"] for row in rows}
        assert tables == EXPECTED and not FORBIDDEN & tables
        try:
            await conn.execute(BAD_OWNER)
            raise AssertionError("PostgreSQL accepted running without owner")
        except asyncpg.CheckViolationError:
            pass
        print("postgres: 7 contract tables; running without owner rejected")
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        await conn.close()


async def mysql_check():
    parsed = urlparse(os.environ["POWER_LOOP_TASK_MYSQL_DSN"])
    name = "design215_r1_" + secrets.token_hex(4)
    conn = await aiomysql.connect(
        host=parsed.hostname, port=parsed.port, user=unquote(parsed.username),
        password=unquote(parsed.password), db=parsed.path.lstrip("/"),
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE `{name}`")
            await cur.execute(f"USE `{name}`")
            await cur.execute((DDL / "mysql.sql").read_text())
            await cur.execute("SHOW TABLES")
            tables = {row[0] for row in await cur.fetchall()}
            assert tables == EXPECTED and not FORBIDDEN & tables
            try:
                await cur.execute(BAD_OWNER)
                raise AssertionError("MySQL accepted running without owner")
            except aiomysql.OperationalError:
                pass
        print("mysql: 7 contract tables; running without owner rejected")
    finally:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{name}`")
        conn.close()


async def main():
    sqlite_check()
    await postgres_check()
    await mysql_check()


if __name__ == "__main__":
    asyncio.run(main())
