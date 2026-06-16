"""39 · 可插拔后端 + 无状态恢复 / Pluggable backends + stateless resume

power-loop 的 loop 是一个**无状态句柄**：会话状态全部在 store 里。所以 loop 可以随意
创建、随意恢复 —— 只靠一个 DSN + session_id 就能在另一个进程冷启动续上。store 本身是
**可插拔**的：默认 SQLite（零依赖），换个 DSN 即可用 PostgreSQL / MySQL。

The loop is a STATELESS handle — all session state lives in the store — so it is cheap to
create and trivially resumed from just a DSN + a session id. The store is PLUGGABLE: SQLite
by default (zero infra), or PostgreSQL/MySQL by swapping the DSN.

What you learn / 你将学到
--------------------------
- ``dsn=`` selects the backend: a path / ``sqlite://`` → SQLite, ``postgresql://`` →
  PostgreSQL (``power-loop[postgres]``), ``mysql://`` → MySQL (``power-loop[mysql]``).
- A brand-new loop (cold start, empty cache) RESUMES a session by id — no state to carry.
- ``SchemaPolicy``: ``AUTO_CREATE`` (default) provisions tables; ``VERIFY`` only checks and,
  if the schema is missing, raises ``StoreSchemaError`` carrying the exact DDL to run.
- ``loop.cache_stats`` — the per-session active-window cache (a pure accelerator).

Run / 运行
----------
    python examples/39_pluggable_backends_and_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _helpers import make_llm

from power_loop import SchemaPolicy, StatefulAgentLoop, StoreSchemaError


async def main() -> str:
    tmp = Path(tempfile.mkdtemp()) / "app.db"
    dsn = f"sqlite://{tmp}"  # ← swap for "postgresql://u:p@host/app" or "mysql://u:p@host/app"

    # ── Loop A: create a session and establish a fact (AUTO_CREATE provisions the schema) ──
    loop_a = StatefulAgentLoop(llm=make_llm(), dsn=dsn)
    sid = await loop_a.new_session()
    await loop_a.send("Remember: the launch code is BLUEBIRD-7. Acknowledge briefly.", session_id=sid)
    await loop_a.aclose()  # process "exits"

    # ── Loop B: a BRAND-NEW loop in a fresh state resumes the SAME session by id ──
    # Nothing was carried over but the DSN and the session id. schema=VERIFY because the
    # tables already exist — no DDL rights needed here.
    loop_b = StatefulAgentLoop(llm=make_llm(), dsn=dsn, schema=SchemaPolicy.VERIFY)
    await loop_b.prewarm(sid)  # optional: warm the window so the first send skips a reload
    answer = (await loop_b.send("What was the launch code?", session_id=sid)).final_text
    print("resumed answer:", answer)
    print("cache_stats:", loop_b.cache_stats)  # the active-window cache (accelerator only)
    await loop_b.aclose()

    # ── VERIFY against a fresh DB prints the exact provisioning DDL instead of auto-creating ──
    fresh = Path(tempfile.mkdtemp()) / "empty.db"
    try:
        StatefulAgentLoop(llm=make_llm(), dsn=f"sqlite://{fresh}", schema=SchemaPolicy.VERIFY)
        # (the store opens lazily on first async use)
        await StatefulAgentLoop(
            llm=make_llm(), dsn=f"sqlite://{fresh}", schema=SchemaPolicy.VERIFY
        ).new_session()
    except StoreSchemaError as e:
        print(f"\nVERIFY on a fresh DB → StoreSchemaError ({len(e.ddl)} DDL statements to run):")
        print(e.ddl[0])  # e.g. CREATE TABLE IF NOT EXISTS pl_schema_migrations (...)

    return answer


if __name__ == "__main__":
    asyncio.run(main())
