"""11 · Cross-process resume: ``db_path="./real_file.db"`` persists the session

What you learn
--------------
- ``StatefulAgentLoop(db_path=…)`` writes to a **real file** (not ``":memory:"``);
  the entire session lives in that SQLite file: messages / pending / usage /
  compactions.
- After the process exits, **as long as you have** ``session_id`` **+ the same
  db path**, a new process can instantiate ``StatefulAgentLoop`` and call
  ``send(..., session_id=sid)`` — the LLM sees the full history.
- No special "resume" API needed: ``SessionStore.open(path)`` seamlessly
  reconnects to an existing file; WAL + busy_timeout make single-file serial
  reuse safe (do NOT let multiple processes write concurrently).

How to run
----------
This example forks a child process to demonstrate "real process swap":

    python examples/11_cross_process_resume.py

It runs in two phases:

1. **phase1**: parent creates a session, tells the LLM a fact ("My name is
   Alan, my favorite number is 37"), closes the loop, writes the sid to
   stdout, then exits.
2. **phase2**: parent uses ``subprocess.run`` to re-invoke *this file* with
   ``phase2 <db_path> <sid>``. The child process has **zero knowledge of
   phase1's memory** — it only opens the same db file →
   ``send("What is my name? What is my favorite number?")`` → LLM should
   answer "Alan / 37".

If the db path is wrong in the child (e.g. ``other.db``), it raises
``SessionNotFoundError`` — proving the persistence anchor is the file, not
memory.

Run
---
    python examples/11_cross_process_resume.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop

SYSTEM = (
    "You are a concise assistant. When the user tells you facts about themselves, "
    "remember them and use them verbatim in later answers."
)


async def phase1(db_path: str) -> str:
    """Parent process: create session, store a fact, close, return sid."""
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=200),
        db_path=db_path,
        config=AgentLoopConfig(system_prompt=SYSTEM, max_rounds=1, compactor=None),
    )
    try:
        sid = loop.new_session()
        r = await loop.send(
            "Remember: my name is Alan and my favorite number is 37. Confirm briefly.",
            session_id=sid,
        )
        print(f"[phase1] sid={sid}")
        print(f"[phase1] reply={r.final_text}")
        return sid
    finally:
        loop.close()


async def phase2(db_path: str, sid: str) -> str:
    """Child process: only has db_path + sid, resumes directly."""
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=200),
        db_path=db_path,
        config=AgentLoopConfig(system_prompt=SYSTEM, max_rounds=1, compactor=None),
    )
    try:
        r = await loop.send(
            "What is my name? What is my favorite number? One sentence only.",
            session_id=sid,
        )
        print(f"[phase2] reply={r.final_text}")
        return r.final_text
    finally:
        loop.close()


async def _run_parent() -> None:
    # Use a real file in a temp directory to avoid polluting the repo root.
    with tempfile.TemporaryDirectory(prefix="power_loop_p11_") as tmp:
        db_path = str(Path(tmp) / "real_file.db")
        sid = await phase1(db_path)
        assert Path(db_path).exists(), "phase1 should have left a file on disk"

        print(f"\n--- parent exits phase1, db left at {db_path} ---")
        print("--- spawning child process for phase2 ---\n")

        cp = subprocess.run(
            [sys.executable, __file__, "phase2", db_path, sid],
            check=True,
            text=True,
        )
        if cp.returncode != 0:
            raise SystemExit(f"phase2 subprocess failed: rc={cp.returncode}")


def main() -> None:
    # Child process entry: argv = [..., "phase2", db_path, sid]
    if len(sys.argv) >= 4 and sys.argv[1] == "phase2":
        _, _, db_path, sid = sys.argv[:4]
        asyncio.run(phase2(db_path, sid))
        return
    # Parent process entry
    asyncio.run(_run_parent())


if __name__ == "__main__":
    # Child process runs from this same file, so ensure relative imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
