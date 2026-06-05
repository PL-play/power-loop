"""11 · 跨进程恢复：``db_path="./real_file.db"`` 真的把会话续上

What you learn
--------------
- ``StatefulAgentLoop(db_path=…)`` 落到一个**真实文件**（不是 ``":memory:"``），
  整段会话都活在那个 SQLite 文件里：messages / pending / usage / compactions。
- 进程退出后，**只要拿着** ``session_id`` **+ 同一个 db 路径**，新进程实例化
  ``StatefulAgentLoop`` 并 ``send(..., session_id=sid)``，LLM 看到的就是完整历史。
- 不需要任何「恢复」API：``SessionStore.open(path)`` 命中已存在文件即原样接上；
  WAL + busy_timeout 让单文件被串行复用是安全的（不要并发多进程同写）。

如何运行
--------
本例自己 fork 一个子进程演示「真的换进程」：

    python examples/11_persistence.py

它会做两段事：

1. **phase1**：父进程创建 session，问一个能立刻让 LLM 记住的事实（"我叫
   阿岚，最喜欢的数字是 37"），关闭 loop、写 sid 到 stdout，进程返回。
2. **phase2**：父进程用 ``subprocess.run`` 重新拉起 *本文件*，传入
   ``phase2 <db_path> <sid>``。子进程**完全不知道 phase1 的内存**，只打开同一个
   db 文件 → ``send("我叫什么？最喜欢的数字是多少？")`` → LLM 应当答出
   「阿岚 / 37」。

如果跨进程把 db 路径打错（例如改成 ``other.db``），子进程会因为 session 不存在
直接抛 ``SessionNotFoundError`` —— 这正是「持久化锚点是文件，不是内存」的证据。

Run
---
    python examples/11_persistence.py
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
    """父进程：建 session、塞一条事实、关闭、把 sid 返回给上层。"""
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=200),
        db_path=db_path,
        config=AgentLoopConfig(system_prompt=SYSTEM, max_rounds=1, compactor=None),
    )
    try:
        r = await loop.send("记住：我叫阿岚，最喜欢的数字是 37。回一句确认就好。")
        print(f"[phase1] sid={r.session_id}")
        print(f"[phase1] reply={r.final_text}")
        return r.session_id
    finally:
        loop.close()


async def phase2(db_path: str, sid: str) -> str:
    """子进程：只拿到 db_path + sid，直接续。"""
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=200),
        db_path=db_path,
        config=AgentLoopConfig(system_prompt=SYSTEM, max_rounds=1, compactor=None),
    )
    try:
        r = await loop.send("我叫什么？最喜欢的数字是多少？请只用一句话回答。", session_id=sid)
        print(f"[phase2] reply={r.final_text}")
        return r.final_text
    finally:
        loop.close()


async def _run_parent() -> None:
    # 用临时目录里的真实文件，避免污染仓库根。
    with tempfile.TemporaryDirectory(prefix="power_loop_p11_") as tmp:
        db_path = str(Path(tmp) / "real_file.db")
        sid = await phase1(db_path)
        assert Path(db_path).exists(), "phase1 应当在磁盘上留下文件"

        print(f"\n--- 父进程退出 phase1，db 留在 {db_path} ---")
        print("--- 重新拉起子进程跑 phase2 ---\n")

        cp = subprocess.run(
            [sys.executable, __file__, "phase2", db_path, sid],
            check=True,
            text=True,
        )
        if cp.returncode != 0:
            raise SystemExit(f"phase2 subprocess failed: rc={cp.returncode}")


def main() -> None:
    # 子进程入口：argv = [..., "phase2", db_path, sid]
    if len(sys.argv) >= 4 and sys.argv[1] == "phase2":
        _, _, db_path, sid = sys.argv[:4]
        asyncio.run(phase2(db_path, sid))
        return
    # 父进程入口
    asyncio.run(_run_parent())


if __name__ == "__main__":
    # 子进程从同一文件执行，需保证相对 import (_helpers) 找得到。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
