"""30 · Subprocess isolation / 子进程隔离: each workflow leaf in its own process + DB

What you learn / 你将学到
--------------------------
- ``SubprocessExecutor`` runs **every workflow leaf in a separate OS process**
  against its **own SQLite file** — the one-writer-per-file rule then holds
  trivially (no shared-write coordination), and a crashed / killed / timed-out
  leaf becomes a non-``completed`` result the resume tier re-runs. A fault in one
  leaf can't corrupt the orchestrator or its siblings.
  / 每个叶子在独立进程 + 独立 SQLite 中运行；某个叶子崩溃也不会污染编排器或兄弟叶子。
- ``WorkerBootstrap(llm_from_env=True)``: the child **rebuilds the real provider
  from env** — deps are reconstructed from *config alone*, nothing live crosses
  the process boundary.
  / 子进程从环境变量重建真实 provider，依赖只从配置重建。
- The ``WorkerLauncher`` seam: ``build()`` is called **per leaf** with that leaf's
  ``AgentSpec``, so an integrator can wrap each child in a sandbox
  (``docker run`` / ``runsc`` / ``firejail``), deciding isolation per leaf from
  ``spec.tools``. This is the **process-level analog of the ``ShellBackend`` seam**
  in example 28 (there it wraps the *shell*; here, the *whole worker*).
  / ``WorkerLauncher`` 是进程级沙箱缝（example 28 的 shell 级类比）。
- Each leaf's db is left on disk for **post-hoc inspection** (GC via ``reap_runs``
  / ``cleanup_run``).
  / 叶子库留盘可事后审查，用 ``reap_runs`` / ``cleanup_run`` 回收。

Requires ``POWER_LOOP_*`` env (a real provider). Each leaf spawns a real
``python -m power_loop.workflow.worker`` process.

Run / 运行
----------
    python examples/30_subprocess_isolation.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from _helpers import load_env

from power_loop import (
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    create_llm_service_from_env,
)
from power_loop.runtime.spec import AgentSpec
from power_loop.workflow import (
    DirectWorkerLauncher,
    SubprocessExecutor,
    WorkerBootstrap,
    WorkflowSpec,
)
from power_loop.workflow.engine import WorkflowEngine

# Two independent leaves with DISTINCT ids → two processes, two db files. A
# `parallel` node runs its branches concurrently (a barrier fan-in).
SPEC = {
    "name": "capitals_isolated",
    "input": "(unused)",
    "root": {
        "type": "parallel",
        "id": "both",
        "branches": [
            {"type": "agent", "id": "france",
             "spec": {"name": "geographer", "system_prompt": "Answer in one short sentence."},
             "input": "What is the capital of France?"},
            {"type": "agent", "id": "japan",
             "spec": {"name": "geographer", "system_prompt": "Answer in one short sentence."},
             "input": "What is the capital of Japan?"},
        ],
    },
}


class RecordingWorkerLauncher:
    """Wraps the default launcher to RECORD each per-leaf launch.

    This proves the seam fires once per leaf with that leaf's ``AgentSpec`` (so a
    real sandbox launcher could pick an isolation policy per leaf from
    ``spec.tools``). A *production* launcher would instead RETURN a wrapped
    command, e.g.::

        return (["docker", "run", "--rm", "-i", "--network", "none",
                 "-v", f"{db_path}:{db_path}", IMAGE, *base_argv], base_env)

    See example 28 for the shell-level analog (``DockerShellBackend``).
    """

    def __init__(self) -> None:
        self.launches: list[dict] = []
        self._inner = DirectWorkerLauncher()

    def build(self, *, base_argv, base_env, spec: AgentSpec, db_path, workspace_dir):
        self.launches.append(
            {"spec_name": spec.name, "tools": tuple(spec.tools or ()), "db_path": db_path}
        )
        return self._inner.build(
            base_argv=base_argv, base_env=base_env, spec=spec,
            db_path=db_path, workspace_dir=workspace_dir,
        )


async def main() -> dict:
    load_env()
    runs_dir = tempfile.mkdtemp(prefix="pl-subproc-")
    launcher = RecordingWorkerLauncher()
    executor = SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True),  # child rebuilds provider from env
        runs_dir=runs_dir,
        timeout_s=120,
        launcher=launcher,  # the per-leaf sandbox seam (here: recording only)
    )
    loop = StatefulAgentLoop(
        llm=create_llm_service_from_env(),
        db_path=":memory:",
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3),
    )
    result = await WorkflowEngine(loop, executor=executor, run_id="capitals").run(
        WorkflowSpec.from_json(SPEC)
    )

    fr = result.results.get("france")
    jp = result.results.get("japan")

    # Each leaf ran in its OWN process + OWN db file → distinct db paths.
    distinct_dbs = {r.db_path for r in (fr, jp) if r and r.db_path}

    # The orchestrator can open a leaf's private db afterward and inspect it —
    # the histories live in separate stores, not one shared one.
    inspected_assistant = False
    if fr and fr.db_path and os.path.exists(fr.db_path):
        store = SessionStore.open(fr.db_path)
        try:
            inspected_assistant = any(
                m.role == "assistant" for m in store.load_all_messages(fr.session_id)
            )
        finally:
            store.close()

    out = {
        "status": result.status,
        "france_text": fr.text if fr else "",
        "japan_text": jp.text if jp else "",
        "distinct_db_count": len(distinct_dbs),
        "launch_count": len(launcher.launches),
        "launched_specs": sorted(x["spec_name"] for x in launcher.launches),
        "inspected_assistant": inspected_assistant,
    }
    print(f"status            : {out['status']}")
    print(f"France leaf       : {out['france_text']!r}")
    print(f"Japan leaf        : {out['japan_text']!r}")
    print(f"distinct db files : {out['distinct_db_count']}  (one per leaf)")
    print(f"per-leaf launches : {out['launch_count']}  via the WorkerLauncher seam")
    print(f"leaf db inspected : {out['inspected_assistant']}")

    # Tidy up the per-leaf dbs (in production: reap_runs(runs_dir, older_than_s=…)).
    shutil.rmtree(runs_dir, ignore_errors=True)
    return out


if __name__ == "__main__":
    asyncio.run(main())
