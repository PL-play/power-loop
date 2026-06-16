"""09 · 全量审计日志 / Audit log: subscribe to all events → JSONL

What you learn / 你将学到
--------------------------
- ``bus.subscribe(None, fn)`` 订阅**所有**类型的事件——常用于审计 / 调试 / OTel
  / subscribes to **all** event types — commonly used for audit / debugging / OTel
- ``AgentEvent.payload`` 是 dict 形式（``data.to_dict()`` 自动填充），方便 JSON 序列化
  / ``AgentEvent.payload`` is a dict (auto-populated via ``data.to_dict()``), easy to JSON serialize
- ``AgentEvent.{type, session_id, round_index, stream_id, source}`` 是路由字段
  / are routing fields
- 订阅者错误默认隔离（``suppress_subscriber_errors=True``），不会污染主循环
  / subscriber errors are isolated by default, won't affect the main loop
- 适合 hook 到 ELK / Datadog / 自家 audit pipeline
  / suitable for hooking into ELK / Datadog / custom audit pipelines

Output / 输出
-------------
跑完后会有一个 ``audit.jsonl`` 文件，每行一个 JSON 事件。
/ After running, an ``audit.jsonl`` file is created, one JSON event per line.

Run / 运行
----------
    python examples/09_audit_log.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from _helpers import make_llm

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentLoopConfig,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)

# ── 简单 echo 工具，让 audit 里出现 TOOL_* 事件 ───────────────────────────
#    Simple echo tool to produce TOOL_* events in the audit log


def echo_handler(**kwargs) -> str:
    return f"echo: {kwargs.get('text', '')}"


ECHO_TOOL = ToolDefinition(
    name="echo",
    description="Echo a string back.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    required_params=("text",),
)


def attach_audit_writer(bus: AgentEventBus, path: Path) -> None:
    """全量订阅 → 写 JSONL / Subscribe to all → write JSONL."""

    # mode="a" 允许多次 run 追加到同一文件；测试里用 tempfile
    # mode="a" allows multiple runs to append; tests use tempfile
    fh = open(path, "a", encoding="utf-8")

    def on_event(event: AgentEvent) -> None:
        record = {
            "ts": time.time(),
            "type": event.type.value,
            "session_id": event.session_id,
            "round_index": event.round_index,
            "stream_id": event.stream_id,
            "source": event.source,
            "payload": event.payload,
        }
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        fh.flush()                                              # demo 用：方便 tail -f
                                                                 # demo: convenient for tail -f

    # bus.subscribe(None, ...) 订阅所有事件类型 / subscribe to all event types
    bus.subscribe(None, on_event)


async def main() -> Path:
    audit_path = Path(tempfile.gettempdir()) / "power_loop_audit.jsonl"
    audit_path.unlink(missing_ok=True)

    bus = AgentEventBus(suppress_subscriber_errors=True)
    attach_audit_writer(bus, audit_path)

    registry = ToolRegistry()
    registry.register(ECHO_TOOL, echo_handler)

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=256, temperature=0),
        db_path=":memory:",
        event_bus=bus,
        tool_registry=registry,
        config=AgentLoopConfig(
            system_prompt=(
                "You have an `echo` tool. Use it to repeat any phrase the user gives you."
            ),
            max_rounds=3,
            compactor=None,
        ),
    )
    sid = await loop.new_session()
    r = await loop.send("Echo back the phrase: 'audit log demo'.", session_id=sid)
    print(f"\n[reply] {r.final_text}\n")

    # 看一眼 audit 文件长什么样 / Peek at the audit file
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    print(f"[audit] wrote {len(lines)} events to {audit_path}")
    type_counter: dict[str, int] = {}
    for line in lines:
        t = json.loads(line)["type"]
        type_counter[t] = type_counter.get(t, 0) + 1
    print("[audit] event type histogram:")
    for t, n in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"        {n:4d}  {t}")

    return audit_path


if __name__ == "__main__":
    asyncio.run(main())
