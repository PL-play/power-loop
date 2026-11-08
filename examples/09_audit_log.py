"""09 · Current worker events to JSONL; close the subscriber and file explicitly."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from _runtime import demo_runtime

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.runtime.pricing import Free

# ── 简单 echo 工具，让 audit 里出现 TOOL_* 事件 ───────────────────────────
#    Simple echo tool to produce TOOL_* events in the audit log


def echo_handler(**kwargs) -> str:
    return f"echo: {kwargs.get('text', '')}"


ECHO_TOOL = ToolDefinition(
    pricing=Free("local example tool"),
    name="echo",
    description="Echo a string back.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    required_params=("text",),
)


def attach_audit_writer(bus: AgentEventBus, path: Path):
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
    def close():
        bus.unsubscribe(on_event)
        fh.close()
    return close


class EchoModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m['role'] == 'tool' for m in request.messages):
            return LLMResponse(raw_text='audit log demo')
        return LLMResponse(raw_text='', tool_calls=[{'id':'echo','type':'function','function':{
            'name':'echo','arguments':'{"text":"audit log demo"}'}}])

    async def close(self):
        pass


async def main():
    directory = Path(tempfile.mkdtemp(prefix='power_loop_audit_test_'))
    audit_path = directory/'audit.jsonl'
    bus = AgentEventBus()
    close = attach_audit_writer(bus, audit_path)
    registry = ToolRegistry()
    registry.register(ECHO_TOOL, echo_handler)
    try:
        async with demo_runtime(EchoModel(), registry, event_bus=bus,
                config=AgentLoopConfig(max_rounds=3, compactor=None)) as (runtime, loop, target, actor):
            result = await loop.send(target, 'echo demo', message_id='audit', actor=actor,
                                     deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
    finally:
        close()
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(row['type'] == 'tool_call_completed' for row in rows)
    print('audit:', audit_path, 'events:', len(rows))
    return audit_path


if __name__ == '__main__':
    asyncio.run(main())
