# 可观测性

[English](../../en/user-guide/observability.md) | [用户指南](../index.md)

每个生命周期时刻都是一个带类型的[事件](events.md)。本页讲的是怎么把这些事件**导出去**——持久化、出指标、连 tracing——以及怎么不让一个 sink 卡住 agent 循环。

## 事件信封

每个 `AgentEvent` 在带类型 payload 之外还带一个信封:

| 字段 | 含义 |
|---|---|
| `seq` | 进程级单调序号——跨所有 session/子代理的全序 |
| `ts` | 墙钟创建时间(epoch 秒)——可读可导出,但非单调 |
| `mono` | `perf_counter` 秒——做时长/延迟用它(不受 NTP/墙钟回拨影响) |

`event.to_dict()` / `AgentEvent.from_dict(d)` 往返整个信封。`from_dict` 原样保留序列化的 `seq`/`ts`/`mono`,且**不**推进进程计数器——所以回放的事件保持原有顺序。

## 持久化事件 + 回放

进程内总线是易失的。把事件持久化到大小轮转的 JSONL 文件,之后再读回:

```python
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay

sink = attach_jsonl_sink(bus, "events.jsonl")   # 全部事件;或 events={...}
...                                              # 跑循环
sink.close()

for event in replay("events.jsonl"):             # 旧→新,跨轮转
    print(event.seq, event.type, event.payload)
```

payload 会被截断 + 密钥脱敏(与 logging sink 共享策略;传 `redact_keys=()` 关闭)。按大小轮转(`max_bytes`、`backup_count`)。

标准库 **logging sink**(`attach_logging_sink`)按行输出 JSON(现在含 `seq`/`ts`)到 `logging.Logger`——可接 jq/Loki/CloudWatch。

## 指标

`attach_metrics_sink` 通过一个极小的 `MetricsBackend` Protocol(`incr`/`observe`/`gauge`)把带类型事件映射成计数/观测——自带后端或自带实现:

```python
from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

attach_metrics_sink(bus, PrometheusBackend())    # power-loop[prometheus]
# 或 StatsDBackend()                              # power-loop[statsd]
```

输出(前缀 `power_loop`):`…_llm_calls`(+`…_llm_call_duration_ms`)、`…_llm_retries`、`…_tool_calls`(标签:tool、success)、`…_rounds`、`…_errors`、`…_usage_total_tokens`。映射本身无依赖;只有出厂后端惰性导入各自客户端。

## Tracing(OpenTelemetry)

`attach_otel_sink` 把成对的 `*_STARTED`/`*_COMPLETED` 事件变成 span 树——`session` → `round` → `llm_call` / `tool_call`——可导出到任意 OTel 后端:

```python
from power_loop.contrib.otel_sink import attach_otel_sink   # power-loop[otel]

bridge = attach_otel_sink(bus)     # 用全局 tracer provider
...                                 # 跑循环 → 导出 span
bridge.close()                      # 结束所有未闭合 span
```

span 带 `model`/`duration_ms`/`success`;失败的工具/LLM 调用置 error 状态。`opentelemetry` 惰性导入,所以无该 extra 也能 import 本模块。

## 背压——别卡住循环

**同步**订阅者在发布线程(通常是 agent 的事件循环)上**内联**执行,所以**必须快**。慢活儿优先用 `async def` 处理器(被调度为 task,永不卡循环)。如果非得跑慢的*同步* sink,可选后台分发:

```python
bus = AgentEventBus(sync_dispatch="thread", queue_maxsize=1000, on_overflow="drop_newest")
...
bus.shutdown()   # 先排空队列再停 worker
```

thread 模式下同步订阅者由后台线程经有界队列消费,`publish()` 立即返回;队列满按 `on_overflow`(`drop_newest`/`drop_oldest`/`block`)丢弃并计入 `bus.dropped`。异步订阅者仍调度到循环。默认 `inline`(行为不变)。

端到端示例见 [`examples/36_observability.py`](../../../examples/36_observability.py)(JSONL sink + 指标 + 回放)。
