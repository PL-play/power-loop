# 配置

[English](../../en/user-guide/configuration.md) | [用户手册](../index.md)

所有可调参数 —— `AgentLoopConfig`、环境变量、`LLMProviderConfig`。

## AgentLoopConfig

```python
from power_loop import AgentLoopConfig

config = AgentLoopConfig(
    system_prompt="你是一个有帮助的助手。",
    max_rounds=24,           # 每次 send() 的最大 LLM 调用次数
    temperature=0.0,         # 0 = 确定性的
    max_tokens=8000,         # 每次请求的 token 上限
    compactor=DefaultCompactor(),  # 默认开启；None 关闭
    retry_policy=None,       # None = 不重试（快速失败）
    memory=None,             # None = 无跨会话记忆
    memory_budget_tokens=1500,
)
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `system_prompt` | `str \| None` | `None` | 每次 LLM 请求前添加的系统消息 |
| `max_rounds` | `int` | `24` | 每次 `send()` 的最大 LLM + 工具轮数。1 = 单条回复，无工具 |
| `temperature` | `float \| None` | `0.0` | LLM 温度 |
| `max_tokens` | `int \| None` | `8000` | 每次请求的 token 上限 |
| `compactor` | `Compactor \| None` | `DefaultCompactor()` | 上下文压缩；`None` 关闭 |
| `retry_policy` | `LLMRetryPolicy \| None` | `None` | LLM 瞬时错误重试 |
| `memory` | `MemoryProvider \| None` | `None` | 跨会话记忆 provider |
| `memory_budget_tokens` | `int` | `1500` | 传给 `memory.recall()` 的 token 预算 |
| `memory_position` | `str` | `"tail"` | 内置 hook 把召回的记忆注入到何处：`"tail"`（历史之后；保持先前历史前缀逐字节稳定、可做前缀缓存）或 `"front"`（旧位置；置于前导 system 消息之后） |
| `builtin_memory_hook` | `bool` | `True` | 设置了 `memory` 时自动注册内置 `MemoryRecallHook`；设为 `False` 则由你自己通过 `LLM_BEFORE` hook 注入记忆 |
| `microcompact_enabled` | `bool` | `False` | 启用 microcompact：把较旧的超大工具输出溢写到磁盘 + 留一个简短指针（仅逐字模式；与 LLM 摘要折叠正交） |
| `microcompact_size_limit` | `int` | `1000` | 超过此字节数的旧工具输出会被溢写（环境变量 `CONTEXT_MICRO_SIZE_LIMIT`） |
| `microcompact_hot_tail` | `int` | `10` | 最近多少轮保持逐字、绝不溢写（环境变量 `CONTEXT_MICRO_HOT_TAIL`） |
| `microcompact_spill_dir` | `str \| None` | `None` | 溢写输出的目录；`None` → 运行时 home 的 `.cache` |
| `distributed_sessions` | `bool` | `False` | 用数据库租约在**多进程之间**协调同一 session（仅服务端后端有意义）。默认关闭——单进程宿主已由进程内的锁覆盖。见 [伸缩](scaling.md#多进程共享一个-store) |
| `session_lease_ttl_s` | `float` | `90.0` | 失败检测窗口：租约在不续约时能存活多久（后台任务每 TTL/3 续一次，与轮边界无关——长轮本身不构成威胁）。取值要大于 event loop 可能被饿死的最长时间；它同时决定持有者崩溃后 session 被锁住多久 |

## 环境变量

配置 LLM 凭证的推荐方式：

```bash
# 必填
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-…
POWER_LOOP_MODEL=gpt-4o-mini

# 可选
POWER_LOOP_PROVIDER=openai          # 遥测标签
POWER_LOOP_TIMEOUT_S=180            # HTTP 超时
POWER_LOOP_MAX_TOKENS=8000          # 每次请求上限
POWER_LOOP_TEMPERATURE=0.0
POWER_LOOP_MAX_RETRIES=3
```

旧 `OPENAI_COMPAT_*` 名称仍支持作为回退。

### 一行构建服务

```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()
# 或自定义前缀
llm = create_llm_service_from_env(prefix="MY_APP")
```

## LLMProviderConfig（编程式）

```python
from power_loop import LLMProviderConfig, create_llm_service_from_config

cfg = LLMProviderConfig(
    provider="dashscope",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-…",
    model="qwen-plus",
    max_tokens=4096,
    temperature=0.0,
)
llm = create_llm_service_from_config(cfg)
```

必填字段缺失 → 构造时 `ValueError`（不是等 `complete()` 时才报——提前暴露配置错误）。

详见 [Providers](providers.md) 的各 provider 片段（OpenAI / DashScope / DeepSeek / 本地）。

## 压缩器调优

```python
from power_loop.runtime.compact import DefaultCompactor

compactor = DefaultCompactor(
    trigger_ratio=0.75,       # token 超过 max_tokens 的 75% 时触发压缩
    keep_last_n=4,            # 始终保留最后 4 轮对话
    summary_max_tokens=512,   # 摘要 LLM 调用的最大 token 数
)
```

或通过环境变量设置绝对阈值：`CONTEXT_COMPACT_THRESHOLD=6000`

关闭压缩：`AgentLoopConfig(compactor=None)`。

## 重试策略

```python
from power_loop import LLMRetryPolicy

retry = LLMRetryPolicy(
    max_attempts=3,           # 1 次初始 + 2 次重试
    backoff_initial=0.5,      # 第二次尝试前等待秒数
    backoff_max=8.0,          # 指数退避上限
    total_timeout=60.0,       # 跨所有尝试的 wall-clock 超时
    retry_on=(Exception,),    # 默认：所有 Exception 子类
)

config = AgentLoopConfig(retry_policy=retry, ...)
```

详见 [重试与取消](retry-cancel.md) 了解完整重试生命周期。

## 日志卫生

`import power_loop` 会给 `power_loop` 根 logger 挂一个 `logging.NullHandler`，所以在你的应用配置
日志之前，库保持静默（所有模块 logger 都归在 `power_loop.*` 子树下）。

要做结构化事件日志，挂上 JSON-lines sink —— 每个事件一行，写到 `power_loop.events` logger：

```python
from power_loop.contrib.logging_sink import attach_logging_sink
attach_logging_sink(bus)                          # 全部事件、INFO、默认脱敏
```

它**默认对密钥名的值脱敏**（`api_key` / `authorization` / `secret` / `password` / `*_token`，
大小写不敏感子串；故意不含裸 `token`，以免误伤 `prompt_tokens`/`completion_tokens` 计数）。可覆盖或关闭：

```python
attach_logging_sink(bus, redact_keys=("api_key", "x-internal-secret"))  # 自定义 denylist
attach_logging_sink(bus, redact_keys=())                                 # 不脱敏
```

## 下一步

- [会话](sessions.md) — 理解会话生命周期
- [工具](tools.md) — 给 Agent 能力