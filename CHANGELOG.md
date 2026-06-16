# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

1.0 起,对 **STABLE** API 的破坏性变更只能出现在 major 版本,
并且必须在此文件中显式列出受影响的 Public API。

## [Unreleased]

## [2.0.0] — 2026-06-16

**Pluggable storage + an async, stateless, resumable loop.** The store became a
backend-neutral **async** facade (SQLite by default; PostgreSQL/MySQL by DSN) and the
public API went fully async. These are breaking changes to the **STABLE** surface — hence
the major bump. Single-file SQLite stays the zero-infrastructure default; nothing else is
required to upgrade beyond `await`-ing the now-async calls.

### Breaking (STABLE API)

- **The public API is async.** `StatefulAgentLoop` session-management methods that were
  synchronous are now coroutines and must be `await`ed: `new_session`, `close_session`,
  `get_messages`, `get_pending`, `resolve_system_prompt`, `abort_pending`, `schedule_timer`,
  `cancel_timer`, `list_timers`, `get_session_stats`, `list_session_stats` (plus the new
  `prewarm`). Every `SessionStore` method is a coroutine, and `SessionStore.open(...)` is now
  `await SessionStore.open(...)`. `send` / `follow_up` / `resume` / `submit_input` were already
  async; the `send_sync` / `follow_up_sync` wrappers stay synchronous. STABLE symbol *names*
  are unchanged (the SemVer name-guard still passes); the break is the sync→async signatures.
- **Store schema changed — no on-disk migration from 1.x.** Tables now carry a configurable
  prefix (`pl_` by default); the `session_runtime_state` / `shared_state` `key` column was
  renamed to `state_key` (a MySQL reserved word); the SQLite `PRAGMA user_version` ladder was
  replaced by a portable, backend-neutral `pl_schema_migrations` version table. A pre-2.0 `.db`
  is not read by 2.0 — start fresh or `export_session` → `import_session`.
- **Removed the read-only WAL connection pool (`read_pool_size`).** The async SQLite backend
  offloads each statement to a worker thread under a single writer lock; PG/MySQL are natively
  async. Scale by choosing a server backend or sharding SQLite files across processes.

### Added

- **Pluggable storage backends.** One store written once against a tiny async
  `Database` / `Dialect` port, with **SQLite** (default, zero-dependency), **PostgreSQL**
  (`power-loop[postgres]`, asyncpg) and **MySQL** (`power-loop[mysql]`, aiomysql). `open_store(dsn)`
  and `StatefulAgentLoop(dsn=, table_prefix=, schema=)` select the backend by DSN scheme; PG/MySQL
  are real multi-writer servers (per-session seq via `SELECT … FOR UPDATE`). A backend-agnostic
  conformance suite runs against all three. New top-level exports: `open_store`, `SchemaPolicy`,
  `StoreSchemaError`.
- **`SchemaPolicy` provisioning.** `AUTO_CREATE` (default) creates tables if missing; `VERIFY`
  only checks and raises `StoreSchemaError` — whose `.ddl` carries (and prints) the complete
  per-backend provisioning script. `create_schema: bool` kept as a deprecated alias.
- **Stateless, resumable loop + per-session window cache.** The loop holds no authoritative
  session state, so a cold/fresh loop resumes any session from `dsn` + `session_id`;
  `await loop.prewarm(session_id)` pre-loads the active window; an LRU active-window cache
  (`session_cache_size`, default 256, `0` disables; `loop.cache_stats`) accelerates hot paths
  as a pure, validated accelerator that never changes what the model sees.
- New example `39_pluggable_backends_and_resume.py`; a **Storage backends** user-guide page
  (EN + ZH) with the exact per-backend DDL.

### Fixed

- Window cache could serve a stale, row-missing window after an out-of-band durable append
  (`resume` / `submit_input` / `abort_pending` / `heal_pending`, or a second loop sharing the
  store) — fixed with a contiguity guard in the cache; covered by warm-vs-cold regression tests
  (caught by an adversarial code-review pass).
- Carries the post-1.0 deep-review hardening merged on `main` (sandbox/import/workflow-cluster
  fixes + latent-finding guards) and the async `_bind_handler` runtime-env fix + restored
  `max_spawn_depth` validation surfaced during the swap.

### Docs

- README / README.zh rewritten around **loop engineering**, pluggable storage, and
  statelessness; the full EN/ZH user-guide + tutorials swept to the async API; a new Storage
  backends page; an async-API + storage migration note.

## [1.0.0] — 2026-06-16

**First stable release.** The `STABLE` public API is now under SemVer: a break to it
requires a major bump (`2.0.0`), enforced by the frozen-baseline guard. The post-0.14.1
hardening roadmap is complete — durability (0.15), scale (0.16), observability (0.17),
ecosystem (0.18) — and the release-readiness audit's blockers are resolved.

### Changed

- **STABLE API — construction closure (the 1.0 gate).** Promoted the LLM contract into
  `STABLE_API` (frozen): `LLMService`, `LLMRequest`, `LLMResponse`, `LLMStreamChunk`,
  `LLMProviderConfig`, and `create_llm_service_from_env` / `create_llm_service_from_config`.
  The flagship `StatefulAgentLoop` can now be **built, used, and given a custom provider
  using STABLE-only symbols** — previously its mandatory `llm=` collaborator was Provisional,
  making the freeze hollow. Error `.code` strings are now a frozen contract too
  (`test_stable_error_codes_are_frozen`).
- **Post-1.0 SemVer everywhere.** Classifier → `Production/Stable`; CONTRIBUTING / CHANGELOG /
  SECURITY / README / API-reference prose updated from the 0.x "breaks-in-a-minor" model to
  "a STABLE break needs a major bump". Stale doc counts (hooks 18→17, events 24→30) and the
  last "depends only on certifi" onboarding line (the core is zero-dependency) corrected.

### Added

- **Real MCP server test + example.** `StdioMCPClient` is now validated end-to-end against
  a live FastMCP stdio server (`test_mcp_real_server.py` — not just a fake source), and
  `examples/38_mcp_tools.py` shows an agent calling a real MCP server's tool. `mcp` added to
  the dev/CI extras so this runs in CI.

## [0.18.0] — 2026-06-16

迈向 1.0 的硬化路线图**第四阶段(收官):生态/供应链/治理**。MCP 工具适配器、vendored
`llm_client` 溯源、**核心零运行时依赖**、扩展工具手册、`SECURITY.md`、可复现发布流程。
**破坏性变更见 Changed**(移除 `certifi` 基础依赖)。至此 0.15.0(持久化)→ 0.16.0(扩展)→
0.17.0(可观测性)→ 0.18.0(生态)四阶段全部落地。

### Changed

- **（ECO-3)核心零运行时依赖。** 删除 vendored 死代码 `qwen_image.py` + `web_search.py`
  (power-loop 从未导入;`qwen_image` 是 `certifi` 的**唯一**导入者),从 `dependencies` 移除
  `certifi` → 基础依赖集为空。受支持的用法不受影响(transports 经 extras 自带 CA 处理);已用
  import-without-extras + 真机 HTTPS 冒烟验证。
- **（ECO-4)修正过期覆盖率目标。** `--cov=llm_client` → 仅 `--cov=power_loop`(vendored 子包
  随之被统计),ci.yml 与 pyproject 同步;覆盖率门槛 74% > 70%。

### Added

- **（ECO-6)扩展工具手册 + 示例。** `docs/{en,zh}/user-guide/extending-tools.md`:自定义工具配方
  (`ToolDefinition`+handler+`register`)、按调用白名单、把 MCP 作为外部连接器路径、以及"为什么
  不捆绑连接器"的内核理念;新增示例 `37_custom_retrieval_tool`(进程内确定性检索工具)。
- **（ECO-5)`SECURITY.md`。** 支持版本、私密漏洞上报渠道(尽力而为、无 SLA),以及"编排而非隔离"
  的安全模型(内置 bash/file 工具**不是**沙箱 → 用 `ShellBackend`/`SubprocessExecutor`;密钥留在
  编排层;磁盘 SQLite 为明文)。
- **（ECO-7)bus-factor surrogates。** CONTRIBUTING 增加可复现的 `Releasing` 流程;README 增加
  "Used by" 与 "Project status & governance"(单维护者、MIT、可分叉、机器化 API 稳定性守卫)。

- **（ECO-1)MCP 工具适配器。** `contrib/mcp`:无依赖的 `MCPToolSource` Protocol +
  `register_mcp_tools`——把 MCP 工具的 `inputSchema` 直接映射成 `ToolDefinition`(`required`
  驱动缺参校验),注册的异步 handler 把调用代理给 source;外加 `[mcp]` extra 之后惰性导入 `mcp`
  SDK 的默认 `StdioMCPClient`(stdio MCP server)。映射本身可用假 source 测试,无需 `mcp` 依赖。
- **（ECO-2)vendored llm_client 溯源。** `_vendor/llm_client/VENDOR.md`(来源、MIT 许可、
  vendored 日期、本地修改含 0.18.0 的删除)+ `scripts/sync_vendor.sh`(重新 vendor:拷贝、
  裁剪未用模块、改写 import 到 vendored 路径)。

## [0.17.0] — 2026-06-15

迈向 1.0 的硬化路线图**第三阶段:可观测性**。事件信封序列化 + 单调时钟、持久可回放的 JSONL
事件流、事件总线背压、指标(Prometheus/StatsD)与 OpenTelemetry span 桥。全部增量;新后端在
可选 extras 之后惰性导入,核心仍 SDK-free。

### Added

- **（OBS-1）事件信封序列化。** `AgentEvent.to_dict()/from_dict()` 携带 `ts`/`seq`/`mono`,
  作为持久化与外部导出的基础;`from_dict` 对时序字段做**存在性检查**(非真值检查)——序列化的
  `seq` 权威保留,既不重新盖章也不推进进程级 `_event_seq` 计数器。`logging_sink` 现在也输出
  信封序号(`seq`/`ts`),日志行可排序、可与持久事件流对账(此前丢弃信封)。
- **（OBS-6）单调时钟字段。** `AgentEvent.mono`(`perf_counter` 秒,进程相对)用于跨事件延迟/
  span 计算,不受 NTP/墙钟回拨影响(`ts` 仍为可读可导出的墙钟时间)。
- **（OBS-2）持久化 JSONL 事件 sink + 回放。** `attach_jsonl_sink(bus, path, …)` 把完整信封
  (经脱敏/截断)按行写入大小轮转文件;`replay(path)` 跨轮转按 `seq` 顺序还原成 `AgentEvent`。
  脱敏策略抽到共享的 `contrib/_redact`(logging 与 jsonl 复用)。
- **（OBS-3）事件总线背压。** 文档化硬契约:同步订阅者必须快、不可阻塞(否则卡住 agent 循环)。
  新增 opt-in `AgentEventBus(sync_dispatch="thread", queue_maxsize=…, on_overflow=…)`:同步订阅者
  改由后台线程经有界队列消费,`publish()` 立即返回,慢订阅者不再卡循环;队列满按 `on_overflow`
  (`drop_newest`/`drop_oldest`/`block`)处理并计入 `bus.dropped`;`shutdown()` 先排空再停线程。
  默认 `inline`,行为不变。异步订阅者仍调度到事件循环(不下放到无 loop 的线程)。
- **（OBS-4)指标 sink。** `contrib/metrics_sink`:无依赖的 `MetricsBackend` Protocol +
  事件→指标映射(llm 调用/重试、工具调用成败、轮次、错误、token 用量),映射本身不依赖任何
  第三方库(可用假后端测试);出厂 `PrometheusBackend`(`[prometheus]`)与 `StatsDBackend`
  (`[statsd]`)惰性导入各自客户端。
- **（OBS-5)OpenTelemetry span 桥。** `contrib/otel_sink`:把成对的 `*_STARTED`/`*_COMPLETED`
  事件映射成 session→round→llm/tool 的 span 树,接入任意 OTel 后端;在 `[otel]` extra 之后,
  `opentelemetry` 惰性导入(无依赖也可 import 本模块)。`close()` 结束所有未闭合 span。

### Changed

- 新增可选 extras:`prometheus` / `statsd` / `otel`,并并入 `all`;`dev` 增加
  `prometheus-client` + `opentelemetry-sdk` 以便 CI 跑 OBS-4/5 后端测试。

## [0.16.0] — 2026-06-15

迈向 1.0 的硬化路线图**第二阶段:扩展性**。把"推理出的"单进程上限变成"测出来的"(自带 `bench/`
压测台),并在读路径去瓶颈:只读 WAL 连接池、把每次 send 的历史读卸载出事件循环、压缩触发的
token 估算从每轮 O(history) 降为 O(1)。**全部增量,无破坏性变更。**

### Added

- **（SCALE-1）基准/压测台。** 新增**不随 wheel 发布**的 `bench/` 包:确定性 `FakeLLM`(可调人工
  延迟,不打真实 provider)驱动真实 `StatefulAgentLoop`+`SessionStore`,三个场景(FANOUT 并发会话 /
  BIG-HISTORY 大历史 / THROUGHPUT 持续吞吐)产出 JSON 报告(sessions/sec、p50/p99 读写延迟)。
  `python -m bench [--smoke]` 运行;`tests/bench/test_bench_smoke.py` 烟囱测试 + 非阻塞 CI
  (`.github/workflows/bench.yml`)。把"推理出的"单进程上限变成"测出来的",并已暴露 BIG-HISTORY
  的 O(history) 每轮成本(SCALE-4 的目标)。
- **（SCALE-3）卸载每次 send 的活动历史读取。** `_run_loop` 里 `load_active_messages`(O(active-history)
  的 SQLite 读 + 逻辑重排)改走 `asyncio.to_thread`,大会话加载不再卡住事件循环上的其它任务。
- **（SCALE-4）压缩触发的 token 估算从每轮 O(history) 降为 O(1)。** pipeline 维护一个自失效的
  增量 token 估算(append 增量更新;fold/recall/hook 替换历史时失效并重算,**永远等于全量重算**),
  经新增的 `CompactionContext.current_tokens` 交给 compactor 做触发判定,避免每轮重扫全历史
  (实测 5ms@1万 / 26ms@5万 每轮 → O(1))。对自定义 compactor 完全向后兼容(不传则照旧全扫)。

- **（SCALE-2）只读 WAL 连接池(opt-in)。** `SessionStore.open(read_pool_size=N)` 开 N 个额外的
  只读连接(`query_only=ON`),读操作(`load_active_messages`/`load_all_messages`)从池中取连接、
  与唯一写连接并发执行,不再排队等写锁——读密集 fan-out 下显著降低读延迟。写入仍由单写连接+RLock
  串行(正确性不变);WAL 保证池读看到读开始前已提交的全部写入。默认关闭;`:memory:` 自动回退
  (内存库连接不可共享)。含持写锁时池读不被阻塞的并发回归测试。

- **（SCALE-5）扩展性文档 + 示例。** 新增 `docs/{en,zh}/user-guide/scaling.md`:单写模型、读连接池、
  保留/VACUUM、多进程(一文件一进程)模式、调优旋钮,全部基于 `bench/` 实测数据(fan-out 在
  ~1000 sessions/sec 见顶;大历史每次 send 成本随历史线性增长——压缩使其平坦),并诚实标注上限。
  新增示例 `34_durability_lifecycle`(裁剪/VACUUM/导出导入/优雅停机)与
  `35_scaling_and_read_pool`(读池 + 并发会话 + 压测台)。

### Changed

- `CompactionContext` 新增可选字段 `current_tokens`(增量 token 估算提示;附加、向后兼容)。
- `SessionStore.open` / 构造器新增可选 `read_pool_size`(默认 0,行为不变)。
- `bench` fanout 场景支持 `db_path` + `read_pool_size`,以测量 SCALE-2 的读并发收益。

## [0.15.0] — 2026-06-15

迈向 1.0 的硬化路线图（见 `ROADMAP_1.0.md`）**第一阶段:持久化**。把长期存活的磁盘
`SessionStore`(本库的核心卖点)做成生产可用:可随版本升级、可回收空间、可导出归档、可优雅停机。
绝大多数为**纯增量** API;存储层的破坏性变更集中于本版本一次(见 Changed)。真机端到端验证见
`tests/real/test_real_durability.py`。

### Changed

- **（破坏性·存储）schema 版本网关。** `SessionStore.open()` 现在用 `PRAGMA user_version`
  门控:**拒绝打开版本高于本构建的 `.db`** 并给出清晰报错(对手改库者是行为变更)。这是后续一切
  schema 变更的前提,确保 ≤0.14.1 的旧库(`user_version=0`)被识别为 legacy 并经迁移步骤 1 升级,
  而非静默保持旧结构。所有存储层变更都集中在本版本这一次"存储形态拐点"。
- **（破坏性·存储)新建库默认 `auto_vacuum=INCREMENTAL`**(仅影响新建文件的空闲页行为;既有文件
  保持原样,不在 open 时做阻塞式全量 VACUUM)。

### Added

- **（OPS-1）SessionStore schema 版本网关 + 迁移阶梯。** 引入 `PRAGMA user_version` 门控与
  有序、幂等、纯增量的 `MIGRATIONS` 阶梯（`CURRENT_SCHEMA_VERSION` + `_apply_migrations`）。
  `open()` 在建表前探测是否全新库:全新库直接盖章到当前版本,既有库按 `target > user_version`
  顺序跑迁移(单事务,失败回滚则不前进版本),**版本高于本构建的库直接拒绝打开**并给出清晰报错。
  原先硬编码、仅针对 `timers` 的 `_micro_migrate` 收编为迁移步骤 1(对 legacy `user_version=0`
  幂等升级)。这是后续一切 schema 变更的**前提**——没有它,既有 `.db` 在升级时会静默保持旧结构
  (`CREATE TABLE IF NOT EXISTS` 永不改表)。回归测试见 `tests/unit/test_session_store_migrations.py`。
- **（OPS-2）按需保留/裁剪。** `SessionStore` 新增**调用方驱动、绝不隐式**的清理方法:
  `prune_compacted_messages`(删折叠出的 `compacted_out` 原文,保留 `compact_note`/active,
  支持 `older_than_ms`/`keep_recent`,**不可逆**)、`prune_usage_rounds`、`prune_timers`(仅终态)。
- **（OPS-3）空间回收。** 新建库默认 `auto_vacuum=INCREMENTAL`;新增 `vacuum(incremental=…)`
  与 `checkpoint(mode=…)`——配合 OPS-2/`close_session` 真正缩小磁盘文件、回收 WAL。
- **（OPS-4）会话导出/导入 + 整库备份。** `export_session` 把单会话全部持久态序列化为带
  `schema_version` 的 JSON,`import_session` 落到新 id(拒绝更高版本/已存在 id);`backup()`
  走 SQLite 在线备份 API 产出可直接打开的整库快照。支持「先归档再裁剪」与跨库迁移。
- **（OPS-5）优雅异步停机。** `StatefulAgentLoop.aclose()` + `async with`:先拒收新 send,
  再逐个获取 per-session 锁等待在飞 send 落盘完成(修复 `close()` 与 `to_thread` 写竞争导致的
  `ProgrammingError`),drain 待决异步事件订阅者(`AgentEventBus.drain()`),checkpoint 后关库。
  同步 `close()` 保留但标注为非优雅。回归:`test_session_store_retention.py` /
  `test_session_store_export.py` / `test_stateful_loop_aclose.py` + 真机 `tests/real/test_real_durability.py`
  端到端走通「压缩→裁剪→VACUUM→导出→aclose→重开(迁移网关)→导入」全链路。

## [0.14.1] — 2026-06-15

修复一个在 0.14.0 中发现的**高危持久化损坏**（C1 的同进程二次压缩遗漏分支）。纯 bug 修复，
无 Public API 破坏。

### Fixed

- **同一次运行内的第二次压缩会损坏持久化状态（C1 续）。** `compact_note` 被分配一个全新的**高**
  身份 `seq`（来自 `next_seq`），却被放在内存历史的**低**逻辑位置，使 `SQLiteSink._history_seqs`
  这个 index→seq 映射**非单调**。后果有二：
  - **数据损坏**：同一次运行内的第二次折叠把折叠边界经非单调映射翻译后，以
    `from_seq > to_seq` 调用 `record_compaction` → `UPDATE … WHERE seq BETWEEN from_seq AND to_seq`
    在 DB 里**一行都不标记**（内存却照折）→ 内存历史与持久化 active 集**分叉**，并向
    `compactions` 审计表写入倒置的 `(from,to)` 行。
  - **重排错误**：即使只折叠一次，note 因高 `seq` 在重载（`load_active_messages ORDER BY seq`）
    时**沉到 kept 尾部之后**，旧轮次的摘要出现在较新消息之后。
  - **修复**：把**身份**（高 `seq`，保持 append-only 与 recall 语义）与**逻辑位置**解耦。
    `record_compaction` 现在按**显式 seq 集合**（`seq IN (…)`，不再用 BETWEEN 区间，对非单调映射
    免疫）标记折叠行，并把 note 的逻辑位置存为 `meta['ord']`，审计区间用 min/max（不再倒置）。
    `SQLiteSink` 新增并行的 `_history_ord`（逐槽逻辑位置）；`load_active_messages` 改按**逻辑序**
    返回（`compact_note` 按 `meta['ord']` 排，否则按 `seq`）。这条直接修正了 0.14.0 的
    `HARDENING_PLAN` 验收声明“no compacted_out mis-map under recall+compaction”遗漏的同进程二次
    压缩场景——该场景此前**完全无测试**。
  - **回归测试**（`tests/unit/test_compaction_double_fold.py`，11 例）：同一次运行内二/三次折叠
    的内存↔DB 一致性、无倒置审计行、单次/中段折叠后 note 重载位置、reload-then-fold 恢复链路、
    显式集合标记、纯占位符折叠的映射对齐，以及一个 200 例 hypothesis 随机 append/fold 调度不变式
    （reload 必须逐字复现内存 active 历史），外加一个端到端「一次 send 内折叠两次」的真实管线用例。

## [0.14.0] — 2026-06-15

硬化计划 `HARDENING_PLAN.md`（0.13.1 → 1.0）的一次大批量推进：**全部已确认正确性 bug
C1–C8 修复**(各配红前/绿后回归测试) + 新增 H7 压缩联动轨道 + H2 测试/CI 加固(并由此发现并
修复一个真实安全漏报) + H3 打包到 1.0(featherweight 核心、vendored `llm_client`、稳定 API
机器化守卫) + H4 可观测性(每调用 LLM 事件、错误码、日志卫生) + H5 注入接缝修复。
**破坏性变更见 Changed**(`llm_client` 收编 / 安装方式 / 核心依赖)。

### Added

- **（H4.4）机器可读错误码**：每个 `PowerLoopError` 子类带稳定的类级 `code`(点分串,如
  `llm.timeout` / `session.pending` / `tool.not_found` / `spec.invalid`),调用方可按
  `exc.code` 分支而非类身份——重构友好、便于日志/翻译。
- **（H4.5）日志卫生**：`import power_loop` 给包根 logger 挂 `NullHandler`(无 handler 噪声、
  应用未配置即不输出);两处硬编码 logger 名改 `getLogger(__name__)`(全树归于 `power_loop.*`);
  `attach_logging_sink` 新增 `redact_keys`——默认对 `api_key`/`authorization`/`secret`/
  `password`/`*_token` 等密钥名的值脱敏为 `***`(故意不含裸 `token`,避免误伤 `*_tokens` 计数),
  可传 `()` 关闭或自定义。
- **（H4.1）每次 LLM 调用的观测事件**：`call_llm` 现在每个 attempt 发
  `LLM_CALL_STARTED` / `LLM_CALL_COMPLETED`(按 `call_id` 配对),带 round/attempt/model、
  `duration_ms`(perf_counter)、成功/失败 + `error_type`、以及**本次调用**的
  token usage(区别于 `USAGE_UPDATED` 的累计值)——重试因此逐次可见。新增两个 payload
  与枚举值,顶层 re-export。OTel 桥接的基石(配合 H4.2 的 `ts`/`seq`)。
- **（H3.4）顶层 re-export LLM 契约**：`LLMService` / `LLMRequest` / `LLMResponse` /
  `LLMStreamChunk` / `LLMTokenUsage` / `OpenAICompatibleChatConfig` / `AnthropicChatConfig`
  现可 `from power_loop import …`(PROVISIONAL),写 `llm.*` hook 或自定义 `LLMService` 不必再
  伸进内部 transport 包。
- **（H3.6）`STABLE_API` 成为稳定层的单一事实源 + SemVer 守卫**：docstring 不再重复罗列(消除
  三方漂移),`FollowUpQueued` 归入 STABLE;新增测试校验 STABLE_API 与 `__all__`/模块属性一致,
  并冻结 v0 基线——未升 major 删除/改名 STABLE 符号即测试失败。
- **（H2 测试/CI 加固）覆盖率门禁 + 严格 marker + 示例冒烟 + 免-extras 导入腿**：
  CI 的 pytest 现在跑 `--cov=power_loop --cov=llm_client --cov-fail-under=70`(当前 72.6%);
  pytest `addopts` 加 `--strict-markers --strict-config`(typo marker / 未知 ini key 直接报错);
  新增 `tests/unit/test_examples_smoke.py`——逐个 import 全部 35 个 `examples/NN_*.py`,public API
  改名会让 CI 立刻红(语义校验仍留夜间真实 LLM);新增 CI job `import-without-extras`——只装核心
  (不装 `[openai]`/`[anthropic]`)后 `import power_loop` + 跑惰性导入测试,守住「零 SDK 可导入」。
  dev 依赖加 `pytest-cov` / `hypothesis`。
- **（H7 Phase 2）`Compactor` 协议加可选 `CompactionContext`——折叠前可联动记忆**：
  `maybe_compact` 现在可**选**接收 `context: CompactionContext`(暴露注入的
  `MemoryProvider` + `session_id` + 只读 `fetch_messages`),自定义压缩器可在折叠前把要点
  `remember` 进记忆,跨 session 留存。**向后兼容**:pipeline 按签名判断,只对接受 `context`
  的压缩器传(老签名压缩器照常工作);`DefaultCompactor` 忽略它,行为不变。新增
  `power_loop.runtime.compact.CompactionContext`(PROVISIONAL)。单测:签名内省门、
  context-aware 压缩器收到完整 context、**老签名压缩器仍可用**(两方向红前/绿后);真实 LLM
  示例 `examples/33_coordinating_compactor.py`(折叠时捕获的事实跨新 session 经 recall 存活)。
- **（H7 Phase 1）`recall_compacted` 工具——按需取回被压缩折叠的细节**：压缩把旧消息折叠成
  `compact_note` 并标 `compacted_out`,但原文**没删**(仍在 store 里)。新默认工具
  `recall_compacted(query?, from_seq?, to_seq?, limit?)` 让 agent 在摘要缺具体细节时把原文
  捞回来——**只读、仅当前会话**、按关键词/seq 过滤、按 `limit` 取最近若干条。属 `full` preset
  (也可 `include=["recall_compacted"]` 单挑)。8 个单测(过滤/空/会话隔离/排除 active/截断)+
  真实 LLM 示例 `examples/32_recall_compacted.py`(把编码埋进被折叠的轮次,用极小 summary 预算
  逼出工具调用)+ en/zh 文档。设计见 `docs/compaction-coordination-design.md`(H7 轨道)。
- **（H4.2）`AgentEvent` 增加 `ts` + 单调 `seq` 信封字段**：每个事件自动盖上墙钟时间与
  进程内单调序号（`itertools.count`，CPython 原子），从而可时间戳化、可全序化——这是
  OTel span 桥接与重建交错子代理/工作流事件流的基石。两字段排除在相等性之外，不定义事件身份。
- **（H3.3）发布 PEP 561 `py.typed` 标记**：下游 mypy/pyright 现在能看到 power-loop 的类型
  注解（此前整套带注解的 Public API 对类型检查器不可见）。
- 新增回归测试：`reap_runs` 并发 unlink、eager_wake 失败重挂、journal 终态冻结、
  AGENT_ERROR 终结事件、事件 `ts/seq`、provider 惰性导入、py.typed 装车、以及
  **默认 OpenAI 流式 transport 的单元测试**（`tests/unit/test_openai_transport.py`，
  把默认 provider 从仅夜间真实 LLM 覆盖提升为每-PR 覆盖，H2.2）。

### Changed

- **（H3.2，打包）`llm_client` 收编进 `power_loop._vendor`**：wheel 不再发布一个裸的顶层
  `llm_client` 包(消除与他人 PyPI 包/本地模块的命名抢注/冲突风险)——`top_level.txt` 现在只有
  `power_loop`。内部引用改走 `power_loop._vendor.llm_client.*`(包内仍用相对导入,无需动)。
  **若你此前直接 `from llm_client.interface import …`**:改用顶层 re-export
  `from power_loop import LLMRequest, LLMResponse, …`(H3.4);工厂类属内部,改用
  `create_llm_service_from_config` / `create_llm_service_from_env`。
- **（H3.5，安装方式）核心依赖瘦身为仅 `certifi`**：`socksio`(从未直接 import,httpx 按需
  传递)移除;`python-dotenv`(仅 examples/tests)移入 dev;`pyyaml` → `[skills]` extra
  (缺失时 `load_skill` 优雅降级、不报错)、`pypdf` → `[pdf]` extra(PDF 输入,懒加载);
  `[all]` 现含两家 transport + skills + pdf。删除 `requirements.txt`(pyproject 为单一事实源)。
  classifier 升 `4 - Beta`(H3.7)。
- **（H3.1）transport 惰性导入 + 可选 extras**：`anthropic` / `openai` 从硬依赖移入
  `[project.optional-dependencies]`；`power_loop.runtime.provider` 仅在真正构造对应 provider
  时才导入其 SDK。`import power_loop` 现在零 SDK 即可成功（featherweight 名副其实）。
  **安装方式变化**：请改用 `pip install 'power-loop[openai]'` / `[anthropic]` / `[all]`；
  缺失所选 SDK 时构造 provider 会抛出带安装提示的清晰 `ImportError`。README 同步更新。

### Fixed

- **（H5.1）绑定默认工具注册表会遮蔽外层注入的 `ShellBackend`/`Blackboard`**：`bind=True` 时
  `_bind_handler` 在调用期把 runtime_env 重置为「仅路径」快照(`shell_backend=None`),悄悄
  defeats 宿主在 `runtime_env_context(shell_backend=sandbox, …)` 里注入的沙箱/board。改为调用期
  **合并**:绑定的是路径(workspace/home/skills),而 ShellBackend/Blackboard 继承外层上下文
  (注册表若显式设置则其优先);`create_default_tool_registry` 新增 `shell_backend`/`blackboard`/
  `blackboard_id` 参数。
- **（H1.10 / C12）`close_session` 不清理 per-session 内存锁**：长生命周期 loop 轮换大量 session
  会按 session id 泄漏 `asyncio.Lock`。`close_session` 现在 pop 掉 `_locks` /
  `_follow_up_queue_locks` / `_follow_up_queues` 三个字典的对应键。
- **（H2.4 / C14，安全）`bash` 危险命令守卫漏过 `rm -rf /<系统目录>`**：写安全分支测试时发现
  `_dangerous_command_reason` 的 rm 正则只匹配**裸**根/家目录(`/` / `~` / `$HOME` 且后接
  空白或行尾),`rm -rf /etc`、`rm -rf /usr/local`、`rm -rf /var/lib` 这类**子路径删除全部漏过**
  (真实 false-negative)。改正则:阻断根/家目录及 `/(bin|boot|dev|etc|home|lib|opt|proc|root|run|sbin|srv|sys|usr|var)`
  系统目录(含子路径与 `~/…` / `$HOME/…`),同时仍放行 `/tmp` 与相对路径;flag 组可重复(`rm -r -f /x` 也拦)。
  新增 `tests/unit/test_bash_guards.py`(37 例):各阻断/放行命令 + `_validate_bash_command_scope`
  的家目录读写/allowlist/越界全覆盖(此前零覆盖)。
- **（H1.9 / C8）同步 SQLite 写阻塞事件循环 → 多会话互相拖死**：pipeline 的写路径
  store/sink 调用同步执行,某会话一次有竞争的写(`busy_timeout` 最高 5s)会卡住整个
  事件循环、拖住其它所有会话。修复:把**写路径** sink/store 调用(`on_message_appended` /
  `on_compaction` / `on_round_started` / `on_assistant_tool_calls` / `on_round_ended` /
  `bump_session_stats`)用 `asyncio.to_thread` 下放到线程(RLock 已保证线程安全);**读保持内联**
  (快、少竞争)。`NullSink` 无 I/O,跳过线程跳转零开销。新增 `tests/unit/test_store_offload.py`:
  阻塞写期间并发 ticker 仍推进(红前 ticker≈停摆 / 绿后照常);`StatefulAgentLoop` 并发文档
  同步更正。
- **（H1.7 / C6）同步 `publish()` 静默吞掉 async 订阅者异常**：有运行中的 loop 时,async
  handler 被 `loop.create_task` fire-and-forget,`suppress_subscriber_errors=False` 的
  re-raise 发生在脱离的 task 里,只剩一条 "Task exception was never retrieved" 的 GC 告警;
  且 task 未被引用,可被 GC。修复:保留 task(防 GC)+ done-callback 取回异常——未抑制时
  在 ERROR 级别大声记录(async 订阅者要内联处理异常请用 `publish_async`)。新增
  `tests/unit/test_event_bus_async.py`(保留→排空、抑制吞掉、未抑制记 ERROR;红前/绿后)。
- **（H1.2 / C2）`parallel`/`foreach` 在 `on_error="halt"` 下不取消在飞的兄弟分支**：
  `asyncio.gather(return_exceptions=False)` 首个失败即 re-raise,但**不取消**其它仍在跑的
  分支——它们继续烧真实 LLM 调用,迟到的 `record_step` 还能污染已 finalize 的 journal。新增
  `WorkflowEngine._gather_branches`:halt 时首个失败即 `task.cancel()` 其余兄弟、置
  `self._cancelled`、best-effort 翻 `self._cancel`,排空后再 re-raise;`continue` 行为不变。
  (journal 污染那半已由 H1.3 终态冻结堵住。)新增 `tests/unit/test_workflow_fanout.py`:
  parallel/foreach halt 取消兄弟 + 无遗留任务 + continue 仍收集全部错误(红前/绿后)。
- **（H1.1 / C1，最高严重度）记忆召回与压缩的 `_history_seqs` 错位 → 压缩标错 DB 行**：
  `_maybe_recall` 把 `memory_*` 消息直接插进 `pipeline.history`（绕过 sink），使
  `sink._history_seqs` 与 history 错位 `len(recalled)`，随后 `on_compaction` →
  `record_compaction` 按错位索引把**错误的行**标 `compacted_out`（静默、持久、会级联）。
  修复：新增 `sink.on_messages_inserted`，召回时为每条非持久化消息插入占位以保持
  index↔seq 对齐；并给 `on_compaction` 加 `expected_history_len` 对齐安全网——一旦
  映射失准（如 `SESSION_START` hook 整体替换历史，C9），**跳过本次压缩持久化**而非
  标错行（内存折叠照常，active 行不动，resume 仍正确）。新增 `examples/31_memory_with_compaction.py`
  （真实 LLM）+ 跨「有/无召回」等价性回归测试。
- **（H1.5）未捕获异常逃逸时既无 `SESSION_ENDED` 也无错误事件**：`pipeline.run()` 中
  raise 的 hook / sink / store I/O 直接抛出，看过 `SESSION_STARTED` 的订阅者被悬挂，且
  「文档声称会发」的 `AGENT_ERROR` 通道实为死代码。现在在调用点捕获 → 发 `AGENT_ERROR` +
  终结 `_finalize("error")`（`SESSION_ENDED`）→ 原样 re-raise；`_finalize` 改为幂等。
- **（H1.4）`eager_wake` 触发未跟踪的 follow_up 任务，可被 GC → 永久丢失父唤醒**：claim
  woke 后用裸 `create_task` 触发、句柄丢弃（CPython 仅持弱引用），且 woke 已 claim 会压制
  durable timer。现在保留任务引用，失败/取消时经 done-callback 重开 woke 并重挂 durable
  timer，父代理仍恰好被唤醒一次。
- **（H1.3）finalized journal 的迟到写回退**：孤儿叶子（`on_error="halt"`，见 H1.2）在 run
  终态后的 `record_step`/`update` 会用陈旧整 blob 把 `status` 退回 `running`、`result` 置空。
  journal 达到终态后冻结 status/result/steps（`record_step` 在写前重读最新 blob 并合并）；
  唤醒/resume 等正当写入用 `allow_terminal=True` 显式放行。
- **（H1.6）`reap_runs` 遇并发 unlink 会中止整轮 GC**：未保护的 `f.stat()` 在 worker /
  `delete_on_success` 并发删 db/WAL 时抛 `FileNotFoundError`，使后续所有 run 目录漏回收。
  改为逐文件 + 逐目录吞 `OSError` 并继续（对齐 `_remove_db`）。

## [0.13.1] — 2026-06-15

修复版本：一次系统性核心能力 bug 审计（5 个并行 agent）发现的全部 16 个问题，
逐一修复并补回归测试。无 Public API 破坏性变更，纯修复 + 一个新示例。

### Fixed

- **（BLOCKER）压缩越界孤儿 `tool` 消息**：`_compactable_span` 会把折叠终点回退越过
  尾部 `tool` 消息，在“工具回合后用户继续说话”时留下没有配对 assistant 的孤儿
  `tool`，导致下一次请求 HTTP 400。移除该回退逻辑（边界已由 `_expand_back_to_atomic`
  保证），旧测试方向写反（误判通过）也一并改正。
- **（CRITICAL）`SessionStore` 非原子写**：sqlite 以 `isolation_level=None`（autocommit）
  打开，`with self._conn:` 从不真正开启事务 → 多语句写入（如 `append_message` 推进
  `next_seq`）非原子。改为延迟事务（`isolation_level=""`）。
- **轮次上限收尾绕过统一调用路径**：达到 `max_rounds` 的总结调用直接 `llm.complete`，
  绕过取消 / 重试 / 超时 / 每-loop 模型 / 流事件。改为走 `call_llm`，并加预调用取消检查点
  与同主循环一致的 degrade 处理。
- **流事件不配对**：LLM 调用失败 / 重试耗尽 / 取消时只发了 `STREAM_STARTED` 没有终结
  事件，订阅者悬挂。`STREAM_COMPLETED` 改到 `finally` 发出，必然配对。
- **重试退避不可取消、且会溢出**：退避 `asyncio.sleep` 不响应取消（与文档承诺相悖）→
  改为分片轮询 token 的 `_cancellable_sleep`；`2**(attempt-1)` 对超大 `max_attempts`
  会在 `min()` 前就 `OverflowError` → 先把指数 cap 在 32。
- **定时器 stale 恢复重复触发**：一次比 `stale_firing_s`（默认 120s）更久的“live”投递
  会被周期性恢复扫描重新 arm 并二次触发。新增 `SessionStore.heartbeat_firing_timer`，
  投递期间后台心跳持续刷新 `firing` 行（节流为 stale 窗口的 1/4，可经新构造参数
  `heartbeat_interval_s` 覆盖）。
- **`grep` 在 rg 与 Python 回退路径下结果分叉**：rg 仅排除少数目录、且 root-anchored、
  又排在 include glob 之前（会被覆盖）；改为对每个 `_COMMON_SKIP_DIRS` 生成
  `!**/<dir>/**`（任意深度）并置于 include glob 之后（rg 后者优先），与回退一致。
  顺带修正截断计数（按实际展示的非空行计数）。
- **结构化输出**：`_extract_first_json_object` 遇到首个 `{` 之前的游离 `}` 会让深度变负
  从而拒绝合法 JSON；忽略深度 0 处的 `}`。文档漂移修正：声称的“单引号修复”代码从未
  实现（正则也无法安全实现），文档对齐为保守的尾逗号修复。
- **工作流**：`from_json` 接受重复 agent id（resume 时会重放错节点）→ 拒绝重复；
  `eager_wake` 会重复唤醒父 agent（绕过 `TIMER_FIRE` 使 wake-guard 无法去重）→ eager
  路径先 claim journal `woke`。
- **早退工具循环留下悬挂 `tool_calls`**：`TOOL_AFTER` BREAK、以及 `request_user_input`
  与其它工具同批时，未执行的工具调用没有配对 `tool` 结果 → 非法序列 / 幻影 pending；
  新增 `_resolve_skipped_tool_calls` 补合成 `[skipped]` 结果。

### Added

- **示例 `28_docker_shell_backend.py`**：通过 `ShellBackend` 缝把内置 bash 换成
  `docker exec`，模型写的 shell 在隔离容器内执行（真实 LLM 验证）。

## [0.13.0] — 2026-06-14

### Added

- **动态工作流（`power_loop.workflow`，可选子模块）**：声明式 `WorkflowSpec` JSON
  DSL（`agent`/`sequence`/`parallel`/`foreach`/`branch`，创建即严格校验、问题一次性聚合），
  确定性 in-process 引擎，叶子是普通子代理。
  - **detached 执行 + 完成回调唤醒主 agent**（`run_detached` + `register_wake_guard`，
    经 durable timer → `follow_up`），`SharedBudget` 跨子代理 token 池，
    LLM-facing `create_workflow` / `workflow_status` 工具。
  - **跨进程重启的编排级 resume**（`resume_run` / `resume_detached`）：journal 持久化
    spec + 每步 text/payload，重放已完成步、只重跑未完成尾；`foreach` 以 aggregate 原子
    重放；幂等 key（`run_id:node_id`）注入叶子 metadata。
- **进程外执行器（subprocess executor）**：`run_spec_isolated` + `WorkerBootstrap`
  （每个子代理独立 SQLite 库，依赖只从配置重建）、`SubprocessExecutor`（每叶一进程，
  插现有 `Executor` 缝；取消=SIGTERM→SIGKILL、超时/崩溃→failed→resume 重跑、
  子库保留/GC `cleanup_run`/`reap_runs`）、**`WorkerLauncher` 缝**（按叶子注入
  runsc/docker 等进程级沙箱，fail-closed）。
- **作用域共享黑板（`runtime.blackboard`）**：`Blackboard` 异步 Protocol + 默认
  `SqliteBlackboard`（新 `shared_state` 表，append + 按条目更新/删除）、`RuntimeEnv`
  新增 `blackboard`/`blackboard_id` 注入缝、通用 `board_*` 工具
  （`register_blackboard_tools`，kinds/statuses 由宿主定策略）。
- **config 可选离线 echo provider**（`provider="echo"`）：确定性、无网络，便于子进程/
  集成测试。

### Changed

- `AgentSpec` 新增 `output_schema`（→ provider `response_format` + `parse_structured`）；
  `AgentLoopConfig` 新增 `model` / `response_format`（**每子代理/工作流步可覆盖全局模型**）。
- `run_agent_spec` 现在**转发 `stop_event`**（协作式取消子代理）、**surface `result.usage`**，
  并发布此前一直未接线的 `SUBAGENT_*` 生命周期事件（带 `AgentEvent.source="subagent"`）。
- `MAX_SPAWN_DEPTH` 由硬常量改为**每 store 可配**（`SessionStore.open(max_spawn_depth=)` /
  `StatefulAgentLoop(max_spawn_depth=)`，默认仍 3）。
- `SessionStore` 新增 `shared_state` 表与 `get/set/delete_shared_state`（owner-keyed JSON，
  不绑定 session）。

## [0.12.0] — 2026-06-12

### Added

- **周期性定时任务（一等语义，创建时声明）**：`timers` 表新增 `interval_s` /
  `fire_count` / `last_fired_at`（旧库自动微迁移补列）。
  - `interval_s IS NULL` = 一次性（`firing → fired`）；设置 = 每次投递后
    `firing → armed`，`due_at = 触发时刻 + interval`（**fixed-delay**：
    停机期间漏掉的周期坍缩成一次，不会补发风暴）；`cancel` 是周期任务唯一出口。
  - 工具 `schedule_wakeup` 新增可选 `every_seconds`；`loop.schedule_timer`
    新增 `interval_s`；`list_wakeups` 显示周期与已触发次数。
  - hook 语义随之落位：SKIP = 跳过本次、周期照常排下次；BREAK = 终止整个周期。
- **真实 LLM 测试** `tests/real/test_real_timers.py`：模型自排唤醒并在被叫醒后
  执行 note（暗号复述）；周期 timer 连续两次真实投递后 cancel 终止。

## [0.11.0] — 2026-06-12

### Added

- **持久定时唤醒（durable timers）**：store 新表 `timers`（timer = 数据非任务，
  跨重启存活，随 session 级联删除）。
  - Agent 侧默认工具：`schedule_wakeup(delay_seconds, note)` /
    `list_wakeups` / `cancel_wakeup(timer_id)` / `current_time`（不在任何
    preset 里，按需 `get_tool_definitions(include=[...])` 注册）。
  - 宿主侧 API：`loop.schedule_timer(sid, delay_s=|due_at_ms=, note=)` /
    `loop.cancel_timer` / `loop.list_timers`（与工具写同一批行）。
  - **`TimerRunner(loop)`**：进程内扫描器——`start()` 时回收 stale `firing`
    行（at-least-once，可能二次投递），到期 CAS 认领后经 **`follow_up`**
    投递（空闲 = send，运行中 = 轮边界注入；进会话只有一条路）。
    不启动 runner（或外部调度器轮询 `store.due_timers()`）则永不触发。
  - **`HookPoint.TIMER_FIRE`** + `TimerFireCtx`：投递前编排否决点——
    CONTINUE 投递 / SKIP 跳过 / BREAK 取消 / `postpone_s` 改期，可改写
    投递文本；无 hook 默认投递。
  - 事件 `timer_fired`（`TimerFiredPayload`，outcome: delivered / queued /
    skipped / cancelled / postponed / error）。
  - `loop.hooks` / `loop.event_bus` 公开只读属性。
- 新示例 `examples/26_timers.py`；新增单测 `tests/unit/test_timers.py`（9 个）。

## [0.10.0] — 2026-06-12

### Added

- **`AgentLoopConfig.max_tokens_per_run`**：per-run 真实 token 预算护栏。轮边界
  检查（越界的那一轮完整结束，不留未决 tool_calls），命中后 status =
  `budget_exceeded`（新 LoopStatus 值），发 `status_changed`
  （`BudgetExceededStatusPayload`，kind="budget_exceeded"）。默认关闭。
- **Session 统计**：store 新表 `session_stats`（每次 send 结束累加一次：sends /
  rounds / llm_calls / tool_calls / prompt / completion / total tokens /
  first_send_at / last_send_at），随
  `close_session` 级联删除；新 API `loop.get_session_stats(sid)` /
  `loop.list_session_stats()`（`SessionStatsRow`）。注意 `usage_rounds` 表按
  (session, round_index) 覆盖、跨 send 不可累计——记账请用 session_stats。
- **`power_loop.contrib.logging_sink.attach_logging_sink(bus)`**：标准结构化
  日志 sink，每个事件一行 JSON（stdlib-only，长字段截断），消灭每个接入方
  重写"事件→日志"胶水的重复劳动。

- `AgentLoopResult.tool_calls` / `StatefulResult.tool_calls`：本次 run 执行的
  工具调用次数（`ContextManager.tool_calls` 计数）。

### Changed

- **同步工具 handler 现在跑在工作线程**（`asyncio.to_thread`，contextvars 正常
  传播）：慢的同步工具不再阻塞事件循环和同进程的其它 session。需要留在事件循环
  线程的 handler 请改 `async def`。
- `@phase` 装饰器发布的 start/end 事件改为携带 `PhaseEventPayload`（typed
  `data`）——至此**所有**内部事件发射路径都保证 `event.data` 非 None（新增契约
  测试）。
- README：补「一个 store 文件 = 一个进程」的多进程边界声明（跨进程并发安全
  暂不实现，先文档约束）。

## [0.9.0] — 2026-06-12

### Added

- **`StatefulResult.usage` / `AgentLoopResult.usage`**：每次 `send`/run 返回累计
  token 用量（对该 run 的全部 LLM 调用求和：`prompt_tokens` / `completion_tokens` /
  `cache_read_tokens` / `reasoning_tokens` / `total_tokens` / `calls`）。此前只能
  订阅 `usage_updated` 事件自行累加（事件是单次调用量、覆盖式），编排方做成本
  记账需要自建 tracker——现在直接读返回值。
- **`ContextManager.usage_totals`**：`update_usage` 在保持 `token_usage`（末次
  调用）语义不变的同时累计总量。
- **`send(..., heal_pending=True)`**（含 `send_sync`）：session 因上一个 run
  被杀死在 tool-call 中途而带有未决 `tool_calls` 时，自动 `abort_pending` 后
  继续本次 send，不再抛 `SessionPendingError`。默认仍为 raise（自愈会丢弃
  未完成的工具结果，应由调用方显式选择）。
- `SessionPendingError` 报错信息补充三条恢复路径指引（`resume` /
  `abort_pending` / `heal_pending=True`）。
- 文档：README 增加 token 记账与 heal_pending 小节、明确「无内置定时器」的
  范围边界；events 文档明确 `usage_updated` 为单次调用量、整 run 总量应读
  `result.usage`、handler 中 `event.payload` 按 dict 取值。
- 新示例 `examples/25_token_usage.py`；新增单测 `tests/unit/test_usage_and_heal.py`。

### Notes

- 本版本由 DeepTalk 多 agent 编排落地反推：token 成本面板需要 per-run 用量、
  人类中断 run 后 session 被未决 tool_calls 卡死，是两处真实暴露的不足。

## [0.8.1] — 2026-06-11

### Fixed

- `SQLiteNoteMemory.remember` 的签名对齐 `MemoryProvider` 协议（keyword-only
  `snapshot=` / `session_id=`）；0.8.0 的位置参数签名会在 session 结束时触发
  `MEMORY_FAILED`（软失败，不影响回复，但有噪音日志）。

## [0.8.0] — 2026-06-11

### Added

- **Agent-authored notes（自主记忆）**：模型用工具自己维护的持久笔记，存在 session store 新增的
  `notes` 表里（按 session 隔离，`close_session` 级联删除）。
  - 新默认工具 `note_add` / `note_update` / `note_delete`（进入 `full` preset；`core`/`explore` 不含）。
  - `SQLiteNoteMemory`：实现 `MemoryProvider` 协议，`recall()` 把该 session 的笔记渲染成一条
    `memory_notes` system 消息每次 send 注入；`remember()` no-op（写入由工具实时完成）。
  - `NotesPolicy(max_notes=50, max_note_chars=1000, inject_max_chars=8000, eviction="reject")`：
    默认**拒绝式**容量——满了报错并提示模型先删/合并（静默遗忘是 agent 记忆最糟的失败模式）；
    `eviction="fifo"` 切换为队列式淘汰（pinned 永不被自动淘汰）。注入超预算时优先隐藏最老的
    未 pin 条目并在文本中声明隐藏数量。
  - `AgentLoopConfig.notes_policy` 字段；顶层导出 `NotesPolicy` / `NotesFullError` /
    `SQLiteNoteMemory` / `DEFAULT_NOTES_POLICY` / `render_notes`。
  - `SessionStore` 新方法：`add_note` / `update_note` / `delete_note` / `list_notes` / `count_notes`，
    `NoteRow` dataclass。旧数据库自动建表，完全向后兼容。
  - 新示例 `examples/24_agent_notes.py`；单测 `tests/unit/test_notes.py`（18 例）。

## [0.7.2] — 2026-06-11

### Fixed

- Export `runtime_env_context` from the top-level package so the documented `bind=False` flow works without an internal import.
- Forward `tools=` and `system_prompt=` through `send_sync()` and idle `follow_up_sync()` as well as the async APIs.

### Docs and tests

- Document per-call overrides, unbound registries, and `ShellBackend` in the English and Chinese guides/API reference.
- Extend example 23 with a real unbound-registry invocation across two runtime workspaces.
- Add regression coverage for sync overrides and runtime resolution of unbound handlers.

## [0.7.1] — 2026-06-11

### Docs

- New **example 23** (`examples/23_per_send_overrides.py`) demonstrating per-call `tools=` allowlisting and `system_prompt=` override, plus a README "Per-call overrides" section. No code changes vs 0.7.0.

## [0.7.0] — 2026-06-11

### Added — Per-call overrides & cleaner public surface

- **`StatefulAgentLoop.send(..., tools=, system_prompt=)`** and **`follow_up(..., tools=, system_prompt=)`** — per-call overrides that do not mutate loop/session state. `tools` accepts a sequence of tool names (allowlisted from the loop registry) or a `ToolRegistry`; the model only *sees* the permitted subset. `system_prompt` overrides for that run only (precedence: per-call > session > config). Enables multi-tenant reuse of one cached loop without runtime hook gating.
- **`ToolRegistry.subset(names)`** and **`ToolRegistry.names()`** — derive a restricted registry.
- **`create_default_tool_registry(..., bind=False)`** — build an **unbound** registry whose handlers read the current `RuntimeEnv` at call time (caller supplies it per call via `runtime_env_context`); no eager workspace requirement. `DEFAULT_TOOL_HANDLERS` is now part of the public API.
- **`ShellBackend.session_key(workspace_dir)`** — the persistent `BashSession` is now cached by the backend's execution-target key, so swapping backends (e.g. local ↔ sandbox, or distinct sandbox containers) no longer needs ad-hoc rebuilds.

### Fixed

- Follow-up dropped on a terminal round (0.6.0) — see below; plus `__init__` export/lint hygiene (`FollowUpQueued`, `DEFAULT_TOOL_HANDLERS` now exported).

### Docs

- README: explicit **"orchestration, not isolation"** scope note — built-in `bash`/file tools run in-process and are not a security boundary; sandbox via the `ShellBackend` seam.

## [0.6.0] — 2026-06-11

### Fixed — Follow-up on a terminal round

- A follow-up enqueued during an otherwise-terminal round (model returned a final answer with no tool calls) was never drained — the queue only drained at the *next* round start, which never came. The loop now drains pending follow-ups before completing and runs another round to address them, so absorbed steering input is always processed.

## [0.5.0] — 2026-06-11

### Added — Pluggable shell backend

- **`runtime.exec_backend`** (`ShellBackend` protocol, `LocalShellBackend`, `DEFAULT_SHELL_BACKEND`) and **`RuntimeEnv.shell_backend`** — host code can route the persistent shell into an isolated sandbox (e.g. `docker exec`) instead of an in-process `/bin/bash`, without changing tool implementations. Default behavior unchanged.

## [0.4.1] — 2026-06-08

### Added — In-flight steering (`follow_up`)

- **`StatefulAgentLoop.follow_up()` / `follow_up_sync()`** — enqueue steering input while a session run is in flight; idle sessions degrade to `send()`.
- **`FollowUpQueued`** — immediate return shape when input is queued for the next pipeline round.
- **Round-boundary drain** — merged follow-ups append as a wrapped `<follow_up>` user message before `prepare_round`.
- **Example 22**, bilingual docs, and unit/real tests for the steering path.

### Added — M2.8 Anthropic Messages API 传输（2026-06-06）

- **`AnthropicMessagesLLMService`**（`llm_client.anthropic_factory`）—— 新增原生 Anthropic Messages API transport，复用统一 `LLMRequest` / `LLMResponse`。
- **`LLMProviderConfig.provider` 成为路由键**：`provider="anthropic"` / `"claude"` / `"dashscope-anthropic"` 使用 Anthropic transport；其它 provider 仍使用 OpenAI-compatible transport。
- **消息转换**：OpenAI-style `tool_calls` → Anthropic `tool_use` blocks；`role="tool"` → `tool_result` blocks；返回的 `tool_use` 统一转回 `LLMResponse.tool_calls`，pipeline 无需分支。
- **测试配置**：real LLM helper 改为 `create_llm_service_from_env()`，支持 `POWER_LOOP_*` 与 legacy `OPENAI_COMPAT_*` 两组环境变量。
- **版本**：`power_loop.__version__ = "0.4.0"`。

### Public API（M2.8 新增）

`AnthropicChatConfig` / `AnthropicMessagesLLMService` 可从子模块导入；顶层 `LLMProviderConfig` 的 `provider` 字段现在影响 transport 路由。

### Changed — M2.7 显式 Session 创建（2026-06-06）

- **`StatefulAgentLoop.new_session(metadata=None, system_prompt=None) -> str`** —— 新增显式会话创建入口。调用方先拿到 `session_id`，再传给每次 `send()` / `send_sync()`。
- **Breaking**：`StatefulAgentLoop.send(user_input, session_id, *, stop_event=None)` 与 `send_sync(...)` 现在必须传入 `session_id`；不再在首次 `send()` 时隐式创建 session。
- **Breaking**：`metadata` 从 `send(metadata=...)` 移到 `new_session(metadata=...)`。这样会话级信息在会话创建时固定，避免首条消息和会话生命周期耦合。
- **文档 / 示例 / 测试**：README、双语 docs、examples、unit/real 测试全部改为 `sid = loop.new_session(); await loop.send(..., session_id=sid)`。
- **版本**：`power_loop.__version__ = "0.3.0"`。

### Public API（M2.7 变更）

`StatefulAgentLoop.new_session` 顶层入口新增；`StatefulAgentLoop.send / send_sync` 签名破坏性变更，`session_id` 必填。

### Added — M1.1 LLM 重试 / 超时 / 取消（2026-06-05）

- **`LLMRetryPolicy`**（`power_loop.runtime.retry`）—— 配置 `max_attempts` / `backoff_initial` / `backoff_max` / `total_timeout` / `retry_on`。指数退避（capped），跨所有 attempt 共享总超时；退避 sleep 是 cancel-aware 的（cancel 触发时不会傻等到底）。
- **`with_retry(call, *, policy, token, on_retry=None)`** —— 库内通用 helper，pipeline 用它包 `await self.llm.complete(...)`。``CancellationRequested`` / ``asyncio.CancelledError`` 直接透传，不会被吞。
- **`CancellationToken`**（`power_loop.runtime.cancellation`）—— 统一 cancel 形状：`from_any(...)` 接受 `asyncio.Event` / `threading.Event` / `Callable[[], bool]` / 已存在的 token / `None`。自带 owned 模式（`token.cancel(reason)`），供 hook `HookDirective.CANCEL`（M1.5）和外部 controller 使用。`is_cancelled()` 对用户 callable 抛出做容错（**绝不让 cancel 检查本身污染主循环控制流**）。
- **`AgentLoopConfig.retry_policy: LLMRetryPolicy | None = None`** —— 默认 None（保持现有 fail-fast 行为）；显式赋值即开启。
- **新事件**：
  - `AgentEventType.LLM_RETRY_ATTEMPTED` + `LlmRetryAttemptedPayload(attempt, max_attempts, error_type, error_message, next_sleep_seconds)`
  - `AgentEventType.LLM_DEGRADED` + `LlmDegradedPayload(reason, attempts, error_type, error_message)` —— `reason ∈ {"retry_exhausted", "timeout"}`
  - `AgentEventType.LOOP_CANCELLED` + `LoopCancelledPayload(reason, round_index)`
- **新错误**（`power_loop.contracts.errors`，全部 `PowerLoopError` 子类）：
  - `LLMTimeout(elapsed, attempts, total_timeout)`
  - `LLMRetryExhausted(attempts, last_error)`（`__cause__` 保留 last error）
  - `CancellationRequested(reason)`
  - `CompactionFailed`（M2.5 占位）
- **`LoopStatus`** 新增 `"degraded"`。Pipeline 在 `call_llm` 抛 `LLMRetryExhausted` / `LLMTimeout` 时：append 一条合成的 `assistant` 消息（`[degraded: …]`），emit `LLM_DEGRADED`，`status="degraded"` 返回。`CancellationRequested` 翻译为 `status="cancelled"` + `LOOP_CANCELLED`。
- **Pipeline 内部统一**：`stop_event` 仍接受任意 cancel-like 对象（API 向后兼容），但内部统一存为 `CancellationToken`；`StatefulAgentLoop.send / send_sync / resume` 的 `stop_event` 类型放宽为 `CancellationLike`。
- **测试**：`tests/unit/test_retry_cancel.py`（12 个，覆盖 `with_retry` 直测 + token 各形态 + pipeline 端到端三条路径）；`tests/real/test_real_retry.py`（2 个真实 LLM 集成 —— 注入 transient 失败后真实 complete 通；全失败走 degraded 不打真实网络）。

### Public API（M1.1 新增）

`LLMRetryPolicy` / `with_retry` / `CancellationToken` / `CancellationLike` / `LLMTimeout` / `LLMRetryExhausted` / `CancellationRequested` / `CompactionFailed` / `LlmRetryAttemptedPayload` / `LlmDegradedPayload` / `LoopCancelledPayload` 全部从 `power_loop` 顶层导出；`AgentEventType.LLM_RETRY_ATTEMPTED` / `LLM_DEGRADED` / `LOOP_CANCELLED` 已加入枚举。

### Added — M1.9 MemoryProvider 协议（2026-06-05）

> 库内**零实现**：定义协议 + 接线 + 注入位置不变量。具体后端（SQLite / HTTP API / 向量库）一律留在调用方或 `examples/`。

- **`MemoryProvider` Protocol**（`power_loop.runtime.memory`）—— 两个方法：
  - `async recall(*, messages, session_id, budget_tokens=1500) -> list[dict]`
  - `async remember(*, snapshot: MemorySnapshot, session_id) -> None`
- **`MemorySnapshot`** dataclass —— `session_id / messages / final_text / rounds / status / metadata`，在 SESSION_END 时传给 `remember`。
- **`tag_as_memory(messages)`** —— 工具函数，把任意 dict 列表规范化成 `role=system, name=memory_*`。Pipeline 在注入前自动调用，业务方不必关心。
- **`AgentLoopConfig.memory: MemoryProvider | None = None`** + **`memory_budget_tokens: int = 1500`**。默认 None（保持原有行为）。
- **注入位置不变量**：召回结果插在 ``self.history`` 的「最长 leading `role=system` 段」之后、对话历史之前。这与 `compact_note` 同区，受压缩器系统区保留保护。
- **失败模型**（库强制不破坏主流程）：
  - `recall` 抛 → 视为返回 `[]`，emit `MEMORY_FAILED(phase="recall")`，loop 照常跑。
  - `remember` 抛 → emit `MEMORY_FAILED(phase="remember")`，`StatefulResult` 原样返回。
- **新 hook**：`HookPoint.MEMORY_RECALLED` + `MemoryRecalledCtx(recalled, session_id, budget_tokens)`。业务可在注入前 redact / 去敏 / `directive=SKIP` 跳过整批注入（典型场景：双方授权 gate）。
- **新事件**：`AgentEventType.MEMORY_RECALLED` + `MemoryRecalledPayload(returned, injected, budget_tokens)`；`AgentEventType.MEMORY_FAILED` + `MemoryFailedPayload(phase, error_type, error_message)`。
- **Pipeline 内部**：`_finalize` 多了一个 `rounds` 形参，使 `MemorySnapshot.rounds` 正确反映已完成回合数；老调用点（cancelled 早出）保留默认行为。
- **测试**：`tests/unit/test_memory.py`（6 个，覆盖注入位置 + tag 规范化 + recall 软失败 + remember 软失败 + 快照内容 + MEMORY_RECALLED SKIP）。
- **example**：`examples/13_memory_sqlite.py` —— SQLite 事实 KV，跨 session 把「我叫阿岚 / 喜欢 37」记忆带回。

### Public API（M1.9 新增）

`MemoryProvider` / `MemorySnapshot` / `tag_as_memory` / `MemoryRecalledCtx` / `MemoryRecalledPayload` / `MemoryFailedPayload` 顶层导出；`HookPoint.MEMORY_RECALLED`、`AgentEventType.MEMORY_RECALLED` / `MEMORY_FAILED` 入枚举。

### Added — M1.3 结构化输出（2026-06-05）

- **`LLMRequest.response_format: dict[str, Any] | None = None`** —— OpenAI 兼容 `response_format` 字段；`llm_factory._request_kwargs` 与 `_build_resume_request` 透传。
- **`StructuredOutputSpec(name, schema, strict=True, description=None, examples=...)`**（`power_loop.runtime.structured`）—— 声明式包装；`.to_openai_response_format()` 渲染成 `{"type":"json_schema","json_schema":{name, schema, strict, description}}`。
- **`parse_structured(output, *, schema=None) -> dict`** —— 四级修复链：
  1. 直接 `json.loads`
  2. markdown ```json``` 围栏剥离
  3. 抓出第一个**括号平衡**的 `{...}` 子串（跳过字符串里的引号）
  4. 修补**尾逗号** `,]` / `,}`
- **`StructuredOutputError(reason, raw_text, detail)`** —— 失败原因机器可读：`no_json` / `invalid_json` / `not_object` / `missing_required:<field>`，`raw_text` 截断到 1000 字符方便调试。
- **本地 schema 校验有限**：仅强制 `type=="object"` 与顶层 `required` 字段存在。更深的 type / enum / pattern 留给 provider 在 strict mode 服务端校验，**避免本地实现与 provider 静默分歧**。
- **测试**：`tests/unit/test_structured.py`（14 个）+ `tests/real/test_real_structured.py`（1 个真实 LLM —— card 抽取来回跑通）。
- **example**：`examples/14_structured_card.py` —— 真实 LLM 抽取 → 修复带噪 JSON → schema 缺字段失败三段。

### Public API（M1.3 新增）

`StructuredOutputSpec` / `parse_structured` / `StructuredOutputError` 顶层导出。`LLMRequest.response_format` 已在 `llm_client` 层落地。

### Added — M1.6 ToolRegistry async-handler 工效学（2026-06-05）

- **`async def` 自动识别**：`ToolRegistry.register()` 用 `inspect.iscoroutinefunction` 在登记时缓存 `RegisteredTool.is_async`，覆盖普通 `async def` 与 `async __call__` callable 对象两种形态。
- **`invoke()`（sync）对 async 处理器抛 `AsyncToolInSyncContext`**：取代之前「silently 返回未 await 的 coroutine」的隐式坑，错误信息明确指向 `invoke_async`。
- **`invoke_async()` 是通用入口**：async handler 直接 `await tool.handler(...)`；sync handler 跑完后若返回 awaitable 仍会被自动 await（保留向后兼容）。
- pipeline 早已用 `invoke_async`，业务侧无需改动；只有显式调 `invoke()` 把 async 当 sync 用的旧代码会立刻看到清晰报错。
- ROADMAP 里提到的 `tests/real/test_real_streaming_subagent.py` 的 `get_event_loop().is_running()` hack 已在 stateful 重构时随旧测试一并删除，本次 polish 把上游 API 工效学也补齐。
- **测试**：`tests/unit/test_tool_registry_async.py`（7 个）—— async 检测、callable 对象、sync-on-async 报错、双形态 invoke_async、sync-returning-awaitable 兼容。

### Public API（M1.6 新增）

`AsyncToolInSyncContext` 顶层导出。`ToolRegistry.invoke / invoke_async` 行为变更（前者更严格，对 async handler 抛清晰错；后者更优雅，省一次 `inspect.isawaitable`）—— 既有调 `invoke_async` 的代码完全不受影响。

### Added — M1.4 LLMProviderConfig 统一（2026-06-05）

- **`LLMProviderConfig`**（`power_loop.runtime.provider`）—— provider-agnostic 配置：`base_url` / `api_key` / `model` 必填，`provider` 标签（informational，今天只走 openai-compatible 一条 transport，预留 M3 多 transport 路由 key），加 `timeout_s` / `max_tokens` / `temperature` / `max_retries` 等默认值。
- **`LLMProviderConfig.from_env(prefix="POWER_LOOP", fallback_prefix="OPENAI_COMPAT", env=None)`** —— 读 `POWER_LOOP_*` 环境变量，缺则回退 `OPENAI_COMPAT_*`，**老 `.env` 无须改字段**。`env` 形参用于测试（注入 dict）。
- **`create_llm_service_from_config(cfg)` / `create_llm_service_from_env(*, prefix=…)`** —— 一行造服务；内部通过 `to_openai_compatible()` 适配现有 `OpenAICompatibleChatLLMService`。
- **失败模式**：必填字段缺失 → 构造时 `ValueError`（不是首个 `complete()` 时），让配置错误在 pytest 阶段就暴露。
- **docs/providers.md** —— 环境变量表 + 4 个 provider snippet（OpenAI / DashScope / DeepSeek / 本地 OpenAI-compatible）+ 老调用方式迁移指引。
- **测试**：`tests/unit/test_provider.py`（11 个）—— 必填守卫 / 主前缀 / 回退前缀 / 主前缀优先 / 三家 provider 参数化建造 / `from_env` 一行入口 / `to_openai_compatible` 适配回环。

### Public API（M1.4 新增）

`LLMProviderConfig` / `create_llm_service_from_config` / `create_llm_service_from_env` 顶层导出。`OPENAI_COMPAT_*` 环境变量名继续可用，仅为回退；新代码请用 `POWER_LOOP_*`（或自定义 prefix）。

### Added — M1.2 trim_history（2026-06-05）

- **`trim_history(messages, max_tokens, *, keep_system=True, keep_last_n=2)`**（`power_loop.runtime.budget`）—— 纯裁剪 helper：保留 leading system + 最后 N 个 user-bounded 交换，从中间删消息直到落在 token 预算内。不调 LLM（不摘要），仅是业务侧调用前裁剪。
- **不变量**：
  1. 预算已够 → 返回原 list（不复制）。
  2. `keep_system=True` → 所有 leading `role=system` 消息保留；`keep_last_n` 个 user-bounded 交换在尾部保留。
  3. `assistant(tool_calls) ↔ tool(tool_call_id=...)` 对永不拆分 — 裁剪边界通过 tool_call_id 配对检测自动调整。
  4. 当 system + tail 都放不下时，降级为 tail-only（丢 system）再按需从尾部裁剪。
  5. 不修改输入（返回新 list）。
- **测试**：`tests/unit/test_budget.py`（9 个）—— 已合预算 / 空 / 零预算 / 系统保留 / 去系统 / 工具对原子性 / 工具对在边界 / 仅 tail / 非突变。
- `estimate_tokens` / `estimate_text_tokens` / `trim_history` 从 `power_loop` 顶层导出。

### Public API（M1.2 新增）

`trim_history` / `estimate_tokens` / `estimate_text_tokens` 顶层导出。

### Added — M2.5 错误体系收口（2026-06-05）

- **`ToolNotFound(tool_name)`** —— `ToolRegistry.invoke / invoke_async` 找不到工具时 raise。
- **`ToolValidationError(tool_name, message)`** —— 参数校验失败时 raise。
- **`SpecValidationError(message, *, field=None)`** —— 新的规范验证错误；`AgentSpecError` 现继承于它（而它继承 `PowerLoopError`），旧代码 `except AgentSpecError` 继续有效，新代码 `except SpecValidationError` 或 `except PowerLoopError` 一把抓。
- `ToolRegistry.invoke / invoke_async` 对 unknown tool 和 invalid args 现在 **raise 异常而非 return 字符串**；`invoke` 也 raise 而非 return 字符串作为错误。Pipeline 的 `execute_tool` 内部 catch 这两个异常并返回 `(str(exc), True)` 给 LLM 看到（保持向后兼容）。
- **`ToolRegistry.validate` 保留为 internal legacy**（仍返回 `str | None`），管线仍用它做第一层检测，但新代码应直接 invoke + catch。
- 所有 `__init__.py` 和 `__all__` 同步更新。

### Added — M2.1 Public API 稳定性约定（2026-06-05）

- **README §5** 新增 "Public API 稳定性约定" 节：**STABLE**（24 符号，跨 minor 保证兼容 + CHANGELOG 独立条目）、**PROVISIONAL**（顶层导入但 0.x 可调）、**INTERNAL**（无版本承诺）。与 `power_loop/STABLE_API` 元组同步。
- Examples 表补齐 11–14（persistence / retry-cancel / memory-sqlite / structured-card）。
- §7 环境变量节更新为 `POWER_LOOP_*` 优先 + `create_llm_service_from_env()` 一行法。

### Added — M2.2 Hook/Event 全表文档（2026-06-05）

- **`docs/hooks.md`** §3.9 新增 `memory.recalled` hook 点文档（Ctx 字段 / SKIP directive / 双方授权示例）。
- **`docs/events.md`** 新增 §2.7 LLM retry/cancel lifecycle（`llm_retry_attempted` / `llm_degraded` / `loop_cancelled`）和 §2.8 Memory（`memory_recalled` / `memory_failed`），含完整 payload 字段表、触发时机、典型订阅者。

## [0.2.0] — 2026-06-05

Stateful refactor. The library now revolves around `StatefulAgentLoop` and a SQLite-backed `SessionStore`; the stateless `AgentLoop` is removed. **Hard break — no compatibility shim.**

### Added

- **`StatefulAgentLoop`** — the only public entry point. `new_session()` / `send(user_input, session_id)` / `send_sync` / `resume(sid)` / `abort_pending(sid)` / `close_session(sid, cascade=True)` / `close()` / `get_messages(sid)` / `get_pending(sid)`. Per-session `asyncio.Lock` so one instance can drive any number of sessions concurrently.
- **`SessionStore`** (`power_loop.runtime.session_store`) — SQLite-backed, the **only** thing that writes to disk. Five tables: `sessions` / `messages` / `compactions` / `usage_rounds` / `session_state`. Single connection + `threading.RLock`; WAL + busy_timeout. Public API surface for sessions, messages, compactions, usage, lifecycle.
- **`MessageSink`** Protocol + `NullSink` + `SQLiteSink` — pipeline persistence hook. SQLiteSink owns the in-memory `_history_seqs` list that mirrors `pipeline.history` so the compactor can translate fold indices back to store rows.
- **Pending state machine** — `assistant(tool_calls)` falling-into-store immediately marks `session_state.pending`; each matching `tool` message clears it. Mid-tool crash leaves a recoverable state. Next `send` raises `SessionPendingError`; caller picks `resume()` (replay remaining tools) or `abort_pending(sid, reason=…)` (synthesize `<aborted>` tool messages).
- **Subagent on top of `SessionStore`** — `spawn_agent` rewritten as a thin shell over the shared store. Children get their own row with `parent_session_id` / `spawn_tool_call_id` / `spawn_depth` (`MAX_SPAWN_DEPTH=3` enforced at insert time).
- **`AgentSpec`** (`power_loop.runtime.spec`) — strict-schema declarative subagent: `name / system_prompt / tools / max_rounds / max_tokens / temperature / model / lifecycle / metadata`. Unknown fields → `AgentSpecError`. `from_dict` / `from_json` factories.
- **`run_agent` meta-tool** — declarative companion to `spawn_agent`. The parent LLM submits a full `AgentSpec` JSON; the library validates and dispatches via `run_agent_spec`. Both meta-tools registered by a single `register_spawn_agent(registry)`.
- **`SubagentLifecycle`** enum — `EPHEMERAL` (default, deleted on success, preserved on failure for debug) / `LINKED` (cascade-deleted with parent) / `DETACHED` (independent of parent's lifecycle).
- **`Compactor`** Protocol + **`DefaultCompactor`** (`power_loop.runtime.compact`) — pluggable LLM-summary compaction. Trigger at `max_tokens × trigger_ratio` (default 0.75) or absolute `CONTEXT_COMPACT_THRESHOLD`. Preserves all `role=system`, last `keep_last_n` user-bounded exchanges, and the `assistant(tool_calls) ↔ tool` atomic pair. Soft-fails to `None` on summary errors so the loop degrades gracefully.
- **`runtime/budget.py`** — `estimate_tokens(messages)` heuristic (≈4 chars/token, stdlib-only) used by the compactor's trigger logic.
- **Error hierarchy** — `PowerLoopError` base + `SessionNotFoundError` + `SessionPendingError(session_id, assistant_seq, pending_tool_calls)`. Caller catches the base class to handle every library-raised exception.
- **`_current_loop` contextvar** (`power_loop.core.agent_context`) — threads the active `StatefulAgentLoop` through tool invocations so meta-tools like `spawn_agent` find their parent without ambient state.
- **Examples 00–05** — progressive tutorial: minimal send → multi-turn → tool calling → subagent → compaction → pending recovery. Each file introduces exactly one new concept. `examples/_helpers.py` shares `.env` loading + LLM construction.
- **Real-LLM test suite** — `tests/real/test_real_stateful_loop.py` / `test_real_tool_use.py` / `test_real_subagent.py` / `test_real_pending_resume.py` / `test_real_compaction.py` / `test_examples.py` (6 examples). `tests/real/judge.py` provides an **LLM-as-judge** helper: tests assert `await assert_passes(question, answer, rubric)` and a separate power-loop evaluator returns `{passed, reason}` JSON, solving the LLM-non-determinism assertion problem.

### Changed — Breaking

- **Removed `AgentLoop` and `agent_loop_async`**. Replace `AgentLoop(llm, config).run(messages=…)` with `StatefulAgentLoop(llm=…, db_path=…, config=…).send(user_input, session_id=…)`. The stateless model is gone — callers no longer ship the full messages list per turn; power-loop loads history from the store.
- **`AgentLoopConfig.compactor: Compactor | None = DefaultCompactor()`** — default-on. Pass `None` to disable.
- **`CompactBeforeCtx`** loses `input_tokens` and `compact_threshold` — neither carried useful data after the runtime/compact.py rewrite. **`AutoCompactStatusPayload`** trades them for `before_tokens` / `after_tokens` (sourced from the `CompactionPlan`).
- **`ContextManager`** loses `compact_async` / `should_compact` / `compact_threshold` / `last_input_tokens` / `_compact_count` / `reset_usage`. It now owns only `update_usage` (telemetry parsing), `microcompact` (large tool-output spill-to-disk), `recent_files`, and `TodoManager`. LLM-summary compaction has moved to `runtime/compact.py`.
- **`power_loop.contracts.errors`** is now a real module (was unused).

### Removed

- `power_loop/agent/loop.py` — `AgentLoop` shell.
- `power_loop/core/agent.py` — `agent_loop_async` entry point.
- Six stale integration / real tests that depended on the removed API; replaced by the new `tests/real/test_real_*.py` suite + the 6 example-driven tests.

### Migration

| Before (0.1.x) | After (0.2.0) |
|---|---|
| `AgentLoop(llm, config).run(messages=[…])` | `sid = loop.new_session(); await loop.send(user_input, session_id=sid)` |
| Caller manages `messages` list | Library loads from `SessionStore` by `session_id` |
| No persistence | `db_path` (default `./power_loop_sessions.db`); `":memory:"` for tests |
| No pending detection | Crash mid-tool → next `send` raises `SessionPendingError`; pick `resume()` or `abort_pending()` |
| Compaction via `ContextManager.compact_async` | `AgentLoopConfig.compactor = DefaultCompactor()` (default-on); pluggable via `Compactor` protocol |
| `spawn_agent` with private `AgentLoop` | `register_spawn_agent(registry)` + `run_agent` meta-tool; shared `SessionStore` with parent linking |
| No declarative subagent | `AgentSpec` + `run_agent_spec(spec, input, parent_loop=…)` |

### Documentation

- README rewritten around `StatefulAgentLoop`. Sections: 1. what it is/isn't · 2. install · 3. quickstart (mirrors examples 00→05) · 4. core concepts (Session / SessionStore / Sink / Compactor / Pending / Subagent / Hooks vs Events) · 5. flat API reference · 6. examples table · 7. env-var config · 8. pipeline ASCII trace + persistence/seq notes + pending state machine · 9. tests (including the LLM-as-judge pattern) · 10. roadmap pointer.
- `docs/hooks.md` — every `HookPoint` with its typed Ctx fields + accepted directives + typical use cases.
- `docs/events.md` — every `AgentEventType` with its payload fields + when fired + typical subscriber.

### Added — M0 工程化基线（2026-06-05）
- `power_loop.__version__ = "0.1.0"` 单一来源 + `STABLE_API` 元组声明稳定面（`AgentLoop / AgentLoopConfig / AgentLoopResult / AgentHooks / AgentEventBus / HookPoint / HookDirective / ToolRegistry / ToolDefinition`）。
- `pyproject.toml` 补 license / classifiers / urls / dynamic version；新增 dev extras (`ruff` / `mypy`)；统一 `ruff` (line=120, E/F/I/UP/B) 与 `pytest` marker (`unit` / `integration` / `real_llm`)。
- `CHANGELOG.md` 与 `ROADMAP.md` 落地（M0–M3 + 关键短板拆解）。
- `tests/` 重排为 `unit/ integration/ real/`；`tests/conftest.py` 实现 `real_llm` 默认 ON、`--no-real` 反向跳、缺 env 自动 skip；`tests/real/conftest.py` 自动给 `tests/real/*` 打 marker。
- GitHub Actions：`ci.yml`（ruff + mypy 非阻塞 + pytest --no-real，3.10/3.12 矩阵）+ `real-llm-nightly.yml`（凭 repo secrets 跑真实 LLM）。
- `examples/00_minimal.py` —— DeepTalk Agent MVP 同款单回合用法；`tests/real/test_examples.py` 把 example 作为活文档锁定。
- README 新增 "实现亮点 / 卖点"（含 M1.7a 压缩策略、M1.9 记忆分层）与 STABLE / PROVISIONAL / INTERNAL 三层稳定性约定。

### Changed
- `ruff --fix` 自动现代化（PEP 585/604、import 排序、unused import 清理）共 458 处；剩余 28 处为真实代码债，按规则码登记到 `pyproject.toml` ignore，统一 M1 接触模块时清理。

### Fixed — M0.8 / M0.9（同日）
- ruff 残留 28 → 0：清掉 19 处 `__init__.py` E402（`STABLE_API` 重排到 imports 之后）；修真实 bug `llm_client/llm_factory.py:855` 未定义的 `e`；修闭包变量延迟绑定（B023）；`Union[…]` → `X | Y`；`zip(...)` 加 `strict=`；其余移除未用变量与 E402。
- mypy 残留 32 → 0：`LLMRequest.messages/tools` 类型放宽为 `list[dict[str, Any]]`（list 不变性问题）；`default_tools.py` 4 处隐式 Optional 改 `int | None` / `str | None`；Popen None 守卫；3 处 var-annotated；`pipeline.execute_tool` 加 ToolRegistry None 守卫；`phase.py` event_meta 显式 `dict[str, Any]`；`system_prompt.py` lambda 改 def 以保留类型推导。
- CI 中 mypy 改为阻塞（不再 `|| true`）。

### Known limitations
- M1 起的所有短板（重试 / 取消 / 压缩 / 持久化 / 记忆 / spec sub-agent / 动态工具）仍未实现。

## [0.1.0] — 2026-06-05

首个基线版本（M0 起点的现状快照）。

### Added
- `AgentLoop` 主外观类 + `AgentLoopConfig` / `AgentLoopResult`。
- `AgentPipeline`：按 phase 拆分的主循环，含 LLM 调用 / 工具调用 / 压缩 hook 点。
- `AgentHooks`：有序 sync/async hook 管理，支持 typed Ctx 与 legacy dict 两种回调。
- `AgentEventBus`：事件订阅 / 发布，订阅者错误隔离。
- `HookPoint` 15 个：`session.* / round.* / llm.* / tools.* / tool.* / compact.* / message.append`。
- `HookDirective`：`CONTINUE / SKIP / BREAK / SHORT_CIRCUIT` 控制流。
- `ToolRegistry`：动态工具注册 + JSON Schema 必填校验 + OpenAI tool schema 导出。
- 默认工具集：`run_bash / run_read / run_write / run_edit / run_grep / run_glob / apply_patch`，含路径白名单。
- `spawn_agent` 工具：命令式子代理，含深度守卫与父总线事件冒泡。
- `llm_client/`：OpenAI 兼容（含 DashScope）+ Anthropic 双家底；多模态、web search、tool-call 解析。
- `SystemPromptBuilder` + `SystemPromptContext`：可拼装的系统提示。
- `runtime/skills.py`：SKILL.md frontmatter 加载（粗糙版）。
- `runtime/env.py`：WORKSPACE / AGENT_DIR 路径白名单。
- 测试 3.7k 行：契约单测、fake LLM 集成、真实 DashScope 端到端 showcase。

### Documentation
- `README.md`：定位、目录结构、安装、env 配置、最小用法、Hook & Event 模型、工具系统、与 DeepTalk 分工表、Roadmap。
- `ROADMAP.md`：M0–M3 四阶段规划，含上下文压缩、AgentSpec、动态工具注册等关键短板。

### Known limitations
- Public API 边界尚未文档化（M0 中解决）。
- 缺乏 LLM 重试 / 超时 / 取消的统一策略（M1.1）。
- 历史窗口 / token 预算修剪缺工具（M1.2）。
- `LLMRequest` 无 `response_format`（结构化输出，M1.3）。
- Provider 实例化散落在 `llm_factory.py`，无 unified `LLMProviderConfig`（M1.4）。
- `compact.*` hook 点存在但**无默认 compactor 实现**，长会话会 hard-fail 在 context overflow（**M1.7a 必须**）。
- 无声明式 sub-agent（`AgentSpec`，M1.8）。
- 无运行时工具注册 meta-tool（M2.6）。
