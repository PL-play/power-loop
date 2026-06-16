# 扩展性

[English](../../en/user-guide/scaling.md) | [用户指南](../index.md)

power-loop 是基于**可插拔 store** 的**可嵌入**内核：默认 SQLite（零基础设施），或按 DSN 选 PostgreSQL/MySQL。本页直白地说明并发模型，给出自带压测台的**实测**数据，列出调优旋钮，并描述如何横向扩展。它对上限诚实：单个 SQLite 写者有上限——服务器后端是越过它的办法。

## 模型

- **每会话一个写者。** 异步 store 把阻塞的 SQLite I/O 卸载到 worker 线程，在单一写者锁下执行——这正是保证 `next_seq` 不碰撞、每个多语句写入原子的关键。（开启 WAL，`journal_mode=WAL` / `synchronous=NORMAL`，读因而不会阻塞写者。）在 PostgreSQL/MySQL 上，每会话序号分配通过 `SELECT … FOR UPDATE` 行锁做到多写者安全。
- **每进程一个 asyncio 事件循环。** 单个 `StatefulAgentLoop` 可驱动任意数量并发会话，每个由自己的 `asyncio.Lock` 串行。SQLite 工作跑在 worker 线程，慢的写/读不会冻结循环；PostgreSQL/MySQL 的 driver 原生异步。
- **loop 不持有任何权威状态。** 状态全在 store 里，因此 loop 创建廉价，任何 session 都能从 `dsn` + `session_id` 恢复（适合 web handler / worker / 冷启动）。每会话的活动窗口缓存（`session_cache_size`，默认 256，`0` 关闭；用 `loop.cache_stats` 查看）只是对持久投影的纯加速器——它从不改变模型看到的内容。

**单个 SQLite 文件**不是多写者 store。多个进程写同一个 SQLite 文件不在范围内。要越过单个写者有两条路：**把 SQLite 文件按进程分片**（见下），或**把 DSN 指向 PostgreSQL/MySQL**——真正的多写者服务器，同样的代码、同样的一致性测试套件（见 [存储后端](storage-backends.md)）。

## 调优旋钮

| 旋钮 | 位置 | 作用 |
|---|---|---|
| 后端选择 | `StatefulAgentLoop` / `open_store(...)` 上的 `dsn=` | SQLite（默认，单写者文件）vs PostgreSQL/MySQL（多写者服务器）。单个写者不够用时的第一根杠杆。见 [存储后端](storage-backends.md)。 |
| `session_cache_size` | `StatefulAgentLoop(session_cache_size=N)` | 每会话活动窗口缓存的 LRU；在热的多次 send 路径上跳过重复读取活动历史。默认 `256`；`0` 关闭。纯加速器。 |
| 压缩 | `AgentLoopConfig(compactor=DefaultCompactor(...))` | 让**活动历史保持有界**（≈ `max_tokens`），从而限定每轮的读 + token 估算成本。长期会话最大的杠杆。 |
| 保留/裁剪 | `prune_compacted_messages` / `prune_usage_rounds` / `prune_timers` + `vacuum()` | 回收被折叠原文 / 旧用量行占用的磁盘；opt-in、调用方驱动。（`vacuum`/`checkpoint` 仅 SQLite；在 PG/MySQL 上是 no-op——用它们的原生工具。） |
| `max_tokens` | `AgentLoopConfig` | 上下文预算；开压缩时也大致是活动历史大小的上限。 |

store 自己卸载阻塞 I/O——SQLite 语句在 worker 线程、单一写者锁下执行，PostgreSQL/MySQL 原生异步——所以读不需要任何 opt-in 连接池。你通过选服务器后端或把 SQLite 文件按进程分片来扩展读写吞吐，而不是调连接池。

## 实测数据

自己跑自带压测台——确定性 `FakeLLM`（无 provider）驱动真实存储，数字反映的是*存储/循环*开销：

```bash
python -m bench            # 全量扫描 → JSON
python -m bench --smoke    # 快速子集（也是 CI 烟囱）
```

下面的数字记录于一台开发 VM（Python 3.12、内存库、`latency_s=0`）——**仅供参考，非规范**。请在你的代表性硬件上记录自己的数字；CI 只断言压测台能跑通且数字单调，从不卡绝对阈值（runner 噪声大）。

**Fan-out**（N 个并发会话，各一次 send）：

| 会话数 | sessions/sec | send p50 | send p99 |
|---:|---:|---:|---:|
| 1 | ~198 | 5.0 ms | 5.0 ms |
| 8 | ~679 | 10.1 ms | 11.4 ms |
| 32 | ~1035 | 24.9 ms | 27.6 ms |
| 128 | ~1031 | 109 ms | 122 ms |
| 512 | ~986 | 450 ms | 503 ms |

吞吐在 **~1000 sessions/sec** 附近见顶（单写者），超过后每次 send 延迟随并发增长——符合单写者上限预期。

**大历史**（每次 send 成本 vs 活动历史大小，无压缩）：

| 活动消息数 | send p50 | send p99 |
|---:|---:|---:|
| 1,000 | 9.6 ms | 11.4 ms |
| 10,000 | 92 ms | 102 ms |
| 50,000 | 511 ms | 559 ms |

每次 send 成本随活动历史大小近似线性增长——因为每次 send 都要加载整个活动窗口。**这正是压缩所避免的：** 开压缩后，无论总轮数多少，活动窗口都保持 ≈ `max_tokens`，此成本因而保持平坦。长期单会话**不开**压缩会持续退化（压测台的顺序吞吐场景显示同样的漂移）。长期会话请开压缩。

## 横向扩展

越过单个写者有两条路。

**A. 把 SQLite 文件按进程分片。** 用 SQLite 时模型是一个文件一个写者。要用更多核 / 扛更多负载，就跑 N 个进程，**每个有自己的 DB 文件**和自己的 `StatefulAgentLoop`：

```
进程 A → loopA → dsn="shard-a.db"
进程 B → loopB → dsn="shard-b.db"
```

把某个会话路由到固定进程（如对 session id 取哈希）。**不要**让两个进程写同一个 SQLite 文件——store 不协调跨进程写者（多出来的第二个写者只会被 `(session_id, seq)` 主键以 `IntegrityError` 抓到，而非被预防）。

**B. 用服务器后端。** 把 DSN 指向 PostgreSQL/MySQL（`dsn="postgresql://…"` / `dsn="mysql://…"`）。现在多个进程可以并发写**同一个**逻辑 store——每会话序号分配通过 `SELECT … FOR UPDATE` 行锁做到多写者安全，因此不同 session 可以并行追加而不碰撞。某个 session 的 pending 状态机仍假设同一时刻只有一个写者，所以请在你的 dispatcher/queue 层把该 session 的 send 串行化。置备与前置条件见 [存储后端](storage-backends.md)。

## 诚实声明

- **上面的实测上限是单 SQLite 写者且环境敏感的。** 磁盘 fsync 延迟与 CPU 主导；请在你的参考硬件上记录权威数字。
- **多写者横向扩展是后端选择，不是魔法。** PostgreSQL/MySQL 让多个进程写同一个逻辑 store，但单个 session 仍同一时刻串行到一个写者（其上的 dispatcher/queue 层是你的）。上面的数字是针对 SQLite 记录的。
- **压缩是长期会话的主要扩展杠杆**；不开则每次 send 成本随历史增长。
- **上限是否"够用"由你判断**，对照你预期的并发会话负载——以及你把 DSN 指向哪个后端。
