# 扩展性

[English](../../en/user-guide/scaling.md) | [用户指南](../index.md)

power-loop 是基于单个 SQLite 文件的**可嵌入、单进程**内核。本页直白地说明并发模型，给出自带压测台的**实测**数据，列出调优旋钮，并描述多进程模式。它对上限诚实：上限是存在的。

## 模型

- **一个 SQLite 文件，一个写者。** 所有写入都走单一连接，由进程内 `threading.RLock` 串行——这正是保证 `next_seq` 不碰撞、每个多语句写入原子的关键。开启 WAL（`journal_mode=WAL`、`synchronous=NORMAL`）。
- **每进程一个 asyncio 事件循环。** 单个 `StatefulAgentLoop` 可驱动任意数量并发会话，每个由自己的 `asyncio.Lock` 串行。写路径上阻塞的 SQLite I/O 用 `asyncio.to_thread` 卸载，避免拖慢循环。
- **读可以并发**（opt-in，见[读连接池](#读连接池)）——WAL 允许多个读者与唯一写者并存。

它**不是**：多写者 / 横向扩展的存储。多个进程写同一个逻辑存储**不在范围内**。要更高吞吐就跑更多进程——每进程一个独立 DB 文件（见[多进程](#多进程)）。

## 调优旋钮

| 旋钮 | 位置 | 作用 |
|---|---|---|
| `read_pool_size` | `SessionStore.open(read_pool_size=N)` | N 个额外只读连接，读不再排在写者锁后面（仅文件库；`:memory:` 会拒绝）。默认 `0`。 |
| 压缩 | `AgentLoopConfig(compactor=DefaultCompactor(...))` | 让**活动历史保持有界**（≈ `max_tokens`），从而限定每轮的读 + token 估算成本。长期会话最大的杠杆。 |
| 保留/裁剪 | `prune_compacted_messages` / `prune_usage_rounds` / `prune_timers` + `vacuum()` | 回收被折叠原文 / 旧用量行占用的磁盘；opt-in、调用方驱动。 |
| `max_tokens` | `AgentLoopConfig` | 上下文预算；开压缩时也大致是活动历史大小的上限。 |

### 读连接池

```python
store = SessionStore.open("app.db", read_pool_size=4)
loop = StatefulAgentLoop(llm=llm, store=store)
```

读操作（`load_active_messages`、`load_all_messages`）从 N 个只读（`query_only=ON`）连接中取一个，与写者并发执行，不再排在写锁后面。WAL 下的池读看到读开始前已提交的全部事务——这正是每次 send 加载历史所需的一致性。默认关闭（`0`）；`:memory:` 会忽略（每个内存连接是**独立**数据库）。读密集 fan-out 下值得开启。

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

## 多进程

存储模型是**一文件一进程**。要用更多核 / 扛更多负载，就跑 N 个进程，**每个有自己的 DB 文件**和自己的 `StatefulAgentLoop`：

```
进程 A → loopA → SessionStore.open("shard-a.db")
进程 B → loopB → SessionStore.open("shard-b.db")
```

把某个会话路由到固定进程（如对 session id 取哈希）。**不要**让两个进程写同一个文件——存储不协调跨进程写者（多出来的第二个写者只会被 `(session_id, seq)` 主键以 `IntegrityError` 抓到，而非被预防）。

## 诚实声明

- **实测上限是单进程且环境敏感的。** 磁盘 fsync 延迟与 CPU 主导；请在你的参考硬件上记录权威数字。
- **多写者横向扩展不在 1.0 范围内。** 交付的是*实测*的单进程上限 + 上面的一文件一进程模式。
- **压缩是长期会话的主要扩展杠杆**；不开则每次 send 成本随历史增长。
- **上限是否"够用"由你判断**，对照你预期的并发会话负载。
