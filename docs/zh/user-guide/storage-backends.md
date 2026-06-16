# 存储后端（SQLite · PostgreSQL · MySQL）

power-loop 的整个存储层——会话、消息、定时器、压缩日志、用量统计、子代理树、共享黑板——
全部**只写一次**，面向一个极小的异步 `Database` + `Dialect` 端口实现。你用一个 **DSN**
选择后端；存储层之上的任何代码都不变。SQLite 是零基础设施的默认后端；PostgreSQL 和 MySQL
是真正的多写入者服务器，藏在可选的驱动 extra 之后。

```python
from power_loop import StatefulAgentLoop, open_store, SchemaPolicy

# 通过 loop（它会在首次使用时惰性打开一个自有的 store）：
StatefulAgentLoop(llm=llm, dsn="app.db")                          # SQLite（默认）
StatefulAgentLoop(llm=llm, dsn="postgresql://u:p@host:5432/app")  # PostgreSQL
StatefulAgentLoop(llm=llm, dsn="mysql://u:p@host:3306/app")       # MySQL

# 或者直接打开一个 store（例如想在多个 loop 之间共享）：
store = await open_store("postgresql://u:p@host/app", table_prefix="pl_")
loop = StatefulAgentLoop(llm=llm, store=store)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `dsn`（别名 `db_path`） | `./power_loop_sessions.db` | DSN，或一个裸路径 / `sqlite://` 路径。scheme 决定后端。 |
| `table_prefix` | `pl_` | 每张表/索引的前缀——在共享数据库上隔离 power-loop。 |
| `schema` | `SchemaPolicy.AUTO_CREATE` | 打开时如何置备表（见下文）。 |

为你使用的后端安装对应驱动：

```bash
pip install 'power-loop[postgres]'   # asyncpg
pip install 'power-loop[mysql]'      # aiomysql（纯 Python）
```

> 要么传一个已打开的 `store=`，**要么**传一组 store 配置（`dsn`/`table_prefix`/`schema`）——不能两者都传（会抛错）。SQLite 是零依赖核心唯一无需额外 extra 即可打开的后端。

---

## 如何选择后端

| | **SQLite**（默认） | **PostgreSQL** | **MySQL** |
|---|---|---|---|
| 基础设施 | 无——一个文件 | 一台服务器 | 一台服务器 |
| 驱动 extra | 无（标准库） | `[postgres]`（asyncpg） | `[mysql]`（aiomysql） |
| 写入者 | **每个文件一个写入进程**（跨文件分片） | 多写入者¹ | 多写入者¹ |
| 适合 | 本地应用、演示、嵌入式、按租户分文件 | 共享服务器、多个应用实例 | 共享服务器、MySQL 体系 |
| 维护操作 | `vacuum()` / `checkpoint()`（WAL） | 空操作² | 空操作² |

¹ 每会话的序号分配通过 `SELECT … FOR UPDATE` 行锁实现多写入者安全，所以两个进程可以并发地向
*不同*会话追加而永不冲突。但 **pending 状态机仍假定同一时刻只有一个写入者驱动某个给定会话**——见
[前置条件](#前置条件)。
² `vacuum`/`checkpoint`/`backup` 是 SQLite 的 `Maintenance` 能力；在 PG/MySQL 上它们是空操作
（请使用你数据库的原生工具）。

同一套与后端无关的**一致性测试套件**（`tests/unit/test_store_parity*.py`）会对 SQLite、
PostgreSQL 和 MySQL 运行每一项存储行为——无间隙的序号分配、压缩排序、级联删除、upsert 累加、
定时器 CAS、原子回滚——并以遗留的本地 store 作为 oracle 参照。

---

## 表结构置备（`SchemaPolicy`）

置备是打开时选择的一项策略：

| 策略 | 行为 |
|---|---|
| `AUTO_CREATE`（默认） | 探测版本表；若不存在，则创建每张表 + 索引并写入版本号。如果 DDL 执行失败（例如该角色没有 `CREATE` 权限），抛出 `StoreSchemaError`，并携带**完整的 DDL 脚本**供你手动运行。 |
| `VERIFY` | 仅探测。如果表结构缺失或版本不一致，抛出 `StoreSchemaError`（携带 DDL）。对于**没有 DDL 权限**的数据库角色——先带外置备，再用 `VERIFY` 打开。 |

```python
from power_loop import SchemaPolicy, StoreSchemaError, open_store

# 零基础设施：表会在首次使用时出现。
store = await open_store("postgresql://app@host/app")  # AUTO_CREATE（默认）

# 受限角色：只校验，并在表结构缺失时打印确切的 DDL 交给 DBA。
try:
    store = await open_store("postgresql://readonly@host/app", schema=SchemaPolicy.VERIFY)
except StoreSchemaError as e:
    print(e)            # 错误信息 + 完整可运行的置备脚本
    print(e.ddl)        # CREATE/INSERT 语句的 list[str]
```

`create_schema: bool` 是为 1.x 系列保留的一个已废弃别名（`True → AUTO_CREATE`，
`False → VERIFY`）。

要在不打开 store 的情况下以编程方式获取 DDL（例如用于生成迁移）：

```python
from power_loop.runtime.store.schema import provisioning_ddl
from power_loop.runtime.store.backends.postgres import PostgresDatabase  # 或 sqlite/mysql

# （或者直接对一个全新的数据库执行 VERIFY 打开，捕获 StoreSchemaError.ddl）
```

---

## 各后端的 DDL

下面正是 `AUTO_CREATE` 所运行的（也是 `StoreSchemaError.ddl` 所打印的），针对默认的 `pl_`
前缀。它在运行时由 dialect 生成，因此永远不会与代码产生偏差。12 张表 + 一张版本表；
SQLite/PostgreSQL 单独声明索引，MySQL 内联声明（它没有 `CREATE INDEX IF NOT EXISTS`）。

### SQLite

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, system_prompt TEXT, model TEXT, config_json TEXT, status TEXT NOT NULL DEFAULT 'active', kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT, spawn_tool_call_id TEXT, spawn_depth INTEGER NOT NULL DEFAULT 0, lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT);
CREATE INDEX IF NOT EXISTS pl_idx_sessions_parent ON pl_sessions(parent_session_id);
CREATE TABLE IF NOT EXISTS pl_messages (session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT, round_index INTEGER, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, seq));
CREATE INDEX IF NOT EXISTS pl_idx_messages_session_state ON pl_messages(session_id, state, seq);
CREATE TABLE IF NOT EXISTS pl_compactions (session_id TEXT NOT NULL, compact_seq INTEGER NOT NULL, note_seq INTEGER NOT NULL, from_seq INTEGER NOT NULL, to_seq INTEGER NOT NULL, before_tokens INTEGER, after_tokens INTEGER, round_index INTEGER, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, compact_seq));
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id TEXT NOT NULL, round_index INTEGER NOT NULL, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, model TEXT, created_at INTEGER NOT NULL, PRIMARY KEY (session_id, round_index));
CREATE TABLE IF NOT EXISTS pl_session_state (session_id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL DEFAULT 1, round_index INTEGER NOT NULL DEFAULT 0, last_compact_seq INTEGER NOT NULL DEFAULT 0, pending_json TEXT);
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, state_key));
CREATE TABLE IF NOT EXISTS pl_shared_state (owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at INTEGER NOT NULL, PRIMARY KEY (owner, state_key));
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL, return_code INTEGER, output_tail TEXT, output_path TEXT, last_seen_at INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, task_id));
CREATE INDEX IF NOT EXISTS pl_idx_background_tasks_session_status ON pl_background_tasks(session_id, status, updated_at);
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id TEXT PRIMARY KEY, sends INTEGER NOT NULL DEFAULT 0, rounds INTEGER NOT NULL DEFAULT 0, llm_calls INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0, first_send_at INTEGER, last_send_at INTEGER, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_timers (session_id TEXT NOT NULL, timer_id INTEGER NOT NULL, due_at INTEGER NOT NULL, note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s INTEGER, fire_count INTEGER NOT NULL DEFAULT 0, last_fired_at INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, timer_id));
CREATE INDEX IF NOT EXISTS pl_idx_timers_due ON pl_timers(status, due_at);
CREATE TABLE IF NOT EXISTS pl_notes (session_id TEXT NOT NULL, note_id INTEGER NOT NULL, content TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (session_id, note_id));
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

### PostgreSQL

形状相同；epoch-毫秒时间戳与计数器列由 `INTEGER → BIGINT`，JSON 仍以 `TEXT` 保存
（由 store 负责序列化/反序列化），`pinned SMALLINT`。

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id TEXT PRIMARY KEY, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, system_prompt TEXT, model TEXT, config_json TEXT, status TEXT NOT NULL DEFAULT 'active', kind TEXT NOT NULL DEFAULT 'root', parent_session_id TEXT, spawn_tool_call_id TEXT, spawn_depth BIGINT NOT NULL DEFAULT 0, lifecycle TEXT NOT NULL DEFAULT 'ephemeral', metadata_json TEXT);
CREATE INDEX IF NOT EXISTS pl_idx_sessions_parent ON pl_sessions(parent_session_id);
CREATE TABLE IF NOT EXISTS pl_messages (session_id TEXT NOT NULL, seq BIGINT NOT NULL, role TEXT NOT NULL, name TEXT, content TEXT, tool_calls_json TEXT, tool_call_id TEXT, round_index BIGINT, state TEXT NOT NULL DEFAULT 'active', meta_json TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq));
CREATE INDEX IF NOT EXISTS pl_idx_messages_session_state ON pl_messages(session_id, state, seq);
CREATE TABLE IF NOT EXISTS pl_compactions (session_id TEXT NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL, from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT, after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, compact_seq));
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id TEXT NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT, completion_tokens BIGINT, total_tokens BIGINT, model TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index));
CREATE TABLE IF NOT EXISTS pl_session_state (session_id TEXT PRIMARY KEY, next_seq BIGINT NOT NULL DEFAULT 1, round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0, pending_json TEXT);
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key));
CREATE TABLE IF NOT EXISTS pl_shared_state (owner TEXT NOT NULL, state_key TEXT NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key));
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id TEXT NOT NULL, task_id TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT, last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id));
CREATE INDEX IF NOT EXISTS pl_idx_background_tasks_session_status ON pl_background_tasks(session_id, status, updated_at);
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id TEXT PRIMARY KEY, sends BIGINT NOT NULL DEFAULT 0, rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0, tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0, completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0, first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL);
CREATE TABLE IF NOT EXISTS pl_timers (session_id TEXT NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL, note TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'armed', interval_s BIGINT, fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, timer_id));
CREATE INDEX IF NOT EXISTS pl_idx_timers_due ON pl_timers(status, due_at);
CREATE TABLE IF NOT EXISTS pl_notes (session_id TEXT NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL, pinned SMALLINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id));
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

### MySQL

字符串主键/索引列变为 `VARCHAR(255)`（MySQL 无法在不指定前缀长度的情况下对 `TEXT` 建索引/主键），
枚举式小列用 `VARCHAR(32)`、`utf8mb4`，且索引**内联**声明（MySQL 没有
`CREATE INDEX IF NOT EXISTS`）。`key` 列改名为 `state_key`（保留字）。版本行的写入使用
`INSERT IGNORE`；upsert 使用 `INSERT … AS new_row ON DUPLICATE KEY UPDATE`（MySQL 8.0.19+）。

```sql
CREATE TABLE IF NOT EXISTS pl_schema_migrations (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pl_sessions (session_id VARCHAR(255) NOT NULL, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, system_prompt TEXT, model VARCHAR(255), config_json TEXT, status VARCHAR(32) NOT NULL DEFAULT 'active', kind VARCHAR(32) NOT NULL DEFAULT 'root', parent_session_id VARCHAR(255), spawn_tool_call_id VARCHAR(255), spawn_depth BIGINT NOT NULL DEFAULT 0, lifecycle VARCHAR(32) NOT NULL DEFAULT 'ephemeral', metadata_json TEXT, PRIMARY KEY (session_id), KEY pl_idx_sessions_parent (parent_session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_messages (session_id VARCHAR(255) NOT NULL, seq BIGINT NOT NULL, role VARCHAR(32) NOT NULL, name VARCHAR(255), content TEXT, tool_calls_json TEXT, tool_call_id VARCHAR(255), round_index BIGINT, state VARCHAR(32) NOT NULL DEFAULT 'active', meta_json TEXT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, seq), KEY pl_idx_messages_session_state (session_id, state, seq)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_compactions (session_id VARCHAR(255) NOT NULL, compact_seq BIGINT NOT NULL, note_seq BIGINT NOT NULL, from_seq BIGINT NOT NULL, to_seq BIGINT NOT NULL, before_tokens BIGINT, after_tokens BIGINT, round_index BIGINT, created_at BIGINT NOT NULL, PRIMARY KEY (session_id, compact_seq)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_usage_rounds (session_id VARCHAR(255) NOT NULL, round_index BIGINT NOT NULL, prompt_tokens BIGINT, completion_tokens BIGINT, total_tokens BIGINT, model VARCHAR(255), created_at BIGINT NOT NULL, PRIMARY KEY (session_id, round_index)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_state (session_id VARCHAR(255) NOT NULL, next_seq BIGINT NOT NULL DEFAULT 1, round_index BIGINT NOT NULL DEFAULT 0, last_compact_seq BIGINT NOT NULL DEFAULT 0, pending_json TEXT, PRIMARY KEY (session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_runtime_state (session_id VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, state_key)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_shared_state (owner VARCHAR(255) NOT NULL, state_key VARCHAR(255) NOT NULL, value_json TEXT, updated_at BIGINT NOT NULL, PRIMARY KEY (owner, state_key)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_background_tasks (session_id VARCHAR(255) NOT NULL, task_id VARCHAR(255) NOT NULL, command TEXT NOT NULL, status VARCHAR(32) NOT NULL, return_code BIGINT, output_tail TEXT, output_path TEXT, last_seen_at BIGINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, task_id), KEY pl_idx_bgtasks_session_status (session_id, status, updated_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_session_stats (session_id VARCHAR(255) NOT NULL, sends BIGINT NOT NULL DEFAULT 0, rounds BIGINT NOT NULL DEFAULT 0, llm_calls BIGINT NOT NULL DEFAULT 0, tool_calls BIGINT NOT NULL DEFAULT 0, prompt_tokens BIGINT NOT NULL DEFAULT 0, completion_tokens BIGINT NOT NULL DEFAULT 0, total_tokens BIGINT NOT NULL DEFAULT 0, first_send_at BIGINT, last_send_at BIGINT, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_timers (session_id VARCHAR(255) NOT NULL, timer_id BIGINT NOT NULL, due_at BIGINT NOT NULL, note TEXT NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'armed', interval_s BIGINT, fire_count BIGINT NOT NULL DEFAULT 0, last_fired_at BIGINT, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, timer_id), KEY pl_idx_timers_due (status, due_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS pl_notes (session_id VARCHAR(255) NOT NULL, note_id BIGINT NOT NULL, content TEXT NOT NULL, pinned TINYINT NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, PRIMARY KEY (session_id, note_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO pl_schema_migrations (id, version) VALUES (1, 1);
```

> 全文把 `pl_` 替换成你的 `table_prefix`。`pl_schema_migrations` 是版本表，它让 `VERIFY` 得以
> 工作，并拒绝一个比代码更新的数据库。

---

## 可恢复的 loop 与活动窗口缓存

`StatefulAgentLoop` **不持有任何权威会话状态**——这些状态全部存在 store 里。所以 loop 创建起来很
廉价，你可以从一个冷进程按 id 恢复任意会话：

```python
loop = StatefulAgentLoop(llm=create_llm_service_from_env(), dsn=DSN)   # 廉价；惰性打开
await loop.prewarm(session_id)                # 可选：预加载活动窗口
result = await loop.send(user_text, session_id=session_id)
```

为了避免每次 send 都重新读取整个活动历史，loop 会为每个会话保留一个**活动窗口缓存**——但它只缓存
*持久化的*投影（`load_active_messages` 返回的那些行），并以一个单调的 `next_seq` 令牌作为键，
每次 send 都会重新构建工作副本（recall、微压缩）。所以它是一个纯加速器：一个空缓存的冷 loop 喂给
模型的提示词与有缓存时逐字节相同（由 warm-vs-cold 一致性测试验证）。它受 LRU 限制
（`session_cache_size`，默认 256，`0` 表示禁用），并通过 `loop.cache_stats` 暴露。

---

## 前置条件

- **每会话单写入者。** 每会话的锁是进程内的；它不提供跨进程互斥。使用 SQLite 时，每个文件运行
  一个写入进程。使用 PostgreSQL/MySQL 时，序号分配是多写入者安全的，但某个给定会话的
  pending 状态机仍假定同一时刻只有一个写入者——请在你的 dispatcher/队列层把一个会话的多次 send
  串行化。（窗口缓存无论如何都是数据安全的：陈旧的令牌会强制重新加载，绝不会提供错误数据。）
- **并发首次启动。** `AUTO_CREATE` 是幂等的，并在重试时自愈，但它在 MySQL 上不是原子的
  （DDL 会自动提交），也不获取跨进程锁。如果可能有许多实例同时针对一个*全新*的服务器表结构启动，
  请带外置备（运行上面的 DDL）并用 `SchemaPolicy.VERIFY` 打开。

另见：[会话](sessions.md) · [扩展性](scaling.md) · 设计说明见
`docs/design/storage-backends.md`。
