# Runtime 容量与共享 worker

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

Runtime.capacity 限制活跃执行数量。独立会话和子任务使用同一容量机制；工作流等待不占模型槽位。默认使用进程内协调与传输，并取得 store 独占启动锁，第二个独立默认进程会被拒绝。纯 Client 发布者不取得该锁。

共享部署设置 shared_execution=True，并显式提供共享所有权与消息传输适配器。只把数据库换成 PostgreSQL/MySQL 不会让默认协调器变成共享。共享模式拒绝进程级适配器。所有 worker 必须具备同一组 required_profiles；模型、工具、hooks、workspace、skills 和后端端口在接单前预检，不靠固定路由找到持有某个对象的进程。

共享资源有稳定 resource_profile_ref、resource_scope="shared"，以及宿主对真实共享存储的映射。目录同名不代表内容共享。工厂在当前节点重建资源。所有权丢失后停止执行，已经开始的外部动作可能保留 unknown。

性能测量覆盖受理、执行、持久化和资源关闭，并记录数据库、并发、历史、工具、输入大小和版本。SQL 数量不能单独证明延迟或吞吐改善。当前测量以对应交付报告为准，旧示意数字不是 v11 基线。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [10_concurrent_sessions.py](../../../examples/10_concurrent_sessions.py) | 并发会话 |
| [35_scaling_and_read_pool.py](../../../examples/35_scaling_and_read_pool.py) | 容量与存储 |
| [36_observability.py](../../../examples/36_observability.py) | 可观测性 |

```bash
.venv/bin/python examples/10_concurrent_sessions.py
```
