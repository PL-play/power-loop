# 存储后端

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

通过 open_store 或 SessionStore.open 显式打开存储，再交给 Runtime。SQLite 是默认本地后端；PostgreSQL、MySQL 需要对应驱动 extra。测试使用 _test 数据库和专用 table_prefix。

SchemaPolicy.AUTO_CREATE 创建全新的当前结构；VERIFY 只校验已有当前结构，不执行迁移。SDK 11 使用任务结构 15、协议 42、描述符结构 1。旧库明确拒绝，不在运行时转换或回退兼容。需要人工建库时可读取 StoreSchemaError 的建表 DDL。

会话序号分配和黑板原子更新由数据库事务保证。公开的 StoreBlackboard 包括首次建行的原子处理；围绕读写加单实例内存锁不能解决跨进程竞争。DeepTalk 的业务黑板和定时器继续通过 API 后端写入，不改为 SDK 写业务库。

保留期清理显式执行，覆盖已压缩历史、用量和终态定时器。定时驱动按索引分批扫描并条件领取。共享执行还需要共享协调与传输，服务器数据库不能替代它们。Runtime 关闭其拥有的 store；独立拥有的读取方使用自己的连接。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | 当前库重开 |
| [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | 原子黑板 |
| [34_durability_lifecycle.py](../../../examples/34_durability_lifecycle.py) | 生命周期 |
| [39_pluggable_backends_and_resume.py](../../../examples/39_pluggable_backends_and_resume.py) | 后端选择 |

```bash
.venv/bin/python examples/11_cross_process_resume.py
```
