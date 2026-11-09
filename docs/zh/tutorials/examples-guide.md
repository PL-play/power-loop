# 示例指南（11.0.0）

默认示例使用确定模型，实际运行 Runtime、工具和存储，无需模型凭证。18 是真实供应商示例，需配置模型与密钥；28 需要 Docker CLI/daemon；38 需要 MCP 2.x（`pip install 'power-loop[mcp]'`）。缺少这些依赖会明确报错，不算验证通过。

`_runtime.py` 展示本地工厂装配。宿主在示例中持有模型，执行期间借用；部署工厂应按稳定 profile 重新取得自己的模型、工具与资源。Runtime 负责启动默认定时 driver，纯 Client 不启动。

从 SDK 目录运行 `python examples/00_hello_world.py`。可执行入口由 `tests/unit/test_examples_run_208.py` 检查；真实模型的格式、摘要、结构化回归在 `tests/real/test_delivery_203.py`。

| 编号 | 文件 | 能力 |
|---|---|---|
| 00 | [00_hello_world.py](../../../examples/00_hello_world.py) | 最小受理与结果 |
| 01 | [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | 多轮历史 |
| 02 | [02_tool_calling.py](../../../examples/02_tool_calling.py) | 自定义工具 |
| 03 | [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | 容量 1 的子任务 |
| 04 | [04_compaction.py](../../../examples/04_compaction.py) | 投影折叠 |
| 05 | [05_pending_recovery.py](../../../examples/05_pending_recovery.py) | 未知结果后新输入继续 |
| 06 | [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | 宿主命令顺序子任务 |
| 07 | [07_human_approval.py](../../../examples/07_human_approval.py) | 人工批准钩子 |
| 08 | [08_streaming.py](../../../examples/08_streaming.py) | 流式事件 |
| 09 | [09_audit_log.py](../../../examples/09_audit_log.py) | 事件审计 |
| 10 | [10_concurrent_sessions.py](../../../examples/10_concurrent_sessions.py) | 并发会话与批准 |
| 11 | [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | 跨进程重开 |
| 12 | [12_retry_and_cancel.py](../../../examples/12_retry_and_cancel.py) | 未知调用与取消 |
| 13 | [13_memory_sqlite.py](../../../examples/13_memory_sqlite.py) | SQLite 记忆 |
| 14 | [14_structured_card.py](../../../examples/14_structured_card.py) | 结构化输出 |
| 15 | [15_skills_from_markdown.py](../../../examples/15_skills_from_markdown.py) | 技能文件 |
| 16 | [16_custom_compactor.py](../../../examples/16_custom_compactor.py) | 自定义折叠策略 |
| 17 | [17_custom_memory_provider.py](../../../examples/17_custom_memory_provider.py) | 宿主记忆接口 |
| 18 | [18_multi_provider.py](../../../examples/18_multi_provider.py) | 真实模型装配 |
| 19 | [19_full_chatbot.py](../../../examples/19_full_chatbot.py) | 工具、记忆与重开 |
| 20 | [20_default_tools.py](../../../examples/20_default_tools.py) | 文件、shell 与后台工具 |
| 22 | [22_follow_up_steering.py](../../../examples/22_follow_up_steering.py) | 忙时追加指引 |
| 23 | [23_per_send_overrides.py](../../../examples/23_per_send_overrides.py) | 单次请求选项 |
| 24 | [24_agent_notes.py](../../../examples/24_agent_notes.py) | 笔记 |
| 25 | [25_token_usage.py](../../../examples/25_token_usage.py) | 用量统计 |
| 26 | [26_timers.py](../../../examples/26_timers.py) | 默认定时器 |
| 27 | [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | 容量 1 的 workflow |
| 28 | [28_docker_shell_backend.py](../../../examples/28_docker_shell_backend.py) | Docker shell |
| 29 | [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | 共享黑板 |
| 31 | [31_memory_with_compaction.py](../../../examples/31_memory_with_compaction.py) | 记忆与折叠共存 |
| 32 | [32_recall_compacted.py](../../../examples/32_recall_compacted.py) | 取回折叠前原文 |
| 33 | [33_coordinating_compactor.py](../../../examples/33_coordinating_compactor.py) | 折叠前保存记忆 |
| 34 | [34_durability_lifecycle.py](../../../examples/34_durability_lifecycle.py) | 保留清理与导出导入 |
| 35 | [35_scaling_and_read_pool.py](../../../examples/35_scaling_and_read_pool.py) | 并发容量 |
| 36 | [36_observability.py](../../../examples/36_observability.py) | 事件与指标 |
| 37 | [37_custom_retrieval_tool.py](../../../examples/37_custom_retrieval_tool.py) | 检索工具 |
| 38 | [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | 本地 MCP 服务 |
| 39 | [39_pluggable_backends_and_resume.py](../../../examples/39_pluggable_backends_and_resume.py) | 存储与重建 |
| 40 | [40_send_context_projection.py](../../../examples/40_send_context_projection.py) | 投影和原文查询 |
| 41 | [41_custom_async_tool.py](../../../examples/41_custom_async_tool.py) | 宿主异步结果唤醒 |
| 42 | [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | 自建工具 |
| 43 | [43_memory_hook_override.py](../../../examples/43_memory_hook_override.py) | 覆盖记忆钩子 |
| 44 | [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | 临时注入审计 |
| 45 | [45_custom_projection_render.py](../../../examples/45_custom_projection_render.py) | 自定义投影展示 |

高级示例：投影、可信工具上下文、工具参数钩子及事件订阅，使用相同 Runtime，均为本地模拟操作。 [advanced_runtime](../../../examples/advanced_runtime/README.md).
