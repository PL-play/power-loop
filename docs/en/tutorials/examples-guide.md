# Examples (11.0.0)

Default examples use fixed model responses but execute the actual Runtime, tools and store. No model credentials are needed. Example 18 uses a real provider and requires model configuration/credentials. Example 28 requires a working Docker CLI/daemon. Example 38 requires MCP 2.x (`pip install 'power-loop[mcp]'`). Missing dependencies are errors, not successful validation.

`_runtime.py` composes a local factory. The demo host owns and lends its model; deployment factories should resolve their own provider, tools and resources by stable profile. Runtime starts the default timer driver; pure clients do not.

Run from the SDK directory: `python examples/00_hello_world.py`. Runnable entries are checked by `tests/unit/test_examples_run_208.py`; real format/summary/structured cases are in `tests/real/test_delivery_203.py`.

| Number | File | Behavior |
|---|---|---|
| 00 | [00_hello_world.py](../../../examples/00_hello_world.py) | One Runtime input and its durable result; no credentials or Redis. |
| 01 | [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | Reuse a TargetRef and unique message IDs; inspect persisted history. |
| 02 | [02_tool_calling.py](../../../examples/02_tool_calling.py) | Custom free tool through the Runtime; the demo model is deterministic. |
| 03 | [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | Capacity-one parent accepts a child, finishes, and receives its result. |
| 04 | [04_compaction.py](../../../examples/04_compaction.py) | Fold old projected sends; preserve the project code in the summary. |
| 05 | [05_pending_recovery.py](../../../examples/05_pending_recovery.py) | Align an interrupted tool response as unknown; never repeat the action. |
| 06 | [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | A trusted factory command delegates with a frozen AgentSpec. |
| 07 | [07_human_approval.py](../../../examples/07_human_approval.py) | 中断 / 用户确认 / Human approval: confirm before executing bash |
| 08 | [08_streaming.py](../../../examples/08_streaming.py) | 流式渲染 / Streaming: subscribe to STREAM_DELTA for typewriter effect |
| 09 | [09_audit_log.py](../../../examples/09_audit_log.py) | Current worker events to JSONL; close the subscriber and file explicitly. |
| 10 | [10_concurrent_sessions.py](../../../examples/10_concurrent_sessions.py) | Three Runtime sessions and an independent asynchronous approval queue. |
| 11 | [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | Sequential processes reopen the same store and TargetRef. |
| 12 | [12_retry_and_cancel.py](../../../examples/12_retry_and_cancel.py) | Unknown calls are not replayed; explicit new input and cancellation. |
| 13 | [13_memory_sqlite.py](../../../examples/13_memory_sqlite.py) | User-scoped SQLite memory across two independent Runtime sessions. |
| 14 | [14_structured_card.py](../../../examples/14_structured_card.py) | 结构化输出 / Structured output: LLM emits JSON cards directly |
| 15 | [15_skills_from_markdown.py](../../../examples/15_skills_from_markdown.py) | Parse skill instructions and supply the resulting config in the factory. |
| 16 | [16_custom_compactor.py](../../../examples/16_custom_compactor.py) | A custom FoldStrategy that deliberately discards old details. |
| 17 | [17_custom_memory_provider.py](../../../examples/17_custom_memory_provider.py) | A host memory adapter over an explicitly supplied API client. |
| 18 | [18_multi_provider.py](../../../examples/18_multi_provider.py) | 多 Provider / Multi-provider: use three LLM providers or switch on demand |
| 19 | [19_full_chatbot.py](../../../examples/19_full_chatbot.py) | Tools, memory, hooks and persisted history on a rebuilt local Runtime. |
| 20 | [20_default_tools.py](../../../examples/20_default_tools.py) | File, shell, skill and background tools with explicit workspace grants. |
| 21 | [21_request_user_input.py](../../../examples/21_request_user_input.py) | A paused task is answered by JobRef, interaction ID and wait round. |
| 22 | [22_follow_up_steering.py](../../../examples/22_follow_up_steering.py) | Append guidance to a running input; duplicate message IDs are recalled. |
| 23 | [23_per_send_overrides.py](../../../examples/23_per_send_overrides.py) | RequestOptions limits visible tools and overrides this input's prompt. |
| 24 | [24_agent_notes.py](../../../examples/24_agent_notes.py) | Notes use the current execution's store; a later input recalls them. |
| 25 | [25_token_usage.py](../../../examples/25_token_usage.py) | Result usage is per input; stored session statistics accumulate. |
| 26 | [26_timers.py](../../../examples/26_timers.py) | Default Runtime-owned timers; no Redis or model credentials required. |
| 27 | [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | Declarative workflow through the common runtime, with an offline example model. |
| 28 | [28_docker_shell_backend.py](../../../examples/28_docker_shell_backend.py) | An explicit factory environment runs bash inside a dedicated Docker container. |
| 29 | [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | Two current-runtime sessions use one database-atomic board (no credentials). |
| 31 | [31_memory_with_compaction.py](../../../examples/31_memory_with_compaction.py) | Recalled memory stays ephemeral while persisted sends are folded. |
| 32 | [32_recall_compacted.py](../../../examples/32_recall_compacted.py) | Retrieve original detail after folding with recall_send. |
| 33 | [33_coordinating_compactor.py](../../../examples/33_coordinating_compactor.py) | A FoldStrategy stores useful detail in a host memory provider before folding. |
| 34 | [34_durability_lifecycle.py](../../../examples/34_durability_lifecycle.py) | 持久化生命周期 / Durability lifecycle (0.15.0 — OPS-2..5) |
| 35 | [35_scaling_and_read_pool.py](../../../examples/35_scaling_and_read_pool.py) | Explicit worker capacity bounds concurrent model calls. |
| 36 | [36_observability.py](../../../examples/36_observability.py) | 可观测性 / Observability (0.17.0 — OBS-1/2/4) |
| 37 | [37_custom_retrieval_tool.py](../../../examples/37_custom_retrieval_tool.py) | 自定义检索工具 / Custom retrieval tool (ECO-6) |
| 38 | [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | MCP 工具 / MCP tools (ECO-1) |
| 39 | [39_pluggable_backends_and_resume.py](../../../examples/39_pluggable_backends_and_resume.py) | Rebuild a Runtime from a store DSN, stable profiles and TargetRef. |
| 40 | [40_send_context_projection.py](../../../examples/40_send_context_projection.py) | Projected history folds by input; recall_send reads the original detail. |
| 41 | [41_custom_async_tool.py](../../../examples/41_custom_async_tool.py) | A host job schedules a read-only result check through the timer backend. |
| 42 | [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | Custom tools on the current Runtime ports. |
| 43 | [43_memory_hook_override.py](../../../examples/43_memory_hook_override.py) | Factory-owned hooks: default memory, disabled memory, custom injection. |
| 44 | [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | Hook 注入审计 / Auditing ephemeral hook injection (record_hook_events) |
| 45 | [45_custom_projection_render.py](../../../examples/45_custom_projection_render.py) | 自定义投影渲染 / Custom projection render (ProjectionRenderConfig + subclass) |

Advanced examples cover projectors, trusted tool context, argument hooks and event subscriptions through the same Runtime, using local simulated actions. [advanced_runtime](../../../examples/advanced_runtime/README.md).
