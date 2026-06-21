# Examples

[Back to docs](../docs/README.md) | [Detailed Guide](../docs/en/tutorials/examples-guide.md) | [详细指南](../docs/zh/tutorials/examples-guide.md)

Each file in this directory is a standalone, runnable Python script that teaches **one concept**. They are ordered progressively — start at `00` and move forward.

A representative subset is validated by `tests/real/test_examples.py` against a real LLM; every example is import/parse-clean and runnable as `python examples/NN_*.py` with your `.env` configured.

## List

| # | File | Concept | Key APIs |
|---|---|---|---|
| 00 | [`hello_world.py`](00_hello_world.py) | Explicit session + send | `StatefulAgentLoop`, `new_session()`, `send()` |
| 01 | [`multi_turn_chat.py`](01_multi_turn_chat.py) | Multi-turn conversation | `session_id`, `get_messages()` |
| 02 | [`tool_calling.py`](02_tool_calling.py) | Custom tool + JSON Schema | `ToolRegistry`, `ToolDefinition` |
| 03 | [`subagent_delegation.py`](03_subagent_delegation.py) | Imperative sub-agent | `spawn_agent`, `EPHEMERAL` |
| 04 | [`compaction.py`](04_compaction.py) | Auto context compaction | `DefaultCompactor`, `SessionStore` |
| 05 | [`pending_recovery.py`](05_pending_recovery.py) | Crash recovery mid-tool | `SessionPendingError`, `resume()`, `abort_pending()` |
| 06 | [`declarative_subagent.py`](06_declarative_subagent.py) | Declarative sub-agent | `AgentSpec`, `run_agent_spec()` |
| 07 | [`human_approval.py`](07_human_approval.py) | User confirmation gate | `TOOL_BEFORE` hook, `HookDirective.SKIP` |
| 08 | [`streaming.py`](08_streaming.py) | Real-time token streaming | `STREAM_DELTA` event |
| 09 | [`audit_log.py`](09_audit_log.py) | Subscribe to all events | `bus.subscribe(None, …)` |
| 10 | [`concurrent_sessions.py`](10_concurrent_sessions.py) | Multiple sessions in parallel | `asyncio.Lock`, `asyncio.Queue` |
| 11 | [`cross_process_resume.py`](11_cross_process_resume.py) | Resume after process restart | `db_path`, `subprocess` |
| 12 | [`retry_and_cancel.py`](12_retry_and_cancel.py) | Retry policy + cancel | `LLMRetryPolicy`, `CancellationToken` |
| 13 | [`memory_sqlite.py`](13_memory_sqlite.py) | Cross-session SQLite memory | `MemoryProvider`, `recall()` / `remember()` |
| 14 | [`structured_card.py`](14_structured_card.py) | Structured JSON extraction | `StructuredOutputSpec`, `parse_structured()` |
| 15 | [`skills_from_markdown.py`](15_skills_from_markdown.py) | SKILL.md → system prompt | Frontmatter parsing, domain knowledge injection |
| 16 | [`custom_compactor.py`](16_custom_compactor.py) | Custom Compactor protocol | `Compactor` Protocol, `CompactionPlan` |
| 17 | [`custom_memory_provider.py`](17_custom_memory_provider.py) | HTTP-backed MemoryProvider | `MemoryProvider` Protocol, HTTP API mock |
| 18 | [`multi_provider.py`](18_multi_provider.py) | Multiple LLM providers | `LLMProviderConfig`, `create_llm_service_from_config()` |
| 19 | [`full_chatbot.py`](19_full_chatbot.py) | **Flagship**: all features on | session + tools + hooks + events + memory + compaction |
| 20 | [`default_tools.py`](20_default_tools.py) | Built-in filesystem/search/bash tools | `create_default_tool_registry(preset="full")` |
| 21 | [`request_user_input.py`](21_request_user_input.py) | Resumable human input | `request_user_input`, `submit_input()` |
| 22 | [`follow_up_steering.py`](22_follow_up_steering.py) | In-flight steering | `follow_up()`, `FollowUpQueued`, `<follow_up>` |
| 23 | [`23_per_send_overrides.py`](23_per_send_overrides.py) | Per-call tool/prompt overrides | `send(tools=, system_prompt=)`, `ToolRegistry.subset` |
| 24 | [`24_agent_notes.py`](24_agent_notes.py) | Agent-authored notes | `note_add/update/delete`, `SQLiteNoteMemory`, `NotesPolicy` |
| 25 | [`25_token_usage.py`](25_token_usage.py) | Token usage accounting | `result.usage`, `get_session_stats`, `max_tokens_per_run`, `usage_updated` |
| 26 | [`26_timers.py`](26_timers.py) | Durable timers / self wake-ups | `schedule_wakeup`, `TimerRunner`, `HookPoint.TIMER_FIRE` |
| 27 | [`27_dynamic_workflow.py`](27_dynamic_workflow.py) | Declarative multi-agent workflow | `power_loop.workflow`, `WorkflowSpec`, `create_workflow`, `register_workflow_tools` |
| 28 | [`28_docker_shell_backend.py`](28_docker_shell_backend.py) | **Sandbox**: model-authored bash inside Docker | `ShellBackend`, `RuntimeEnv(shell_backend=…)`, `runtime_env_context` |
| 29 | [`29_shared_blackboard.py`](29_shared_blackboard.py) | **Coordination**: two agents share a scoped board | `SqliteBlackboard`, `register_blackboard_tools`, `RuntimeEnv(blackboard=…)` |
| 30 | [`30_subprocess_isolation.py`](30_subprocess_isolation.py) | **Isolation**: each workflow leaf in its own process + DB | `SubprocessExecutor`, `WorkerBootstrap`, `WorkerLauncher` |
| 31 | [`31_memory_with_compaction.py`](31_memory_with_compaction.py) | Cross-session memory recall alongside the default compactor | `MemoryProvider`, `DefaultCompactor` |
| 32 | [`32_recall_compacted.py`](32_recall_compacted.py) | Pull back exact turns that compaction folded out | `recall_compacted` tool |
| 33 | [`33_coordinating_compactor.py`](33_coordinating_compactor.py) | Custom compactor that remembers must-keep detail before folding | `CompactionContext`, `Compactor` |
| 34 | [`34_durability_lifecycle.py`](34_durability_lifecycle.py) | **Durability**: prune folded originals, VACUUM, export/import, graceful `aclose()` | `prune_compacted_messages`, `vacuum`, `export_session`/`import_session`, `aclose` |
| 35 | [`35_scaling_and_read_pool.py`](35_scaling_and_read_pool.py) | **Scale**: concurrent sessions on the async store; the `bench/` harness | `asyncio.gather` fan-out, `python -m bench` |
| 36 | [`36_observability.py`](36_observability.py) | **Observability**: durable JSONL sink + `replay` + metrics | `attach_jsonl_sink`, `replay`, `attach_metrics_sink` |
| 37 | [`37_custom_retrieval_tool.py`](37_custom_retrieval_tool.py) | **Extending**: register a custom (retrieval) tool — the no-bundled-connectors recipe | `ToolDefinition`, `ToolRegistry`, `tools=` allowlist |
| 38 | [`38_mcp_tools.py`](38_mcp_tools.py) | **MCP**: wire a real MCP server's tools into the agent | `StdioMCPClient`, `register_mcp_tools` (`power-loop[mcp]`) |
| 39 | [`39_pluggable_backends_and_resume.py`](39_pluggable_backends_and_resume.py) | **Storage**: pick a backend by DSN (SQLite/PG/MySQL), resume a session from a cold loop, schema policy | `dsn=`, `SchemaPolicy`, `StoreSchemaError`, `prewarm()`, `cache_stats` |
| 40 | [`40_send_context_projection.py`](40_send_context_projection.py) | **Projection**: feed finished sends as plain text (`pl_project_messages`) instead of verbatim history; fold + `recall_send` | `HistoryProjector`, `DefaultDeterministicProjector`, `history_projector=`, `recall_send` |
| 41 | [`41_custom_async_tool.py`](41_custom_async_tool.py) | **Custom async-wake tool**: start async work, return immediately, wake the agent when done — the extension recipe | `ToolDefinition`, `get_tool_runtime_context`, `loop.schedule_timer`, `TimerRunner`, `follow_up` |

## Running

```bash
# Set up credentials (once)
cp .env.example .env
# edit .env with POWER_LOOP_BASE_URL / API_KEY / MODEL

# Run any example
python examples/00_hello_world.py
```

## Shared helper

[`_helpers.py`](_helpers.py) provides `make_llm()` — loads `.env` and builds the configured LLM client. Each example imports it. If you copy an example into your own project, inline the two lines from `make_llm()` and drop the import.

## Advanced runtime examples

[`advanced_runtime/`](advanced_runtime/) contains focused examples for runtime-bound tools: custom `RuntimeProjector`s, `get_tool_runtime_context()`, hooks-based control flow, and event-bus observability. These examples call your configured real LLM, so make sure `.env` is set and watch provider usage/cost.

## Next

After `19`, explore the [User Guide](../docs/en/user-guide/index.md) for deep-dive reference docs, or the [Tutorials](../docs/en/tutorials/index.md) for step-by-step project builds.
