# Examples

[Back to docs](../docs/README.md) | [Detailed Guide](../docs/en/tutorials/examples-guide.md) | [详细指南](../docs/zh/tutorials/examples-guide.md)

Each file in this directory is a standalone, runnable Python script that teaches **one concept**. They are ordered progressively — start at `00` and move forward.

All examples are automatically validated by `tests/real/test_examples.py` against a real LLM.

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

## Next

After `19`, explore the [User Guide](../docs/en/user-guide/index.md) for deep-dive reference docs, or the [Tutorials](../docs/en/tutorials/index.md) for step-by-step project builds.
