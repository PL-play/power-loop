# Architecture

[中文](../zh/architecture.md) | [Back to docs](../README.md)

power-loop's internal collaboration: module boundaries, the `send()` full chain, pipeline phases, pending state machine, compaction, sub-agents, and key invariants.

> **9 Mermaid diagrams** — GitHub renders them natively. No external image tooling needed.

## 1. Module Boundaries

```mermaid
flowchart TB
    subgraph Public["Public API"]
        SAL[StatefulAgentLoop]
        Config[AgentLoopConfig]
        SR[StatefulResult]
        Store[SessionStore]
        TR[ToolRegistry]
        Hooks[AgentHooks]
        Bus[AgentEventBus]
    end

    subgraph Core["Core (internal)"]
        Pipeline[AgentPipeline]
        Runner[AgentRunner]
        CM[ContextManager]
    end

    subgraph Runtime["Runtime (internal)"]
        Compact[DefaultCompactor]
        Retry[LLMRetryPolicy]
        Cancel[CancellationToken]
        Memory[MemoryProvider]
        Structured[StructuredOutputSpec]
        Budget[trim_history]
    end

    subgraph Contracts["Contracts"]
        Errors[PowerLoopError...]
        HooksCtx[HookContexts]
        Events[EventPayloads]
    end

    SAL --> Pipeline
    SAL --> Runner
    SAL --> Store
    Config --> Compact
    Config --> Retry
    Config --> Memory
    Pipeline --> TR
    Pipeline --> Hooks
    Pipeline --> Bus
    Pipeline --> CM
```

## 2. send() Full Chain

```mermaid
sequenceDiagram
    participant Caller
    participant SAL as StatefulAgentLoop
    participant Store as SessionStore
    participant Pipeline as AgentPipeline
    participant LLM
    participant Tool as ToolRegistry

    Caller->>SAL: send(user_input, session_id=?)
    SAL->>Store: load_active_messages(sid)
    SAL->>Pipeline: run(history)
    Pipeline->>Pipeline: session.start hook
    Pipeline->>Pipeline: memory.recall → inject
    loop round 0..max_rounds
        Pipeline->>Pipeline: round.start hook
        Pipeline->>Pipeline: compactor.maybe_compact
        Pipeline->>Pipeline: llm.before hook
        Pipeline->>LLM: complete(messages, tools)
        LLM-->>Pipeline: response
        Pipeline->>Pipeline: llm.after hook
        Pipeline->>Store: append assistant msg
        alt has tool_calls
            Pipeline->>Pipeline: round.decide hook
            Pipeline->>Pipeline: tool.before hook
            Pipeline->>Tool: invoke_async(name, args)
            Tool-->>Pipeline: result
            Pipeline->>Pipeline: tool.after hook
            Pipeline->>Store: append tool msg
        else no tools
            Pipeline->>Pipeline: round.end → session.end
            Pipeline-->>SAL: result
        end
    end
    SAL-->>Caller: StatefulResult
```

## 3. Pipeline Single Round

```mermaid
flowchart TD
    A[round.start] --> B[prepare_round]
    B --> B1{todo reminder?}
    B1 -->|every 5 rounds| B2[inject todo snapshot]
    B1 -->|no| C
    B2 --> C{compaction needed?}
    C -->|yes| C1[compactor.maybe_compact]
    C -->|no| D
    C1 --> D[llm.before]
    D --> E[LLM.complete]
    E --> F[llm.after]
    F --> G[append assistant msg]
    G --> H{has tool_calls?}
    H -->|no| I[round.end → session.end]
    H -->|yes| J[round.decide]
    J --> K[tools.batch.before]
    K --> L[tool.before → invoke → tool.after]
    L --> M[append tool msg]
    M --> N{more tools?}
    N -->|yes| L
    N -->|no| O[tools.batch.after]
    O --> P[round.end]
    P --> A
```

## 4. Sink-Store

```mermaid
classDiagram
    class MessageSink {
        <<Protocol>>
        on_message_appended(msg, round_index)
        on_assistant_tool_calls(assistant_seq, tool_calls, round_index)
        on_round_started(round_index)
        on_round_ended(round_index, usage)
        on_compaction(fold_start_idx, fold_end_idx, note_seq)
    }

    class NullSink {
        No-op
    }

    class SQLiteSink {
        -store: SessionStore
        -sid: str
        -_history_seqs: list[int]
        init_history_seqs(seqs)
        on_message_appended(...)
        on_assistant_tool_calls(...)
        on_compaction(...)
    }

    class SessionStore {
        -conn: sqlite3.Connection
        -lock: threading.RLock
        open(path) SessionStore
        create_session(...) str
        append_message(...) int
        load_active_messages(sid) list[MessageRow]
        load_all_messages(sid) list[MessageRow]
        set_pending(sid, pending)
        get_pending(sid) dict
        record_compaction(...)
        close_session(sid, cascade) int
        close()
    }

    MessageSink <|.. NullSink
    MessageSink <|.. SQLiteSink
    SQLiteSink --> SessionStore
```

## 5. Pending State Machine

```mermaid
stateDiagram-v2
    [*] --> Clean: session created
    Clean --> Pending: assistant(tool_calls) stored
    Pending --> Pending: tool msg stored, some remain
    Pending --> Clean: last tool msg stored
    Pending --> Recovered: resume() called
    Pending --> Aborted: abort_pending() called
    Recovered --> Clean: remaining tools executed
    Aborted --> Clean: <aborted> msgs synthesized
```

## 6. Compaction

```mermaid
flowchart TD
    A[round.start] --> B{estimate_tokens > threshold?}
    B -->|no| Z[continue]
    B -->|yes| C[compact.before hook]
    C --> D{hook SKIP?}
    D -->|yes| Z
    D -->|no| E[find foldable span]
    E --> F[expand to atomic boundary]
    F --> G[LLM summary call]
    G --> H{summary success?}
    H -->|yes| I[replace span with compact_note]
    H -->|no| J[soft-fail: continue uncompacted]
    I --> K[persist compaction → store.record_compaction]
    K --> Z
    J --> Z
```

## 7. Sub-Agent Sequence

```mermaid
sequenceDiagram
    participant Parent as Parent Pipeline
    participant SAL as StatefulAgentLoop
    participant Store as SessionStore
    participant Child as Child Pipeline

    Parent->>SAL: spawn_agent(task, preset)
    SAL->>Store: create_session(parent=parent_sid, depth=d+1)
    Store-->>SAL: child_sid
    SAL->>Child: run(child_history)
    Child->>Child: child loop runs
    Child-->>SAL: child_result
    SAL->>Store: close_session(child_sid) or mark complete
    SAL-->>Parent: subagent_text event
```

## 8. Session Tree

```mermaid
flowchart TD
    P[Parent Session<br/>sess_abc] --> C1[Child: researcher<br/>sess_def]
    P --> C2[Child: reviewer<br/>sess_ghi]
    C1 --> GC1[Grandchild: searcher<br/>sess_jkl]
```

All children share the parent's `SessionStore`. `close_session(parent_sid, cascade=True)` recursively deletes the entire tree. `MAX_SPAWN_DEPTH = 3`.

## 9. Concurrency and Isolation

```mermaid
flowchart LR
    subgraph SAL[StatefulAgentLoop]
        L1[asyncio.Lock: sid_1]
        L2[asyncio.Lock: sid_2]
        L3[asyncio.Lock: sid_3]
    end
    subgraph Store[SessionStore]
        DB[(SQLite WAL)]
    end
    L1 --> DB
    L2 --> DB
    L3 --> DB
```

One `StatefulAgentLoop` can drive any number of sessions concurrently. Each session is protected by its own `asyncio.Lock`. The `SessionStore` uses a single connection + `threading.RLock` for write serialization.

## 10. Key Invariants

| Invariant | Enforced at | Cost of violation |
|---|---|---|
| `assistant(tool_calls)` stored → `set_pending` immediately | `SQLiteSink.on_assistant_tool_calls` | Protocol-invalid LLM request |
| Pending cleared on last `tool` msg | `SQLiteSink.on_message_appended` | Permanent pending state |
| `next_seq` monotonic, unique per session | `SessionStore.append_message` (tx read+increment) | Message order corruption |
| `messages.state ∈ {active, compacted_out}` | Schema + Sink/Store | Wrong history loaded |
| Compaction never splits `assistant(tool_calls) ↔ tool` | `DefaultCompactor._expand_back_to_atomic` | Protocol-invalid LLM request |
| Compaction fails soft → `None` | `DefaultCompactor.maybe_compact` try/except | Long session hard-fails |
| `MAX_SPAWN_DEPTH = 3` | `SessionStore.create_session` | Deep recursion stack overflow |
| `_history_seqs` 1:1 with `pipeline.history` | `SQLiteSink` methods | Compaction hits wrong seq range |
| `close_session` physically deletes 5 tables | `SessionStore._delete_session_tree` | Data leak / orphan rows |
| Sub-agent shares parent's `SessionStore` | `run_agent_spec` passes `parent_loop.store` | Broken parent-child link |

Detailed data flow and test coverage: `tests/unit/test_session_store.py`, `tests/unit/test_stateful_loop.py`, `tests/unit/test_compact.py`, `tests/unit/test_subagent.py`.

## 11. Retry State Machine

```mermaid
stateDiagram-v2
    [*] --> Attempting: LLM call starts
    Attempting --> Success: LLM responds
    Attempting --> Retryable: retry_on exception
    Retryable --> Backoff: sleep(backoff)
    Backoff --> Attempting: attempt < max_attempts
    Backoff --> Cancelled: CancellationToken fires
    Retryable --> Timeout: total_timeout exceeded
    Retryable --> Exhausted: attempt == max_attempts
    Cancelled --> [*]: status=cancelled
    Timeout --> [*]: status=degraded
    Exhausted --> [*]: status=degraded
    Success --> [*]: status=completed
```

## 12. Memory Lifecycle

```mermaid
flowchart TD
    A[send] --> B[session.start]
    B --> C{memory?}
    C -->|No| F[round loop]
    C -->|Yes| D[recall]
    D --> E{raises?}
    E -->|Yes| E1[emit MEMORY_FAILED]
    E1 --> F
    E -->|No| E2[tag + MEMORY_RECALLED hook]
    E2 --> E3{hook SKIP?}
    E3 -->|Yes| E4[skip]
    E4 --> F
    E3 -->|No| E5[inject after system block]
    E5 --> F
    F --> G[session.end]
    G --> H{memory?}
    H -->|Yes| I[remember]
    I --> J{raises?}
    J -->|Yes| J1[emit MEMORY_FAILED]
    J -->|No| K[done]
    H -->|No| K
```

## 13. Hook Decision Tree

```mermaid
flowchart TD
    Q["I want to..."] --> A1[init before loop]
    Q --> A2[check before each round]
    Q --> A3[modify LLM request]
    Q --> A4[block/limit tools]
    Q --> A5[intercept single tool]
    Q --> A6[post-process tool result]
    Q --> A7[stop after LLM reply]
    Q --> A8[skip compaction]
    Q --> A9[modify each message]
    Q --> A10[filter memory]

    A1 --> H1["session.start"]
    A2 --> H2["round.start"]
    A3 --> H3["llm.before"]
    A4 --> H4["round.decide / tools.batch.before"]
    A5 --> H5["tool.before"]
    A6 --> H6["tool.after / tool.error"]
    A7 --> H7["llm.after"]
    A8 --> H8["compact.before"]
    A9 --> H9["message.append"]
    A10 --> H10["memory.recalled"]
```