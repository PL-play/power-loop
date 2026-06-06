# 架构总览

[English](en/architecture.md) | [回到文档站](README.md)

本文概览 power-loop 的内部协作：模块边界、`send()` 全链路、pipeline 阶段、pending 状态机、压缩、子代理、并发隔离和关键不变量。

> 这些图使用保守 Mermaid 语法，适合 GitHub 原生渲染。

## 1. 模块边界

```mermaid
flowchart TB
    subgraph Public["Public API"]
        SAL["StatefulAgentLoop"]
        Config["AgentLoopConfig"]
        Result["StatefulResult"]
        Store["SessionStore"]
        Registry["ToolRegistry"]
        Hooks["AgentHooks"]
        Bus["AgentEventBus"]
    end

    subgraph Core["Core internal"]
        Pipeline["AgentPipeline"]
        Runner["AgentRunner"]
        Context["ContextManager"]
    end

    subgraph Runtime["Runtime internal"]
        Compact["DefaultCompactor"]
        Retry["LLMRetryPolicy"]
        Cancel["CancellationToken"]
        Memory["MemoryProvider"]
        Structured["StructuredOutputSpec"]
        Budget["trim_history"]
    end

    SAL --> Pipeline
    SAL --> Runner
    SAL --> Store
    Config --> Compact
    Config --> Retry
    Config --> Memory
    Pipeline --> Registry
    Pipeline --> Hooks
    Pipeline --> Bus
    Pipeline --> Context
```

业务方主要依赖 `StatefulAgentLoop`、`AgentLoopConfig`、`SessionStore`、工具、hooks 和 events。`core` 与大部分 `runtime` 模块是内部实现细节。

## 2. send 全链路

```mermaid
sequenceDiagram
    participant Caller
    participant Loop as StatefulAgentLoop
    participant Store as SessionStore
    participant Pipeline as AgentPipeline
    participant LLM
    participant Tools as ToolRegistry

    Caller->>Loop: new_session()
    Loop->>Store: create_session()
    Store-->>Loop: sid
    Caller->>Loop: send(user_input, session_id=sid)
    Loop->>Store: append user message
    Loop->>Store: load active messages
    Loop->>Pipeline: run(history)
    Pipeline->>Pipeline: session.start hook
    loop each round
        Pipeline->>Pipeline: round.start hook
        Pipeline->>Pipeline: maybe compact and recall memory
        Pipeline->>LLM: complete(messages, tools)
        LLM-->>Pipeline: response
        Pipeline->>Store: append assistant message
        alt response has tool calls
            Pipeline->>Tools: invoke_async(name, args)
            Tools-->>Pipeline: result
            Pipeline->>Store: append tool message
        else final answer
            Pipeline->>Pipeline: session.end hook
            Pipeline-->>Loop: AgentLoopResult
        end
    end
    Loop-->>Caller: StatefulResult
```

## 3. Pipeline 单轮

```mermaid
flowchart TD
    A["round.start"] --> B["prepare round"]
    B --> C{"compaction needed"}
    C -->|"yes"| D["compactor.maybe_compact"]
    C -->|"no"| E["llm.before"]
    D --> E
    E --> F["LLM.complete"]
    F --> G["llm.after"]
    G --> H["append assistant message"]
    H --> I{"has tool calls"}
    I -->|"no"| J["round.end and session.end"]
    I -->|"yes"| K["round.decide"]
    K --> L["tools.batch.before"]
    L --> M["tool.before, invoke, tool.after"]
    M --> N["append tool message"]
    N --> O{"more tools"}
    O -->|"yes"| M
    O -->|"no"| P["tools.batch.after"]
    P --> Q["round.end"]
    Q --> A
```

## 4. Sink 和 Store

```mermaid
flowchart LR
    Pipeline["AgentPipeline"] --> Sink["MessageSink protocol"]
    Sink --> SQLiteSink["SQLiteSink"]
    Sink --> NullSink["NullSink"]
    SQLiteSink --> Store["SessionStore"]
    Store --> Sessions["sessions"]
    Store --> Messages["messages"]
    Store --> State["session_state"]
    Store --> Compactions["compactions"]
    Store --> Usage["usage_rounds"]
```

`SQLiteSink` 把 pipeline 的内存 history 映射回 SQLite seq，保证消息追加、pending 清理和 compaction 审计都落在同一个 `SessionStore`。

## 5. Pending 状态机

```mermaid
stateDiagram-v2
    [*] --> Clean: session created
    Clean --> Pending: assistant tool calls stored
    Pending --> Pending: tool message stored, some remain
    Pending --> Clean: last tool message stored
    Pending --> Recovered: resume called
    Pending --> Aborted: abort_pending called
    Recovered --> Clean: remaining tools executed
    Aborted --> Clean: aborted tool messages stored
```

如果进程在 assistant tool calls 已落库、tool 消息未全部落库时退出，下次 `send()` 会抛 `SessionPendingError`。调用方可以 `resume()` 跑完剩余工具，或 `abort_pending()` 合成中止 tool 消息。

## 6. 压缩流程

```mermaid
flowchart TD
    A["round.start"] --> B{"tokens over threshold"}
    B -->|"no"| Z["continue"]
    B -->|"yes"| C["compact.before hook"]
    C --> D{"hook skipped"}
    D -->|"yes"| Z
    D -->|"no"| E["find foldable span"]
    E --> F["expand to atomic boundary"]
    F --> G["LLM summary call"]
    G --> H{"summary success"}
    H -->|"yes"| I["replace span with compact_note"]
    H -->|"no"| J["continue without compaction"]
    I --> K["persist compaction audit"]
    K --> Z
    J --> Z
```

压缩不切开 `assistant(tool_calls)` 和对应 `tool` 消息，失败时软降级为未压缩 history。

## 7. 子代理

```mermaid
sequenceDiagram
    participant Parent as Parent Pipeline
    participant Loop as StatefulAgentLoop
    participant Store as SessionStore
    participant Child as Child Pipeline

    Parent->>Loop: spawn_agent(task, preset)
    Loop->>Store: create child session
    Store-->>Loop: child_sid
    Loop->>Child: run(child_history)
    Child-->>Loop: child result
    Loop->>Store: close or keep child session
    Loop-->>Parent: subagent result text
```

```mermaid
flowchart TD
    P["Parent Session sess_abc"] --> C1["Child researcher sess_def"]
    P --> C2["Child reviewer sess_ghi"]
    C1 --> GC1["Grandchild searcher sess_jkl"]
```

所有子代理共享父代理的 `SessionStore`。`close_session(parent_sid, cascade=True)` 会级联删除 `LINKED` 子树。

## 8. 并发与隔离

```mermaid
flowchart LR
    subgraph Loop["StatefulAgentLoop"]
        L1["asyncio.Lock sid_1"]
        L2["asyncio.Lock sid_2"]
        L3["asyncio.Lock sid_3"]
    end
    subgraph Store["SessionStore"]
        DB["SQLite WAL"]
    end
    L1 --> DB
    L2 --> DB
    L3 --> DB
```

一个 `StatefulAgentLoop` 可以并发驱动多个 session；每个 session 有独立 `asyncio.Lock`，底层 SQLite 连接由 `threading.RLock` 串行化写入。

## 9. 关键不变量

| 不变量 | 执行位置 |
|---|---|
| `assistant(tool_calls)` 落库后立即设置 pending | `SQLiteSink.on_assistant_tool_calls` |
| 最后一个 `tool` 消息落库后清 pending | `SQLiteSink.on_message_appended` |
| `(session_id, seq)` 单调唯一 | `SessionStore.append_message` |
| compaction 不切开工具调用原子对 | `DefaultCompactor` |
| compaction 失败软降级 | `DefaultCompactor.maybe_compact` |
| 子代理深度不超过 `MAX_SPAWN_DEPTH` | `SessionStore.create_session` |
| `SQLiteSink._history_seqs` 与 pipeline history 一一对应 | `SQLiteSink` |
| 子代理共享父代理的 `SessionStore` | `run_agent_spec` |

更多细节见 [Hooks](hooks.md)、[Events](events.md) 和对应单元测试。
