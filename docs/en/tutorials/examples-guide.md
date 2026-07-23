# Examples Guide / 示例指南

[Back to docs](../index.md) | [中文](../../zh/tutorials/examples-guide.md)

This guide walks through every example in `examples/` with code explanation, real output, and key takeaways. Each example teaches **one concept** — they are ordered progressively.

All examples share `_helpers.py` which loads `.env` and builds an LLM client. To copy an example into your own project, inline the two lines from `make_llm()` and drop the import.

---

## Table of Contents

| # | File | Concept |
|---|---|---|
| [00](#00-hello-world) | `hello_world.py` | Minimal: one send, one reply |
| [01](#01-multi-turn-chat) | `multi_turn_chat.py` | Multi-turn conversation with session_id |
| [02](#02-tool-calling) | `tool_calling.py` | Custom tools + JSON Schema |
| [03](#03-sub-agent-delegation) | `subagent_delegation.py` | Imperative sub-agent via `spawn_agent` |
| [04](#04-compaction) | `compaction.py` | Auto context compaction |
| [05](#05-pending-recovery) | `pending_recovery.py` | Crash recovery mid-tool-call |
| [06](#06-sub-agent-overrides-and-declarative-spec) | `declarative_subagent.py` | spawn_agent overrides + declarative run_agent_spec |
| [07](#07-human-approval) | `human_approval.py` | User confirmation gate via hooks |
| [08](#08-streaming) | `streaming.py` | Real-time token streaming |
| [09](#09-audit-log) | `audit_log.py` | Full event audit → JSONL |
| [10](#10-concurrent-sessions) | `concurrent_sessions.py` | Parallel sessions + async approval queue |
| [11](#11-cross-process-resume) | `cross_process_resume.py` | Resume after process restart |
| [12](#12-retry-and-cancel) | `retry_and_cancel.py` | Retry policy + cancellation |
| [13](#13-memory-sqlite) | `memory_sqlite.py` | Cross-session SQLite memory |
| [14](#14-structured-output) | `structured_card.py` | Structured JSON extraction |
| [15](#15-skills-from-markdown) | `skills_from_markdown.py` | SKILL.md → system prompt |
| [16](#16-custom-compactor) | `custom_compactor.py` | Custom compaction strategy |
| [17](#17-custom-memory-provider) | `custom_memory_provider.py` | HTTP-backed MemoryProvider |
| [18](#18-multi-provider) | `multi_provider.py` | Multiple LLM providers |
| [19](#19-full-chatbot) | `full_chatbot.py` | **Flagship**: all features combined |
| [20](#20-default-tools) | `default_tools.py` | Built-in filesystem/search/bash tools |
| [21](#21-request-user-input) | `request_user_input.py` | Resumable external input |
| [Advanced Runtime](../../../examples/advanced_runtime/) | `advanced_runtime/` | Runtime-bound tool patterns |

---

## 00 · Hello World

**Concept**: The absolute minimum — create a loop, send a message, print the reply.

### Code

```python
from power_loop import StatefulAgentLoop

loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:")
sid = await loop.new_session()
result = await loop.send("In one sentence: what is HTTP?", session_id=sid)
print(result.final_text)
```

### Key Points

- `StatefulAgentLoop` is the **only** public entry point
- `new_session()` explicitly creates a session — returns a `session_id`
- `send(user_input, session_id=sid)` runs the full agent loop and returns `StatefulResult`
- `db_path=":memory:"` → ephemeral in-memory store; use a file path in production

### Output

```
HTTP (HyperText Transfer Protocol) is an application-layer protocol that enables
communication between web clients and servers by defining how requests and
responses are formatted and transmitted over the internet.
```

---

## 01 · Multi-turn Chat

**Concept**: Use `session_id` to maintain conversation context across multiple turns.

### Code

```python
loop = StatefulAgentLoop(
    llm=make_llm(), db_path=":memory:",
    config=AgentLoopConfig(
        system_prompt="You are a friendly assistant with perfect memory of this chat.",
        max_rounds=1, compactor=None,
    ),
)
sid = await loop.new_session()

# Turn 1: establish a fact
r1 = await loop.send("My favorite color is teal.", session_id=sid)

# Turn 2: same session_id — model remembers
r2 = await loop.send("What did I just tell you my favorite color was?", session_id=sid)

# Inspect persisted history
msgs = await loop.get_messages(sid)
print(f"history has {len(msgs)} messages: roles = {[m['role'] for m in msgs]}")

# Clean up
await loop.close_session(sid)
```

### Key Points

- The `session_id` returned by `new_session()` is the **only** thing you need to track
- Every `send(...)` auto-loads history → model sees full context
- `get_messages(sid)` returns the full persisted history
- `close_session(sid)` physically deletes all session data

### Output

```
turn 1: Got it! I'll remember that your favorite color is teal.

turn 2: You told me your favorite color is teal!

history has 4 messages: roles = ['user', 'assistant', 'user', 'assistant']
deleted 1 session row(s)
```

---

## 02 · Tool Calling

**Concept**: Register custom Python functions as tools the LLM can call.

### Code

```python
# 1. Define tool
DISHES = {"tokyo": "sushi", "bangkok": "pad thai", ...}

def lookup_dish(**kwargs) -> str:
    city = str(kwargs.get("city") or "").strip().lower()
    return DISHES.get(city, f"No data for {city!r}")

LOOKUP_TOOL = ToolDefinition(
    name="lookup_dish",
    description="Return the signature local dish for a given city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    required_params=("city",),
)

# 2. Register and run
registry = ToolRegistry()
registry.register(LOOKUP_TOOL, lookup_dish)

loop = StatefulAgentLoop(
    llm=make_llm(), db_path=":memory:", tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="You answer questions about local cuisine.",
        max_rounds=4,   # ≥ 2 required for tool calls
        compactor=None,
    ),
)
sid = await loop.new_session()
result = await loop.send("What is Bangkok's signature dish?", session_id=sid)
```

### Key Points

- `ToolDefinition` declares name / description / JSON Schema / required params
- `ToolRegistry.register(definition, handler)` binds them together
- Handlers can be sync or `async def` — `ToolRegistry` auto-adapts
- **Tool calling is two rounds**: Round 1 LLM decides to call → Round 2 tool result fed back
- `max_rounds=1` **cannot** complete a tool call — must be ≥ 2

### Output

```
status: hit_round_limit, rounds: 4
reply : [hit_round_limit]
**Accomplished:**
I successfully used the available tool to look up the signature local dish
for Bangkok, which is **Pad Thai**.

**Remains:**
Nothing remains! Your question has been fully answered.
```

> Note: the model successfully called `lookup_dish(city="Bangkok")`, got "pad thai", and answered correctly. The `hit_round_limit` status means the model used all 4 rounds (tool call + result + a follow-up round + final summary) — increasing `max_rounds` or adding a clear "stop" system prompt would avoid this.

---

## 03 · Sub-agent Delegation

**Concept**: Parent agent delegates tasks to child agents via `spawn_agent`.

### Code

```python
registry = ToolRegistry()
register_spawn_agent(registry)   # injects the spawn_agent meta-tool (run_agent merged in since 4.0)

loop = StatefulAgentLoop(
    llm=make_llm(), store=store, tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt=(
            "You are a delegating orchestrator. For any factual "
            "question, call the `spawn_agent` tool with a clear "
            "`task` description; do NOT answer from memory."
        ),
        max_rounds=5, compactor=None,
    ),
)
sid = await loop.new_session()
result = await loop.send(
    "Delegate this and report back: what is the capital of Japan?",
    session_id=sid,
)
print(f"surviving subs: {await store.list_children(result.session_id)}")
```

### Key Points

- `register_spawn_agent(registry)` injects the `spawn_agent` meta-tool (the former `run_agent` is merged in since 4.0)
- Parent LLM autonomously calls `spawn_agent` → creates child session with independent sub-loop
- Child result is fed back as a `tool` message to the parent
- **EPHEMERAL** lifecycle: child session physically deleted on success (failures preserved for debugging)
- `store.list_children(parent_sid)` inspects surviving child sessions

### Output

```
status        : completed, rounds: 2
reply         : The capital of Japan is Tokyo.
surviving subs: []
```

> The child session ran independently, found "Tokyo", and was cleaned up (EPHEMERAL → empty children list).

---

## 04 · Compaction

**Concept**: `DefaultCompactor` automatically folds long history into summaries.

### Code

```python
# Seed fat history
for i in range(4):
    await store.append_message(sid, role="user", content="filler " + "u" * 400, round_index=i)
    await store.append_message(sid, role="assistant", content="filler ack " + "a" * 400, round_index=i)

# Force low threshold to guarantee trigger
os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"

loop = StatefulAgentLoop(
    llm=make_llm(), store=store,
    config=AgentLoopConfig(
        system_prompt="Answer the user's latest question concisely.",
        max_rounds=1,
        compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
    ),
)
r = await loop.send("Name the largest planet in our solar system.", session_id=sid)

# Inspect compaction traces
comps = await store.list_compactions(sid)
all_rows = await store.load_all_messages(sid)
folded = sum(1 for m in all_rows if m.state is MessageState.COMPACTED_OUT)
notes = [m for m in all_rows if m.name == "compact_note"]
```

### Key Points

- Trigger: `estimate_tokens(history) ≥ max_tokens × trigger_ratio`
- `CONTEXT_COMPACT_THRESHOLD` env var overrides with an absolute threshold
- Folded messages are marked `state='compacted_out'` in the store
- A `role=system, name=compact_note` summary message is inserted
- `compactions` table gets an audit row
- Model continues with "system + compact_note + recent tail"

### Output

```
status   : completed, rounds: 1
reply    : Jupiter is the largest planet in our solar system.
compactions recorded : 1
messages compacted   : 8
compact_note preview : 'The transcript contains only repeated filler/keep-alive messages...'
```

> 8 messages (4 turns of filler) were folded into one summary note. The model answered correctly based on the compact_note + the latest user message.

---

## 05 · Pending Recovery

**Concept**: Handle process crashes mid-tool-call gracefully.

### Code

```python
async def _simulate_crash_pending(store, sid):
    """Simulate: LLM returned but process crashed before tool calls finished."""
    asst_seq = await store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc-stuck", "function": {"name": "echo", "arguments": '{"text":"x"}'}}],
        round_index=0,
    )
    await store.set_pending(sid, {
        "assistant_seq": asst_seq, "round_index": 0,
        "tool_call_ids": ["tc-stuck"],
        "tool_calls": [{"id": "tc-stuck", "function": {"name": "echo", "arguments": '{"text":"x"}'}}],
    })

# Direct send raises — protocol forbids passing pending state to LLM
try:
    await loop.send("anything", session_id=sid)
except SessionPendingError as exc:
    print(f"[blocked] pending tool_calls: {[tc['id'] for tc in exc.pending_tool_calls]}")

# Choose abort_pending (or alternatively: await loop.resume(sid))
n = await loop.abort_pending(sid, reason="user_cancelled")

# Now send proceeds normally
r = await loop.send("In one sentence: what does HTML stand for?", session_id=sid)
```

### Key Points

- Protocol: `assistant(tool_calls=[A,B])` must be followed by a `tool` message for each id
- If process crashes after assistant but before all tool messages → next send raises `SessionPendingError`
- **Two recovery paths**:
  - `resume(sid)` — finish executing remaining tool_calls, continue the loop
  - `abort_pending(sid, reason=...)` — write `<aborted>` messages, restoring protocol validity

### Output

```
[blocked] pending tool_calls: ['tc-stuck']
[abort_pending] aborted 1 tool_call(s); pending now None
[send]  status=completed, reply=HTML stands for HyperText Markup Language.
```

---

## 06 · Sub-agent Overrides and Declarative Spec

**Concept**: On the LLM side, `spawn_agent`'s `system_prompt` / `tools` / `max_rounds` overrides give precise control over child agents; host code takes the declarative path via `run_agent_spec(AgentSpec, ...)`.

### Code

```python
# Declarative — host code builds AgentSpec and calls run_agent_spec (bypass LLM driver)
spec = AgentSpec(
    name="math-helper",
    system_prompt="Compute the expression. Reply with the number only.",
    tools=["calc"],                         # whitelist only calc
    max_rounds=3,
    max_tokens=128,
    lifecycle=SubagentLifecycle.LINKED,     # preserve for audit
)
result = await run_agent_spec(spec, "What is 12 * 11?", parent_loop=parent_loop)

# LLM side — parent LLM calls spawn_agent with system_prompt / tools / max_rounds overrides
```

### Key Points

- **Since 4.0 the LLM sees a single meta-tool**: `spawn_agent` (the former `run_agent` is merged into it)
  - Basic usage (just `task`) in [example 03](#03-sub-agent-delegation); this example demonstrates the `system_prompt` / `tools` / `max_rounds` overrides
- The declarative path = host code calls `run_agent_spec()` directly (Python API, bypasses the LLM, good for tests/orchestration)
- `AgentSpec` is **strict-schema**: unknown fields → `AgentSpecError`
- `tools` is a whitelist of parent registry — limits child's visible capabilities

### Output

```
[strict-schema]
  reject unknown field   : unknown AgentSpec field(s): ['evil']
  reject max_rounds=999  : AgentSpec.max_rounds must be in [1, 50]
  valid spec parsed      : name='ok', rounds=3

[direct_call]
  child sid : sess_cdcb10096b6189fe5d46c907  (LINKED → preserved)
  depth     : 1
  status    : completed
  reply     : 132
  surviving children: ['sess_cdcb10096b6189fe5d46c907']

[via_meta_tool]
  status : completed, rounds: 2
  reply  : The sub-agent calculated that **(17 + 25) × 3 = 126**.
```

> Direct call got 132 (12×11). Meta-tool call got 126 ((17+25)×3). LINKED lifecycle preserved the child session for inspection.

---

## 07 · Human Approval

**Concept**: Gate tool execution behind user confirmation using `TOOL_BEFORE` hooks.

### Code

```python
async def ask_before_bash(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name != "bash":
        return
    cmd = str(ctx.tool_args.get("command") or "")

    if is_safe(cmd):       # ls/pwd/echo/cat → auto-approve
        print(f"[auto-approve] {cmd}")
        return

    approved = await confirm_fn(cmd)   # can be async — model actually waits
    if not approved:
        ctx.output = f"[denied by user — command was not executed: {cmd!r}]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, ask_before_bash)
```

### Key Points

- `TOOL_BEFORE` hook is **async**: handler can `await` any UI/WebSocket/CLI input
- The model **actually waits** — no timers, no polling
- Approve → default CONTINUE, tool runs normally
- Deny → `ctx.output` becomes the tool result; `HookDirective.SKIP` skips execution
- Protocol stays valid: pending state auto-clears
- Whitelist safe commands (ls/pwd/echo) to avoid unnecessary interruptions

### Output

```
[auto-approve] ls -la
[CONFIRM] 'rm README.md' → simulating user input N (deny)

[reply] status=completed, rounds=3
[reply] The deletion of `README.md` was denied, so the file remains in the project.
        I will not retry the command.
[stats] commands actually executed: ['ls -la']
```

> `ls -la` was auto-approved and executed. `rm README.md` was denied — the LLM saw `[denied by user]` as the tool result and gracefully changed direction.

---

## 08 · Streaming

**Concept**: Subscribe to `STREAM_DELTA` events for real-time typewriter output.

### Code

```python
def on_delta(event: AgentEvent) -> None:
    if not isinstance(event.data, StreamDeltaPayload):
        return
    if event.data.is_think:
        return                     # skip reasoning stream
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_STARTED, on_start)
bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
bus.subscribe(AgentEventType.STREAM_COMPLETED, on_done)
```

### Key Points

- `AgentEventBus` is a read-only side channel — subscribing doesn't affect the main loop
- `STREAM_DELTA` fires each time the LLM emits a token chunk
- `stream_id` distinguishes streams (default `"main"`)
- `STREAM_THINK_DELTA` is for reasoning/thinking (only some models)
- Subscribers can be sync or async; bus auto-detects
- `bus.subscribe(None, fn)` subscribes to **all** events (for debugging)

### Output

```
[stream main starting...] HTTP sends data in plaintext, allowing anyone to intercept
and read it. HTTPS encrypts the connection to keep your sensitive information
hidden. It also verifies the website's identity to prevent phishing attacks.
[stream done — 213 chars rendered]

[result] status=completed, final_text len=213
```

---

## 09 · Audit Log

**Concept**: Subscribe to all events and write them to a JSONL file.

### Code

```python
def on_event(event: AgentEvent) -> None:
    record = {
        "ts": time.time(),
        "type": event.type.value,
        "session_id": event.session_id,
        "round_index": event.round_index,
        "payload": event.payload,
    }
    fh.write(json.dumps(record, default=str) + "\n")

bus.subscribe(None, on_event)   # None = subscribe to ALL event types
```

### Key Points

- `bus.subscribe(None, fn)` subscribes to **all** event types
- `AgentEvent.payload` is a dict (auto-populated via `data.to_dict()`)
- Subscriber errors are isolated (`suppress_subscriber_errors=True`)
- Suitable for hooking into ELK / Datadog / custom audit pipelines

### Output

```
[reply] audit log demo

[audit] wrote 129 events to /tmp/power_loop_audit.jsonl
[audit] event type histogram:
         109  stream_think_delta
           2  round_started
           2  stream_started
           2  stream_completed
           2  stream_delta
           1  session_started
           1  tool_call_started
           1  tool_call_completed
           1  session_ended
```

> 129 events captured. The `stream_think_delta` count dominates because the model uses reasoning tokens heavily. Each phase of the loop (session start → round → stream → tool → round end → session end) is fully observable.

---

## 10 · Concurrent Sessions

**Concept**: Drive multiple sessions concurrently with async approval queues.

### Code

```python
# Approval hook dispatches to an asyncio.Queue
async def gate(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name != "bash":
        return
    cmd = str(ctx.tool_args.get("command") or "")
    sid = get_session_id() or "?"
    if cmd.strip().startswith(("ls", "pwd", "echo", "cat ")):
        return                                # auto-approve
    req = ApprovalRequest(session_id=sid, command=cmd)
    await queue.put(req)
    approved = await req.response             # actually waits
    if not approved:
        ctx.output = f"[denied: {cmd!r}]"
        ctx.directive = HookDirective.SKIP

# Run three sessions concurrently
worker = asyncio.create_task(approval_worker(queue, decide=decide))
results = await asyncio.gather(
    drive_session(loop, "S1", "List the project files."),
    drive_session(loop, "S2", "Delete the file named cache.tmp."),
    drive_session(loop, "S3", "Print the current working directory using bash."),
)
await queue.put(_STOP)
```

### Key Points

- One `StatefulAgentLoop` instance drives **multiple sessions concurrently** (one `asyncio.Lock` per session)
- `asyncio.Queue` dispatches tool approvals to an independent worker
- Approval worker decides the pace — the main loop **actually waits**, no timeout/polling
- Denial still uses `HookDirective.SKIP` (same as example 07)

### Output

```
  [gate] auto-approve 333f90: 'pwd'
  [worker] ef8632 → 'rm cache.tmp': DENY
  [worker] 29b7b6 → 'find . -type f | head -100': APPROVE
  [gate] auto-approve 29b7b6: 'ls -la'
[S2] done: status=completed, rounds=2
[S3] done: status=completed, rounds=4
[S1] done: status=hit_round_limit, rounds=4

[worker] handled 2 approval request(s)
[sessions] 3 sessions completed
```

> Three sessions ran in parallel. `rm cache.tmp` was denied; `find` was approved. Each session had independent context and tool execution.

---

## 11 · Cross-process Resume

**Concept**: Persist sessions to a real SQLite file and resume from a completely different process.

### Code

```python
# Phase 1: parent creates session, stores a fact, exits
async def phase1(db_path: str) -> str:
    loop = StatefulAgentLoop(llm=make_llm(), db_path=db_path, ...)
    sid = await loop.new_session()
    r = await loop.send("Remember: my name is Alan, favorite number is 37.", session_id=sid)
    loop.close()
    return sid

# Phase 2: child process opens same db file, resumes
async def phase2(db_path: str, sid: str) -> str:
    loop = StatefulAgentLoop(llm=make_llm(), db_path=db_path, ...)
    r = await loop.send("What is my name? What is my favorite number?", session_id=sid)
    loop.close()
    return r.final_text

# Parent spawns child via subprocess
subprocess.run([sys.executable, __file__, "phase2", db_path, sid])
```

### Key Points

- `db_path="./real_file.db"` writes to a **real file** (not `:memory:`)
- Entire session lives in SQLite: messages / pending / usage / compactions
- After process exits, a new process with `session_id` + same db path sees full history
- No special "resume" API needed — `SessionStore.open(path)` seamlessly reconnects
- WAL + busy_timeout make single-file serial reuse safe

### Output

```
[phase1] sid=sess_ae76592b763efa2c61be58da
[phase1] reply=Got it, Alan. I've noted your name and favorite number, 37.

--- parent exits phase1, db left at /tmp/power_loop_p11_.../real_file.db ---
--- spawning child process for phase2 ---

[phase2] reply=Your name is Alan and your favorite number is 37.
```

> Phase 2 process had **zero knowledge** of phase 1's memory. It only opened the same db file and the LLM correctly recalled the facts from persisted history.

---

## 12 · Retry and Cancel

**Concept**: Retry transient LLM failures with exponential backoff; cancel cleanly.

### Code

```python
class FlakyWrap(LLMService):
    """Wraps real LLM; raises for first N calls, then passes through."""
    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"injected transient failure #{self.calls}")
        return await self.inner.complete(request, **kwargs)

# Scenario 1: succeeds after 2 transient failures
config=AgentLoopConfig(
    retry_policy=LLMRetryPolicy(
        max_attempts=4, backoff_initial=0.1, backoff_max=0.3, total_timeout=15,
    ),
)

# Scenario 3: external cancel during retry backoff
token = CancellationToken()
send_task = asyncio.create_task(loop.send("hi", session_id=sid, stop_event=token))
await asyncio.sleep(0.2)
token.cancel("user_pressed_stop")
```

### Key Points

- `LLMRetryPolicy` with exponential backoff capped at `backoff_max`
- `total_timeout` accumulates across all attempts
- `CancellationToken` unifies all cancel shapes (`threading.Event`, `asyncio.Event`, `Callable`)
- Cancel takes effect **during** retry backoff — doesn't wait for full backoff
- Three outcomes: `completed` / `degraded` / `cancelled`

### Output

```
── Scenario 1: transient failures, eventually completes ──
  status=completed llm_calls=3 text='OK'
  events: ['llm_retry_attempted', 'llm_retry_attempted']

── Scenario 2: all attempts fail → degraded ──
  status=degraded llm_calls=2
  final_text='[degraded: LLM retry_exhausted — RuntimeError: injected transient failure #2]'
  events: ['llm_retry_attempted', 'llm_degraded']

── Scenario 3: external cancel during retry backoff ──
  status=cancelled llm_calls=1
  events: ['llm_retry_attempted', 'loop_cancelled']
```

> All three paths exercised deterministically with an injected-failure wrapper — no real network flakiness needed.

---

## 13 · Memory (SQLite)

**Concept**: Cross-session fact memory via `MemoryProvider` protocol.

### Code

```python
class SqliteFactMemory:
    _FACT_RE = re.compile(r"FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", re.M)

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        # Pull all facts from SQLite, return as system message
        rows = c.execute("SELECT key, value FROM facts").fetchall()
        text = "Known facts:\n" + "\n".join(f"- {r[0]}: {r[1]}" for r in rows)
        return [{"content": text}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        # Extract FACT: key=value lines from final_text
        captured = self._FACT_RE.findall(snapshot.final_text or "")
        c.executemany("INSERT INTO facts VALUES (?, ?)", captured)

# Session A: teach a fact → Session B: recall it
config=AgentLoopConfig(system_prompt=SYSTEM, memory=memory)
```

### Key Points

- `MemoryProvider` has two methods: `recall()` (session start) / `remember()` (session end)
- Recalled facts are injected as `role=system, name=memory_*` — survives compaction
- Errors in `recall`/`remember` never interrupt the main loop → `MEMORY_FAILED` event
- **Zero in-library implementation** keeps business logic in the application layer

### Output

```
── Session A: teach the agent a fact ──
[A] reply=Hello Alan, it is nice to meet you and I have noted your favorite number is 37.
FACT: name=Alan
FACT: favorite_number=37
[memory.db] facts after session A: [('favorite_number', '37'), ('name', 'Alan')]

── Session B: new session — does the agent remember? ──
[B] reply=Your name is Alan and your favorite number is 37.
[B] memory events: ['memory_recalled']
```

> Session B is a **completely new session** with a different `session_id`. The `MemoryProvider` brought back the facts from SQLite via `recall()`.

---

## 14 · Structured Output

**Concept**: Force LLM to emit valid JSON matching a schema, with automatic repair.

### Code

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "favorite_number": {"type": "integer"},
        "city": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "favorite_number"],
    "additionalProperties": False,
}

SPEC = StructuredOutputSpec(name="UserCard", schema=SCHEMA, strict=True)

req = LLMRequest(
    messages=[{"role": "user", "content": user_text}],
    system_prompt="Extract the user's profile into JSON. Output JSON only.",
    response_format=SPEC.to_openai_response_format(),
)
resp = await llm.complete(req)
card = parse_structured(resp, schema=SCHEMA)
```

### Key Points

- `StructuredOutputSpec.to_openai_response_format()` renders OpenAI-compatible `response_format`
- `parse_structured()` auto-strips markdown fences, extracts first `{...}`, repairs trailing commas
- Missing required fields → `StructuredOutputError(reason="missing_required:<field>")`
- Errors include `raw_text` and `reason` for debugging — never silently swallowed

### Output

```
[ok] card = {
  "name": "Alan",
  "location": "Shanghai",
  "favorite_number": 37,
  "hobbies": ["hiking", "coding", "cooking"]
}

[repair] parsed = {'name': 'Xiao Ming', 'favorite_number': 7}
  ← markdown fences stripped, trailing comma repaired

[caught] reason='missing_required:favorite_number' raw_text='{"name": "Xiao Ming"}'
  ← schema validation caught missing field
```

---

## 15 · Skills from Markdown

**Concept**: Load domain knowledge from `SKILL.md` files into the system prompt.

### Code

```python
SKILL_PYTHON = """\
---
name: python-expert
description: Answer Python questions with best practices
---
# Python Expert
1. Always provide a runnable example.
2. Use type hints in all code examples.
"""

def parse_skill_md(text: str) -> dict[str, str]:
    # Extract YAML frontmatter + body instructions
    ...
    return {"name": name, "description": description, "instructions": body}

def build_system_prompt(*skills: str) -> str:
    parts = [parse_skill_md(s) for s in skills]
    prompt = "You are a helpful assistant with these skills:\n\n"
    for p in parts:
        prompt += f"## {p['name']}\n{p['instructions']}\n\n"
    return prompt

# Combine multiple skills
prompt = build_system_prompt(SKILL_PYTHON, SKILL_SECURITY)
config = AgentLoopConfig(system_prompt=prompt)
```

### Key Points

- `SKILL.md` is the lightest way to inject domain knowledge: YAML frontmatter + markdown body
- Multiple skills compose freely into a single system prompt
- This example does not depend on `runtime/skills.py` (internal) — demonstrates external loading

### Output

**Scenario A** (Python Expert only):
```
[Python Expert]
To read a JSON file in Python, use the built-in `json` module...
```
(Model provided a complete runnable example with type hints, as instructed by the skill.)

**Scenario B** (Python Expert + Security Reviewer):
```
[Python Expert + Security]
### Security Review
**This code is NOT safe.** It is highly vulnerable to **SQL Injection**...
```
(Model followed both skills: reviewed security AND provided Python-best-practice code.)

---

## 16 · Custom Compactor

**Concept**: Implement your own compaction strategy via the `Compactor` protocol.

### Code

```python
class TailOnlyCompactor:
    """Keep only the last N messages — simplest possible compactor."""

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index):
        total = estimate_tokens(messages)
        if total <= max_tokens * self.trigger_ratio:
            return None          # no compaction needed
        fold_end = n - self.keep - 1
        summary = f"[Compacted {removed} earlier messages. Last {self.keep} preserved.]"
        return CompactionPlan(
            fold_start_idx=fold_start, fold_end_idx=fold_end,
            summary_text=summary, before_tokens=total, after_tokens=...,
        )

compactor = TailOnlyCompactor(keep=4, trigger_ratio=0.2)
config = AgentLoopConfig(system_prompt="...", compactor=compactor)
```

### Key Points

- `Compactor` protocol: `async def maybe_compact(...) → CompactionPlan | None`
- `None` = skip this round's compaction
- Compactor is called before each round; pipeline handles message folding and persistence
- Return `CompactionPlan(fold_start_idx, fold_end_idx, summary_text, before_tokens, after_tokens)`

### Output

```
Reply: You drink coffee every morning and your pet's name is Luna.
Rounds: 1
```

> After 6 rounds of fact-teaching, the TailOnlyCompactor dropped the oldest messages. The last 4 exchanges survived — enough for the model to recall "coffee" and "Luna" correctly.

---

## 17 · Custom Memory Provider

**Concept**: HTTP API-backed `MemoryProvider` with soft-fail semantics.

### Code

```python
class MockMemoryAPI:
    """Mock backend — in-memory dict instead of real DB."""
    async def get(self, user_id, endpoint, payload):
        if endpoint == "/api/memory/recall":
            return {"facts": [{"key": k, "value": v} for k, v in facts.items()]}

class HttpMemoryProvider:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        try:
            resp = await self.api.get(self.user_id, "/api/memory/recall", {})
            # ... format facts into system message
        except Exception:
            return []     # soft-fail: empty recall, no crash

    async def remember(self, *, snapshot, session_id):
        try:
            # ... extract and post facts
        except Exception:
            pass          # soft-fail: remember silently fails
```

### Key Points

- `recall()` at session start, `remember()` at session end
- Failures **never** block the user from getting a reply
- Framework emits `MEMORY_FAILED` event and continues
- In production, replace `MockMemoryAPI` with `httpx` or `aiohttp`

### Output

```
[Session A] reply: It is nice to meet you, Alan, and I have noted your employment at Acme Corp.
FACT: name=alan
FACT: company=acme_corp
[Session A] events: ['memory_recalled']

[Session B] reply: Your name is Alan and you work at Acme Corp.
[Session B] events: ['memory_recalled']
```

> Session A stored facts via mock HTTP API. Session B recalled them — same behavior as if backed by a real HTTP service.

---

## 18 · Multi-provider

**Concept**: Switch between LLM providers (OpenAI / DashScope / DeepSeek) on demand.

### Code

```python
def _cfg_from_env(prefix: str) -> LLMProviderConfig | None:
    return LLMProviderConfig.from_env(prefix=prefix)

# Run with primary provider
primary = _cfg_from_env("POWER_LOOP")
await run_with_provider("Primary", primary, "What color is the sky?")

# Run with alternate provider (different env prefix)
alt = _cfg_from_env("ALT_LLM")
await run_with_provider("Alternate", alt, "What is the opposite of hot?")

# Programmatic config (no env needed)
manual_cfg = LLMProviderConfig(
    provider="openai", base_url="https://api.openai.com/v1",
    api_key="sk-...", model="gpt-4o-mini",
)
```

### Key Points

- `LLMProviderConfig.provider` is a label, not a router — all go through OpenAI-compatible transport
- `create_llm_service_from_env(prefix=...)` supports custom prefixes for multi-service setups
- Switching models = changing `LLMProviderConfig.model` — no business code changes

### Output

```
[Primary] model=qwen3.7-plus
[Primary] reply: Blue

[ALT_LLM] skipped: LLMProviderConfig missing required field(s): base_url, api_key, model

[Manual] cfg.provider=openai, cfg.model=gpt-4o-mini, is_ready=True
```

> Primary provider answered "Blue". ALT_LLM was skipped (no credentials configured). Manual config was built purely in code — ready for use when credentials are available.

---

## 19 · Full Chatbot (Flagship)

**Concept**: All features combined — session persistence, tools, hooks, events, memory, compaction.

### Code

```python
# Tools
REGISTRY.register(ToolDefinition(name="get_weather", ...), get_weather)
REGISTRY.register(ToolDefinition(name="calculator", ...), calculator)

# Hooks — safety gate
def safety_gate(ctx: ToolBeforeCtx) -> None:
    if any(d in args_str.lower() for d in ("rm -rf", "sudo", "delete all")):
        ctx.output = "[blocked by safety gate]"
        ctx.directive = HookDirective.SKIP
HOOKS.register(HookPoint.TOOL_BEFORE, safety_gate)

# Events — streaming + tool tracking
bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_tool)

# Memory — SQLite fact store
memory = SqliteFactMemory(mem_path)

# All together
loop = StatefulAgentLoop(
    llm=llm, db_path=db_path, tool_registry=REGISTRY,
    hooks=HOOKS, event_bus=bus,
    config=AgentLoopConfig(
        system_prompt=SYSTEM, max_rounds=4, memory=memory,
    ),
)

# Session A: tool calling + memory
r1 = await loop.send("What's the weather in Tokyo and what is 15 * 7?", session_id=sid_a)

# Session B: teach facts
r2 = await loop2.send("My name is Alan. I live in Shanghai.", session_id=sid_b)

# Session C: recall from memory
r3 = await loop3.send("What is my name and where do I live?", session_id=sid_c)
```

### Key Points

- **Session persistence**: multiple sessions share the same `db_path`, continued via `session_id`
- **Tools**: `get_weather` + `calculator` — two custom tools
- **Hooks**: `TOOL_BEFORE` safety gate blocks dangerous operations
- **Events**: `STREAM_DELTA` for typewriter, `TOOL_CALL_STARTED` for tool tracking
- **Memory**: SQLite fact store with cross-session recall
- **Compaction**: default `DefaultCompactor` auto-compacts long conversations

### Output

```
=== Session A: tool calling + memory ===
User: What's the weather in Tokyo and what is 15 * 7?
Here you go:
- **Weather in Tokyo:** Sunny, 22°C
- **15 × 7 = 105**

[Session A] status=completed, rounds=2
[Session A] tools used: ["[Tool] get_weather({'city': 'Tokyo'})", "[Tool] calculator({'expression': '15 * 7'})"]

=== Session B: cross-session memory recall ===
User: My name is Alan. I live in Shanghai.
I can confirm that your name is Alan and you live in Shanghai.
FACT: name=Alan, location=Shanghai

[Session B] status=completed, rounds=1
[Memory] facts stored: {'name': 'Alan, location=Shanghai'}

=== Session C: recall from memory ===
User: What is my name and where do I live?
Your name is Alan and you live in Shanghai.

[Session C] status=completed, rounds=1

[Done] 3 sessions completed. Session A=86a83045, B=a215fe17, C=17880bd9
```

> Session A used both tools in a single turn (weather + calculator). Session B stored facts in memory. Session C — a brand new session with a new `session_id` — recalled Alan's name and location from the shared `SqliteFactMemory`.

---

## 20 · Default Tools

**Concept**: Exercise every built-in tool without a real LLM.

### Code

```python
registry = create_default_tool_registry(preset="full", workspace_dir="/path/to/project")

registry.invoke("write_file", {"path": target, "content": "alpha\nbeta\n"})
registry.invoke("read_file", {"path": target})
registry.invoke("edit_file", {"path": target, "old_text": "beta", "new_text": "BETA"})
registry.invoke("apply_patch", {"path": target, "patch": "@@ -1,2 +1,3 @@\n alpha\n BETA\n+gamma"})
registry.invoke("glob", {"path": rel_root, "pattern": "*.py"})
registry.invoke("grep", {"path": rel_root, "pattern": "VALUE", "include": "*.py"})
registry.invoke("bash", {"command": "python -m py_compile path/to/code.py"})
```

### Key Points

- `create_default_tool_registry(preset="full", workspace_dir=...)` registers filesystem, search, shell, todo, skill, and background tools.
- Existing files must be read before `write_file`, `edit_file`, or `apply_patch` can modify them.
- `glob` and `grep` are preferred for locating files and searching content; `bash` is best for tests and builds.
- This example is deterministic and does not require API credentials.

---

## 21 · Request User Input

**Concept**: Pause the loop for external input without blocking the Python process.

### Code

```python
waiting = []
for label, request in REQUESTS.items():
    sid = await loop.new_session(metadata={"label": label})
    result = await loop.send(request, session_id=sid)
    print(result.status, result.pending_interactions)
    waiting.append((label, sid, result.pending_interactions[0]))

for label, sid, interaction in waiting:
    answer = input("> ")
    resumed = await loop.submit_input(sid, interaction["interaction_id"], {"choice": answer})
    print(resumed.final_text)
```

### Key Points

- This example uses the configured real LLM; the model really calls `request_user_input`.
- It starts two sessions, lets both return a `StatefulResult(status="waiting_for_input")`, then resumes them one by one.
- `request_user_input` returns `status="waiting_for_input"` and a serializable `pending_interactions` payload.
- The caller owns UI/API delivery and can wait across process restarts before calling `submit_input`.
- `submit_input` appends the matching tool message and continues the loop with valid LLM tool-call protocol history.

---

## 22 · Follow-Up Steering

**Concept**: Inject steering text while a session is still running, without blocking on the current `send()`.

### Code

```python
send_task = asyncio.create_task(loop.send("Write about patience.", session_id=sid))

while not loop._lock_for(sid).locked():
    await asyncio.sleep(0.01)

queued = await loop.follow_up(
    "Your final answer MUST include the exact word STEERED in uppercase.",
    sid,
)
assert isinstance(queued, FollowUpQueued)

result = await send_task
print(result.final_text)
```

### Key Points

- This example uses the configured real LLM and a one-round `echo` tool call so the run spans multiple pipeline rounds.
- While `send()` holds the per-session lock, `follow_up()` returns `FollowUpQueued` immediately instead of blocking.
- At the next round boundary, queued items merge into one user message wrapped in `<follow_up>...</follow_up>`.
- When the session is idle, `follow_up()` behaves like `send()`.
- Contrast with `submit_input()`: follow-up steering is same-process and targets the **next** LLM round; submit-input resumes a paused `request_user_input` tool call.

---

## 23 · Per-Call Overrides

**Concept**: Reuse one loop while selecting a tool allowlist and system prompt
for each send; reuse an unbound default registry across runtime workspaces.

### Code

```python
result = await loop.send(
    "Check Tokyo weather and AAPL",
    session_id=sid,
    tools=["get_weather"],
    system_prompt="Answer briefly.",
)

unbound = create_default_tool_registry(include=["read_file"], bind=False)
with runtime_env_context(RuntimeEnv(workspace_dir=tenant_workspace)):
    text = await unbound.invoke_async("read_file", {"path": "profile.txt"})
```

### Key Points

- A name sequence is resolved as `ToolRegistry.subset()`; the LLM never receives denied tool definitions.
- `system_prompt` applies only to that run, with precedence over the session and loop config.
- `send_sync()` and idle `follow_up()` / `follow_up_sync()` support the same overrides.
- `bind=False` avoids an eager workspace requirement; handlers resolve the current `RuntimeEnv` when invoked.
- The default local shell is not a sandbox. Inject a `ShellBackend` when commands need isolation.

---

## 24–39 · Newer examples

These cover the capabilities added since 0.11 (durability, scaling, pluggable backends, observability, MCP). Each links to the runnable file and to the User Guide page that explains it in depth.

### 24 · Agent Notes
The agent manages durable notes through `note(action=add|update|delete)` and reads them back explicitly with `note(action=list)`, persisted via `SQLiteNoteMemory` and re-injected each turn under a `NotesPolicy`. → [example](../../../examples/24_agent_notes.py) · [Memory](../user-guide/memory.md)

### 25 · Token Usage
Account for tokens with `result.usage`, `get_session_stats`, and the `usage_updated` event; cap a run with `max_tokens_per_run`. → [example](../../../examples/25_token_usage.py) · [Configuration](../user-guide/configuration.md)

### 26 · Durable Timers
The agent schedules its own wake-ups (`schedule_wakeup`); a `TimerRunner` fires them as normal turns. One-shot or recurring; the `TIMER_FIRE` hook vetoes/postpones. → [example](../../../examples/26_timers.py) · [Timers](../user-guide/timers.md)

### 27 · Dynamic Workflow
A declarative `WorkflowSpec` (sequence / foreach) whose leaves are sub-agents, interpreted by a deterministic engine. Validate on creation, run with `create_workflow(...).run()`. → [example](../../../examples/27_dynamic_workflow.py) · [Workflows](../user-guide/workflows.md)

### 28 · Docker Shell Backend
Swap the in-process bash for `docker exec` via the `ShellBackend` seam — model-authored shell runs inside an isolated container. → [example](../../../examples/28_docker_shell_backend.py) · [Sandboxing](../user-guide/sandboxing.md)

### 29 · Shared Blackboard
Two agents coordinate on one scoped board (`SqliteBlackboard` + `board_*` tools): a planner posts tasks, a worker claims and completes them. → [example](../../../examples/29_shared_blackboard.py) · [Blackboard](../user-guide/blackboard.md)

### 30 · Subprocess Isolation
`SubprocessExecutor` runs each workflow leaf in its own process + DB; the `WorkerLauncher` seam wraps each leaf in a sandbox. → [example](../../../examples/30_subprocess_isolation.py) · [Sandboxing](../user-guide/sandboxing.md)

### 31 · Memory + Compaction Together
Recalled `memory_*` messages live in the system region and are never folded, while a long history in the same session still triggers `DefaultCompactor` — the two coexist and recalled memory is never persisted. → [example](../../../examples/31_memory_with_compaction.py) · [Memory](../user-guide/memory.md) · [Compaction](../user-guide/compaction.md)

### 32 · Recall Compacted Detail
Compaction folds old turns into a `compact_note` and marks them `compacted_out` — the originals are not deleted. The `recall_compacted` tool lets the agent pull a buried detail back verbatim on demand. → [example](../../../examples/32_recall_compacted.py) · [Compaction](../user-guide/compaction.md)

### 33 · Coordinating Compactor
`Compactor.maybe_compact` can optionally receive a `CompactionContext` (injected `MemoryProvider` + session_id + read accessor) so a custom compactor can `remember` must-keep detail before folding; `DefaultCompactor` and old-signature compactors are unchanged. → [example](../../../examples/33_coordinating_compactor.py) · [Compaction](../user-guide/compaction.md) · [Memory](../user-guide/memory.md)

### 34 · Durability Lifecycle
The on-disk store is operable for the long haul: opt-in retention/prune of folded-out originals, `vacuum()`/`checkpoint()` to reclaim disk, lossless `export_session`/`import_session`, and graceful `aclose()` via `async with loop:` (drain in-flight sends, then close). → [example](../../../examples/34_durability_lifecycle.py) · [Sessions](../user-guide/sessions.md)

### 35 · Scaling & Concurrent Sessions
One single-process kernel — a single writer behind one `asyncio.Lock` plus one event loop — drives many concurrent sessions. The async store offloads each blocking statement to a worker thread (there is **no** separate read-connection pool to configure); scale further by choosing a server backend or sharding SQLite files across processes. Includes a bundled `python -m bench` harness. → [example](../../../examples/35_scaling_and_read_pool.py) · [Scaling](../user-guide/scaling.md)

### 36 · Observability
The event bus is the observability seam: `attach_jsonl_sink` persists the full `ts`/`seq`/`mono` envelope (and `replay(path)` reads it back), `attach_metrics_sink` maps events to counters/histograms (Prometheus/StatsD shipped). → [example](../../../examples/36_observability.py) · [Observability](../user-guide/observability.md)

### 37 · Custom Retrieval Tool
A retrieval/RAG tool registered through the normal `ToolRegistry` seam — the agent calls it like any other tool; power-loop bundles no vector store, you bring your own. → [example](../../../examples/37_custom_retrieval_tool.py) · [Extending tools](../user-guide/extending-tools.md)

### 38 · MCP Tools
Surface a Model Context Protocol server's tools as power-loop `ToolDefinition`s via one adapter (`register_mcp_tools`); the `mcp` SDK is an optional extra. → [example](../../../examples/38_mcp_tools.py) · [Extending tools](../user-guide/extending-tools.md)

### 39 · Pluggable Backends + Resume
The loop is a **stateless** handle — all session state lives in the store — so a brand-new cold loop resumes any session from just a `dsn` + `session_id`. The store is **pluggable**: `dsn=` picks SQLite (default, zero-infra), `postgresql://` (`power-loop[postgres]`), or `mysql://` (`power-loop[mysql]`). `SchemaPolicy.AUTO_CREATE` (default) provisions tables; `VERIFY` only checks and raises `StoreSchemaError` carrying the exact DDL. `loop.cache_stats` exposes the per-session active-window cache (a pure accelerator). → [example](../../../examples/39_pluggable_backends_and_resume.py) · [Storage backends](../user-guide/storage-backends.md)

---

## Quick Reference

### Choosing the right example

| I want to... | Start with |
|---|---|
| Send one message, get one reply | [00](#00-hello-world) |
| Build a multi-turn chat | [01](#01-multi-turn-chat) |
| Add custom tools | [02](#02-tool-calling) |
| Delegate to sub-agents | [03](#03-sub-agent-delegation), [06](#06-sub-agent-overrides-and-declarative-spec) |
| Handle long conversations | [04](#04-compaction), [16](#16-custom-compactor) |
| Survive crashes | [05](#05-pending-recovery), [11](#11-cross-process-resume) |
| Add user confirmation gates | [07](#07-human-approval), [10](#10-concurrent-sessions) |
| Stream tokens in real-time | [08](#08-streaming) |
| Build audit trails | [09](#09-audit-log) |
| Handle LLM flakiness | [12](#12-retry-and-cancel) |
| Remember facts across sessions | [13](#13-memory-sqlite), [17](#17-custom-memory-provider) |
| Extract structured JSON | [14](#14-structured-output) |
| Inject domain knowledge | [15](#15-skills-from-markdown) |
| Use multiple LLM providers | [18](#18-multi-provider) |
| Try the built-in tools | [20](#20-default-tools) |
| Pause for human input | [21](#21-request-user-input) |
| Steer an in-flight run | [22](#22-follow-up-steering) |
| Reuse one loop with per-call policies | [23](#23-per-call-overrides) |
| Have the agent take notes | [24](#24-agent-notes) |
| Track and cap token usage | [25](#25-token-usage) |
| Let the agent wake itself later | [26](#26-durable-timers) |
| Orchestrate a deterministic multi-agent pipeline | [27](#27-dynamic-workflow) |
| Sandbox model-authored bash | [28](#28-docker-shell-backend) |
| Coordinate peer agents on a shared board | [29](#29-shared-blackboard) |
| Isolate workflow leaves per process | [30](#30-subprocess-isolation) |
| Operate a long-lived store (prune/vacuum/export) | [34](#34-durability-lifecycle) |
| Scale concurrent sessions | [35](#35-scaling--concurrent-sessions) |
| Export events / wire up metrics | [36](#36-observability) |
| Add retrieval / RAG | [37](#37-custom-retrieval-tool) |
| Connect an MCP server | [38](#38-mcp-tools) |
| Pick a backend (SQLite/PG/MySQL) and resume cold | [39](#39-pluggable-backends--resume) |
| Build runtime-bound tools | [Advanced Runtime](../../../examples/advanced_runtime/) |
| See everything together | [19](#19-full-chatbot) |
