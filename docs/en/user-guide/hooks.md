# Hooks

[中文](../../zh/user-guide/hooks.md) | [User Guide](../index.md)

Hooks are the **control-flow** channel. Unlike [Events](events.md) (read-only observation), hooks can modify messages, skip tools, short-circuit LLM calls, or end the loop.

> **Full reference**: [docs/hooks.md](../../hooks.md) — every `HookPoint` with typed Ctx fields, directives, and code snippets.

## Quick Example

```python
from power_loop import AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def block_dangerous(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash" and "rm -rf" in str(ctx.tool_args):
        ctx.output = "[blocked: destructive command]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_dangerous)

loop = StatefulAgentLoop(llm=llm, hooks=hooks, config=config)
```

## Hook Point Overview

| Hook Point | When | Common Use |
|---|---|---|
| `session.start` | Before the first round | Inject warmup messages |
| `session.end` | After the loop ends | Audit, close connections |
| `round.start` | Before each round | Budget check, `BREAK` to stop |
| `round.end` | After each round | Per-round metrics |
| `llm.before` | Before LLM call | Modify request, cache check (`SHORT_CIRCUIT`) |
| `llm.after` | After LLM response | Content moderation, `BREAK` to stop |
| `round.decide` | Before tool execution | Skip all tools (`SKIP`) |
| `tools.batch.before` | Before tool batch | Batch-level gate |
| `tools.batch.after` | After tool batch | Read-only observation |
| `tool.before` | Before each tool | **Security gate**, user confirmation |
| `tool.after` | After successful tool | Trim output, modify result |
| `tool.error` | On tool exception | Swallow error (`SKIP`) or retry (`SHORT_CIRCUIT`) |
| `compact.before` | Before compaction | Skip compaction this round |
| `compact.after` | After compaction | Read-only observation |
| `message.append` | Before message is stored | PII redaction, metadata injection |
| `memory.recalled` | After memory recall | Filter/redact memory; `SKIP` to drop |

## Directives

| Directive | Effect |
|---|---|
| `CONTINUE` | Default. Pipeline proceeds normally. |
| `SKIP` | Skip the current phase. Exact behavior varies by hook point. |
| `BREAK` | End the loop immediately. `status="completed"`. |
| `SHORT_CIRCUIT` | Replace the real operation with a fake result (e.g., cached LLM response). |

## Common Patterns

### Security Gate

```python
def block_destructive(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash":
        cmd = ctx.tool_args.get("command", "")
        if any(bad in cmd for bad in ("rm -rf", "sudo", ":(){:|:&};:")):
            ctx.output = "[blocked: destructive]"
            ctx.directive = HookDirective.SKIP
```

### LLM Cache (Short-Circuit)

```python
def cache_check(ctx: LlmBeforeCtx) -> None:
    key = hash(str(ctx.messages))
    if (cached := cache.get(key)):
        ctx.output = LLMResponse(raw_text=cached)
        ctx.directive = HookDirective.SHORT_CIRCUIT
```

### Budget Cap

```python
total_cost = 0

def cap_budget(ctx: RoundStartCtx) -> None:
    if total_cost > 0.50:  # dollars
        ctx.reason = "budget_exhausted"
        ctx.directive = HookDirective.BREAK
```

### User Confirmation (Async)

```python
async def ask_before_bash(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name != "bash":
        return
    if not await your_ui.confirm(ctx.tool_args.get("command", "")):
        ctx.output = "[denied by user]"
        ctx.directive = HookDirective.SKIP
```

### Content Moderation

```python
def block_pii(ctx: LlmAfterCtx) -> None:
    text = getattr(ctx.output, "raw_text", "") or ""
    if PII_PATTERN.search(text):
        ctx.directive = HookDirective.BREAK
```

### Memory Gate

```python
def gate_memory(ctx: MemoryRecalledCtx) -> None:
    if not user_has_consented(ctx.session_id):
        ctx.directive = HookDirective.SKIP
```

## Registration

```python
hooks = AgentHooks()
hooks.register(HookPoint.TOOL_BEFORE, sync_handler)
hooks.register(HookPoint.TOOL_BEFORE, async_handler)  # async also works

loop = StatefulAgentLoop(llm=llm, hooks=hooks, ...)
```

Handlers run in registration order. If any handler sets a non-`CONTINUE` directive, subsequent handlers for the same hook point are **not** executed.

## Hook vs Event

| | Hook | Event |
|---|---|---|
| Can change control flow? | Yes (`SKIP`, `BREAK`, `SHORT_CIRCUIT`) | No |
| Can modify data? | Yes (messages, tools, LLM request) | No |
| Error handling | Handler errors propagate (stop the chain) | Subscriber errors are suppressed |
| Use for | Security, routing, caching, moderation | Observability, audit, streaming |

## Next

- [Events](events.md) — observe the loop without interfering
- [Full Hook Reference](../../hooks.md) — every Ctx field and directive combo