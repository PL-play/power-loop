# Tool context policy — asynchronous summaries (provisional)

Opt in using `AgentLoopConfig(tool_context=ToolContextConfig(...))`. Tool definitions declare
`ToolContextPolicy`; application overrides change specified fields only. Host metadata is not
part of the model-facing tool schema. The default remains disabled.

```python
from power_loop import ToolContextConfig
config = ToolContextConfig(
    cache_scope="tenant:123:permission-version:4",  # otherwise isolated by session
    overrides={"read_file": {"general": "llm_summary"}},
    cache_ttl_s=604800, cache_max_entries=10000, cache_max_bytes=67108864,
)
```

Rules per tool: `general` (between sends and fuse fallback), `insend_fuse` (fuse level 1),
`age_retirement`; transformations: `preserve`, `rule`, `llm_summary`, `truncate`, `archive`.
See docs/*/user-guide/send-context-projection.md, "one mechanism". Rule precedence: explicit host rule, ToolDefinition.project, representation adapter.
Bash's explicit `context_policy=keep` preserves parameters/results within the current send (both fuse levels).
The load_skill definition preserves in-send evidence. This API does not change bash's schema or
output capture. Existing cross-send fold remains a separate mechanism.

## Critical path

After a tool result is durably appended, eligible summary work is enqueued. A shared store worker
performs model calls independently (two global active jobs, one per owner session). No model
request or usage delivery is awaited by a send. At a context boundary, ready cache -> rules ->
full original. Rules must shrink safely or the full paired arguments/result remain. Selected
representations are persisted; late results do not rewrite them. Cached text excludes recall
coordinates; the current source's coordinate is appended when used.

A result's `usage` is an as-of-return snapshot, including known auxiliary attempts started in that
invocation and `auxiliary_pending`. It cannot promise final cost for work that outlives the send.
The durable attempt ledger/session statistics are the eventual accounting authority; already
returned workflow results are not retroactively mutated. Session ownership and source send are
separate. An application can rebuild aggregate reports from the attempt ledger.

## Storage and deduplication

Store schema v9 adds `context_cache` (queue + derived cache) and `context_attempts` (request ledger).
The key hashes permission scope, tool name, canonical full arguments/result, prompt/policy version,
model/endpoint identity and output budget. Call IDs/seq/session coordinates are excluded. A unique
key and transactionally serialized admission prevent simultaneous duplicate requests. `cache_scope`
is an application permission boundary: never share it across principals with different access.
Summary hashes are verified on reads. API keys are never stored in keys or job payloads.

Queued payloads freeze source/model/prompt and attribution. Provider attempts are recorded before
calling; terminal receipts survive cache eviction and session deletion. Expired running leases,
cancellation and ambiguous transport failures are uncertain and are not automatically reissued,
even after cache eviction. Explicit HTTP 429/503 gets one retry after 30 seconds, each attempt
separately recorded. Permanent failures and invalid/nonshrinking summaries fall back. Bundled SDK
hidden retries are disabled; custom transports must make one provider attempt per complete().

TTL and approximate LRU bound cache storage, with entry/byte admission limits. Active jobs are not
evicted; capacity pressure rejects new summary work without affecting tool execution. Summary
completion can temporarily overshoot byte capacity until the next bounded sweep. Access timestamps
are updated at most once a minute. Ledger rows are never part of cache eviction. `ContextCache.stats()`
reports statuses, counts and bytes. The legacy summary_budget_s field is retained for compatibility;
it no longer makes sends wait for generation.

## Lifecycle and recovery

The worker belongs to the store, not a send or child loop. Close the store to stop workers cleanly.
Outbox delivery must be idempotent on `(attempt_id, revision)` and return only after durable acknowledgement.
Keep one charge row per attempt_id; accept only newer revisions. A late provider result can correct
an interrupted/unknown receipt without representing a second request. Session stats apply the same
versioned delta. Pre-6.54.1 stats receipts without usage snapshots remain immutable; their ledger can
still receive corrections, so historical aggregate repair requires a separate reconciliation.
`ContextCache.reconcile(receiver)` repairs stats and pending delivery independently of model work.
DeepTalk runs a separate ten-second scan that rebinds pending jobs to unchanged model/config and
retries delivery even when no new send arrives. Generic hosts must restore worker bindings after a
restart; credentials/callback implementations are not serialized. Missing/changed bindings leave
jobs pending until TTL, not executed against a different provider.

In-process children inherit policies. SubprocessExecutor explicitly rejects enabled policies:
its isolated SQLite execution does not yet transport these callbacks and jobs. Existing fold
accounting remains unchanged. Already committed historical workflow result snapshots are not
rewritten by late billing; query the ledger for final totals.

Upgrade all processes sharing a store before provisioning v9; it adds tables without rewriting
messages. Structural checks do not prove summary quality. Validate task artifacts before broad
activation; the historical 1014 benchmark has not yet been rerun with this implementation.

## Workflow ownership (6.54.2)

Receipts keep the actual owner_session_id and direct workflow_run_id/node_id/node_attempt.
`workflow_ancestry` is an ordered list of enclosing workflow leaf identities, frozen in leaf
metadata before execution; it survives synthetic driver cleanup and propagates through ordinary
subagents. Detached resumes recover it from retained sessions. Older sessions without this metadata
cannot reconstruct a deleted driver chain. Inclusive billing must deduplicate attempt_id across
this chain; it is not an instruction to add a new charge at every ancestor.

### Summary thinking configuration

Summary requests use their own request options; the main agent's requests are unchanged.
For DeepSeek models the default is `extra_body.thinking.type=disabled`.
`ToolContextConfig.summary_request_extra` may override these provider-specific options;
`None` selects defaults and `{}` explicitly inherits the transport settings. Unknown model
families receive no guessed vendor flags. Hosts must supply supported options for them.
Effective summary options participate in model/cache/executor identity and are frozen in
queued job payloads, so recovery cannot silently mix thinking and non-thinking jobs.
Already queued jobs retain their original options.

`summary_min_chars` defaults to **8000 Unicode characters of the original tool result**
(not tokens, bytes, or JSON-escaped length). Below this threshold `llm_summary` preserves
both arguments and result, without enqueueing or falling back to a lossy rule. Exactly
8000 is eligible. The check applies both to prefetch and every distillation trigger,
including delegated background tool results. Rule-only tools are unaffected. Hosts may
set this nonnegative integer to 0 to disable the admission threshold.
