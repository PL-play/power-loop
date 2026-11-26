# Runtime 12: sessions and inputs

SDK **12.0.0**, schema **17**, protocol **43**, descriptor **2**. This is a local source delivery;
publication, deployment, business cutover and the final three-node drill are separate steps.

## Start and send

Install this checkout with `python -m pip install -e '.[dev]'`. Open a fresh `SessionStore`
(`schema="tasks"` is the existing schema provisioning option; it provisions the session schema).
Register a versioned factory definition and its profiles, construct `Runtime`, and start its worker.
`runtime.client.create_session(descriptor, actor=actor)` returns a `TargetRef`.
The descriptor freezes the definition, model/tool/hook/resource profiles, configuration and host context.
Resolve credentials and resources in the factory on the executing node.

```python
receipt = await runtime.client.send(
    target, "hello", message_id="request-1", actor=actor, priority="normal",
)
result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=deadline)
```

The host supplies the authenticated actor and stable request ID. A receipt means accepted, not completed.
Repeat the same ID and content to retry acceptance; changed content with that ID fails explicitly.
`TargetRef(store_id, session_id)` addresses a session. `InputRef(store_id, session_id, message_id)`
addresses one accepted input. `send_index` identifies the session's execution batch.
All identity comparisons include the store. There is one session execution path.

## Input, result and cancellation

An idle session starts from its pending inputs. At every model/tool round boundary the worker reads
the latest inbox version; it merges the current pending snapshot into one user message. Later inputs
remain for the next boundary or batch. `priority="urgent"` requests prompt interruption according to
the current tool's interrupt policy. No notification payload is the durable authority.

Use `inspect(InputRef)`, `wait(InputRef)`, `messages(TargetRef)` and `list_children(TargetRef)`.
`cancel(InputRef)` addresses that input; `cancel(TargetRef, scope="subtree")` follows linked children.
Independent children retain their own work. New input to an existing child uses ordinary `send`.

Accept, claim and final each have one atomic transaction. The `sessions` row owns the execution state,
active send and ownership fence. Final writes an ordinary result message and terminalizes inputs.
Results route as ordinary parent/driver inputs, with `result_message_id` deriving their stable ID.
Unknown external effects are queried or reported unknown; they are not replayed automatically.

## Children, background work and workflows

`spawn_agent` returns `target_ref` and `input_ref`; the child is an agent session with a direct parent.
A background operation is a `background` session that freezes the original tool and arguments.
A workflow is an independent `workflow_driver` child of its caller; its nodes are agent sessions.
Graph progress lives in the driver's `session_runtime_state`, while results remain messages/inputs.
`workflow:run:*` and `workflow:node:*` are journal keys, not a second result ledger.
Model-visible tools do not supply trusted identity. Tool/model allowlists and child configuration
narrow the parent's authority. Ordinary tool validation, hooks, events and context processing remain.

## Shared execution and shutdown

The default composition uses in-process transport and database session ownership; it needs no Redis.
For shared execution pass `shared_execution=True` and shared ownership/transport implementations.
`DatabasePollingTransport` is an included shared adapter. The DSN alone does not enable distribution.
All participants share the store and profiles, with distinct node IDs. Missing capabilities fail closed.
Shutdown drains work up to its deadline; interrupted inputs remain visible for recovery by another worker.
Call `Runtime.aclose` to close owned resources. A client close does not delete a session.

Run the complete deterministic examples:

```bash
.venv/bin/python examples/design215_single_process.py
.venv/bin/python examples/design215_shared.py
```

The shared example uses two workers and independent database connections on one host. It does not
substitute for a real three-node deployment test.

## Store policy and retention

Old schema/protocol stamps and descriptor schema 2 are rejected. There is no runtime migration,
dual read/write or old-result fallback. Initialize a fresh isolated store or prefix; reopening the
current schema verifies it. `SchemaPolicy.VERIFY` does not create missing tables.

`SessionStore.close_if_idle` and bounded deletion read lifecycle from `sessions` and serialize on
the `sessions` row. Running/queued/recovering work, valid ownership, unfinished inputs and unclosed tools
protect the session. Linked children participate in close/delete; independent children survive.
Resource guards must also allow deletion. A partial delete leaves the session closing and rejects input.

`purge_expired(now_ms, limit=...)` has a maximum of 1,000 row writes per batch. Categories are
`sessions`, `messages`, `inbox`, `runtime`, `projections`. Terminal inbox entries may expire; active
inputs and open-session context remain protected. Closed-session messages/journal/projections are
removed only after input references are gone. The host schedules maintenance; SDK billing writes are
zero. DeepTalk billing uses the host's design 214 `BillingChargeEvent` path.

The complete frozen export set is `power_loop.STABLE_API`, verified by `tests/unit/test_packaging.py`.
The P7 report records executed checks and remaining blockers; final full suites and real three nodes
belong to P8.

Notes, default timers and shared blackboards store their documents in session_runtime_state. No auxiliary schema tables are required.

`runtime_transport.PROTOCOL_VERSION=42` is used by `TransportPacket` for internal notifications; it is separate from the store `schema_migrations.protocol_version=43`.
