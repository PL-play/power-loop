# Storage backends

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Open the store explicitly with open_store or SessionStore.open, then pass it to Runtime. SQLite is the default local backend. PostgreSQL and MySQL require their driver extras. Isolate tests by _test database and dedicated table_prefix.

SchemaPolicy.AUTO_CREATE creates a fresh current store; VERIFY checks an existing current store without migration. SDK 11 uses task schema 15, protocol 42 and descriptor schema 1. Old stores are rejected. There is no runtime conversion or compatible fallback. StoreSchemaError exposes provisioning DDL when manual provisioning is needed.

Database transactions own session sequence allocation and atomic blackboard updates. StoreBlackboard is the public database-backed implementation, including atomic first-row creation. A memory lock around a read followed by write is insufficient across processes. DeepTalk uses its API backend for business blackboards and timers rather than SDK business writes.

Retention is explicit and covers compacted history, usage and terminal timers. Current timer drivers use indexed due batches and conditional claims. Shared execution additionally needs shared coordination/transport; a server database does not remove that requirement. Runtime shutdown closes its owned store; independently owned readers should have separate connections.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | Current-store reopening |
| [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | Atomic blackboard |
| [34_durability_lifecycle.py](../../../examples/34_durability_lifecycle.py) | Lifecycle |
| [39_pluggable_backends_and_resume.py](../../../examples/39_pluggable_backends_and_resume.py) | Backend selection |

```bash
.venv/bin/python examples/11_cross_process_resume.py
```
