# Runtime capacity and shared workers

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Runtime.capacity bounds active execution. Independent sessions and children use the same admission mechanism; workflow waiting does not reserve a model slot. The default composition uses in-process coordination and transport. It acquires an exclusive store startup lock, so a second independent default process is rejected. Client-only publishers do not acquire this lock.

A shared deployment sets shared_execution=True and supplies shared ownership and message transport adapters. Merely switching to PostgreSQL/MySQL does not make the default coordinator shared. Process-only adapters are rejected. All workers implement the same required_profiles; declare and preflight models, tools, hooks, workspace, skills and backend ports before accepting. Do not rely on sticky routing to a process that owns an object.

Shared resources have a stable resource_profile_ref, resource_scope="shared", and a host mapping to actual shared storage. Matching directory names alone do not establish shared contents. Factories rebuild resources on the current node. Ownership loss stops the execution; already-started external work may remain unknown.

Measure the complete request, including acceptance, execution, persistence and resource close. Record database, concurrency, history, tools, payload sizes and versions. SQL counts alone do not establish a latency or throughput improvement. Current delivery measurements belong to the versioned delivery report; old illustrative numbers are not a v11 baseline.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [10_concurrent_sessions.py](../../../examples/10_concurrent_sessions.py) | Concurrent sessions |
| [35_scaling_and_read_pool.py](../../../examples/35_scaling_and_read_pool.py) | Capacity and storage |
| [36_observability.py](../../../examples/36_observability.py) | Observability |

```bash
.venv/bin/python examples/10_concurrent_sessions.py
```
