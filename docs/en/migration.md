# SDK 12 store boundary

SDK 12 uses schema 17, protocol 43 and descriptor 2. Open a fresh store with a new store ID; old databases, protocols and descriptors are rejected. No automatic migration, aliases or dual writes are supported.

Register a runtime factory, open a session from a frozen descriptor and trusted actor, then submit inputs through RuntimeClient. Use TargetRef, InputRef, send_index and message IDs for identity. Ownership belongs to the session worker.

See [Runtime 12](../runtime-12.md), [Getting started](getting-started.md), and the runnable examples in `examples/design215_single_process.py` and `examples/design215_shared.py`.
