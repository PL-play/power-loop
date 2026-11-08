# power-loop 11.0.0

An embeddable agent runtime. Main agents, children, background work and workflows
use one ID-addressed client/worker execution path. The default composition is
single-process, with no Redis dependency. Shared execution requires explicit
shared ownership and messaging adapters.

Current task schema: **15**. Protocol: **42**. Descriptor schema: **1**.
Use a fresh SDK database or table prefix; historical tasks are not imported.
This branch is a source delivery, not a published or deployed release. See the
P7 delivery report for the full regression status.

- [Current API and startup guide](docs/runtime-11.md)
- [Runnable single-process example](examples/design203_single_process.py)
- [Runnable default timers](examples/26_timers.py)
- [Changelog](CHANGELOG.md)
- [Authoritative stable exports](power_loop/__init__.py)
- [中文](README.zh.md)

Install the checkout with `python -m pip install -e '.[dev]'`. From an isolated
test directory, run `python /absolute/path/to/examples/design203_single_process.py`.
The demo uses a fixed model; a real deployment resolves models, tools, hooks and
resources in its registered `RuntimeFactory`.

`StatefulAgentLoop` accepts `runtime=`. Direct LLM/DSN construction is retired.
An acceptance receipt is not completion: wait for its original `InputRef`.
Normal authorization, accounting, context and cleanup remain required. Failures
may leave interrupted/unknown results; unknown external actions are not replayed.

Finite child waits use `WaitPolicy` and `register_wait_handoff_tool`: the factory
sets a deadline, the child releases capacity, and its original main explicitly
continues, defers or cancels the same task. Configure the timer backend/driver in
the Runtime; see [the contract and assembly](docs/runtime-11.md#finite-child-review).
