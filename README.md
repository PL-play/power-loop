# power-loop 14.1.0

An embeddable agent runtime. Main agents, children, background work and workflows
use one ID-addressed client/worker execution path. The default composition is
single-process, with no Redis dependency. Shared execution requires explicit
shared ownership and messaging adapters.

Current schema: **18**. Protocol: **43**. Descriptor schema: **2**.
Use a fresh SDK database or table prefix; historical tasks are not imported.
Published on PyPI as `power-loop==14.1.0` and used by the DeepTalk `agent` service.
The SDK is feature-frozen: changes are driven by issues found in real business use.

- [Current API and startup guide](docs/runtime-12.md)
- [Runnable single-process example](examples/design215_single_process.py)
- [Runnable default timers](examples/26_timers.py)
- [Changelog](CHANGELOG.md)
- [Authoritative stable exports](power_loop/__init__.py)
- [中文](README.zh.md)

Install the checkout with `python -m pip install -e '.[dev]'`. From an isolated
test directory, run `python /absolute/path/to/examples/design215_single_process.py`.
The demo uses a fixed model; a real deployment resolves models, tools, hooks and
resources in its registered `RuntimeFactory`.

`StatefulAgentLoop` accepts `runtime=`. Direct LLM/DSN construction is retired.
Runtime claim/recover is the only ownership installation path. The worker binds
that same `SessionOwnership` handle before hooks and the factory build, for agent,
workflow driver and background executions. It retires the handle and resets its
binding before post-final business cleanup; inherited tasks keep the retired
handle and cannot start more work. Memory and Store coordinators remain supported.
An acceptance receipt is not completion: wait for its original `InputRef`.
Normal authorization, accounting, context and cleanup remain required. Failures
may leave interrupted/unknown results; unknown external actions are not replayed.

Shared example: [design215_shared.py](examples/design215_shared.py).
