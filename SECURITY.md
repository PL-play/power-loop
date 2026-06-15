# Security Policy

## Supported versions

Security fixes land on the **latest released `1.x`** only; upgrade to the newest version to
receive them.

## Reporting a vulnerability

Please report security issues **privately**, not in public issues/PRs:

- Use GitHub's **private vulnerability reporting** ("Report a vulnerability" on the
  Security tab of `PL-play/power-loop`), or
- email the maintainer listed in `pyproject.toml` (`[project].authors`).

Include a description, affected version, and a minimal reproduction. This is a
single-maintainer project: triage is **best-effort, with no response-time SLA**.
Please allow time for a fix before any public disclosure.

## Security model — read this before deploying

power-loop **orchestrates; it does not, by itself, isolate.** Knowing where the boundary
is matters more than any single setting:

- **Built-in `bash`/file tools run in-process and inherit the host environment.** They are
  convenient for *trusted, local* use and are **NOT a security boundary**. The
  string/regex guards on `bash` are tripwires that reduce footguns — they are *not* a
  sandbox and can be bypassed.
- **For untrusted or model-authored commands, inject real isolation:**
  - tool-level — supply a sandboxed `ShellBackend` (gVisor/Docker/firejail) via the
    runtime-env seam; or
  - process-level — run workflow leaves through the `SubprocessExecutor` + a
    `WorkerLauncher` that wraps each child in your sandbox.
  See the [Sandboxing](docs/en/user-guide/sandboxing.md) guide.
- **Keep secrets in your orchestrator.** Sandboxed children get no MinIO/API credentials;
  the contrib logging/JSONL sinks redact common secret keys by default, but you own what
  you put into prompts, tool inputs, and event payloads.
- **Event-bus subscribers** run with your process's privileges; a durable/metrics/trace
  sink you attach sees event payloads — apply your own PII/retention policy.
- **The on-disk session store is plaintext SQLite.** Protect it with filesystem
  permissions / disk encryption; power-loop does not encrypt it.

The agent loop is a tool for building agents — the security posture of what you build is
yours to set. The library's job is to make the boundary explicit and the isolation seams
available; it will not silently pretend a convenience is a sandbox.
