# Sandbox and process resources

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

The execution factory returns ExecutionEnvironment(runtime_env=..., process_control=..., workflow_file_io=...). RuntimeEnv supplies workspace, skills and ShellBackend on the node that executes the input. Model-authored shell runs with the permissions of that backend; choose an isolated host-managed container when the product needs isolation.

ShellBackend defines launch_argv, launch_cwd, launch_env and session_key. A Docker adapter must map a stable resource reference to the intended container and workspace. The shell binds its original target once. Later cancellation uses that original target, not a newly resolved container with the same reusable name.

ShellProcessControl.stop_tagged returns a strict confirmed boolean. A missing capability or uncertain original process remains unconfirmed; do not turn it into success or silently fall back to another resource. Output draining and cancellation are bounded. Background commands use Runtime jobs, stay controllable after their parent exits, and preserve original-resource protection.

Default runtime and workflow file IO are explicitly assembled. Shared nodes must verify the shared profile and contents, not merely directory existence. The Docker example needs a working Docker daemon and its declared image; the Python process controller needs the documented target support. These dependencies must fail visibly when absent.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [28_docker_shell_backend.py](../../../examples/28_docker_shell_backend.py) | Docker shell |
| [20_default_tools.py](../../../examples/20_default_tools.py) | Workspace tools |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | Workflow IO |

```bash
.venv/bin/python examples/28_docker_shell_backend.py
```
