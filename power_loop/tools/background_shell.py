"""One private shell process, stopped through its original backend target."""
import shlex
import subprocess
import time

from power_loop.core.agent_context import get_current_cancel_token, get_current_loop
from power_loop.runtime.env import get_runtime_env
from power_loop.runtime.exec_backend import DEFAULT_SHELL_BACKEND
from power_loop.runtime.process_control import OwnedProcess, bind_shell_control, stop_shell_target


def execute(command, timeout=300, **kwargs):
    from .command_policy import command_policy_reason
    from .default_tools import _dangerous_command_reason, _truncate_chars, _validate_bash_command_scope
    env = get_runtime_env()
    error = (_dangerous_command_reason(command, env.command_rules)
        or command_policy_reason(command, env.blocked_command_categories, env.command_rules)
        or _validate_bash_command_scope(command))
    if error:
        raise ValueError(error)
    context = get_current_loop()._runtime_worker_context
    token = get_current_cancel_token()
    token.raise_if_cancelled()
    workspace = env.require_workspace_dir()
    backend = env.shell_backend or DEFAULT_SHELL_BACKEND
    argv, cwd, launch_env = backend.launch_argv(workspace), backend.launch_cwd(workspace), backend.launch_env(workspace)
    target = bind_shell_control(backend, workspace, argv=argv, cwd=cwd, env=launch_env)
    process = OwnedProcess(argv, cwd=cwd, env=launch_env)
    tag = context.job_ref.job_id + "_" + str(context.job_ref.generation)
    payload = "exec env POWER_LOOP_BACKGROUND_JOB=" + tag + " /bin/bash --norc --noprofile -c " + shlex.quote(command) + "\n"
    deadline = time.monotonic() + timeout
    try:
        while True:
            if token.is_cancelled() or time.monotonic() >= deadline:
                process.request_stop()
                stopped = stop_shell_target(target, tag, variable="POWER_LOOP_BACKGROUND_JOB", grace_s=0.1)
                process.drain_stopped_output(timeout=2)
                if not stopped.confirmed:
                    from power_loop.runtime.task_runtime import mark_unconfirmed_resource
                    mark_unconfirmed_resource()
                    raise RuntimeError("original_shell_stop_unconfirmed")
                token.raise_if_cancelled()
                raise TimeoutError("background shell timed out")
            try:
                out, err = process.communicate(input=payload, timeout=0.1)
                return _truncate_chars(f"exit_code={process.returncode}\n{out or ''}{err or ''}")
            except subprocess.TimeoutExpired:
                payload = None
    finally:
        if process.poll() is None:
            process.request_stop()
            process.drain_stopped_output(timeout=2)
