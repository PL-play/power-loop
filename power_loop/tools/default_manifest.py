from __future__ import annotations

from collections.abc import Sequence

from power_loop.contracts.tools import ToolDefinition

DEFAULT_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="write_file",
        description=(
            "Create a new UTF-8 text file or overwrite an existing one with the complete provided content. "
            "Parent directories are created automatically. For safety, overwriting an existing file requires that you "
            "have read it with read_file in this process and that it has not changed since that read; use edit_file or "
            "apply_patch for smaller edits to existing files. Send the entire desired file content in one call, not chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, absolute workspace path, or @workspace/..."},
                "content": {"type": "string", "description": "Complete UTF-8 file content to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether an existing file may be overwritten. Defaults to true, still requiring a prior read.",
                },
            },
            "required": ["path", "content"],
        },
        required_params=("path", "content"),
    ),
    ToolDefinition(
        name="read_file",
        description=(
            "Read a UTF-8 text file with stable line numbers, or list a directory. Use this before edit_file, apply_patch, "
            "or overwriting an existing file. Large files are paged by line range; use offset and limit to continue. "
            "Binary-looking files are refused instead of decoded blindly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, absolute workspace path, or @workspace/..."},
                "offset": {"type": "integer", "description": "1-based starting line. Defaults to 1."},
                "limit": {"type": "integer", "description": "Maximum number of lines to return."},
            },
            "required": ["path"],
        },
        required_params=("path",),
        # 可异步：纯只读。单次快，但一轮里读 8 个文件时并发才有意义——真实日志里一个 send 把 read_file 摊成 8 轮（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Replace text in an existing UTF-8 file using an exact, unique old_text snippet. Read the file first, then "
            "include enough surrounding context that old_text matches exactly once. If multiple identical occurrences "
            "should all change, set replace_all=true. The tool preserves BOM and dominant line endings and returns a diff."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit."},
                "old_text": {"type": "string", "description": "Exact current text to replace. Must not be empty."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every exact occurrence instead of requiring uniqueness. Defaults to false.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        required_params=("path", "old_text", "new_text"),
    ),
    ToolDefinition(
        name="apply_patch",
        description=(
            "Apply one or more unified-diff style hunks to an existing file. Read the file first. Prefer this for "
            "multi-line or multi-hunk edits where exact search/replace would be awkward. Each hunk should include context "
            "lines beginning with a space, deleted lines beginning with '-', and added lines beginning with '+'. Ambiguous "
            "or stale hunks are rejected rather than guessed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to patch."},
                "patch": {
                    "type": "string",
                    "description": "Unified diff hunks, e.g. @@ -1,2 +1,2 @@ followed by context/delete/add lines.",
                },
            },
            "required": ["path", "patch"],
        },
        required_params=("path", "patch"),
    ),
    ToolDefinition(
        name="bash",
        description=(
            "Run a shell command in a persistent bash session rooted at the workspace. Use dedicated tools for file reads, "
            "writes, search, and patches whenever possible; use bash for tests, builds, package managers, git inspection, "
            "and other CLI-only operations. Output is truncated, timeouts restart the shell to prevent leftover commands, "
            "and obviously dangerous privileged/device-level commands are blocked. Set restart=true to reset the session."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "restart": {"type": "boolean", "description": "Restart the persistent bash session instead of running command."},
                "timeout": {"type": "integer", "description": "Seconds to wait before timing out. Range: 1-600."},
            },
            "required": ["command"],
        },
        required_params=("command",),
    ),
    ToolDefinition(
        name="glob",
        description=(
            "Find files or directories by glob pattern, sorted by modification time newest first. Prefer glob over bash/find "
            "when locating paths. A bare name like '*.py' searches recursively as '**/*.py'. Common bulky directories such "
            "as .git, node_modules, build, and dist are skipped. Hidden paths are skipped unless include_hidden=true or the "
            "pattern explicitly mentions a hidden segment."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern. Bare filenames are searched recursively."},
                "path": {"type": "string", "description": "Directory to search. Defaults to workspace root."},
                "max_results": {"type": "integer", "description": "Maximum paths to return. Defaults to 100, max 500."},
                "include_hidden": {"type": "boolean", "description": "Include hidden files and directories. Defaults to false."},
            },
            "required": ["pattern"],
        },
        required_params=("pattern",),
        # 可异步：纯只读的路径匹配（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="grep",
        description=(
            "Search UTF-8 text file contents by regex, returning file:line:content matches. Prefer grep over bash/rg/grep "
            "commands for code search because it has workspace scoping, result caps, and binary/bulky directory filtering. "
            "Use include for a glob filter such as '*.py' or 'src/**/*.ts'. Set literal=true when searching for exact text "
            "rather than regex syntax."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern unless literal=true."},
                "path": {"type": "string", "description": "File or directory to search. Defaults to workspace root."},
                "include": {"type": "string", "description": "Optional glob filter, e.g. '*.py' or 'src/**/*.ts'."},
                "max_results": {"type": "integer", "description": "Maximum matches to return. Defaults to 50, max 500."},
                "literal": {"type": "boolean", "description": "Treat pattern as plain text instead of regex. Defaults to false."},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive search. Defaults to true."},
                "include_hidden": {"type": "boolean", "description": "Include hidden files and directories. Defaults to false."},
            },
            "required": ["pattern"],
        },
        required_params=("pattern",),
        # 可异步：纯只读的内容检索（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="load_skill",
        description="Load specialized knowledge by name.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
        required_params=("name",),
        # 可异步：只读：把技能文件读进来（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="todo",
        description=(
            "Update the current task list. Send the WHOLE list every time (it replaces the old one).\n"
            "每一条有两个维度，别混：`status` 说**进展**（pending/in_progress/completed），"
            "`owner` 说**谁在做**（self=你自己动手；background/subagent/workflow=派出去了）。\n"
            "**同时只能有一件 owner=self 的 in_progress**——你自己一次只做一件事。但派出去的"
            "不限：三件活分别在后台、子 agent、workflow 里跑，就是三条 in_progress，各带各的 owner。\n"
            "owner 不是 self 时必须给 `ref`（task_id / run id / 子会话 id）：说不出派到哪的活"
            "没法回来收，清单上会永远挂着。先真的派出去、拿到句柄，再把这一条标上。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            "owner": {
                                "type": "string",
                                "enum": ["self", "background", "subagent", "workflow"],
                                "description": "谁在做（缺省 self）。不是 self 就必须给 ref。",
                            },
                            "ref": {
                                "type": "string",
                                "description": "派出去的句柄：background 的 task_id / workflow 的 run id / 子 agent 的会话 id。",
                            },
                        },
                        "required": ["id", "text", "status"],
                    }
                }
            },
            "required": ["items"],
        },
        required_params=("items",),
    ),
    ToolDefinition(
        name="note",
        description=(
            "Manage your persistent notes with one action: add, update, delete, or list. Notes "
            "survive context compaction. Use action=add with content for a durable fact, preference, "
            "or ongoing task state; action=list to read notes and obtain their #ids; action=update "
            "with note_id plus content and/or pinned to keep memory current; action=delete with "
            "note_id for stale memory. Keep notes short and self-contained; use pinned=true only for "
            "notes that must never be hidden or auto-evicted.\n"
            "边界：**会过期的运行时状态不要写进 note**——后台 task_id、workflow run id、任务进度，"
            "那些是待办的 owner/ref 与平台自己的台账在管，任务跑完你写的那条就成了没人删的垃圾。"
            "note 只放「以后还成立」的事实与决定。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "delete", "list"],
                    "description": "Operation to perform.",
                },
                "note_id": {
                    "type": "integer",
                    "description": "Required for update/delete; obtain it with action=list.",
                },
                "content": {
                    "type": "string",
                    "description": "Required for add; optional replacement text for update.",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "Optional pin state for add/update.",
                },
            },
            "required": ["action"],
        },
        required_params=("action",),
    ),
    ToolDefinition(
        name="schedule_wakeup",
        description=(
            "Manage your durable wake-up timers with one action: schedule, list, or cancel. "
            "action defaults to schedule when you pass delay_seconds (to cancel when you pass only "
            "timer_id, to list when you pass nothing) — but passing it explicitly is clearer. "
            "action=schedule arms a wake-up — after delay_seconds you receive your note back as "
            "a message and can act on it (check a long task, follow up on a promise); set "
            "every_seconds to make it RECURRING (fires repeatedly until cancelled). action=list "
            "shows your wake-ups (#id, seconds until due, note). action=cancel disarms one by "
            "timer_id. Timers survive restarts. Requires the host to run a TimerRunner; if the "
            "host documented no timer support, don't rely on wake-ups firing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["schedule", "list", "cancel"],
                    "description": "Operation to perform.",
                },
                "delay_seconds": {"type": "integer", "description": "Required for action=schedule: seconds until the FIRST fire (min 5, max 30 days)."},
                "note": {"type": "string", "description": "Required for action=schedule: what future-you should do when woken. Be specific."},
                "every_seconds": {"type": "integer", "description": "Optional for action=schedule: repeat every N seconds after each fire (fixed-delay) until cancelled. Omit for one-shot."},
                "timer_id": {"type": "integer", "description": "Required for action=cancel: the #id from action=list or the schedule confirmation."},
            },
            "required": [],
        },
        required_params=(),
    ),
    ToolDefinition(
        name="current_time",
        description="Current local date and time (with weekday). Use before scheduling wake-ups or reasoning about deadlines.",
        input_schema={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="recall_compacted",
        description=(
            "Retrieve older messages that context compaction summarized out of your active "
            "window. They were folded into a compact_note and marked compacted_out — they "
            "still exist, just aren't shown each turn. Use this ONLY when the compact_note "
            "lacks a specific detail you need (an exact value, path, decision, error). Filter "
            "by query (case-insensitive substring over message content) and/or a from_seq/to_seq "
            "range; limit caps how many are returned (most recent first)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Case-insensitive substring to match in folded message content."},
                "from_seq": {"type": "integer", "description": "Only return folded messages with seq >= this."},
                "to_seq": {"type": "integer", "description": "Only return folded messages with seq <= this."},
                "limit": {"type": "integer", "description": "Max messages to return (most recent first). Default 20, max 50."},
            },
        },
        required_params=(),
        # 可异步：只读：回看被折叠掉的历史（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="recall_send",
        description=(
            "Re-expand an earlier send shown in your context only as a compact summary. Each "
            "summarized tool line carries a coordinate «#N·sS» (send_index N, row seq S). To get "
            "ONE original tool result in full (a file you read, a skill you loaded, a vision answer, "
            "a command's output), call recall_send(send_index=N, seq=S) — that returns just that row "
            "with a large cap. Without seq you get a MAP of send #N — every row with its seq, "
            "bodies trimmed to fit one small budget; use it to FIND the coordinate, then call again "
            "with seq. Read-only, current session."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "send_index": {"type": "integer", "description": "The #N of the past send."},
                "seq": {"type": "integer", "description": "Optional: the sS row coordinate from a summary line → return that single row in full."},
            },
            "required": ["send_index"],
        },
        required_params=("send_index",),
        # 可异步：只读：回看本会话的历史行（design/95）
        async_capable=True,
    ),
    ToolDefinition(
        name="background_run",
        description=(
            "Manage private background tasks with one action: run, tool, or check. action=run "
            "starts a shell command in a private non-interactive background worker and returns its "
            "task_id immediately. action=tool runs one async-capable TOOL in the background "
            "(tool=<name>, args={…}) and returns a task_id at once — use it for long, "
            "side-effect-free calls (image generation, web fetches) whose result you do not "
            "need immediately; you will be notified on completion. action=check reports a "
            "task's status/output by task_id, or lists all your tasks when task_id is omitted."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    # list = check 不带 task_id 的显式别名（6.17.0）。原来「列出全部」这件事
                    # 藏在「check 不带参数」这个隐式约定里，模型要先知道才用得上——同一件事
                    # 该只有一个名字（design/96 §6.4）。
                    "enum": ["run", "tool", "check", "list"],
                    "description": "Operation to perform.",
                },
                "command": {"type": "string", "description": "Required for action=run: shell command to execute."},
                "tool": {"type": "string", "description": "Required for action=tool: name of an async-capable tool (marked ⏳ in its description)."},
                "args": {"type": "object", "description": "action=tool: the tool's arguments, exactly as you would pass them when calling it directly."},
                "task_id": {"type": "string", "description": "For action=check: task to inspect; omit to list all tasks."},
            },
            "required": ["action"],
        },
        required_params=("action",),
    ),
    ToolDefinition(
        name="request_user_input",
        description=(
            "Ask the caller or product UI for external input, then pause the agent loop. "
            "The loop returns status='waiting_for_input' with pending_interactions; resume by calling "
            "StatefulAgentLoop.submit_input(session_id, interaction_id, value). Use this for confirmations, choices, "
            "or text that must come from outside the LLM process."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "Input kind such as text, confirm, or choice. Defaults to text.",
                },
                "prompt": {"type": "string", "description": "User-facing prompt or confirmation request."},
                "options": {
                    "type": "array",
                    "description": "Optional choices for choice/confirm prompts.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "metadata": {
                    "type": "object",
                    "description": "Opaque caller metadata returned unchanged in pending_interactions.",
                },
            },
            "required": ["prompt"],
        },
        required_params=("prompt",),
    ),
]

# Indexed lookup for selective registration.
DEFAULT_TOOL_DEFINITIONS_MAP: dict[str, ToolDefinition] = {d.name: d for d in DEFAULT_TOOL_DEFINITIONS}

# ---------------------------------------------------------------------------
# Tool presets (matching zero-code's BASE_TOOLS / EXPLORE_TOOLS categories)
# ---------------------------------------------------------------------------

# Core coding tools — the minimal set for an agent that reads, writes, and runs code.
CORE_TOOL_NAMES: tuple[str, ...] = (
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "glob",
    "grep",
    "load_skill",
    "request_user_input",
)

# Read-only / exploration tools — no file mutation.  Matches zero-code EXPLORE_TOOLS.
EXPLORE_TOOL_NAMES: tuple[str, ...] = (
    "bash",
    "read_file",
    "glob",
    "grep",
    "load_skill",
    "request_user_input",
)

# Full set — everything including todo and background tasks.
FULL_TOOL_NAMES: tuple[str, ...] = tuple(d.name for d in DEFAULT_TOOL_DEFINITIONS)

TOOL_PRESETS: dict[str, tuple[str, ...]] = {
    "core": CORE_TOOL_NAMES,
    "explore": EXPLORE_TOOL_NAMES,
    "full": FULL_TOOL_NAMES,
}


def get_tool_definitions(
    *,
    preset: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[ToolDefinition]:
    """Return a filtered list of default tool definitions.

    Priority: *include* > *preset* > all.
    *exclude* is applied last regardless.

    Args:
        preset: One of "core", "explore", "full".
        include: Explicit tool names to include (ignores preset).
        exclude: Tool names to drop from the result.
    """
    if include is not None:
        names = list(include)
    elif preset is not None:
        names_tuple = TOOL_PRESETS.get(preset)
        if names_tuple is None:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {', '.join(TOOL_PRESETS)}")
        names = list(names_tuple)
    else:
        names = [d.name for d in DEFAULT_TOOL_DEFINITIONS]

    if exclude:
        exclude_set = set(exclude)
        names = [n for n in names if n not in exclude_set]

    result: list[ToolDefinition] = []
    for name in names:
        defn = DEFAULT_TOOL_DEFINITIONS_MAP.get(name)
        if defn is None:
            raise ValueError(f"Unknown default tool: '{name}'")
        result.append(defn)
    return result
