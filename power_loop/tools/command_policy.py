"""Category-level command policy for ``bash`` / ``background_run`` (rule layer).

`_dangerous_command_reason` in default_tools is the *absolute* deny-list (sudo, mkfs, rm -rf /…).
This module is the layer above it: it classifies a shell command into coarse **categories** —
installing packages, downloading files, piping a download into an interpreter, starting daemons —
and the host decides per agent which categories are blocked (``RuntimeEnv.blocked_command_categories``).

Why a category layer and not one big regex: a sandbox that can reach npm/pypi through the egress
proxy *will* have a model install playwright + chromium the first time a skill points it at a
tool it doesn't have (that is exactly what happened, and it cost 20 minutes and a few hundred MB).
The fix is not to blacklist "playwright" — it is to make "install software" a capability the
operator grants, with a refusal message that tells the model what to do instead.

Lexical only, no semantics: we split on shell operators (shlex with punctuation_chars) and look at
each simple command's program name + subcommand. False negatives are possible (``python -c
"import subprocess; …"``); that is what the host's optional LLM review layer is for. False
positives are cheap: the model gets a tool result telling it the category and the exit.

Library default: nothing blocked except ``pipe_to_shell`` (there is no legitimate use for
``curl … | sh`` inside an agent sandbox). Hosts opt in to more.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

CATEGORIES: tuple[str, ...] = ("package_install", "download", "pipe_to_shell", "daemon")
ALWAYS_BLOCKED: frozenset[str] = frozenset({"pipe_to_shell"})

_OPERATORS = {"|", "||", "&&", ";", "&", "(", ")", ";;", "|&"}
_INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "python", "python3", "node", "perl", "ruby", "php"}
_WRAPPERS = {"env", "nice", "time", "command", "exec", "builtin", "nohup", "setsid", "stdbuf", "timeout"}

_NODE_PM = {"npm", "pnpm", "yarn", "bun"}
_NODE_PM_INSTALL = {"i", "install", "add", "ci", "update", "upgrade", "up", "dlx", "exec"}
_PY_PM = {"pip", "pip3", "pipx", "poetry", "uv", "conda", "mamba", "pipenv"}
_PY_PM_INSTALL = {"install", "add", "sync", "download"}
_SYS_PM = {"apt", "apt-get", "apk", "yum", "dnf", "brew", "pacman", "zypper", "snap"}
_SYS_PM_INSTALL = {"install", "add", "reinstall", "upgrade", "dist-upgrade"}
_LANG_PM = {"cargo": {"install"}, "gem": {"install"}, "go": {"install", "get"}, "cpan": {"install"}}
_DAEMON_PROGS = {"systemctl", "service", "daemonize", "pm2", "forever", "supervisord", "supervisorctl",
                 "screen", "tmux", "start-stop-daemon"}
_SERVER_PROGS = {"node", "python", "python3", "npm", "npx", "pnpm", "yarn", "bun", "deno", "serve",
                 "http-server", "uvicorn", "gunicorn", "flask", "ruby", "php", "caddy", "nginx"}

_PIPE_TO_SHELL_RE = re.compile(
    r"(?:\bcurl\b|\bwget\b)[^|\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|python3?|node|perl|ruby)\b"
    r"|<\(\s*(?:curl|wget)\b"
    r"|\b(?:sh|bash|zsh|python3?|node)\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b",
    re.I,
)


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into simple commands (token lists) on operators."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    segs: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok in _OPERATORS or tok == "\n":
            if cur:
                segs.append(cur)
            cur = []
            if tok == "&":
                segs.append(["&"])  # marker: previous segment was backgrounded
        else:
            cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def _program(seg: list[str]) -> tuple[str, list[str]]:
    """Program basename + remaining args, skipping ``FOO=bar`` assignments and wrappers."""
    i = 0
    while i < len(seg):
        tok = seg[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            i += 1
            continue
        name = Path(tok).name
        if name in _WRAPPERS:
            i += 1
            # `timeout 30 cmd`, `nice -n 5 cmd`: skip their own option/value tokens
            while i < len(seg) and (seg[i].startswith("-") or re.fullmatch(r"\d+[smhd]?", seg[i])):
                i += 1
            continue
        return name, seg[i + 1:]
    return "", []


def _subcommand(args: list[str]) -> str:
    for a in args:
        if not a.startswith("-"):
            return a
    return ""


def classify_command(command: str) -> set[str]:
    """Return the set of policy categories a command touches (empty = plain)."""
    cats: set[str] = set()
    if not command or not command.strip():
        return cats
    if _PIPE_TO_SHELL_RE.search(command):
        cats.add("pipe_to_shell")
    segs = _segments(command)
    for idx, seg in enumerate(segs):
        if seg == ["&"]:
            continue
        prog, args = _program(seg)
        if not prog:
            continue
        sub = _subcommand(args)
        backgrounded = idx + 1 < len(segs) and segs[idx + 1] == ["&"]

        # ── package_install ──
        if prog in _NODE_PM and (sub in _NODE_PM_INSTALL or (prog == "yarn" and not sub)):
            cats.add("package_install")
        elif prog == "npx" or prog == "corepack":
            cats.add("package_install")
        elif prog in _PY_PM:
            subs = {a for a in args if not a.startswith("-")}
            if subs & _PY_PM_INSTALL or (prog == "uv" and "pip" in subs and "install" in subs):
                cats.add("package_install")
        elif prog.startswith("python") and "-m" in args:
            m = args.index("-m")
            if m + 1 < len(args) and args[m + 1] in {"pip", "ensurepip"} and "install" in args:
                cats.add("package_install")
        elif prog in _SYS_PM and sub in _SYS_PM_INSTALL:
            cats.add("package_install")
        elif prog in _LANG_PM and sub in _LANG_PM[prog]:
            cats.add("package_install")
        elif prog == "playwright" and sub == "install":
            cats.add("package_install")

        # ── download ──
        if prog == "curl":
            if any(a in {"-o", "-O", "--output", "--remote-name", "-J", "--remote-header-name"}
                   or a.startswith(("-o", "--output=")) or re.fullmatch(r"-[a-zA-Z]*[oO][a-zA-Z]*", a)
                   for a in args):
                cats.add("download")
        elif prog == "wget":
            to_stdout = any(a in {"-O-", "-qO-", "-O", "--output-document=-"} and (a.endswith("-") or
                            (a == "-O" and "-" in args[args.index(a) + 1:args.index(a) + 2]))
                            for a in args)
            if not to_stdout:
                cats.add("download")
        elif prog in {"aria2c", "axel"}:
            cats.add("download")
        elif prog == "git" and sub in {"clone", "fetch", "pull"}:
            cats.add("download")

        # ── daemon ──
        if prog in _DAEMON_PROGS:
            cats.add("daemon")
        elif backgrounded and prog in _SERVER_PROGS:
            cats.add("daemon")
        # wrappers that only exist to daemonize
        if seg and Path(seg[0]).name in {"nohup", "setsid"}:
            cats.add("daemon")
    return cats


_EXITS = {
    "package_install": (
        "installing packages (npm/pip/apt/cargo…) is disabled for this agent. Use what is preinstalled "
        "in the sandbox; if something is genuinely missing, tell the user so the platform can add it. "
        "For screenshots of HTML prototypes use the render_html tool, not a browser install."
    ),
    "download": (
        "downloading files with curl/wget/git clone is disabled for this agent. Use the platform's "
        "fetch_file / web_read tools for content you need, or ask the user to provide the file."
    ),
    "pipe_to_shell": (
        "piping a download straight into a shell/interpreter (curl … | sh) is never allowed."
    ),
    "daemon": (
        "starting long-running/background daemons is disabled for this agent. Run the command in the "
        "foreground with a timeout, or use the background_run tool for a bounded job."
    ),
}


def command_policy_reason(command: str, blocked: frozenset[str] | set[str] | None) -> str | None:
    """Return a refusal message if ``command`` touches a blocked category, else None."""
    effective = set(ALWAYS_BLOCKED) | set(blocked or ())
    hit = classify_command(command) & effective
    if not hit:
        return None
    cat = next(c for c in CATEGORIES if c in hit)  # stable, most-severe-first order
    return f"Error: Command blocked by sandbox policy ({cat}): {_EXITS[cat]}"


__all__ = ["ALWAYS_BLOCKED", "CATEGORIES", "classify_command", "command_policy_reason"]
