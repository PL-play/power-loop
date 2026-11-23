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
``curl … | sh`` inside an agent sandbox). Hosts opt in to more. Since 6.42.0 the always-blocked set,
extra category regexes and the refusal text are data (``command_rules.CommandRules``) a host can override.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from power_loop.tools.command_rules import (
    CATEGORIES,
    DEFAULT_ALWAYS_BLOCKED,
    DEFAULT_CATEGORY_MESSAGES,
    CommandRules,
    category_message,
    extra_categories,
)

#: Built-in "always blocked" set (hosts can override it via ``CommandRules.always_blocked``, 6.42.0).
ALWAYS_BLOCKED: frozenset[str] = DEFAULT_ALWAYS_BLOCKED

_OPERATORS = {"|", "||", "&&", ";", "&", "(", ")", ";;", "|&"}
_PIPE_OPERATORS = {"|", "|&"}
_INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "python", "python3", "node", "perl", "ruby", "php"}
_WRAPPERS = {"env", "nice", "time", "command", "exec", "builtin", "nohup", "setsid", "stdbuf", "timeout", "sudo"}
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "sudo": frozenset({"-u", "-g", "-C", "-h", "-p", "-r", "-t", "-T", "-U", "--user", "--group"}),
}
_DOWNLOAD_PIPE_SOURCES = {"curl", "wget"}

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
    r"<\(\s*(?:/\S+/)?(?:curl|wget)\b"
    r"|(?:^|[\s;&|(])(?:/\S+/)?(?:sh|bash|zsh|dash|ksh|python\d*(?:\.\d+)?|node|perl|ruby|php)\s+-c\s+[\"']?\$\(\s*(?:/\S+/)?(?:curl|wget)\b",
    re.I,
)


def _is_interpreter(prog: str) -> bool:
    name = prog.lower()
    return name in _INTERPRETERS or bool(re.fullmatch(r"python\d+(?:\.\d+)?", name))


def _segments_with_separators(command: str) -> list[tuple[list[str], str]]:
    """Split a shell command into (token_list, preceding_separator) pairs."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    segs: list[tuple[list[str], str]] = []
    cur: list[str] = []
    pending_sep = ""
    for tok in tokens:
        if tok in _OPERATORS or tok == "\n":
            if cur:
                segs.append((cur, pending_sep))
                cur = []
                pending_sep = ""
            if tok == "&":
                segs.append((["&"], "&"))  # marker: previous segment was backgrounded
                pending_sep = ""
            elif tok in {"(", ")"}:
                continue
            else:
                pending_sep = tok
        else:
            cur.append(tok)
    if cur:
        segs.append((cur, pending_sep))
    return segs


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into simple commands (token lists) on operators."""
    return [seg for seg, _ in _segments_with_separators(command)]


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
            value_flags = _WRAPPER_VALUE_FLAGS.get(name, frozenset())
            split_string = False
            i += 1
            # `timeout 30 cmd`, `nice -n 5 cmd`, `env -S sh`, `env -u FOO sh`: skip wrapper options/values
            while i < len(seg):
                opt = seg[i]
                if opt == "--":
                    i += 1
                    break
                if name == "env" and opt in {"-S", "--split-string"}:
                    split_string = True
                    i += 1
                    continue
                if name == "env" and opt.startswith("--split-string="):
                    payload = opt.split("=", 1)[1].strip()
                    if payload:
                        try:
                            parts = shlex.split(payload, posix=True)
                        except ValueError:
                            parts = payload.split()
                        if parts:
                            return _program(parts + seg[i + 1:])
                    i += 1
                    continue
                if opt in value_flags:
                    i += 2
                    continue
                if opt.startswith("-") or re.fullmatch(r"\d+[smhd]?", opt):
                    i += 1
                    continue
                break
            if split_string and i < len(seg) and any(ch.isspace() for ch in seg[i]):
                try:
                    parts = shlex.split(seg[i], posix=True)
                except ValueError:
                    parts = seg[i].split()
                if parts:
                    return _program(parts + seg[i + 1:])
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
    segs_with_seps = _segments_with_separators(command)
    prev_prog = ""
    for idx, (seg, sep) in enumerate(segs_with_seps):
        if seg == ["&"]:
            prev_prog = ""
            continue
        prog, args = _program(seg)
        if not prog:
            prev_prog = ""
            continue
        if sep in _PIPE_OPERATORS and prev_prog in _DOWNLOAD_PIPE_SOURCES and _is_interpreter(prog):
            cats.add("pipe_to_shell")
        prev_prog = prog.lower()
        sub = _subcommand(args)
        backgrounded = idx + 1 < len(segs_with_seps) and segs_with_seps[idx + 1][0] == ["&"]

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


#: Built-in refusal text per category (kept for backward compatibility; see command_rules).
_EXITS = DEFAULT_CATEGORY_MESSAGES


def command_categories(command: str, rules: CommandRules | None = None) -> set[str]:
    """Built-in lexical classification ∪ the host's extra category regexes (6.42.0)."""
    return classify_command(command) | extra_categories(command, rules)


def command_policy_reason(
    command: str,
    blocked: frozenset[str] | set[str] | None,
    rules: CommandRules | None = None,
) -> str | None:
    """Return a refusal message if ``command`` touches a blocked category, else None.

    ``rules`` (6.42.0) overrides the always-blocked set, adds category regexes and replaces refusal
    text; ``None`` = built-in rules (identical to 6.41.0).
    """
    always = rules.always_blocked if rules is not None else ALWAYS_BLOCKED
    effective = set(always) | set(blocked or ())
    hit = command_categories(command, rules) & effective
    if not hit:
        return None
    cat = next(c for c in CATEGORIES if c in hit)  # stable, most-severe-first order
    return f"Error: Command blocked by sandbox policy ({cat}): {category_message(cat, rules)}"


__all__ = ["ALWAYS_BLOCKED", "CATEGORIES", "classify_command", "command_categories", "command_policy_reason"]
