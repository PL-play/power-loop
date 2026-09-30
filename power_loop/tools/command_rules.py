"""Configurable rule layer for ``bash`` / ``background_run`` (6.42.0).

Before 6.42.0 the rule layer was hard-coded in two places: the absolute deny-list in
``default_tools._dangerous_command_reason`` (privileged programs, ``rm -rf /``-style regexes, raw
device redirection) and the category policy in ``command_policy`` (which categories are always
blocked, the refusal text per category). Hosts could only choose which *grantable* categories to
block (``RuntimeEnv.blocked_command_categories``).

``CommandRules`` turns those constants into data. The built-in defaults below are exactly the old
constants, so ``CommandRules()`` / ``command_rules=None`` behaves byte-for-byte like 6.41.0.
A host overrides field by field (``from_overrides``): a field that is absent or ``None`` keeps the
built-in value; a field that is present **replaces** it (an empty list really means "none").

What is deliberately *not* here: the POWER_LOOP_HOME path guard in ``default_tools`` — it protects
the runtime's own files, not a policy choice about which commands an agent may run.

PROVISIONAL.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

#: Categories the classifier in ``command_policy`` knows about (ordered most-severe first).
CATEGORIES: tuple[str, ...] = ("package_install", "download", "pipe_to_shell", "daemon")


@dataclass(frozen=True)
class DenyPattern:
    """One absolute-deny regex. Matched (``re.search``) against the lowered, whitespace-collapsed command."""

    name: str
    pattern: str
    reason: str


DEFAULT_DENY_PROGRAMS: frozenset[str] = frozenset({
    "sudo", "su", "shutdown", "reboot", "halt", "poweroff", "mkfs", "diskutil", "dd",
})

DEFAULT_DENY_PATTERNS: tuple[DenyPattern, ...] = (
    # Recursive/force rm targeting root, home, or a top-level SYSTEM directory. The target alternatives
    # deliberately allow /tmp and relative paths (scratch is fine) while blocking '/', '~', '$HOME',
    # '~/…', '$HOME/…', and '/etc', '/usr/local', '/var/…' etc. The flag group repeats so
    # 'rm -r -f /x' is caught too.
    DenyPattern(
        name="rm_rf_system",
        pattern=(
            r"\brm\s+(?:-[a-z]*[rf][a-z]*\s+)+"
            r"(?:/(?:\s|$)"
            r"|~(?:/|\s|$)"
            r"|\$home"
            r"|/(?:bin|boot|dev|etc|home|lib|lib64|opt|proc|root|run|sbin|srv|sys|usr|var)(?:/|\s|$))"
        ),
        reason="refusing recursive deletion of a root/home/system path",
    ),
    DenyPattern(
        name="raw_device_redirect",
        pattern=r">\s*/dev/(sd|disk|rdisk|nvme|zero|mem)",
        reason="refusing redirection to raw device paths",
    ),
)

DEFAULT_ALWAYS_BLOCKED: frozenset[str] = frozenset({"pipe_to_shell"})

DEFAULT_CATEGORY_MESSAGES: Mapping[str, str] = {
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


@dataclass(frozen=True)
class CommandRules:
    """The whole rule layer as data. Defaults == the pre-6.42.0 hard-coded constants."""

    #: Program basenames refused anywhere in the command (``sudo``, ``dd``…).
    deny_programs: frozenset[str] = DEFAULT_DENY_PROGRAMS
    #: Absolute-deny regexes.
    deny_patterns: tuple[DenyPattern, ...] = DEFAULT_DENY_PATTERNS
    #: Categories blocked for every agent regardless of its grants.
    always_blocked: frozenset[str] = DEFAULT_ALWAYS_BLOCKED
    #: Extra regexes (``re.search``, case-insensitive) that also put a command into a category —
    #: *added* to the built-in lexical classifier.
    category_patterns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Refusal text per category (falls back to the built-in text for categories not listed).
    category_messages: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CATEGORY_MESSAGES))


DEFAULT_COMMAND_RULES = CommandRules()


@lru_cache(maxsize=512)
def _compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def dangerous_command_reason(command: str, rules: CommandRules | None = None) -> str | None:
    """Absolute deny-list check. Returns the refusal reason, or None when the command is allowed."""
    r = rules or DEFAULT_COMMAND_RULES
    compact = re.sub(r"\s+", " ", command.strip().lower())
    for dp in r.deny_patterns:
        if _compile(dp.pattern).search(compact):
            return dp.reason
    if r.deny_programs:
        import shlex
        from pathlib import Path

        try:
            tokens = shlex.split(command, comments=False, posix=True)
        except ValueError:
            tokens = command.split()
        used = sorted({Path(t).name for t in tokens if t and not t.startswith("-")} & set(r.deny_programs))
        if used:
            return f"refusing privileged or device-level command: {', '.join(used)}"
    return None


def extra_categories(command: str, rules: CommandRules | None = None) -> set[str]:
    """Categories hit by the host's extra regexes (the built-in classifier is separate)."""
    r = rules or DEFAULT_COMMAND_RULES
    hit: set[str] = set()
    for cat, patterns in (r.category_patterns or {}).items():
        if any(_compile(p, re.I).search(command) for p in patterns):
            hit.add(cat)
    return hit


def category_message(category: str, rules: CommandRules | None = None) -> str:
    r = rules or DEFAULT_COMMAND_RULES
    return (r.category_messages or {}).get(category) or DEFAULT_CATEGORY_MESSAGES.get(
        category, f"commands in category {category!r} are disabled for this agent."
    )


# ── host config (JSON) ⇄ CommandRules ────────────────────────────────────────


def _str_list(value: Any, what: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list | tuple):
        errors.append(f"{what} must be a list of strings")
        return None
    out = [str(v).strip() for v in value if str(v).strip()]
    return out


def _check_regex(pattern: str, what: str, errors: list[str]) -> bool:
    try:
        re.compile(pattern)
    except re.error as exc:
        errors.append(f"{what}: invalid regex ({exc})")
        return False
    return True


def validate_overrides(overrides: Mapping[str, Any] | None) -> list[str]:
    """Problems with a host override dict; empty list = valid."""
    errors: list[str] = []
    _build(overrides, errors)
    return errors


def from_overrides(overrides: Mapping[str, Any] | None) -> CommandRules:
    """Build rules from a host override dict. Absent / None fields keep the built-in value.

    Raises ``ValueError`` (all problems joined) on invalid input — hosts validate at save time and
    fall back to ``DEFAULT_COMMAND_RULES`` at run time if a stored value is somehow bad.
    """
    errors: list[str] = []
    rules = _build(overrides, errors)
    if errors:
        raise ValueError("; ".join(errors))
    return rules


def _build(overrides: Mapping[str, Any] | None, errors: list[str]) -> CommandRules:
    o = overrides if isinstance(overrides, Mapping) else {}
    if overrides is not None and not isinstance(overrides, Mapping):
        errors.append("command rules must be an object")
    kw: dict[str, Any] = {}

    if o.get("deny_programs") is not None:
        progs = _str_list(o["deny_programs"], "deny_programs", errors)
        if progs is not None:
            bad = [p for p in progs if "/" in p or " " in p]
            if bad:
                errors.append(f"deny_programs: program names only (no paths / spaces): {bad}")
            kw["deny_programs"] = frozenset(progs)

    if o.get("deny_patterns") is not None:
        items = o["deny_patterns"]
        if not isinstance(items, list | tuple):
            errors.append("deny_patterns must be a list of {name, pattern, reason}")
        else:
            pats: list[DenyPattern] = []
            names: set[str] = set()
            for i, it in enumerate(items):
                if not isinstance(it, Mapping):
                    errors.append(f"deny_patterns[{i}] must be an object")
                    continue
                name = str(it.get("name") or "").strip() or f"pattern_{i + 1}"
                pattern = str(it.get("pattern") or "").strip()
                reason = str(it.get("reason") or "").strip() or f"refusing command matching rule {name}"
                if not pattern:
                    errors.append(f"deny_patterns[{i}] ({name}): pattern is empty")
                    continue
                if name in names:
                    errors.append(f"deny_patterns: duplicate name {name!r}")
                names.add(name)
                if _check_regex(pattern, f"deny_patterns[{i}] ({name})", errors):
                    pats.append(DenyPattern(name=name, pattern=pattern, reason=reason))
            kw["deny_patterns"] = tuple(pats)

    if o.get("always_blocked") is not None:
        cats = _str_list(o["always_blocked"], "always_blocked", errors)
        if cats is not None:
            unknown = [c for c in cats if c not in CATEGORIES]
            if unknown:
                errors.append(f"always_blocked: unknown categories {unknown} (known: {list(CATEGORIES)})")
            kw["always_blocked"] = frozenset(c for c in cats if c in CATEGORIES)

    if o.get("category_patterns") is not None:
        cp = o["category_patterns"]
        if not isinstance(cp, Mapping):
            errors.append("category_patterns must be an object {category: [regex, …]}")
        else:
            out: dict[str, tuple[str, ...]] = {}
            for cat, pats in cp.items():
                if cat not in CATEGORIES:
                    errors.append(f"category_patterns: unknown category {cat!r} (known: {list(CATEGORIES)})")
                    continue
                lst = _str_list(pats, f"category_patterns.{cat}", errors)
                if lst is None:
                    continue
                ok = [p for p in lst if _check_regex(p, f"category_patterns.{cat}", errors)]
                if ok:
                    out[cat] = tuple(ok)
            kw["category_patterns"] = out

    if o.get("category_messages") is not None:
        cm = o["category_messages"]
        if not isinstance(cm, Mapping):
            errors.append("category_messages must be an object {category: text}")
        else:
            msgs = dict(DEFAULT_CATEGORY_MESSAGES)
            for cat, text in cm.items():
                if cat not in CATEGORIES:
                    errors.append(f"category_messages: unknown category {cat!r}")
                    continue
                if str(text or "").strip():
                    msgs[cat] = str(text).strip()
            kw["category_messages"] = msgs

    return CommandRules(**kw)


def defaults_as_dict() -> dict[str, Any]:
    """The built-in rules in the same JSON shape ``from_overrides`` accepts (for host UIs)."""
    return rules_as_dict(DEFAULT_COMMAND_RULES)


def rules_as_dict(rules: CommandRules) -> dict[str, Any]:
    return {
        "deny_programs": sorted(rules.deny_programs),
        "deny_patterns": [{"name": p.name, "pattern": p.pattern, "reason": p.reason} for p in rules.deny_patterns],
        "always_blocked": [c for c in CATEGORIES if c in rules.always_blocked],
        "category_patterns": {k: list(v) for k, v in (rules.category_patterns or {}).items()},
        "category_messages": {c: category_message(c, rules) for c in CATEGORIES},
    }


__all__ = [
    "CATEGORIES",
    "DEFAULT_ALWAYS_BLOCKED",
    "DEFAULT_CATEGORY_MESSAGES",
    "DEFAULT_COMMAND_RULES",
    "DEFAULT_DENY_PATTERNS",
    "DEFAULT_DENY_PROGRAMS",
    "CommandRules",
    "DenyPattern",
    "category_message",
    "dangerous_command_reason",
    "defaults_as_dict",
    "extra_categories",
    "from_overrides",
    "rules_as_dict",
    "validate_overrides",
]
