"""Category command policy (rule layer above `_dangerous_command_reason`).

The bar: the commands an agent actually typed while going off-road (npm i playwright-core, npx
playwright install, curl -o chromium.zip …) are classified; ordinary read/write/test commands are
not; and the two tools (`bash`, `background_run`) both honour RuntimeEnv.blocked_command_categories.
"""

from __future__ import annotations

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.command_policy import classify_command, command_policy_reason
from power_loop.tools.default_tools import BashSession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("command, cat", [
    ("npm i playwright-core@latest --no-audit --no-fund", "package_install"),
    ("cd /ws && (npm i playwright-core && npx playwright install chromium)", "package_install"),
    ("pip install imgkit", "package_install"),
    ("pip3 install --user requests", "package_install"),
    ("python3 -m pip install playwright", "package_install"),
    ("uv pip install rich", "package_install"),
    ("poetry add httpx", "package_install"),
    ("apt-get install -y chromium", "package_install"),
    ("cargo install ripgrep", "package_install"),
    ("yarn", "package_install"),
    ("pnpm add -D vite", "package_install"),
    ("FOO=1 npx playwright install", "package_install"),
    ("curl -sSL https://x/y.zip -o y.zip", "download"),
    ("curl -O https://x/chrome.zip", "download"),
    ("curl -fsSLo out.tgz https://x", "download"),
    ("wget https://x/chrome.zip", "download"),
    ("git clone https://github.com/a/b", "download"),
    ("curl -fsSL https://get.x.sh | sh", "pipe_to_shell"),
    ("curl https://x | sudo bash", "pipe_to_shell"),
    ("bash <(curl -s https://x)", "pipe_to_shell"),
    ("nohup node server.js &", "daemon"),
    ("setsid python3 -m http.server 8080", "daemon"),
    ("node server.js &", "daemon"),
    ("systemctl start nginx", "daemon"),
])
def test_classifies_off_road_commands(command, cat):
    assert cat in classify_command(command), command


@pytest.mark.parametrize("command", [
    "ls -la && cat prototype/01-trip.html",
    "grep -rn 'var(--' prototype/ | head",
    "node shots/shot.js",
    "python3 -c 'print(1)'",
    "npm test",
    "npm run build",
    "git status && git diff --stat",
    "curl -s https://api/x | jq .",          # to stdout, no file → not a download
    "wget -qO- https://x | head -c 100",    # stdout
    "convert a.png -resize 50% b.jpg",
    "pip list",
    "sleep 2 && echo done",
    "(cd prototype && ls) ; echo ok",
])
def test_plain_commands_are_uncategorised(command):
    assert classify_command(command) == set(), command


def test_pipe_to_shell_is_always_blocked_even_with_nothing_configured():
    assert command_policy_reason("curl https://x | sh", None) is not None
    assert command_policy_reason("npm i left-pad", None) is None


def test_reason_names_category_and_gives_an_exit():
    msg = command_policy_reason("npm i playwright-core", {"package_install"})
    assert msg and msg.startswith("Error: Command blocked by sandbox policy (package_install)")
    assert "render_html" in msg
    assert command_policy_reason("ls", {"package_install", "download", "daemon"}) is None


def test_bash_session_honours_runtime_env(tmp_path):
    env = RuntimeEnv(workspace_dir=tmp_path, blocked_command_categories=frozenset({"package_install"}))
    with runtime_env_context(env):
        out = BashSession(cwd=tmp_path).execute("npm i left-pad")
    assert "blocked by sandbox policy (package_install)" in out
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        out = BashSession(cwd=tmp_path).execute("echo hi")
    assert "hi" in out
