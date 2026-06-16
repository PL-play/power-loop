"""Coverage: SkillLoader resolution + robustness — path-traversal safety, unconfigured/explicit-dir validation, malformed frontmatter, reload (probed clean)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_skill_name_path_traversal_is_blocked(tmp_path):
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.skills import SkillLoader

    skills_root = tmp_path / "skills"
    (skills_root / "good").mkdir(parents=True)
    (skills_root / "good" / "SKILL.md").write_text(
        "---\ndescription: Good\n---\nGood body", encoding="utf-8"
    )
    # Secret SKILL.md outside skills_dir that traversal could target.
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "SKILL.md").write_text(
        "---\ndescription: S\n---\nSECRET BODY", encoding="utf-8"
    )
    (tmp_path / "ws").mkdir()
    (tmp_path / "home").mkdir()
    env = RuntimeEnv(
        workspace_dir=tmp_path / "ws",
        home_dir=tmp_path / "home",
        skills_dir=skills_root,
    )
    with runtime_env_context(env):
        loader = SkillLoader()
        for evil in ["../secret", "../../secret", "good/../../secret", "..%2fsecret"]:
            out = loader.handle_load_skill(evil)
            assert "SECRET BODY" not in out
            assert "Unknown skill" in out

def test_skills_dir_unconfigured_raises(tmp_path):
    import pytest

    from power_loop.runtime.env import RuntimeEnv, RuntimeEnvError, runtime_env_context
    from power_loop.runtime.skills import SkillLoader

    env = RuntimeEnv(workspace_dir=tmp_path, home_dir=tmp_path, skills_dir=None)
    with runtime_env_context(env):
        with pytest.raises(RuntimeEnvError):
            SkillLoader()

def test_resolve_skills_dir_validates_explicit_path(tmp_path):
    import pytest

    from power_loop.runtime.skills import _resolve_skills_dir

    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        _resolve_skills_dir(str(missing))

    afile = tmp_path / "afile.txt"
    afile.write_text("x")
    with pytest.raises(FileNotFoundError):
        _resolve_skills_dir(str(afile))

def test_malformed_and_nondict_frontmatter_does_not_crash(tmp_path):
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.skills import SkillLoader

    skills_root = tmp_path / "skills"

    def write(name, text):
        d = skills_root / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(text, encoding="utf-8")

    write("broken", "---\ndescription: [unclosed\n---\nBody")
    write("scalar", "---\njust a string\n---\nBody")
    write("plain", "Just body, no frontmatter")
    (tmp_path / "ws").mkdir()
    (tmp_path / "home").mkdir()
    env = RuntimeEnv(
        workspace_dir=tmp_path / "ws",
        home_dir=tmp_path / "home",
        skills_dir=skills_root,
    )
    with runtime_env_context(env):
        loader = SkillLoader()
        assert set(loader.names) == {"broken", "scalar", "plain"}
        assert loader.skills["scalar"]["meta"] == {}
        assert loader.skills["plain"]["meta"] == {}
        assert loader.skills["plain"]["body"] == "Just body, no frontmatter"
        desc = loader.get_descriptions()
        assert "broken: No description" in desc

def test_reload_picks_up_new_skills_and_get_content(tmp_path):
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.skills import SkillLoader

    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    (tmp_path / "ws").mkdir()
    (tmp_path / "home").mkdir()
    env = RuntimeEnv(
        workspace_dir=tmp_path / "ws",
        home_dir=tmp_path / "home",
        skills_dir=skills_root,
    )
    with runtime_env_context(env):
        loader = SkillLoader()
        assert loader.names == []
        d = skills_root / "late"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\ndescription: Late\n---\nLate body", encoding="utf-8"
        )
        loader.reload()
        assert "late" in loader.names
        out = loader.get_content("late")
        assert "Late body" in out
        assert "late" in out
