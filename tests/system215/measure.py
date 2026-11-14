"""Measure a fixed Git SDK revision using current test instruments, without another checkout.

Only SDK modules are loaded from the immutable Git tree, in memory. Test instruments
remain the current files. Their revision/diff must be recorded with the output.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RevisionLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, revision):
        self.revision = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
        raw = subprocess.check_output(["git", "archive", self.revision, "power_loop"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            self.sources = {m.name: archive.extractfile(m).read() for m in archive if m.isfile() and m.name.endswith(".py")}
        self.loaded = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "power_loop" and not fullname.startswith("power_loop."):
            return None
        relative = fullname.replace(".", "/")
        package = relative + "/__init__.py"
        filename = package if package in self.sources else relative + ".py"
        if filename not in self.sources:
            if any(name.startswith(relative + "/") for name in self.sources):
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(ROOT / relative)]
                return spec
            raise ImportError(f"{fullname} absent from fixed revision {self.revision}")
        return importlib.util.spec_from_file_location(fullname, ROOT / filename, loader=self,
            submodule_search_locations=[str(ROOT / relative)] if filename == package else None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename = str(Path(module.__file__).relative_to(ROOT))
        self.loaded.append(filename)
        exec(compile(self.sources[filename], module.__file__, "exec"), module.__dict__)


if __name__ == "__main__":
    revision, output = sys.argv[1:3]
    loader = RevisionLoader(revision)
    sys.meta_path.insert(0, loader)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ["P0_WRITE_METER"] = "1"
    os.environ["P215_PROFILE_OUT"] = output + "-profile.jsonl"
    os.environ["P215_COMPOSITION_OUT"] = output + "-composition.jsonl"
    import pytest

    selected = sys.argv[3:] or ["tests/unit/test_send_sql_profile.py",
                               "tests/unit/test_composition_sql_profile_215.py"]
    code = pytest.main(["-q", "--no-real", "-p", "no:cacheprovider", *selected])
    Path(output + "-source.json").write_text(json.dumps({"sdk_head": loader.revision,
        "sdk_tree": subprocess.check_output(["git", "rev-parse", loader.revision + ":power_loop"], text=True).strip(),
        "loaded_from_git": sorted(loader.loaded), "exit_code": code,
        "test_instruments": "current workspace tests; SDK source comes only from specified Git revision"}, indent=2))
    raise SystemExit(code)
