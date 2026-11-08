"""Load the shared example Runtime composition without installing examples."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / '_runtime.py'
_spec = importlib.util.spec_from_file_location('_example_runtime', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
demo_runtime = _module.demo_runtime
