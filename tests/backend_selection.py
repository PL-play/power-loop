"""Select a DB's parametrized cases plus every ordinary case, without name matching.

Usage: pytest -p tests.backend_selection --test-backend=postgres tests --collect-only
Non-parametrized tests and cases without an explicit DB parameter run on every pass.
"""
import json
from pathlib import Path

import pytest

_BACKENDS = {'sqlite':'sqlite','postgres':'postgres','postgresql':'postgres','mysql':'mysql'}
# Database fixtures/parameters, not arbitrary strings (e.g. DSN parser expectations).
_BACKEND_PARAMS = {'registry_store', 'pg_registry_store', 'store', 'backend'}


def pytest_addoption(parser):
    parser.addoption('--test-backend',choices=('sqlite','postgres','mysql'))
    parser.addoption('--selection-report',default=None)


def backends(value):
    if isinstance(value,str):
        selected=_BACKENDS.get(value)
        return {selected} if selected else set()
    if isinstance(value,(tuple,list)):
        return set().union(*(backends(part) for part in value))
    return set()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config,items):
    backend=config.getoption('--test-backend')
    report=config.getoption('--selection-report')
    ordinary,matching,excluded=[],[],[]
    for item in items:
        callspec=getattr(item,'callspec',None)
        declared=set().union(*(backends(value) for name, value in callspec.params.items() if name in _BACKEND_PARAMS)) if callspec else set()
        if not declared:
            ordinary.append(item)
        elif backend is None or backend in declared:
            matching.append(item)
        else:
            excluded.append(item)
    selected=set(ordinary+matching)
    if report:
        path=Path(report)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'backend':backend,'collected':len(items),
            'ordinary':[item.nodeid for item in ordinary],
            'database':[item.nodeid for item in matching],
            'deselected':[item.nodeid for item in excluded]},indent=2)+'\n')
    if excluded:
        config.hook.pytest_deselected(items=excluded)
        items[:]=[item for item in items if item in selected]
