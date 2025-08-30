import importlib
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def mkpkg(tmp_path, monkeypatch):
    """
    Create a temporary package on sys.path.
    Usage:
        root = mkpkg("pkgx", {
            "__init__.py": "",
            "registry.py": "from bfm.core.registry import Registry\nitems = Registry()\n",
            "a.py": "from pkgx.registry import items\n@items.register('a')\nclass A: pass\n",
        })
    """
    def _make(pkg_name: str, files: dict[str, str]) -> Path:
        base = tmp_path / pkg_name
        for rel, src in files.items():
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src))
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        return base

    yield _make

    # Cleanup any imported temp modules
    for m in list(sys.modules):
        if m.startswith("pkgx") or m.startswith("myproj"):
            sys.modules.pop(m, None)
    importlib.invalidate_caches()