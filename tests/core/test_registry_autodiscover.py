import sys
import importlib
import textwrap


def test_autodiscover_relative_default(mkpkg):
    src_registry = textwrap.dedent("""\
        from bfm.core.registry import Registry
        registry = Registry()
    """)

    src_items = textwrap.dedent("""\
        from pkgx.registry import registry

        @registry.register("item_name", "item_alias")
        class A:
            def __init__(self, c: int = 64): 
                self.c = c
    """)

    mkpkg("pkgx", {
        "__init__.py": "",
        "registry.py": src_registry,
        "module/__init__.py": "from .items import *",
        "module/items.py": src_items,
    })

    reg_mod = importlib.import_module("pkgx.registry")
    reg = reg_mod.registry

    assert "pkgx.module.items" not in sys.modules

    assert "item_name" in reg.list()
    assert "item_alias" in reg.list_aliases().keys()

    obj1 = reg.resolve("item_alias", c=32)
    obj2 = reg.resolve("item_name", c=16)
    assert obj1.c == 32 and obj2.c == 16


def test_autodiscover_absolute(mkpkg):
    # Create a temp package pkgx.training.losses with a registered loss
    src_registry = """
        from bfm.core.registry import Registry
        # absolute path
        losses = Registry("pkgx.training.losses", relative=False)
    """
    src_losses_init = "from . import focal"
    src_focal = """
        from pkgx.training.registry import losses
        @losses.register("focal", "focal_loss")
        def focal_fn(gamma: float = 2.0): return {"gamma": gamma}
    """

    mkpkg("pkgx", {
        "__init__.py": "",
        "training/__init__.py": "",
        "training/registry.py": src_registry,
        "training/losses/__init__.py": src_losses_init,
        "training/losses/focal.py": src_focal
    })

    reg_mod = importlib.import_module("pkgx.training.registry")
    losses = reg_mod.losses

    # Autodiscovery should pull in pkgx.training.losses.focal
    assert "focal" in losses.list()
    cfg = losses.resolve("focal_loss", gamma=3.3)
    assert cfg == {"gamma": 3.3}

def test_autodiscover_is_cached(mkpkg, monkeypatch):
    """
    Count how many times importlib.import_module is called for our base package.
    The registry calls autodiscover() each time, but it's cached, so submodules
    should be imported only once.
    """
    mkpkg("pkgx", {
        "__init__.py": "",
        "models/__init__.py": "",
        "models/registry.py": "from bfm.core.registry import Registry\nitems = Registry('pkgx.models.items', relative=False)\n",
        "models/items/__init__.py": "from . import a, b\n",
        "models/items/a.py": "from pkgx.models.registry import items\n@items.register('a')\nclass A: pass\n",
        "models/items/b.py": "from pkgx.models.registry import items\n@items.register('b')\nclass B: pass\n",
    })

    # Wrap import_module to count imports under 'pkgx.models.items'
    orig_import_module = importlib.import_module
    calls = {"count": 0}

    def counting_import(name, package=None):
        if name.startswith("pkgx.models.items"):
            calls["count"] += 1
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", counting_import)

    reg_mod = importlib.import_module("pkgx.models.registry")
    items = reg_mod.items

    # First call triggers discovery -> imports pkgx.models.items, a, b
    names1 = items.list()
    assert set(names1) == {"a", "b"}
    first_count = calls["count"]
    assert first_count >= 3  # at least base + two submodules

    # Subsequent calls should NOT re-import due to cached autodiscover()
    items.list()
    items.contains("a")
    try:
        items.get("a")
    except AttributeError:
        pass
    items.list_aliases()

    assert calls["count"] == first_count  # no new imports