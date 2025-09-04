"""
Generic Registry with lazy autodiscovery.

Example usage:

```python
from bfm.core.registry import Registry

items = Registry() # create a registry for the current package

@items.register("my_item")
class MyItem:
    def __init__(self, number: int):
        ...
    
my_item = items.resolve("my_item", number=4)
``` 
"""
import importlib
import inspect
import os
import pkgutil
from collections.abc import Callable
from functools import cache
from typing import Generic, TypeVar, Optional, Dict, List
from bfm.core.logger import get_logger

T = TypeVar("T")
Factory = Callable[..., T]

logger = get_logger(__name__)

def _make_autodiscover(base_pkg: str) -> Callable[[], None]:
    """Create a cached autodiscover() bound to a given package path."""
    @cache
    def _autodiscover():
        if os.getenv("BFM_DISABLE_AUTODISCOVER") == "1":
            return
        pkg = importlib.import_module(base_pkg)
        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:  # not a package -> no-op
            logger.warning(f"Registry base package {base_pkg!r} is not a package, skipping autodiscovery")
            return
        for m in pkgutil.walk_packages(pkg_path, prefix=base_pkg + "."):
            importlib.import_module(m.name)
    return _autodiscover


def _caller_package() -> str:
    frm = inspect.currentframe()
    if frm is None:
        raise RuntimeError("Cannot infer caller frame.")

    this_file = os.path.abspath(__file__)

    frm = frm.f_back  # start at the immediate caller
    while frm:
        mod = inspect.getmodule(frm)
        # Stop when we find a module that's not this file
        if mod and hasattr(mod, "__file__") and os.path.abspath(str(mod.__file__)) != this_file:
            if mod.__package__:
                return mod.__package__
        frm = frm.f_back
        
    raise RuntimeError("Cannot infer caller frame.")


class Registry(Generic[T]):
    """
    Create a registry for all items in a package.

    Args:
        package (Optional[str]): Dotted path of the package to scan for items
            (modules under it should import and call @register). If no package is specified, the caller's package will be used.
        relative (bool): If True, the package is treated as relative to the caller's package.
    """
    def __init__(self, package: Optional[str] = None, relative: bool = True):
        if package is None:
            base_pkg = _caller_package()
        elif relative:
            base_pkg = f"{_caller_package()}.{package}"
        else:
            base_pkg = package
            
        self._store: Dict[str, Factory] = {}
        self._aliases: Dict[str, str] = {}
        self._autodiscover = _make_autodiscover(base_pkg)

    def register(self, name: str, *aliases: str) -> Callable[[Factory], Factory]:
        """Register item under a canonical name and optional aliases."""
        def decorator(factory: Factory) -> Factory:
            canon = name.strip().lower()
            if canon in self._store or canon in self._aliases:
                raise KeyError(f"Key {name!r} already registered")
            self._store[canon] = factory
            for alias in aliases:
                a = alias.strip().lower()
                if a in self._store or a in self._aliases:
                    raise KeyError(f"Alias {alias!r} already registered")
                self._aliases[a] = canon  # map alias -> canonical key
            return factory
        return decorator
    
    def get(self, name: str) -> Factory:
        """Get a factory by name."""
        self._autodiscover()
        key = name.strip().lower()
        key = self._aliases.get(key, key)
        try:
            return self._store[key]
        except KeyError as e:
            raise KeyError(
                f"Unknown key {name!r}. Available: {self.list()}"
            ) from e


    def resolve(self, name: str, **kwargs) -> T:
        """Instantiate by name with kwargs."""
        factory = self.get(name)
        return factory(**kwargs)

    def contains(self, name: str) -> bool:
        """Return True if the registry contains the given name."""
        self._autodiscover()
        key = name.strip().lower()
        return key in self._store or key in self._aliases

    def list(self) -> List[str]:
        """Canonical keys (not aliases)."""
        self._autodiscover()
        return sorted(self._store.keys())

    def list_aliases(self) -> Dict[str, str]:
        """alias → canonical mapping."""
        self._autodiscover()
        return self._aliases