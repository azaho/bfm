"""
Registry for training configs.

Lets you register a method under a string key and resolve it later:
```python
    from training_setup.registry import register, resolve

    @register("sample", "default")
    class SampleSetup(TrainingSetup):
        ...

    setup = resolve("sample")
```
"""
import importlib
import pkgutil
from typing import Callable, List, Dict
from training_setup.training_setup import TrainingSetup

Factory = Callable[..., TrainingSetup]

_REGISTRY: Dict[str, Factory] = {}
_ALIASES: Dict[str, str] = {}

_populated = False

def register(name: str, *aliases: str) -> Callable[[Factory], Factory]:
    """
    Register a training setup under a string key and optional aliases.
    
    Args:
        name (str): The name of the training setup.
        *aliases (str): Optional aliases for the training setup.
    """
    def decorator(setup: Factory) -> Factory:
        canon = name.strip().lower()
        if canon in _REGISTRY or canon in _ALIASES:
            raise KeyError(f"Setup {name!r} already registered")
        _REGISTRY[canon] = setup
        
        for alias in aliases:
            canon = alias.strip().lower()
            if canon in _REGISTRY or canon in _ALIASES:
                raise KeyError(f"Alias {alias!r} already registered")
            _ALIASES[canon] = name
        return setup
    
    return decorator


def resolve(name: str, **kwargs) -> TrainingSetup:
    """
    Return a training setup by name or alias.
    
    Args:
        name (str): Name or alias of the training setup.
        **kwargs: Arguments to pass to the training setup constructor.

    Returns:
        TrainingSetup: The resolved training setup instance.
    """
    autodiscover()
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key in _REGISTRY:
        return _REGISTRY[key](**kwargs)
    raise KeyError(f"Unknown setup: {name!r}. Available: {list_setups()}")


def list_setups() -> List[str]:
    """Return list of registered setups (not aliases)."""
    return sorted(_REGISTRY.keys())


def list_aliases() -> Dict[str, str]:
    """Return alias → canonical mapping."""
    return dict(_ALIASES)


def autodiscover():
    """Auto-discover and import all training setups."""
    global _populated
    if not _populated:
        for m in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
            importlib.import_module(m.name)
        _populated = True