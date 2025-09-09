from typing import Any, Dict, cast
from collections.abc import Mapping 
from omegaconf import DictConfig, OmegaConf

from bfm.model.base import BFModule
from bfm.core.registry import Registry
from bfm.model.registry import REGISTRIES


def _cfg_to_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Convert a DictConfig to a standard dictionary, resolving any references."""
    container = OmegaConf.to_container(cfg, resolve=True)
    
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected dict-like container, got {type(container).__name__}")

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {str(k): _normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize(x) for x in obj]
        return obj

    return cast(Dict[str, Any], _normalize(container))


def build_module(component: str, cfg: DictConfig) -> BFModule:
    """
    Build a component from the configuration.
    
    Expects cfg.components.<component> to have:
      - registry: str  (one of REGISTRIES keys)
      - name: str    (name of the component in the registry)
      - kwargs: dict

    Args:
        component (str): Name of the component to build.    
        cfg (DictConfig): Configuration object with model parameters.

    Returns:
        BFModule: Constructed BFModule.
    """
    component_cfg = getattr(cfg.components, component, None)
    if component_cfg is None:
        raise ValueError(f"Component {component} not found in configuration.")

    registry = REGISTRIES.get(component_cfg.registry)
    if not isinstance(registry, Registry):
        raise ValueError(f"Registry {component_cfg.registry} not found or valid.")

    return registry.resolve(
        component_cfg.name, 
        **_cfg_to_kwargs(component_cfg.kwargs) if "kwargs" in component_cfg else {}
    )
