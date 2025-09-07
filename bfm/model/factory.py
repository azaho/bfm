from typing import List, Optional

import torch.nn as nn
from omegaconf import OmegaConf

from bfm.core.registry import Registry
from bfm.model.base import BFModule
from bfm.model.registry import REGISTRIES


def build_model(module: Optional[nn.Module], cfg, components: List[str]) -> BFModule:
    """
    Assembled named components into a BFModule.

    Expects cfg.components.<component> to have:
      - registry: str  (one of REGISTRIES keys)
      - name: str    (name of the component in the registry)
      - kwargs: dict
    
    Args:
        module: An instance of BFModule to which components will be added. If None, a new BFModule is created.
        cfg: Configuration object with model parameters.
        components: List of component names to build and assemble.

    Returns:
        BFModule: Assembled model with specified components.
        
    Raises:
        ValueError: If a component's configuration is missing or if the specified registry is not found.
    """    
    module = module or BFModule()
    for component in components:

        component_cfg = getattr(cfg.components, component, None)
        if component_cfg is None:
            raise ValueError(f"Component {component} not found in configuration.")
        
        registry = REGISTRIES.get(component_cfg.registry)
        if not isinstance(registry, Registry):
            raise ValueError(f"Registry {component_cfg.registry} not found or valid.")

        
        module_component = registry.resolve(
            component_cfg.name,
            **(OmegaConf.to_container(component_cfg.kwargs, resolve=True) 
            if "kwargs" in component_cfg else {})
        )
        module.add_module(component, module_component)
        
    return module
