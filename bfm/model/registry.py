"""
Contains registries for the following model components:
- encoders
- backbones
- modules
- preprocessors
"""
from typing import Mapping
from bfm.core.registry import Registry


encoders = Registry("encoders")
backbones = Registry("backbones")
modules = Registry("modules")
preprocessors = Registry("preprocessing")


REGISTRIES: Mapping[str, Registry] = {
    "preprocessors": preprocessors,
    "encoders": encoders,
    "backbones": backbones,
    "modules": modules,
}