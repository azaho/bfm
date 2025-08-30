"""
Contains registries for the following model components:
- encoders
- backbones
- modules
- preprocessors
"""
from bfm.core.registry import Registry


encoders = Registry("encoders")
backbones = Registry("backbones")
modules = Registry("modules")
preprocessors = Registry("preprocessors")