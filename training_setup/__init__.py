"""
Defines model architectures and training paradigms for brain foundation models.
"""
import os
import importlib
import pkgutil

def autodiscover():
    """Auto-discover and import all training setup methods."""
    for m in pkgutil.walk_packages(__path__, prefix=__name__ + ".methods."):
        importlib.import_module(m.name)
        

# call by default; allow opt-out via env
if os.getenv("REGISTRY_AUTOFILL", "1") == "1":
    autodiscover()