"""Contains registries for Training Setups

Example usage:
```python
from bfm.training.setup_registry import setups

@setups.register("my_custom_setup")
class MySetup(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        ...

setup = setups.resolve("my_custom_setup", all_subjects=all_subjects, config=config)
```
"""
from bfm.core.registry import Registry

setups = Registry("setups")