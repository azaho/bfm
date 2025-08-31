All custom training setup instances should be in the `bfm.training.setups` module. Such instances should inherit from the `bfm.training.setups.BaseSetup` class and implement the required methods. 

To be discoverable by the registry, each setup instance must be registered to `bfm.training.setup_registry.setups`.

Example:
```python
from bfm.training.training_setup import TrainingSetup
from bfm.training.setup_registry import setups


@setups.register("custom")
class CustomSetup(TrainingSetup):
    def __init__(self):
        super().__init__()

    def initialize_model(self):
        # Custom model initialization logic
        pass

    def calculate_pretrain_loss(self, batch, output_accuracy=True):
        pass

    def generate_frozen_features(self, batch):
        pass


# Downstream use:
setup = setups.resolve("custom")
```