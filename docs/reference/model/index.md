# Model Module

The Model module provides the building blocks for developing foundation model architectures. It includes various components such as encoders, backbones, and preprocessing functions that can be easily customized and extended for specific research needs.

Additionally, the module provides a registry system for managing and retrieving these components, making it easier to assemble and configure complex model architectures through the factory interface.

## Submodules

The following submodules are defined:

- `encoders`: Contains various encoder architectures to transform input data into embeddings.
- `backbones`: Contains backbone architectures for processing encoder outputs.
- `modules`: Contains reusable model components.
- `preprocessing`: Contains data preprocessing utilities.

**Registry**

This module provides the following registry instances:

- `encoders`: Stores encoder components discovered in `bfm.model.encoders`.
- `backbones`: Stores backbone components discovered in `bfm.model.backbones`.
- `modules`: Stores module components discovered in `bfm.model.modules`.
- `preprocessors`: Stores preprocessing components discovered in `bfm.model.preprocessing`.

To use these registry instances, you can import them from the `bfm.model.registry` module:

```python
from bfm.model.registry import encoders

@encoders.register("example")
class CustomEncoder(BFModule):
    ...

encoder = encoders.resolve("example")
```