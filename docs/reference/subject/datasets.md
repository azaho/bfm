The datasets module provides classes and functions for accessing and managing datasets used in brain function modeling.



Example:
```python
from torch.utils.data import Dataset
from bfm.subject.registry import datasets

@datasets.register("example")
class CustomDataset(Dataset):
    ...

dataset = datasets.resolve("example")
```