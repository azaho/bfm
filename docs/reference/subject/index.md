# Subject Module

The subject module contains classes and functions for subject neural data access from various sources (MGH, BrainTreebank, etc.).

**Registry**

This module provides the following registry instances:

- `subjects`: Stores subject components discovered in `bfm.subject.subjects`.
- `datasets`: Stores dataset components discovered in `bfm.subject.datasets`.

To use these registry instances, you can import them from the `bfm.subject.registry` module:

```python
from bfm.subject.registry import subjects

@subjects.register("example")
class CustomSubject(Subject):
    ...

subject = subjects.resolve("example")
```