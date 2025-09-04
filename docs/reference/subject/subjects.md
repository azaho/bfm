# Subjects

All custom subject instances should be in the `bfm.subject.subjects` module. Such instances should inherit from the `bfm.subject.base.Subject` class and implement the required methods.

To be discoverable by the registry, each subject instance must be registered to `bfm.subject.registry.subjects`.

Example:

```python
from bfm.subject.base import Subject
from bfm.subject.registry import subjects


@subjects.register("custom")
class CustomSubject(Subject):
    def __init__(self):
        super().__init__()

    def get_all_electrode_data(self, session_id, window_from=None, window_to=None):
        pass

    ...

# Downstream use:
subject = subjects.resolve("custom")
```