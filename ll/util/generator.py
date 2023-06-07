"""
This module provides a properly typed decorator function to convert a generator to a list.

For example, the following code:
```python
from ll.util.generator import to_list

@to_list
def my_generator():
    yield 1

# The type of my_generator is now Callable[[], List[int]]
```
"""

from functools import wraps
from typing import Callable, Generator, List, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def to_list(func: Callable[P, Generator[R, None, None]]) -> Callable[P, List[R]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> List[R]:
        return list(func(*args, **kwargs))

    return wrapper
