from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar

from numpy import dtype, generic, ndarray


T = TypeVar("T", bound=generic)

Array: TypeAlias = ndarray[tuple[int, ...], dtype[T]]
Number: TypeAlias = int | float
ArrayLike: TypeAlias = Sequence[Number] | Array[Any]
