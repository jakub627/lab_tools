from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar
import numpy as np
from numpy import dtype, generic, ndarray
from os import PathLike

StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
FileDescriptorOrPath: TypeAlias = int | StrOrBytesPath

T = TypeVar("T", bound=generic)

Array: TypeAlias = ndarray[tuple[int, ...], dtype[T]]
Number: TypeAlias = int | float
ArrayLike: TypeAlias = Sequence[Number] | Array[Any] | np.typing.ArrayLike
