import json
from os import PathLike
import pickle
from typing import Any, TypeAlias


StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
FileDescriptorOrPath: TypeAlias = int | StrOrBytesPath


def _check_extension(file: FileDescriptorOrPath, expected_ext: str) -> None:
    """Helper to check file extension."""
    if isinstance(file, int):
        return  # file descriptors do not have extensions
    file_str = str(file)
    if not file_str.endswith(expected_ext):
        ext = file_str.rsplit(".", 1)[-1] if "." in file_str else "<no extension>"
        raise ValueError(
            f"Invalid file type: expected a {expected_ext}, but got .{ext} file."
        )


def pickle_dump(file: FileDescriptorOrPath, data: Any) -> None:
    _check_extension(file, ".pkl")
    with open(file, "wb") as f:
        pickle.dump(data, f)


def pickle_load(file: FileDescriptorOrPath) -> Any:
    _check_extension(file, ".pkl")
    with open(file, "rb") as f:
        return pickle.load(f)


def json_dump(file: FileDescriptorOrPath, data: dict[Any, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError(
            f"Invalid data type: expected dict, but got {type(data).__name__}."
        )
    _check_extension(file, ".json")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def json_load(file: FileDescriptorOrPath) -> dict[Any, Any]:
    _check_extension(file, ".json")
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
