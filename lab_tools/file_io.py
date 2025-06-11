import json
from os import PathLike
import pickle
from typing import Any, Mapping, TypeAlias

import h5py
import numpy as np


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


def txt_dump(file: FileDescriptorOrPath, data: str) -> None:
    if not isinstance(data, str):
        raise TypeError(
            f"Invalid data type: expected str, but got {type(data).__name__}."
        )
    _check_extension(file, ".txt")
    with open(file, "w", encoding="utf-8") as f:
        f.write(data)


def txt_load(file: FileDescriptorOrPath) -> str:
    _check_extension(file, ".txt")
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def hdf5_dump(file: FileDescriptorOrPath, data: dict[str, Any]) -> None:
    """Save a dictionary of arrays or nested structures to an HDF5 file."""
    _check_extension(file, ".h5")
    if not isinstance(data, dict):
        raise TypeError(f"Invalid data type: expected dict, got {type(data).__name__}")

    def write_group(group: h5py.Group, d: Mapping[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, Mapping):
                subgroup = group.create_group(key)
                write_group(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression="gzip")
            elif isinstance(value, (int, float, str, bytes)):
                group.attrs[key] = value  # store scalars as attributes
            else:
                raise TypeError(
                    f"Unsupported value type for key '{key}': {type(value)}"
                )

    with h5py.File(file, "w") as f:
        write_group(f, data)


def hdf5_load(file: FileDescriptorOrPath) -> dict[str, Any]:
    """Load an HDF5 file into a nested dictionary."""
    _check_extension(file, ".h5")

    def read_group(group: h5py.Group) -> dict[str, Any]:
        result: dict[str, Any] = {}
        # read datasets
        for key, val in group.items():
            if isinstance(val, h5py.Group):
                result[key] = read_group(val)
            elif isinstance(val, h5py.Dataset):
                result[key] = val[()]
        # read attributes (scalars)
        for key, val in group.attrs.items():
            result[key] = val
        return result

    with h5py.File(file, "r") as f:
        return read_group(f)
