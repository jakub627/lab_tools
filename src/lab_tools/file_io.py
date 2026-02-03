from abc import ABC, abstractmethod
import json
import pickle
from typing import Any


from .dtypes import FileDescriptorOrPath
from .validate import Validate


class FileIO(ABC):
    @classmethod
    @abstractmethod
    def dump(cls, file: FileDescriptorOrPath, data: Any) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, file: FileDescriptorOrPath) -> Any: ...


class PICKLE(FileIO):
    @classmethod
    def dump(cls, file: FileDescriptorOrPath, data: Any) -> None:
        Validate.file_extension(file, ".pkl")
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file: FileDescriptorOrPath) -> Any:
        Validate.file_extension(file, ".pkl")
        with open(file, "rb") as f:
            return pickle.load(f)


class JSON(FileIO):
    @classmethod
    def dump(cls, file: FileDescriptorOrPath, data: dict[Any, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(
                f"Invalid data type: expected dict, but got {type(data).__name__}."
            )
        Validate.file_extension(file, ".json")
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, file: FileDescriptorOrPath) -> dict[Any, Any]:
        Validate.file_extension(file, ".json")
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)


class TXT(FileIO):
    @classmethod
    def dump(cls, file: FileDescriptorOrPath, data: str) -> None:
        if not isinstance(data, str):
            raise TypeError(
                f"Invalid data type: expected str, but got {type(data).__name__}."
            )
        Validate.file_extension(file, ".txt")
        with open(file, "w", encoding="utf-8") as f:
            f.write(data)

    @classmethod
    def load(cls, file: FileDescriptorOrPath) -> str:
        Validate.file_extension(file, ".txt")
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
