import json
import pickle
from typing import Any


from dtypes import FileDescriptorOrPath
from validate import Validate


class PICKLE:
    @staticmethod
    def dump(file: FileDescriptorOrPath, data: Any) -> None:
        Validate.file_extension(file, ".pkl")
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(file: FileDescriptorOrPath) -> Any:
        Validate.file_extension(file, ".pkl")
        with open(file, "rb") as f:
            return pickle.load(f)


class JSON:
    @staticmethod
    def dump(file: FileDescriptorOrPath, data: dict[Any, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(
                f"Invalid data type: expected dict, but got {type(data).__name__}."
            )
        Validate.file_extension(file, ".json")
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load(file: FileDescriptorOrPath) -> dict[Any, Any]:
        Validate.file_extension(file, ".json")
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)


class TXT:
    @staticmethod
    def dump(file: FileDescriptorOrPath, data: str) -> None:
        if not isinstance(data, str):
            raise TypeError(
                f"Invalid data type: expected str, but got {type(data).__name__}."
            )
        Validate.file_extension(file, ".txt")
        with open(file, "w", encoding="utf-8") as f:
            f.write(data)

    @staticmethod
    def load(file: FileDescriptorOrPath) -> str:
        Validate.file_extension(file, ".txt")
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
