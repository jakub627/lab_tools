import pytest
import tempfile
import os
from file_io import PICKLE, JSON, TXT


class TestPickle:
    def test_pickle_dump_and_load(self, tmp_path):
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.pkl"
        PICKLE.dump(file_path, data)
        loaded_data = PICKLE.load(file_path)
        assert loaded_data == data

    def test_pickle_dump_invalid_extension(self, tmp_path):
        data = [1, 2, 3]
        file_path = tmp_path / "test.txt"  # Złe rozszerzenie
        with pytest.raises(ValueError, match="Invalid file type: expected '.pkl'"):
            PICKLE.dump(file_path, data)


class TestJSON:
    def test_json_dump_and_load(self, tmp_path):
        data = {"name": "test", "value": 123}
        file_path = tmp_path / "test.json"
        JSON.dump(file_path, data)
        loaded_data = JSON.load(file_path)
        assert loaded_data == data

    def test_json_dump_invalid_data_type(self, tmp_path):
        data = "not a dict"
        file_path = tmp_path / "test.json"
        with pytest.raises(TypeError, match="Invalid data type: expected dict"):
            JSON.dump(file_path, data) # type: ignore

    def test_json_dump_invalid_extension(self, tmp_path):
        data = {"key": "value"}
        file_path = tmp_path / "test.pkl"  # Złe rozszerzenie
        with pytest.raises(ValueError, match="Invalid file type: expected '.json'"):
            JSON.dump(file_path, data)


class TestTXT:
    def test_txt_dump_and_load(self, tmp_path):
        data = "Hello, world!"
        file_path = tmp_path / "test.txt"
        TXT.dump(file_path, data)
        loaded_data = TXT.load(file_path)
        assert loaded_data == data

    def test_txt_dump_invalid_data_type(self, tmp_path):
        data = 123  # Nie str
        file_path = tmp_path / "test.txt"
        with pytest.raises(TypeError, match="Invalid data type: expected str"):
            TXT.dump(file_path, data) # type: ignore

    def test_txt_dump_invalid_extension(self, tmp_path):
        data = "text"
        file_path = tmp_path / "test.json"  # Złe rozszerzenie
        with pytest.raises(ValueError, match="Invalid file type: expected '.txt'"):
            TXT.dump(file_path, data)
