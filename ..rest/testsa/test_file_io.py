import pytest
from lab_tools.file_io import pickle_dump, pickle_load, json_dump, json_load


@pytest.fixture
def sample_dict():
    return {"a": 1, "b": [2, 3], "c": {"d": 4}}


def test_pickle_roundtrip(tmp_path, sample_dict):
    path = tmp_path / "data.pkl"
    pickle_dump(path, sample_dict)
    loaded = pickle_load(path)
    assert loaded == sample_dict


def test_json_roundtrip(tmp_path, sample_dict):
    path = tmp_path / "data.json"
    json_dump(path, sample_dict)
    loaded = json_load(path)
    assert loaded == sample_dict


def test_pickle_invalid_extension(tmp_path):
    path = tmp_path / "invalid.txt"
    with pytest.raises(ValueError, match="expected a .pkl"):
        pickle_dump(path, {"x": 1})


def test_json_invalid_extension(tmp_path):
    path = tmp_path / "invalid.txt"
    with pytest.raises(ValueError, match="expected a .json"):
        json_dump(path, {"x": 1})


def test_json_invalid_data_type(tmp_path):
    path = tmp_path / "data.json"
    with pytest.raises(TypeError, match="expected dict"):
        json_dump(path, [1, 2, 3])  # type: ignore # not a dict
