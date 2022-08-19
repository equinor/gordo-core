from unittest.mock import MagicMock

import pytest

from gordo_core.metadata import (
    sensor_tags_from_build_metadata,
    tags_to_json_representation,
)
from gordo_core.sensor_tag import SensorTag


@pytest.fixture
def build_dataset_metadata():
    return {
        "dataset_meta": {
            "tag_loading_metadata": {
                "tags": {
                    "tag1": {"name": "tag1"},
                    "tag2": {"name": "tag2"},
                }
            },
            "data_provider": {},
        }
    }


@pytest.fixture
def build_dataset_metadata_with_storage():
    return {"dataset_meta": {"data_provider": {"storage_name": "test_storage"}}}


@pytest.fixture
def build_dataset_metadata_empty():
    return {"dataset_meta": {}}


def test_sensor_tags_from_build_metadata(build_dataset_metadata):
    result = sensor_tags_from_build_metadata(build_dataset_metadata, {"tag1", "tag2"})
    assert result == {
        "tag2": SensorTag(name="tag2"),
        "tag1": SensorTag(name="tag1"),
    }


def test_sensor_tags_from_build_metadata_exception(
    build_dataset_metadata,
):
    with pytest.raises(ValueError):
        sensor_tags_from_build_metadata(
            build_dataset_metadata,
            {"tag1", "tag2", "tag3"},
        )


def test_sensor_tags_from_build_metadata_exception_empty_metadata():
    with pytest.raises(ValueError):
        sensor_tags_from_build_metadata(
            {},
            {"tag1"},
        )


def test_tags_to_json_representation():
    result = tags_to_json_representation(["tag1", SensorTag("tag2")])
    assert result == {"tag2": {"name": "tag2"}}
