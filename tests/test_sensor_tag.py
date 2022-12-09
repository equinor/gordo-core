import pickle

import pytest

from gordo_core.sensor_tag import (
    SensorTag,
    SensorTagNormalizationError,
    extract_tag_name,
    load_sensor_tag,
    normalize_sensor_tag,
    tag_to_json,
    to_list_of_strings,
    unique_tag_names,
)


def test_sensor_tag():
    sensor_tag = SensorTag("tag", field1="value1", field2=None)
    assert sensor_tag.name == "tag"
    assert sensor_tag.field1 == "value1"
    assert sensor_tag.field2 is None
    assert sensor_tag.to_json() == {"name": "tag", "field1": "value1", "field2": None}


def test_sensor_tag_state():
    serialized = pickle.dumps(SensorTag("tag", field="value"))
    sensor_tag = pickle.loads(serialized)
    assert sensor_tag.name == "tag"
    assert sensor_tag.field == "value"


def test_sensor_tag_eq():
    sensor_tag = SensorTag("tag", field1="value1")
    assert sensor_tag == SensorTag("tag", field1="value1")
    assert sensor_tag != SensorTag("tag1", field1="value1")
    assert sensor_tag != {"name": "tag", "field1": "value1"}
    assert "name" in sensor_tag
    assert "field1" in sensor_tag


def test_sensor_tag_validation_error():
    with pytest.raises(SensorTagNormalizationError):
        fields = {"field1": 1}
        SensorTag("tag", **fields)  # type: ignore


@pytest.mark.parametrize(
    "sensor,field,expected",
    [
        (SensorTag("tag2", field="some"), None, SensorTag("tag2", field="some")),
        ({"name": "tag1", "field": "field1"}, None, SensorTag("tag1", field="field1")),
        ("tag3", "field3", SensorTag("tag3", field="field3")),
        ("tag", None, "tag"),
    ],
)
def test_normalize_sensor_tag_success(sensor, field, expected):
    sensor_tag = normalize_sensor_tag(sensor, required_fields=("field",), field=field)
    assert sensor_tag == expected


@pytest.mark.parametrize(
    "sensor,field",
    [
        ({"name": "tag1"}, None),
        ({"field": "field1"}, None),
        (["tag", "field", "something_else"], None),
        (42, None),
    ],
)
def test_normalize_sensor_tag_failed(sensor, field):
    with pytest.raises(SensorTagNormalizationError):
        normalize_sensor_tag(sensor, required_fields=("field",), field=field)


@pytest.mark.parametrize(
    "sensor,expected",
    [
        ({"name": "tag1", "field": "field1"}, SensorTag("tag1", field="field1")),
    ],
)
def test_load_sensor_tag(sensor, expected):
    assert load_sensor_tag(sensor) == expected


def test_load_sensor_tag_fail():
    with pytest.raises(SensorTagNormalizationError):
        load_sensor_tag(23)  # type: ignore


def test_extract_tag_name():
    assert extract_tag_name("tag1") == "tag1"
    assert extract_tag_name(SensorTag("tag2", field="field2")) == "tag2"


@pytest.mark.parametrize(
    "tags",
    [
        [
            SensorTag("tag1", field="field1"),
            SensorTag("tag1", field="field2"),
        ],
        ["tag1", SensorTag("tag1", field="field")],
        [SensorTag("tag1", field="field"), "tag1"],
    ],
)
def test_unique_tag_names_errors(tags):
    with pytest.raises(ValueError):
        unique_tag_names(tags)


def test_to_list_of_strings():
    sensor_tags = [SensorTag("tag1"), SensorTag("tag2")]
    assert to_list_of_strings(sensor_tags) == ["tag1", "tag2"]


def test_tag_to_json():
    assert tag_to_json("tag") == "tag"
    assert tag_to_json(SensorTag("tag")) == {"name": "tag"}
