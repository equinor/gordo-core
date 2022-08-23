import dateutil.parser
import pandas as pd
import pytest

from gordo_core.data_providers import InfluxDataProvider, RandomDataProvider
from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.exceptions import ConfigException
from gordo_core.sensor_tag import SensorTag, SensorTagNormalizationError


def test_from_dict_default():
    config = {"type": "gordo_core.data_providers.providers.RandomDataProvider"}
    data_provider = GordoBaseDataProvider.from_dict(config)
    assert type(data_provider) is RandomDataProvider


def test_from_dict_simple():
    config = {"type": "InfluxDataProvider", "measurement": "test"}
    data_provider = GordoBaseDataProvider.from_dict(config)
    assert type(data_provider) is InfluxDataProvider


def test_from_dict_full_import():
    config = {
        "type": "gordo_core.data_providers.providers.InfluxDataProvider",
        "measurement": "test",
    }
    data_provider = GordoBaseDataProvider.from_dict(config)
    assert type(data_provider) is InfluxDataProvider


@pytest.mark.parametrize(
    "wrong_type",
    [
        "WrongProvider",
        "my_module.WrongProvider",
        "gordo_core.data_provider.providers.WrongProvider",
    ],
)
def test_from_dict_errors(wrong_type):
    with pytest.raises(ConfigException):
        config = {"type": wrong_type}
        GordoBaseDataProvider.from_dict(config)


def test_to_dict_built_in():
    data_provider = RandomDataProvider()
    config = data_provider.to_dict()
    assert config["type"] == "gordo_core.data_providers.providers.RandomDataProvider"


class CustomRandomDataProvider(RandomDataProvider):
    pass


def test_to_dict_custom():
    data_provider = CustomRandomDataProvider()
    config = data_provider.to_dict()
    assert config["type"] == "tests.data_providers.test_base.CustomRandomDataProvider"


def test_base_get_closest_datapoint():
    data_provider = GordoBaseDataProvider()
    before_time = dateutil.parser.isoparse("2017-01-01T09:11:00+00:00")
    with pytest.raises(NotImplementedError):
        data_provider.get_closest_datapoint(
            "tag1", before_time=before_time, point_max_look_back=pd.Timedelta("10 days")
        )


@pytest.mark.parametrize(
    "sensors,expected",
    [
        (["tag1", "tag2"], [SensorTag("tag1"), SensorTag("tag2")]),
        (
            [{"name": "tag1", "field": "value1"}, {"name": "tag2", "field": "value2"}],
            [SensorTag("tag1", field="value1"), SensorTag("tag2", field="value2")],
        ),
    ],
)
def test_tag_normalizer(sensors, expected):
    data_provider = GordoBaseDataProvider()
    tags_list = data_provider.tag_normalizer(sensors)
    assert tags_list == expected


@pytest.mark.parametrize(
    "sensors,exception",
    [
        ([123, None], SensorTagNormalizationError),
        (
            [{"name": "tag1", "field": "value1"}, {"name": "tag1", "field": "value2"}],
            ValueError,
        ),
        ([["tag1", "value1"], ["tag1", "value2"]], ValueError),
        (["tag1", ["tag1", "value1"]], ValueError),
    ],
)
def test_tag_normalizer_fail(sensors, exception):
    data_provider = GordoBaseDataProvider()
    with pytest.raises(exception):
        data_provider.tag_normalizer(sensors)
