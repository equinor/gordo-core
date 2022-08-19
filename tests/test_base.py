import pytest

from typing import cast
from datetime import datetime

from dateutil.tz import tzutc

from gordo_core.base import GordoBaseDataset
from gordo_core.sensor_tag import SensorTag
from gordo_core.time_series import TimeSeriesDataset
from gordo_core.data_providers import RandomDataProvider
from gordo_core.exceptions import ConfigException


def test_from_dict():
    train_start_date = datetime(2020, 1, 1, tzinfo=tzutc())
    train_end_date = datetime(2020, 3, 1, tzinfo=tzutc())
    tag_list = [SensorTag("tag1"), SensorTag("tag2")]

    config = {
        "type": "TimeSeriesDataset",
        "data_provider": {
            "type": "gordo_core.data_providers.RandomDataProvider",
        },
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "tag_list": tag_list,
    }
    dataset = GordoBaseDataset.from_dict(config)
    assert type(dataset) is TimeSeriesDataset
    assert dataset.train_start_date == train_start_date
    assert dataset.train_end_date == train_end_date
    assert dataset.tag_list == tag_list


def test_from_dict_with_empty_type():
    train_start_date = datetime(2020, 1, 1, tzinfo=tzutc())
    train_end_date = datetime(2020, 3, 1, tzinfo=tzutc())
    tag_list = [SensorTag("tag1"), SensorTag("tag2")]

    config = {
        "data_provider": RandomDataProvider(),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "tag_list": tag_list,
    }
    dataset = GordoBaseDataset.from_dict(config)
    assert type(dataset) is TimeSeriesDataset
    assert dataset.train_start_date == train_start_date
    assert dataset.train_end_date == train_end_date
    assert dataset.tag_list == tag_list


def test_to_dict_build_in():
    train_start_date = datetime(2020, 1, 1, tzinfo=tzutc())
    train_end_date = datetime(2020, 3, 1, tzinfo=tzutc())
    tag_list = [SensorTag("tag1"), SensorTag("tag2")]

    dataset = TimeSeriesDataset(
        data_provider=RandomDataProvider(),
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        tag_list=tag_list,
    )
    config = dataset.to_dict()
    assert config["train_start_date"] == "2020-01-01T00:00:00+00:00"
    assert config["train_end_date"] == "2020-03-01T00:00:00+00:00"
    assert config["tag_list"] == tag_list
    assert config["type"] == "gordo_core.time_series.TimeSeriesDataset"


def test_default_data_provider():
    dataset = cast(
        TimeSeriesDataset,
        GordoBaseDataset.from_dict(
            {
                "data_provider": {},
                "train_start_date": datetime(2020, 1, 1, tzinfo=tzutc()),
                "train_end_date": datetime(2020, 3, 1, tzinfo=tzutc()),
                "tag_list": [SensorTag("tag1")],
            },
            default_data_provider="gordo_core.data_providers.providers.RandomDataProvider",
        ),
    )
    assert type(dataset.data_provider) is RandomDataProvider


def test_wrong_argument():
    with pytest.raises(ConfigException):
        cast(
            TimeSeriesDataset,
            GordoBaseDataset.from_dict(
                {
                    "data_provider": {
                        "type": "gordo_core.data_providers.providers.InfluxDataProvider"
                    },
                    "train_start_date": datetime(2020, 1, 1, tzinfo=tzutc()),
                    "train_end_date": datetime(2020, 3, 1, tzinfo=tzutc()),
                    "tag_list": [SensorTag("tag1")],
                },
            ),
        )
