# -*- coding: utf-8 -*-

from datetime import datetime, timezone
import logging
from typing import Iterable, Optional, Tuple, Union, cast
from unittest.mock import MagicMock
import warnings

import dateutil.parser
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gordo_core.base import GordoBaseDataset
from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.data_providers.providers import RandomDataProvider
from gordo_core.time_series import (
    RandomDataset,
    TimeSeriesDataset,
    NotEnoughDataWarning,
)
from gordo_core.exceptions import (
    EmptyGeneratedDataframeError,
    GlobalExtremaEmptyDataError,
    KnownPeriodsEmptyDataError,
    NuisanceEmptyDataError,
    RowFilterEmptyDataError,
    InsufficientDataError,
    ConfigException,
)
from gordo_core.sensor_tag import Tag, SensorTag, extract_tag_name
from gordo_core.utils import capture_args


def create_timeseries_list():
    """Create three dataframes with different resolution and different start/ends"""
    # Test for no NaNs, test for correct first and last date
    latest_start = "2018-01-03 06:00:00Z"
    earliest_end = "2018-01-05 06:00:00Z"

    index_seconds = pd.date_range(
        start="2018-01-01 06:00:00Z", end="2018-01-07 06:00:00Z", freq="S"
    )
    index_minutes = pd.date_range(
        start="2017-12-28 06:00:00Z", end=earliest_end, freq="T"
    )
    index_hours = pd.date_range(
        start=latest_start, end="2018-01-12 06:00:00Z", freq="H"
    )

    timeseries_seconds = pd.Series(
        data=np.random.randint(0, 100, len(index_seconds)),
        index=index_seconds,
        name="ts-seconds",
    )
    timeseries_minutes = pd.Series(
        data=np.random.randint(0, 100, len(index_minutes)),
        index=index_minutes,
        name="ts-minutes",
    )
    timeseries_hours = pd.Series(
        data=np.random.randint(0, 100, len(index_hours)),
        index=index_hours,
        name="ts-hours",
    )

    return (
        [
            (timeseries_seconds, SensorTag("ts-seconds")),
            (timeseries_minutes, SensorTag("ts-minutes")),
            (timeseries_hours, SensorTag("ts-hours")),
        ],
        latest_start,
        earliest_end,
    )


def extract_gaps_metadata(metadata, columns):
    gaps_metadata = []
    for column in columns:
        column_metadata = metadata[column]
        gaps_metadata.append(
            {
                "first_timestamp": column_metadata["first_timestamp"],
                "last_timestamp": column_metadata["last_timestamp"],
                "gaps": column_metadata["gaps"],
            }
        )
    return gaps_metadata


def _assert_error_equals(first, second):
    assert (type(first), vars(first)) == (type(second), vars(second))


def test_random_dataset_attrs(dataset):
    """
    Test expected attributes
    """

    assert isinstance(dataset, GordoBaseDataset)
    assert hasattr(dataset, "get_data")
    assert hasattr(dataset, "get_metadata")

    X, y = dataset.get_data()
    assert isinstance(X, pd.DataFrame)

    # y can either be None or an numpy array
    assert isinstance(y, pd.DataFrame) or y is None

    metadata = dataset.get_metadata()
    assert isinstance(metadata, dict)


def test_join_to_dataframe(dataset):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()

    assert (
        len(timeseries_list[0][0])
        > len(timeseries_list[1][0])
        > len(timeseries_list[2][0])
    )

    frequency = "7T"
    timedelta = pd.Timedelta("7 minutes")
    resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
    resampling_end = dateutil.parser.isoparse("2018-01-15 08:00:00Z")
    all_in_frame, metadata = dataset._join_to_dataframe(
        timeseries_list, resampling_start, resampling_end, frequency
    )
    metadata_keys = sorted(metadata.keys())
    assert [
        "aggregate_metadata",
        "tags",
        "ts-hours",
        "ts-minutes",
        "ts-seconds",
    ] == metadata_keys

    # Check that first resulting resampled, joined row is within "frequency" from
    # the real first data point
    assert all_in_frame.index[0] >= pd.Timestamp(latest_start) - timedelta
    assert all_in_frame.index[-1] <= pd.Timestamp(resampling_end)


@pytest.mark.parametrize(
    "n_rows, resolution, error, error_value",
    [
        # Frequency passed as zero, resulting in an ZeroDivisionError during aggregation
        (None, "0T", ZeroDivisionError, None),
        # Empty series results in an InsufficientDataError
        (0, "12T", InsufficientDataError, None),
        # When all rows are NaNs and dropped result in InsufficientDataError
        (None, "12T", InsufficientDataError, None),
        # Rows less then or equal to `row_threshold` result in InsufficientDataError
        (6, "12T", InsufficientDataError, None),
        # The same as above, but with a more specific error
        (
            0,
            "12T",
            EmptyGeneratedDataframeError,
            EmptyGeneratedDataframeError(0, 0, ["Tag 1", "Tag 2", "Tag 3"]),
        ),
        (
            None,
            "12T",
            EmptyGeneratedDataframeError,
            EmptyGeneratedDataframeError(0, 0, ["Tag 1", "Tag 2", "Tag 3"]),
        ),
        (
            6,
            "12T",
            EmptyGeneratedDataframeError,
            EmptyGeneratedDataframeError(0, 0, ["Tag 1", "Tag 2", "Tag 3"]),
        ),
    ],
)
def test_join_to_dataframe_empty_series(n_rows, resolution, error, error_value):
    """
    Test that empty data scenarios raise appropriate errors
    """
    train_start_date = dateutil.parser.isoparse("2018-01-01 00:00:00+00:00")
    train_end_date = dateutil.parser.isoparse("2018-01-05 00:00:00+00:00")
    tag_list = [SensorTag(name=n) for n in ["Tag 1", "Tag 2", "Tag 3"]]

    kwargs = {
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "tag_list": tag_list,
        "resolution": resolution,
        "data_provider": MockDataProvider(value=np.NaN, n_rows=n_rows),
    }

    with pytest.raises(error) as excinfo:
        TimeSeriesDataset(**kwargs).get_data()

    if error_value is not None:
        _assert_error_equals(excinfo.value, error_value)


def test_join_to_dataframe_nonutcstart(dataset):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()
    frequency = "7T"
    resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00+07:00")
    resampling_end = dateutil.parser.isoparse("2018-01-12 13:07:00+07:00")
    all_in_frame, metadata = dataset._join_to_dataframe(
        timeseries_list,
        resampling_start,
        resampling_end,
        frequency,
        interpolation_limit="8H",
    )
    metadata_keys = sorted(metadata.keys())
    assert [
        "aggregate_metadata",
        "tags",
        "ts-hours",
        "ts-minutes",
        "ts-seconds",
    ] == metadata_keys
    assert len(all_in_frame) == 481

    assert extract_gaps_metadata(
        metadata, ["ts-hours", "ts-minutes", "ts-seconds"]
    ) == [
        {
            "first_timestamp": 1514958960.0,
            "last_timestamp": 1515737220.0,
            "gaps": {"start": [1514156400.0], "end": [1514958960.0]},
        },
        {
            "first_timestamp": 1514440680.0,
            "last_timestamp": 1515160560.0,
            "gaps": {
                "start": [1514156400.0, 1515160560.0],
                "end": [1514440680.0, 1515737220.0],
            },
        },
        {
            "first_timestamp": 1514786340.0,
            "last_timestamp": 1515333180.0,
            "gaps": {
                "start": [1514156400.0, 1515333180.0],
                "end": [1514786340.0, 1515737220.0],
            },
        },
    ]


def test_join_to_dataframe_with_gaps(dataset):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()

    assert (
        len(timeseries_list[0][0])
        > len(timeseries_list[1][0])
        > len(timeseries_list[2][0])
    )

    remove_from = "2018-01-03 10:00:00Z"
    remove_to = "2018-01-03 18:00:00Z"
    timeseries_with_holes = [
        (ts[(ts.index < remove_from) | (ts.index >= remove_to)], sensor_tag)
        for ts, sensor_tag in timeseries_list
    ]

    frequency = "10T"
    resampling_start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
    resampling_end = dateutil.parser.isoparse("2018-01-12 07:00:00Z")

    all_in_frame, metadata = dataset._join_to_dataframe(
        timeseries_with_holes,
        resampling_start,
        resampling_end,
        frequency,
        interpolation_limit="8h",
    )
    metadata_keys = sorted(metadata.keys())
    assert [
        "aggregate_metadata",
        "tags",
        "ts-hours",
        "ts-minutes",
        "ts-seconds",
    ] == metadata_keys
    assert all_in_frame.index[0] == pd.Timestamp(latest_start)
    assert all_in_frame.index[-1] <= pd.Timestamp(resampling_end)

    assert extract_gaps_metadata(
        metadata, ["ts-hours", "ts-minutes", "ts-seconds"]
    ) == [
        {
            "first_timestamp": 1514959200.0,
            "last_timestamp": 1515740400.0,
            "gaps": {
                "start": [1514181600.0, 1514998800.0],
                "end": [1514959200.0, 1515002400.0],
            },
        },
        {
            "first_timestamp": 1514440800.0,
            "last_timestamp": 1515160800.0,
            "gaps": {
                "start": [1514181600.0, 1515160800.0],
                "end": [1514440800.0, 1515740400.0],
            },
        },
        {
            "first_timestamp": 1514786400.0,
            "last_timestamp": 1515333600.0,
            "gaps": {
                "start": [1514181600.0, 1515333600.0],
                "end": [1514786400.0, 1515740400.0],
            },
        },
    ]


def test_join_to_dataframe_with_interpolation_method_wrong_interpolation_method(
    dataset,
):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()
    resampling_start = dateutil.parser.isoparse("2017-01-01 06:00:00+07:00")
    resampling_end = dateutil.parser.isoparse("2018-02-01 13:07:00+07:00")

    with pytest.raises(ValueError):
        dataset._join_to_dataframe(
            timeseries_list,
            resampling_start,
            resampling_end,
            resolution="10T",
            interpolation_method="wrong_method",
            interpolation_limit="8H",
        )


def test_join_to_dataframe_with_interpolation_method_wrong_interpolation_limit(dataset):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()
    resampling_start = dateutil.parser.isoparse("2017-01-01 06:00:00+07:00")
    resampling_end = dateutil.parser.isoparse("2018-02-01 13:07:00+07:00")

    with pytest.raises(ValueError):
        dataset._join_to_dataframe(
            timeseries_list,
            resampling_start,
            resampling_end,
            resolution="10T",
            interpolation_method="ffill",
            interpolation_limit="1T",
        )


def test_join_to_dataframe_with_interpolation_method_linear_interpolation(dataset):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()
    resampling_start = dateutil.parser.isoparse("2017-01-01 06:00:00+07:00")
    resampling_end = dateutil.parser.isoparse("2018-02-01 13:07:00+07:00")

    all_in_frame, metadata = dataset._join_to_dataframe(
        timeseries_list,
        resampling_start,
        resampling_end,
        resolution="10T",
        interpolation_method="linear_interpolation",
        interpolation_limit="8H",
    )
    metadata_keys = sorted(metadata.keys())
    assert [
        "aggregate_metadata",
        "tags",
        "ts-hours",
        "ts-minutes",
        "ts-seconds",
    ] == metadata_keys
    assert len(all_in_frame) == 337

    assert extract_gaps_metadata(
        metadata, ["ts-hours", "ts-minutes", "ts-seconds"]
    ) == [
        {
            "first_timestamp": 1514959200.0,
            "last_timestamp": 1515765600.0,
            "gaps": {
                "start": [1483225200.0, 1515765600.0],
                "end": [1514959200.0, 1517465220.0],
            },
        },
        {
            "first_timestamp": 1514440800.0,
            "last_timestamp": 1515160800.0,
            "gaps": {
                "start": [1483225200.0, 1515160800.0],
                "end": [1514440800.0, 1517465220.0],
            },
        },
        {
            "first_timestamp": 1514786400.0,
            "last_timestamp": 1515333600.0,
            "gaps": {
                "start": [1483225200.0, 1515333600.0],
                "end": [1514786400.0, 1517465220.0],
            },
        },
    ]


def test_join_to_dataframe_with_interpolation_method_linear_interpolation_no_limit(
    dataset,
):
    timeseries_list, latest_start, earliest_end = create_timeseries_list()
    resampling_start = dateutil.parser.isoparse("2017-01-01 06:00:00+07:00")
    resampling_end = dateutil.parser.isoparse("2018-02-01 13:07:00+07:00")

    all_in_frame, metadata = dataset._join_to_dataframe(
        timeseries_list,
        resampling_start,
        resampling_end,
        resolution="10T",
        interpolation_method="linear_interpolation",
        interpolation_limit=None,
    )
    metadata_keys = sorted(metadata.keys())
    assert [
        "aggregate_metadata",
        "tags",
        "ts-hours",
        "ts-minutes",
        "ts-seconds",
    ] == metadata_keys
    assert len(all_in_frame) == 4177

    assert extract_gaps_metadata(
        metadata, ["ts-hours", "ts-minutes", "ts-seconds"]
    ) == [
        {
            "first_timestamp": 1514959200.0,
            "last_timestamp": 1517464800.0,
            "gaps": {"start": [1483225200.0], "end": [1514959200.0]},
        },
        {
            "first_timestamp": 1514440800.0,
            "last_timestamp": 1517464800.0,
            "gaps": {"start": [1483225200.0], "end": [1514440800.0]},
        },
        {
            "first_timestamp": 1514786400.0,
            "last_timestamp": 1517464800.0,
            "gaps": {"start": [1483225200.0], "end": [1514786400.0]},
        },
    ]


def test_row_filter():
    """Tests that row_filter filters away rows"""
    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-31 06:00:00Z"),
    )
    X, _ = TimeSeriesDataset(**kwargs).get_data()
    assert 833 == len(X)

    X, _ = TimeSeriesDataset(row_filter="`Tag 1` < 5000", **kwargs).get_data()
    assert 8 == len(X)

    X, _ = TimeSeriesDataset(
        row_filter="`Tag 1` / `Tag 3` < 0.999", **kwargs
    ).get_data()
    assert 3 == len(X)


def test_aggregation_methods():
    """Tests that it works to set aggregation method(s)"""

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-31 06:00:00Z"),
    )

    # Default aggregation gives no extra columns
    X, _ = TimeSeriesDataset(**kwargs).get_data()
    assert (833, 3) == X.shape

    # The default single aggregation method gives the tag-names as columns
    assert list(X.columns) == ["Tag 1", "Tag 2", "Tag 3"]

    # Using two aggregation methods give a multi-level column with tag-names
    # on top and aggregation_method as second level
    X, _ = TimeSeriesDataset(aggregation_methods=["mean", "max"], **kwargs).get_data()

    assert (833, 6) == X.shape
    assert list(X.columns) == [
        ("Tag 1", "mean"),
        ("Tag 1", "max"),
        ("Tag 2", "mean"),
        ("Tag 2", "max"),
        ("Tag 3", "mean"),
        ("Tag 3", "max"),
    ]


def test_metadata_statistics():
    """Tests that it works to set aggregation method(s)"""

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-31 06:00:00Z"),
    )

    # Default aggregation gives no extra columns
    dataset = TimeSeriesDataset(**kwargs)
    X, _ = dataset.get_data()
    assert (833, 3) == X.shape
    metadata = dataset.get_metadata()
    assert isinstance(metadata["x_hist"], dict)
    assert len(metadata["x_hist"].keys()) == 3


def test_time_series_no_resolution():
    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
    )

    no_resolution, _ = TimeSeriesDataset(resolution=None, **kwargs).get_data()
    wi_resolution, _ = TimeSeriesDataset(resolution="10T", **kwargs).get_data()
    assert len(no_resolution) > len(wi_resolution)


@pytest.mark.parametrize(
    "tag_list",
    [
        [SensorTag("Tag 1"), SensorTag("Tag 2"), SensorTag("Tag 3")],
        [SensorTag("Tag 1")],
    ],
)
@pytest.mark.parametrize(
    "target_tag_list",
    [
        [SensorTag("Tag 2"), SensorTag("Tag 1"), SensorTag("Tag 3")],
        [SensorTag("Tag 1")],
        [SensorTag("Tag10")],
        [],
    ],
)
def test_timeseries_target_tags(tag_list, target_tag_list):
    start = dateutil.parser.isoparse("2017-12-25 06:00:00Z")
    end = dateutil.parser.isoparse("2017-12-29 06:00:00Z")
    tsd = TimeSeriesDataset(
        start,
        end,
        tag_list=tag_list,
        target_tag_list=target_tag_list,
        data_provider=MockDataProvider(),
    )
    X, y = tsd.get_data()

    assert len(X) == len(y)

    # If target_tag_list is empty, it defaults to tag_list
    if target_tag_list:
        assert y.shape[1] == len(target_tag_list)
    else:
        assert y.shape[1] == len(tag_list)

    # Ensure the order in maintained
    assert [tag.name for tag in target_tag_list or tag_list] == y.columns.tolist()

    # Features should match the tag_list
    assert X.shape[1] == len(tag_list)

    # Ensure the order in maintained
    assert [tag.name for tag in tag_list] == X.columns.tolist()


class MockDataProvider(GordoBaseDataProvider):
    def __init__(self, value=None, n_rows=None, **kwargs):
        """With value argument for generating different types of data series (e.g. NaN)"""
        self.value = value
        self.n_rows = n_rows
        self.last_tag_list = None

    def can_handle_tag(self, tag):
        return True

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        self.last_tag_list = tag_list
        index = pd.date_range(train_start_date, train_end_date, freq="s")
        for i, name in enumerate(sorted([extract_tag_name(tag) for tag in tag_list])):
            # If value not passed, data for each tag are staggered integer ranges
            data = [self.value if self.value else i for i in range(i, len(index) + i)]
            series = pd.Series(index=index, data=data, name=name)
            output_series = series[: self.n_rows] if self.n_rows else series
            yield output_series, SensorTag(output_series.name)


def test_timeseries_dataset_compat():
    """
    There are accepted keywords in the config file when using type: TimeSeriesDataset
    which don't actually match the kwargs of the dataset's __init__; for compatibility
    :func:`gordo_core.time_series.compat` should adjust for these differences.
    """
    dataset = TimeSeriesDataset(
        data_provider=MockDataProvider(),
        train_start_date="2017-12-25 06:00:00Z",
        train_end_date="2017-12-29 06:00:00Z",
        tags=[SensorTag("Tag 1")],
    )
    assert dataset.train_start_date == dateutil.parser.isoparse("2017-12-25 06:00:00Z")
    assert dataset.train_end_date == dateutil.parser.isoparse("2017-12-29 06:00:00Z")
    assert dataset.tag_list == [SensorTag("Tag 1")]


@pytest.mark.parametrize(
    "n_samples_threshold, filter_value, expected_data_length",
    [(10, 5000, 8), (0, 100, 0)],
)
def test_insufficient_data_after_row_filtering(
    n_samples_threshold, filter_value, expected_data_length
):
    """
    Test that dataframe after row_filter scenarios raise appropriate
    InsufficientDataError
    """

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        n_samples_threshold=n_samples_threshold,
    )

    with pytest.raises(InsufficientDataError) as excinfo:
        TimeSeriesDataset(row_filter=f"`Tag 1` < {filter_value}", **kwargs).get_data()

    _assert_error_equals(
        excinfo.value,
        RowFilterEmptyDataError(
            expected_data_length, n_samples_threshold, f"`Tag 1` < {filter_value}", 0
        ),
    )


@pytest.mark.parametrize(
    "n_samples_threshold, high_threshold, low_threshold, expected_data_length",
    [(10, 5000, -1000, 8), (0, 100, 0, 0)],
)
def test_insufficient_data_after_global_filtering(
    n_samples_threshold, high_threshold, low_threshold, expected_data_length
):
    """
    Test that dataframe after row_filter scenarios raise appropriate
    InsufficientDataError
    """

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        n_samples_threshold=n_samples_threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )

    with pytest.raises(InsufficientDataError) as excinfo:
        TimeSeriesDataset(**kwargs).get_data()

    _assert_error_equals(
        excinfo.value,
        GlobalExtremaEmptyDataError(expected_data_length, n_samples_threshold),
    )


def test_insufficient_data_after_known_filter_periods_filtering():
    """
    Test that dataframe after row_filter scenarios raise appropriate
    InsufficientDataError
    """

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        n_samples_threshold=10,
        known_filter_periods=[
            "~('2017-12-25 07:00:00+00:00' <= index <= '2017-12-29 06:00:00+00:00')"
        ],
    )

    with pytest.raises(InsufficientDataError) as excinfo:
        TimeSeriesDataset(**kwargs).get_data()

    _assert_error_equals(excinfo.value, KnownPeriodsEmptyDataError(6, 10))


def test_insufficient_data_after_automatic_filtering():
    """
    Test that dataframe after row_filter scenarios raise appropriate
    InsufficientDataError
    """

    kwargs = dict(
        data_provider=MockDataProvider(),
        tag_list=[
            SensorTag("Tag 1"),
            SensorTag("Tag 2"),
            SensorTag("Tag 3"),
        ],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        n_samples_threshold=560,
        filter_periods={"filter_method": "all"},
    )

    with pytest.raises(InsufficientDataError) as excinfo:
        TimeSeriesDataset(**kwargs).get_data()

    _assert_error_equals(excinfo.value, NuisanceEmptyDataError(559, 560))


def test_trigger_tags():
    data_provider = MockDataProvider()
    dataset = TimeSeriesDataset(
        data_provider=data_provider,
        tag_list=[SensorTag("Tag 1"), SensorTag("Tag 2")],
        target_tag_list=[SensorTag("Tag 5")],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        row_filter="`Tag 3` > 0 & `Tag 4` > 1",
    )
    X, y = dataset.get_data()
    assert X is not None
    assert y is not None
    assert set(data_provider.last_tag_list) == {
        SensorTag("Tag 1"),
        SensorTag("Tag 2"),
        SensorTag("Tag 3"),
        SensorTag("Tag 4"),
        SensorTag("Tag 5"),
    }
    assert set(X.columns.values) == {"Tag 1", "Tag 2"}
    assert set(y.columns.values) == {"Tag 5"}


def test_get_dataset_with_full_import():
    dataset = GordoBaseDataset.from_dict(
        {
            "type": "gordo_core.time_series.RandomDataset",
            "train_start_date": "2017-12-25 06:00:00Z",
            "train_end_date": "2017-12-29 06:00:00Z",
            "tag_list": [
                SensorTag("Tag 1"),
                SensorTag("Tag 2"),
            ],
        }
    )
    assert type(dataset) is RandomDataset


@pytest.mark.skip(reason="Switch this test to checking .get_client_data()")
def test_process_metadata():
    data_provider = MockDataProvider()
    dataset = TimeSeriesDataset(
        data_provider=data_provider,
        tag_list=[SensorTag("Tag 1"), SensorTag("Tag 2")],
        target_tag_list=[SensorTag("Tag 5")],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        row_filter="`Tag 3` > 0 & `Tag 4` > 1",
        process_metadata=False,
    )
    dataset.get_data()
    assert dataset._metadata == {}


class DatasetForTest(GordoBaseDataset):
    @capture_args
    def __init__(self):
        super(DatasetForTest, self).__init__()

    def get_data(
        self,
    ) -> Tuple[
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ]:
        return np.array([]), np.array([])


def test_legacy_to_dict():
    dataset = RandomDataset(
        "2017-12-25 06:00:00Z",
        "2017-12-29 06:00:00Z",
        [SensorTag("Tag 1"), SensorTag("Tag 2")],
    )
    config = dataset.to_dict()
    assert config["type"] == "gordo_core.time_series.RandomDataset"


def test_to_dict_for_test_dataset():
    dataset = DatasetForTest()
    config = dataset.to_dict()
    assert config["type"] == "tests.test_dataset.DatasetForTest"


def test_deprecated_arguments(caplog):
    with caplog.at_level(logging.ERROR):
        TimeSeriesDataset(
            data_provider=MockDataProvider(),
            train_start_date="2017-12-25 06:00:00Z",
            train_end_date="2017-12-29 06:00:00Z",
            tags=[SensorTag("Tag 1")],
            depricated_argument=1,
        )
    assert (
        "Deprecated argument depricated_argument=1 provided for TimeSeriesDataset"
        in caplog.text
    )


def test_warning_data_has_gaps():
    tag1 = SensorTag("tag1")
    tag2 = SensorTag("tag2")
    tag3 = SensorTag("tag3")

    tags = [tag1, tag2, tag3]

    def load_series(*args, **kwargs):
        index = pd.date_range("2022-01-01", periods=144, freq="10T", tz=timezone.utc)
        tag1_data = np.concatenate((np.repeat(np.nan, 100), np.random.rand(44)), axis=0)
        tag2_data = np.concatenate((np.repeat(np.nan, 50), np.random.rand(94)), axis=0)
        tag3_data = np.random.rand(144)
        return iter(
            [
                (pd.Series(tag1_data, name="tag1", index=index), tag1),
                (pd.Series(tag2_data, name="tag2", index=index), tag2),
                (pd.Series(tag3_data, name="tag3", index=index), tag3),
            ]
        )

    def get_closest_datapoint(*args, **kwargs):
        return None

    def tag_normalizer(*args, **kwargs):
        return tags

    data_provider = MagicMock(spec=GordoBaseDataProvider)
    data_provider.load_series = load_series
    data_provider.get_closest_datapoint = get_closest_datapoint
    data_provider.tag_normalizer = tag_normalizer
    dataset = TimeSeriesDataset(
        data_provider=data_provider,
        train_start_date="2022-01-01T00:00:00+00:00",
        train_end_date="2022-01-02T00:00:00+00:00",
        tags=tags,
        resolution="10T",
    )
    expected = (
        "Dataset doesn't have enough data, for training period "
        "'2022-01-01T00:00:00+00:00' to '2022-01-02T00:00:00+00:00': "
        '"tag1", "tag2"'
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _ = dataset.get_data()
        w = recorded[-1]
        assert w.category == NotEnoughDataWarning
        assert str(w.message) == expected


def test_data_provider_as_dict():
    with pytest.raises(ValueError):
        TimeSeriesDataset(
            data_provider={"type": "RandomDataProvider"},
            train_start_date="2022-01-01T00:00:00+00:00",
            train_end_date="2022-01-02T00:00:00+00:00",
            tag_list=["tag1", "tag2"],
        )


def test_all_tags_are_triggered():
    dataset = RandomDataset(
        train_start_date="2022-01-01T00:00:00+00:00",
        train_end_date="2022-01-02T00:00:00+00:00",
        tag_list=["tag1", "tag2"],
        row_filter=["`tag1` > 0.5", "`tag3` < 0.5"],
    )
    X, y = dataset.get_data()
    assert (X.columns == ["tag1", "tag2"]).all()
    assert (y.columns == ["tag1", "tag2"]).all()


def test_deprecated_asset():
    dataset = TimeSeriesDataset.with_data_provider(
        {"type": "RandomDataProvider"},
        dict(
            train_start_date="2022-01-01T00:00:00+00:00",
            train_end_date="2022-01-02T00:00:00+00:00",
            tag_list=["tag1", "tag2"],
            asset="some",
        ),
    )
    assert dataset.default_tag == {"asset": "some"}


def test_deprecated_asset_with_default_tag():
    with pytest.raises(ConfigException):
        TimeSeriesDataset.with_data_provider(
            {"type": "RandomDataProvider"},
            dict(
                train_start_date="2022-01-01T00:00:00+00:00",
                train_end_date="2022-01-02T00:00:00+00:00",
                tag_list=["tag1", "tag2"],
                default_tag={"field": "some"},
                asset="some",
            ),
        )


def test_trigger_tags_metadata():
    class CustomRandomDataProvider(RandomDataProvider):
        tags_required_fields = ("number",)

    data_provider = CustomRandomDataProvider()
    dataset = TimeSeriesDataset(
        data_provider=data_provider,
        tag_list=[SensorTag("Tag 1", number="1"), SensorTag("Tag 2", number="2")],
        additional_tags=[SensorTag("Tag 3", number="3")],
        train_start_date=dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        train_end_date=dateutil.parser.isoparse("2017-12-29 06:00:00Z"),
        row_filter="`Tag 3` > 0 & `Tag 4` > 0",
    )
    dataset.get_data()
    metadata = dataset.get_metadata()
    assert "row_filter_tags" in metadata
    assert set(metadata["row_filter_tags"]) == {"Tag 3", "Tag 4"}
