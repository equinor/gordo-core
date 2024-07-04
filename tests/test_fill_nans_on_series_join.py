"""Tests for filling Nans with points from past on timeseries join.

Test case 1: no data at beginning with data in the prev look-back hours.
    - '26DE603A' - no data at beginning/not all data:
        'first_valid_datapoint': exists,
        'previous_datapoint': exists,
    - '26PST6021' - all data available.
    Result: fully filled tags.

Test case 2: no data at beginning and no data in prev look-back hours.
    - '21PIT1001 - all data available.
    - '21FIT1029' - has gaps at the beginning - and no past point.
        'first_valid_datapoint': exists,
        'previous_datapoint': missing,
    Result: gaps at the beginning.

Test case 3: no data at desired time but data in prev look-back hours.
    In this case there is no data during the fetched time period,
    but datapoints can be found during previous look-back.
    Depending on the look-back and INTERPOLATION_LIMIT length there may be
    null values left in the final dataframe. In this specific test case and
    given the choice of 48 hours, there aren’t.
    If we were to test with 24h then there will be null values for some tags.
    - '27AT1009' - no data at all.
        'first_valid_datapoint': missing,
        'previous_datapoint': exists,
    - '27PDST1222A' - no data at all
        'first_valid_datapoint': missing,
        'previous_datapoint': exists,
    Result: fully filled tags.

Test case 4: No data during time period and data in prev look-back hours
    is out interpolation_limit. Additionally look-back index-rounding is tested.
    The same as for Test case 3 but following changed:
    - for '27AT1009'
        'previous_datapoint': missing,
    - for '27PDST1222A'
        'previous_datapoint': over interpolation limit,
    Result: empty dataframe.

Test case 5: No data at the end.
    - 26XT8019: all data available.
    - 26PDT8005:
        'first_valid_datapoint': exists,
        'previous_datapoint': exists,
    Result: fully filled tags.

Test case 6: no data during desired time and data in look_back at the end
    of interpolation_limit. Similar to Test case 4 but look-back point can fill
    only one Nan due to interpolation limit.
    - '27AT1009':
        'first_valid_datapoint': missing,
        'previous_datapoint': exists,
    - '27PDST1222A':
        'first_valid_datapoint': missing,
        'previous_datapoint': exists,
    Result: Dataframe with one not Nan row.
"""

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
import pytz

from gordo_core.sensor_tag import SensorTag
from tests.data.fill_nans_data.expected_metadata import get_expected_metadata

TESTS_ROOT = Path(__file__).parent.resolve()
TEST_DATA_FOLDER = TESTS_ROOT / "data" / "fill_nans_data"
INTERPOLATION_LIMIT = "48h"
RESOLUTION = "10T"
# use pytz tz not face 'Inferred time zone not equal to passed time zone' error.
COMMON_TZ = pytz.UTC


def _dt(date_time: str) -> datetime:
    """Parse and make UTC aware datetime."""
    return datetime.fromisoformat(date_time).replace(tzinfo=COMMON_TZ)


def _read_series_from_file(file_name: str, series_name: str) -> pd.Series:
    """Read .csv file into Series (1 index and 1 column with values)."""
    series = pd.read_csv(
        TEST_DATA_FOLDER / file_name,
        header=0,
        index_col="time",
        dtype={"time": "str"},
        parse_dates=["time"],
        float_precision="round_trip",
    ).squeeze("columns")
    series.name = series_name
    return series


def _series_iterable(
    series: dict[str, pd.Series]
) -> Iterable[Tuple[pd.Series, SensorTag]]:
    """Make iterable with series data."""
    for tag_name, s in series.items():
        yield s, SensorTag(name=tag_name)


def _read_df_from_file(file_name: str) -> pd.DataFrame:
    """Read .csv file into Series (1 index and i value column)."""
    # setting up tag's column types is required for reading empty Dataframes.
    col_names_without_index = pd.read_csv(
        TEST_DATA_FOLDER / file_name, nrows=0
    ).columns[1:]
    col_types = {col: float for col in col_names_without_index}

    df = pd.read_csv(
        TEST_DATA_FOLDER / file_name,
        header=0,
        index_col=0,
        float_precision="round_trip",
        dtype=col_types,
    )
    # setup datetime index to match dtypes.
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.freq = RESOLUTION  # make same frequency for df matching.
    return df


def _all_nans_series(
    start_time: datetime, end_time: datetime, series_name: str, frequency: str = "7T"
) -> pd.Series:
    """Make Series with Nans and datetime index of the given range."""
    index = pd.date_range(start_time, end_time, freq=frequency, tz=COMMON_TZ)
    return pd.Series(index=index, name=series_name, dtype=float)


def _make_closest_past_points(points_data: dict[str, float]) -> list[pd.Series]:
    """Make return values for 'data_provider.get_closest_datapoint' func."""
    return [
        pd.Series([v], index=[pd.Timestamp(t, tz=COMMON_TZ)])
        for t, v in points_data.items()
    ]


@pytest.mark.parametrize(
    "first_tag,second_tag,start_time,end_time,closest_datapoint,use_nan_series",
    [
        (
            # tc1: no data at beginning but data in the look_back before.
            "26DE603A",
            "26PST6021",
            _dt("2021-08-12 00:00:00+00:00"),
            _dt("2021-08-13 00:00:00+00:00"),
            _make_closest_past_points({"2021-08-11 17:10:00+00:00": 1.1}),
            False,
        ),
        (
            # tc2: no data at beginning and no data in look_back before.
            "21FIT1029",
            "21PIT1001",
            datetime.fromisoformat("2021-10-15 00:00:00+00:00"),
            datetime.fromisoformat("2021-10-16 00:00:00+00:00"),
            [None],
            False,
        ),
        (
            # tc3: no data during desired time but data in look-back before.
            "27AT1009",
            "27PDST1222A",
            _dt("2021-09-16 00:00:00+00:00"),
            _dt("2021-09-17 00:00:00+00:00"),
            _make_closest_past_points(
                {
                    "2021-09-15 11:50:00+00:00": 50033.66015625,
                    "2021-09-16 00:00:00+00:00": 0.09788713604211807,
                },
            ),
            True,
        ),
        (
            # tc4: no data during desired time and no data in look_back before within interpolation_limit.
            "27AT1009",
            "27PDST1222A",
            _dt("2021-09-16 00:00:00+00:00"),
            _dt("2021-09-17 00:00:00+00:00"),
            _make_closest_past_points(
                {
                    "2021-09-15 11:50:00+00:00": 50033.66015625,
                    # found point but longer than 48h, rounded to '23:50:00'.
                    "2021-09-13 23:51:00+00:00": 0.48,
                },
            ),
            True,
        ),
        (
            # tc5: no data at the end but data in look-back before.
            "26PDT8005",
            "26XT8019",
            _dt("2022-01-12 14:00:00+00:00"),
            _dt("2022-01-13 14:00:00+00:00"),
            _make_closest_past_points(
                {"2022-01-12 12:20:00+00:00": 0.059157028794288635}
            ),
            False,
        ),
        (
            # tc6: no data during desired time and data in look_back at the end of interpolation_limit.
            "27AT1009",
            "27PDST1222A",
            _dt("2021-09-16 00:00:00+00:00"),
            _dt("2021-09-17 00:00:00+00:00"),
            _make_closest_past_points(
                {
                    "2021-09-15 11:50:00+00:00": 50033.66015625,
                    # found point to fill only one missing nan, rounded to '00:00:00'.
                    "2021-09-14 00:01:00+00:00": 0.48,
                },
            ),
            True,
        ),
        (
            # statr and end time are not rounded (minutes and seconds are not 00:00).
            "26PDT8005",
            "26XT8019",
            _dt("2022-01-12 21:24:44+00:00"),
            _dt("2022-01-13 06:24:44+00:00"),
            [None],
            False,
        ),
    ],
    ids=["tc1", "tc2", "tc3", "tc4", "tc5", "tc6", "tc7"],
)
def test_look_back_filling_in_join_to_dataframe(
    first_tag,
    second_tag,
    start_time,
    end_time,
    closest_datapoint,
    use_nan_series,
    dataset,
    request,
):
    """See to level docstring for detailed tests explanations."""
    if use_nan_series:
        first_tag_series, second_tag_series = [
            _all_nans_series(start_time=start_time, end_time=end_time, series_name=tag)
            for tag in (first_tag, second_tag)
        ]
    else:
        first_tag_series, second_tag_series = [
            _read_series_from_file(f"{request.node.callspec.id}_{tag}.csv", tag)
            for tag in (first_tag, second_tag)
        ]

    with patch.object(
        dataset.data_provider, "get_closest_datapoint", side_effect=closest_datapoint
    ):
        joined_df, metadata = dataset._join_to_dataframe(
            series_iterable=_series_iterable(
                {first_tag: first_tag_series, second_tag: second_tag_series}
            ),
            resampling_startpoint=start_time,
            resampling_endpoint=end_time,
            resolution=RESOLUTION,
            aggregation_methods="mean",
            interpolation_method="linear_interpolation",
            interpolation_limit=INTERPOLATION_LIMIT,
        )
    expected_metadata = get_expected_metadata(
        request.node.callspec.id, first_tag, second_tag
    )
    expected_joined_df = _read_df_from_file(
        f"{request.node.callspec.id}_expected_df.csv"
    )

    assert metadata == expected_metadata
    pd.testing.assert_frame_equal(joined_df, expected_joined_df)
