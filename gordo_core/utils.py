import functools
import inspect
import logging
from collections import namedtuple
from datetime import datetime
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

DEFAULT_INFLUX_PROXIES = MappingProxyType({"https": "", "http": ""})


def capture_args_ext(ignore: Optional[Iterable[str]] = None):
    """
    Decorator that captures args and kwargs passed to a given method.
    This assumes the decorated method has a self, which has a dict of
    kwargs assigned as an attribute named _params.

    Parameters
    ----------
    ignore: Optional[Iterable[str]]
        List of arguments that need to be ignored during capturing

    Returns
    -------
    Any
        Returns whatever the original method would return
    """

    def decorator(method: Callable):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            sig_params = inspect.signature(method).parameters.items()

            # Get the default values for the method signature
            params = {
                param: value.default
                for param, value in sig_params
                if value.default is not inspect.Parameter.empty and param != "self"
            }

            # Update with arg values provided
            arg_map = dict()
            for arg_val, arg_key in zip(
                args,
                (arg for arg in inspect.getfullargspec(method).args if arg != "self"),
            ):
                arg_map[arg_key] = arg_val

            # Update params with args/kwargs provided in the current call
            params.update(arg_map)
            params.update(kwargs)

            if ignore is not None:
                for arg in ignore:
                    params.pop(arg)

            self._params = params
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


def capture_args(method: Callable):
    """
    `capture_args_ext` without arguments
    """
    return capture_args_ext()(method)


# Prediction result representation, name=str, predictions=dataframe, error_messages=list[str]
PredictionResult = namedtuple("PredictionResult", "name predictions error_messages")


def _parse_influx_uri(uri: str) -> Tuple[str, str, str, str, str, str]:
    """
    Parse an influx URI

    Parameters
    ----------
    uri: str
        Format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>

    Returns
    -------
    (str, str, str, str, str, str)
        username, password, host, port, path, database
    """
    username, password, host, port, *path, db_name = (
        uri.replace("/", ":").replace("@", ":").split(":")
    )
    path_str = "/".join(path) if path else ""
    return username, password, host, port, path_str, db_name


def influx_client_from_uri(
    uri: str,
    api_key: Optional[str] = None,
    api_key_header: Optional[str] = "Ocp-Apim-Subscription-Key",
    recreate: bool = False,
    dataframe_client: bool = False,
    proxies: Mapping[str, str] = DEFAULT_INFLUX_PROXIES,
) -> Union[InfluxDBClient, DataFrameClient]:
    """
    Get a InfluxDBClient or DataFrameClient from a SqlAlchemy like URI

    Parameters
    ----------
    uri: str
        Connection string format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>
    api_key: str
        Any api key required for the client connection
    api_key_header: str
        The name of the header the api key should be assigned
    recreate: bool
        Re/create the database named in the URI
    dataframe_client: bool
        Return a DataFrameClient instead of a standard InfluxDBClient
    proxies: dict
        A mapping of any proxies to pass to the influx client

    Returns
    -------
    Union[InfluxDBClient, DataFrameClient]
    """

    username, password, host, port, path, db_name = _parse_influx_uri(uri)

    Client = DataFrameClient if dataframe_client else InfluxDBClient

    client = Client(
        host=host,
        port=port,
        database=db_name,
        username=username,
        password=password,
        path=path,
        ssl=bool(api_key),
        proxies=proxies,
    )
    if api_key:
        client._headers[api_key_header] = api_key
    if recreate:
        client.drop_database(db_name)
        client.create_database(db_name)
    return client


def fill_series_with_look_back_points(
    series: pd.Series,
    look_back_point: pd.Series,
    end_time: datetime,
    interpolation_limit: str,
    resolution: Union[str, pd.Timedelta],
) -> pd.Series:
    """Fill of Nans of given Series with interpolated look-back points.

    Parameters
    ----------
    series : pd.Series
        Series for Nans filling.
    look_back_point : pd.Series
        Series with one closest point (with time) in past. Contains only one point.
    end_time : datetime
        latest time of the Series till what points might be filled.
    interpolation_limit : str
        time limit for interpolation.
    resolution : str
        resolution of DatetimeIndex of the Series.

    Returns
    -------
    pd.Series
        Newly copied and filled with Nans (if possible) Series.
        Given Series is not affected.
    """
    # round time of the look_back_point to match resolution, otherwise indexes won't match on 'filling'.
    look_back_point.index = look_back_point.index.round(resolution)

    series_copy = series.copy()

    back_filled_series = _make_back_filled_series(
        look_back_point=look_back_point,
        original_series=series_copy,
        end_time=end_time,
        resolution=resolution,
        interpolation_limit=interpolation_limit,
    )

    return series_copy.fillna(back_filled_series)


def _make_back_filled_series(
    look_back_point: pd.Series,
    original_series: pd.Series,
    end_time: datetime,
    resolution: Union[str, pd.Timedelta],
    interpolation_limit: str,
) -> pd.Series:
    """Make Series with points populated from the closest point in the past."""
    resolution = pd.Timedelta(resolution) if isinstance(resolution, str) else resolution
    limit = int(pd.Timedelta(interpolation_limit) / resolution)

    look_back_index = _make_look_back_series_index(
        look_back_point_time=look_back_point.index[0],
        series_first_valid_index=original_series.first_valid_index(),  # type: ignore  # Expected 'Timestamp',got 'type'
        end_time=end_time,
        resolution=resolution,
    )
    back_filled_series = pd.Series(
        index=look_back_index, dtype=float, name=original_series.name
    )

    _set_series_start_end_points(
        series=back_filled_series,
        look_back_value=look_back_point[0],
        first_not_nan_value=_get_first_non_nan_value(series=original_series),
    )

    return back_filled_series.interpolate(method="linear", limit=limit)


def _get_first_non_nan_value(series: pd.Series) -> Optional[float]:
    """Return first not Nan value or None if all values are Nans."""
    series_first_valid_index = series.first_valid_index()
    first_series_not_nan_value = (
        None
        if series_first_valid_index is None
        else series.at[series_first_valid_index]
    )
    return first_series_not_nan_value


def _set_series_start_end_points(
    series: pd.Series, look_back_value: float, first_not_nan_value: Optional[float]
):
    """Inplace set first and last point in given Series.

    If last value is None -> np.nan is used instead.
    """
    first_value = look_back_value
    last_value = first_not_nan_value if first_not_nan_value is not None else np.nan

    series.iloc[0] = first_value
    series.iloc[-1] = last_value


def _make_look_back_series_index(
    look_back_point_time: pd.Timestamp,
    series_first_valid_index: Optional[pd.Timestamp],
    end_time: datetime,
    resolution: Union[str, pd.Timedelta],
) -> pd.DatetimeIndex:
    """Make datetime index for look-back Series in given range.

    Note: pass point's time that is already rounded to particular resolution.
    """
    start = look_back_point_time
    # use 'end_time' if no data was found for tag in given time-range
    end = series_first_valid_index or end_time

    return pd.date_range(start=start, end=end, freq=resolution)


def resample(
    series: pd.Series,
    resampling_startpoint: datetime,
    resampling_endpoint: datetime,
    resolution: str,
    interpolation_limit: str,
    aggregation_methods: Union[str, list[str], Callable] = "mean",
    interpolation_method: str = "linear_interpolation",
):
    """Resample series accordingly to given parameters.

    Note: Nans are NOT dropped in this function anymore after resampling.

    Takes a single series and resamples it.
    See :class:`gordo_core.base.GordoBaseDataset.join_timeseries`
    """

    startpoint_sametz = resampling_startpoint.astimezone(tz=series.index[0].tzinfo)
    endpoint_sametz = resampling_endpoint.astimezone(tz=series.index[0].tzinfo)

    if series.index[0] > startpoint_sametz:
        # Insert a NaN at the startpoint, to make sure that all resampled
        # indexes are the same. This approach will "pad" most frames with
        # NaNs, that will be removed at the end.
        startpoint = pd.Series([np.NaN], index=[startpoint_sametz], name=series.name)
        series = pd.concat([startpoint, series])
        logging.debug(f"Appending NaN to {series.name} " f"at time {startpoint_sametz}")

    elif series.index[0] < resampling_startpoint:
        msg = (
            f"Error - for {series.name}, first timestamp "
            f"{series.index[0]} is before the resampling start point "
            f"{startpoint_sametz}"
        )
        logging.error(msg)
        raise RuntimeError(msg)

    if series.index[-1] < endpoint_sametz:
        endpoint = pd.Series([np.NaN], index=[endpoint_sametz], name=series.name)
        series = pd.concat([series, endpoint])
        logging.debug(f"Appending NaN to {series.name} " f"at time {endpoint_sametz}")
    elif series.index[-1] > endpoint_sametz:
        msg = (
            f"Error - for {series.name}, last timestamp "
            f"{series.index[-1]} is later than the resampling end point "
            f"{endpoint_sametz}"
        )
        logging.error(msg)
        raise RuntimeError(msg)

    logging.debug("Head (3) and tail(3) of dataframe to be resampled:")
    logging.debug(series.head(3))
    logging.debug(series.tail(3))

    resampled = series.resample(resolution, label="left").agg(aggregation_methods)
    # If several aggregation methods are provided, agg returns a dataframe
    # instead of a series. In this dataframe the column names are the
    # aggregation methods, like "max" and "mean", so we have to make a
    # multi-index with the series-name as the top-level and the
    # aggregation-method as the lower-level index.
    # For backwards-compatibility we *dont* return a multi-level index
    # when we have a single resampling method.
    if isinstance(resampled, pd.DataFrame):  # Several aggregation methods provided
        resampled.columns = pd.MultiIndex.from_product(
            [[series.name], resampled.columns], names=["tag", "aggregation_method"]
        )

    if interpolation_method not in ["linear_interpolation", "ffill"]:
        raise ValueError(
            "Interpolation method should be either linear_interpolation of ffill"
        )

    if interpolation_limit is not None:
        limit = int(
            pd.Timedelta(interpolation_limit).total_seconds()
            / pd.Timedelta(resolution).total_seconds()
        )

        if limit <= 0:
            raise ValueError("Interpolation limit must be larger than given resolution")
    else:
        limit = None

    if interpolation_method == "linear_interpolation":
        return resampled.interpolate(limit=limit)
    else:
        return resampled.fillna(method=interpolation_method, limit=limit)


def _to_timestamps(column: list[pd.Timestamp]):
    return [v.timestamp() for v in column]


def gaps_df_to_dict(df: pd.DataFrame) -> dict:
    """
    Extract metadata information from `_find_gaps` DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        Contains columns: start, end
    """
    if df.empty:
        return {}
    df_dict = df.to_dict("list")
    return {
        "start": _to_timestamps(df_dict["start"]),
        "end": _to_timestamps(df_dict["end"]),
    }


def find_gaps(
    values: Union[pd.Series, pd.DataFrame],
    resolution: str,
    resampling_startpoint: datetime,
    resampling_endpoint: datetime,
) -> pd.DataFrame:
    """Find the gaps in a series's index and create a dataframe to store them, with
    columns: start, end
    """

    index = values.index
    td = pd.to_timedelta(resolution)
    s = index.to_series()
    # Add start/end points to catch heading/trailing gaps
    startpoint_sametz = resampling_startpoint.astimezone(tz=index[0].tzinfo)
    endpoint_sametz = resampling_endpoint.astimezone(tz=index[0].tzinfo)
    start_end = pd.Series(
        [startpoint_sametz, endpoint_sametz], index=[startpoint_sametz, endpoint_sametz]
    )
    s = pd.concat([s, start_end]).sort_values()
    diff_ = s.diff()[1:]  # Ignore first NaN
    g = diff_[diff_ > td]

    # Output DataFrame
    gaps = pd.DataFrame(g, columns=["duration"])
    gaps["start"] = gaps.index - gaps["duration"]
    gaps = gaps.rename_axis("end").reset_index()
    return gaps[["start", "end"]]
