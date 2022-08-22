# -*- coding: utf-8 -*-
import collections
import logging
import warnings
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from dateutil.parser import isoparse

from gordo_core.data_providers import GordoBaseDataProvider, RandomDataProvider
from gordo_core.exceptions import (
    ConfigException,
    EmptyGeneratedDataframeError,
    GlobalExtremaEmptyDataError,
    InsufficientDataError,
    KnownPeriodsEmptyDataError,
    NuisanceEmptyDataError,
    RowFilterEmptyDataError,
)
from gordo_core.filters.periods import FilterPeriods
from gordo_core.filters.rows import pandas_filter_rows, parse_pandas_filter_vars
from gordo_core.sensor_tag import (
    Sensor,
    Tag,
    extract_tag_name,
    tag_to_json,
    unique_tag_names,
)
from gordo_core.utils import (
    capture_args,
    fill_series_with_look_back_points,
    find_gaps,
    gaps_df_to_dict,
    resample,
)
from gordo_core.validators import (
    ValidDataProvider,
    ValidDatasetKwargs,
    ValidDatetime,
    ValidTagList,
)

from .base import DatasetWithProvider
from .import_utils import BackCompatibleLocations
from .metadata import sensor_tags_from_build_metadata, tags_to_json_representation

logger = logging.getLogger(__name__)

DEFAULT_INTERPOLATION_LIMIT = "48H"


class NotEnoughDataWarning(RuntimeWarning):
    pass


def compat(init):
    """
    __init__ decorator for compatibility where the Gordo config file's ``dataset`` keys have
    drifted from what kwargs are actually expected in the given dataset. For example,
    using `train_start_date` is common in the configs, but :class:`~TimeSeriesDataset`
    takes this parameter as ``train_start_date``, as well as :class:`~RandomDataset`

    Renames old/other acceptable kwargs to the ones that the dataset type expects
    """

    @wraps(init)
    def wrapper(*args, **kwargs):
        renamings = {
            "from_ts": "train_start_date",
            "to_ts": "train_end_date",
            "tags": "tag_list",
        }
        for old, new in renamings.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        return init(*args, **kwargs)

    return wrapper


class TimeSeriesDataset(DatasetWithProvider):

    train_start_date = ValidDatetime()
    train_end_date = ValidDatetime()
    tag_list = ValidTagList()
    target_tag_list = ValidTagList()
    data_provider = ValidDataProvider()
    kwargs = ValidDatasetKwargs()

    @classmethod
    def with_data_provider(
        cls,
        data_provider: Optional[Union[dict[str, Any], GordoBaseDataProvider]],
        args: dict[str, Any],
        *,
        back_compatibles: Optional[BackCompatibleLocations] = None,
    ):
        if isinstance(data_provider, dict):
            data_provider = GordoBaseDataProvider.from_dict(
                data_provider, back_compatibles=back_compatibles
            )
        return cls(
            **args,
            data_provider=data_provider,
        )

    @compat
    @capture_args
    def __init__(
        self,
        train_start_date: Union[datetime, str],
        train_end_date: Union[datetime, str],
        tag_list: Sequence[Sensor],
        target_tag_list: Optional[Sequence[Sensor]] = None,
        additional_tags: Optional[Sequence[Sensor]] = None,
        default_tag: Optional[dict[str, Optional[str]]] = None,
        data_provider: Optional[GordoBaseDataProvider] = None,
        resolution: Optional[str] = "10T",
        row_filter: Union[str, list] = "",
        known_filter_periods: Optional[list] = None,
        aggregation_methods: Union[str, list[str], Callable] = "mean",
        row_filter_buffer_size: int = 0,
        asset: Optional[str] = None,
        n_samples_threshold: int = 0,
        low_threshold: Optional[int] = -10_000,
        high_threshold: Optional[int] = 500_000,
        interpolation_method: str = "linear_interpolation",
        interpolation_limit: str = DEFAULT_INTERPOLATION_LIMIT,
        filter_periods: Optional[Union[dict, FilterPeriods]] = None,
        **kwargs,
    ):
        """
        Creates a TimeSeriesDataset backed by a provided dataprovider.

        A TimeSeriesDataset is a dataset backed by timeseries, but resampled,
        aligned, and (optionally) filtered.

        Parameters
        ----------
        train_start_date: Union[datetime, str]
            Earliest possible point in the dataset (inclusive)
        train_end_date: Union[datetime, str]
            Earliest possible point in the dataset (exclusive)
        tag_list: Sequence[Union[str, dict, sensor_tag.SensorTag]]
            List of tags to include in the dataset. The elements can be strings,
            dictionaries or SensorTag namedtuples.
        target_tag_list: Sequence[list[Union[str, dict, sensor_tag.SensorTag]]]
            List of tags to set as the dataset y. These will be treated the same as
            tag_list when fetching and pre-processing (resampling) but will be split
            into the y return from ``.get_data()``
        data_provider: Optional[GordoBaseDataProvider]
            A dataprovider which can provide dataframes for tags from train_start_date to train_end_date
        resolution: Optional[str]
            The bucket size for grouping all incoming time data (e.g. "10T").
            Available strings come from
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            **Note**: If this parameter is ``None`` or ``False``,
            then _no_ aggregation/resampling is applied to the data.
        row_filter: str or list
            Filter on the rows. Only rows satisfying the filter will be in the dataset.
            See :func:`gordo_core.filter_rows.pandas_filter_rows` for
            further documentation of the filter format.
        known_filter_periods: list
            List of periods to drop in the format
            [~('2020-04-08 04:00:00+00:00' < index < '2020-04-08 10:00:00+00:00')].
            Note the time-zone suffix (+00:00), which is required.
        aggregation_methods
            Aggregation method(s) to use for the resampled buckets. If a single
            resample method is provided then the resulting dataframe will have names
            identical to the names of the series it got in. If several
            aggregation-methods are provided then the resulting dataframe will
            have a multi-level column index, with the series-name as the first level,
            and the aggregation method as the second level.
            See :py:func::`pandas.core.resample.Resampler#aggregate` for more
            information on possible aggregation methods.
        row_filter_buffer_size: int
            Whatever elements are selected for removal based on the ``row_filter``, will also
            have this amount of elements removed fore and aft.
            Default is zero 0
        asset: Optional[str]
            Asset for which the tags are associated with.
        n_samples_threshold: int = 0
            The threshold at which the generated DataFrame is considered to have too few rows of data.
        interpolation_method: str
            How should missing values be interpolated. Either forward fill (`ffill`) or by linear
            interpolation (default, `linear_interpolation`).
        interpolation_limit: str
            Parameter sets how long from last valid data point values will be interpolated/forward filled.
            If None, all missing values are interpolated/forward filled.
            Also, it's used as max time limit of point for look-back to find
            latest point before window's start (if needed).
        filter_periods: dict
            Performs a series of algorithms that drops noisy data is specified.
            See `filter_periods` class for details.
        kwargs
            Deprecated arguments

        .. deprecated:: 5.0.0
            `asset` will be removed in gordo-dataset 6.0.0
        """
        self.train_start_date = self._validate_dt(train_start_date)
        self.train_end_date = self._validate_dt(train_end_date)

        if self.train_start_date >= self.train_end_date:
            raise ConfigException(
                f"train_end_date ({self.train_end_date}) must be after train_start_date ({self.train_start_date})"
            )

        if data_provider is None:
            raise ConfigException(
                "data_provider is empty. "
                "Make sure data_provider.type is also specified."
            )
        if type(data_provider) is dict:
            raise ConfigException(
                "dict is deprecated for 'data_provider' since gordo-dataset>=5.0.0."
                " Use %s.instantiate() method instead." % self.__class__.__name__
            )
        self.data_provider = data_provider

        if asset is not None:
            if default_tag is not None:
                raise ConfigException(
                    "default_tag and asset can not be specified simultaneously"
                )
            default_tag = {"asset": asset}
        if default_tag is None:
            default_tag = {}
        self.default_tag = default_tag
        self.tag_list = self.data_provider.tag_normalizer(list(tag_list), **default_tag)
        self.target_tag_list = (
            self.data_provider.tag_normalizer(list(target_tag_list), **default_tag)
            if target_tag_list
            else self.tag_list.copy()
        )
        self.additional_tags = (
            self.data_provider.tag_normalizer(list(additional_tags), **default_tag)
            if additional_tags
            else []
        )
        self.resolution = resolution
        self.row_filter = row_filter
        self.aggregation_methods = aggregation_methods
        self.row_filter_buffer_size = row_filter_buffer_size
        self.n_samples_threshold = n_samples_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.interpolation_method = interpolation_method
        self.interpolation_limit = interpolation_limit
        self.filter_periods: Optional[FilterPeriods] = None
        if filter_periods:
            if isinstance(filter_periods, dict):
                self.filter_periods = FilterPeriods(
                    granularity=self.resolution, **filter_periods
                )
            else:
                self.filter_periods = cast(Optional[FilterPeriods], filter_periods)
        self.known_filter_periods = (
            known_filter_periods if known_filter_periods is not None else []
        )

        if not self.train_start_date.tzinfo or not self.train_end_date.tzinfo:
            raise ConfigException(
                f"Timestamps ({self.train_start_date}, {self.train_end_date}) need to include timezone "
                f"information"
            )

        for name, value in kwargs.items():
            logger.error(
                "Deprecated argument %s=%s provided for %s",
                name,
                value,
                self.__class__.__name__,
            )

        super().__init__()

    def get_data_provider(self) -> GordoBaseDataProvider:
        return self.data_provider

    def to_dict(self):
        params = super().to_dict()
        to_str = lambda dt: str(dt) if not hasattr(dt, "isoformat") else dt.isoformat()
        params["train_start_date"] = to_str(params["train_start_date"])
        params["train_end_date"] = to_str(params["train_end_date"])
        return params

    @staticmethod
    def _validate_dt(dt: Union[str, datetime]) -> datetime:
        dt = dt if isinstance(dt, datetime) else isoparse(dt)
        if dt.tzinfo is None:
            raise ValueError(
                "Must provide an ISO formatted datetime string with timezone information"
            )
        return dt

    @staticmethod
    def _get_row_filter_tags(row_filter: Union[str, list]) -> list[str]:
        return parse_pandas_filter_vars(row_filter)

    def _get_tag_list(
        self,
        row_filter_tags: list[str],
        tag_list: list[Tag],
        target_tag_list: list[Tag],
        additional_tags: list[Tag],
    ) -> Tuple[list[Tag], list[Tag]]:
        # TODO better docstring
        default_tag = self.default_tag
        unique_tags = unique_tag_names(tag_list, target_tag_list)

        output_trigger_tags: list[Tag] = []
        if row_filter_tags:
            all_unique_tags = unique_tag_names(unique_tags.values(), additional_tags)
            triggered_tags = []
            for tag_name in row_filter_tags:
                triggered_tags.append(all_unique_tags.get(tag_name, tag_name))
            triggered_tags = self.data_provider.tag_normalizer(
                triggered_tags, **default_tag
            )
            unique_triggered_tags = unique_tag_names(triggered_tags)
            for tag_name, tag in unique_triggered_tags.items():
                if tag_name not in unique_tags:
                    output_trigger_tags.append(tag)
                    unique_tags[tag_name] = tag
        return list(unique_tags.values()), output_trigger_tags

    def _join_timeseries(
        self,
        series_iter: Iterable[Tuple[pd.Series, Tag]],
        process_metadata: bool = True,
    ) -> pd.DataFrame:
        # TODO better docstring
        # Resample if we have a resolution set, otherwise simply join the series.
        if self.resolution:
            data, metadata = self._join_to_dataframe(
                series_iter,
                self.train_start_date,
                self.train_end_date,
                self.resolution,
                aggregation_methods=self.aggregation_methods,
                interpolation_method=self.interpolation_method,
                interpolation_limit=self.interpolation_limit,
            )
        else:
            series_list, tags = [], []
            for series, tag in series_iter:
                series_list.append(series)
                tags.append(tag)
            data = pd.concat(series_list, axis=1, join="inner")
            metadata = {"tags": tags_to_json_representation(tags)}
        if process_metadata:
            self._metadata["tag_loading_metadata"] = metadata
        return data

    def _join_to_dataframe(
        self,
        series_iterable: Iterable[Tuple[pd.Series, Tag]],
        resampling_startpoint: datetime,
        resampling_endpoint: datetime,
        resolution: str,
        aggregation_methods: Union[str, list[str], Callable] = "mean",
        interpolation_method: str = "linear_interpolation",
        interpolation_limit: str = DEFAULT_INTERPOLATION_LIMIT,
    ) -> Tuple[pd.DataFrame, dict]:
        """Resample, aggregate, join Series to Dataframe and drop Nans.

        Parameters
        ----------
        series_iterable: Iterable[pd.Series]
            An iterator supplying series with time index
        resampling_startpoint: datetime.datetime
            The starting point for resampling. Most data frames will not have this
            in their datetime index, and it will be inserted with a NaN as the value.
            The resulting NaNs will be removed, so the only important requirement for this is
            that this resampling_startpoint datetime must be before or equal to the first
            (earliest) datetime in the data to be resampled.
        resampling_endpoint: datetime.datetime
            The end point for resampling. This datetime must be equal to or after the last datetime in the
            data to be resampled.
        resolution: str
            The bucket size for grouping all incoming time data (e.g. "10T")
            Available strings come from
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        aggregation_methods: Union[str, list[str], Callable]
            Aggregation method(s) to use for the resampled buckets. If a single
            resample method is provided then the resulting dataframe will have names
            identical to the names of the series it got in. If several
            aggregation-methods are provided then the resulting dataframe will
            have a multi-level column index, with the series-name as the first level,
            and the aggregation method as the second level.
            See :py:func::`pandas.core.resample.Resampler#aggregate` for more
            information on possible aggregation methods.
        interpolation_method: str
            How should missing values be interpolated. Either forward fill (`ffill`) or by linear
            interpolation (default, `linear_interpolation`).
        interpolation_limit: str
            Parameter sets how long from last valid data point values will be interpolated/forward filled.
            Default is eight hours (`8H`).
            If None, all missing values are interpolated/forward filled.

        Returns
        -------
        pd.DataFrame
            A dataframe without NaNs, a common time index, and one column per
            element in the dataframe_generator. If multiple aggregation methods
            are provided then the resulting dataframe will have a multi-level column
            index with series-names as top-level and aggregation-method as second-level.
        dict
            Matadata information
        """
        resampled_series = []
        missing_data_series = []
        metadata: dict[str, Any] = dict()

        tags: list[Tag] = []
        columns_with_gaps: list[str] = []
        for series, tag in series_iterable:
            series_name = extract_tag_name(tag)
            tags.append(tag)
            original_length = len(series)
            metadata[series_name] = dict(original_length=original_length)
            try:
                resampled = resample(
                    series,
                    resampling_startpoint=resampling_startpoint,
                    resampling_endpoint=resampling_endpoint,
                    resolution=resolution,
                    aggregation_methods=aggregation_methods,
                    interpolation_method=interpolation_method,
                    interpolation_limit=interpolation_limit,
                )
            except IndexError:
                missing_data_series.append(series_name)
            else:
                if (
                    resampled.first_valid_index() is None
                    or resampled.first_valid_index() > resampling_startpoint
                ):
                    resampled = self.fill_series_nans(
                        series=resampled,
                        tag=tag,
                        resampling_startpoint=resampling_startpoint,
                        resampling_endpoint=resampling_endpoint,
                        resolution=resolution,
                        interpolation_limit=interpolation_limit,
                    )

                # drop Nans here, so gaps will be added to metadata-info.
                resampled = resampled.dropna()

                resampled_series.append(resampled)
                metadata[series_name].update(dict(resampled_length=len(resampled)))
                if not resampled.index.empty:
                    first_timestamp = resampled.index.min()
                    last_timestamp = resampled.index.max()
                    gaps_df = find_gaps(
                        resampled,
                        resolution=resolution,
                        resampling_startpoint=resampling_startpoint,
                        resampling_endpoint=resampling_endpoint,
                    )
                    if not gaps_df.empty:
                        columns_with_gaps.append(series_name)
                    metadata[series_name].update(
                        dict(
                            first_timestamp=first_timestamp.timestamp(),
                            last_timestamp=last_timestamp.timestamp(),
                            gaps=gaps_df_to_dict(gaps_df),
                        )
                    )
        if missing_data_series:
            raise InsufficientDataError(
                f"The following features are missing data: {missing_data_series}"
            )

        joined_df = pd.concat(resampled_series, axis=1, join="inner")

        # Before returning, delete all rows with NaN, they were introduced by the
        # insertion of NaNs in the beginning of all timeseries
        dropped_na = joined_df.dropna()

        metadata["aggregate_metadata"] = dict(
            joined_length=len(joined_df), dropped_na_length=len(dropped_na)
        )
        metadata["tags"] = tags_to_json_representation(tags)
        if columns_with_gaps:
            self._warning_data_has_gaps(columns_with_gaps)
        return dropped_na, metadata

    def fill_series_nans(
        self,
        series: pd.Series,
        tag: Tag,
        resampling_startpoint: datetime,
        resampling_endpoint: datetime,
        resolution: str,
        interpolation_limit: str,
    ) -> pd.Series:
        """Try to fill Nans from look-back interpolated point.

        Only uses point from past to Nans filling if it was found not far then
        interpolation limit.

        Returns:
        -------
        pd.Series
            Same not changed Series or Series with attempt to fill Nans.
        """
        try:
            look_back_point = self.data_provider.get_closest_datapoint(
                tag=tag,
                before_time=resampling_startpoint,
                point_max_look_back=pd.Timedelta(interpolation_limit),
            )
        except NotImplementedError:
            look_back_point = None

        if look_back_point is None:
            return series

        return fill_series_with_look_back_points(
            series,
            look_back_point,
            end_time=resampling_endpoint,
            resolution=resolution,
            interpolation_limit=interpolation_limit,
        )

    def _check_number_of_samples(self, data: pd.DataFrame, n_samples_threshold: int):
        # TODO better docstring
        if len(data) <= n_samples_threshold:
            tag_names = list(self._generate_empty_tag_names())
            raise EmptyGeneratedDataframeError(
                len(data), n_samples_threshold, tag_names
            )

    def _generate_empty_tag_names(self):
        metadata = self._metadata.get("tag_loading_metadata", {})
        for tag_name, tag_info in metadata.items():
            if tag_info.get("resampled_length") == 0:
                yield tag_name

    @staticmethod
    def _apply_row_filter_and_known_filter_periods(
        data: pd.DataFrame,
        known_filter_periods: Optional[list],
        row_filter: Union[str, list],
        triggered_tags: list[Tag],
        row_filter_buffer_size: int,
        n_samples_threshold: int,
    ) -> pd.DataFrame:
        # TODO better docstring
        if known_filter_periods:
            data = pandas_filter_rows(data, known_filter_periods, buffer_size=0)
            if len(data) <= n_samples_threshold:
                raise KnownPeriodsEmptyDataError(len(data), n_samples_threshold)

        if row_filter:
            data = pandas_filter_rows(
                data, row_filter, buffer_size=row_filter_buffer_size
            )
            if len(data) <= n_samples_threshold:
                raise RowFilterEmptyDataError(
                    len(data), n_samples_threshold, row_filter, row_filter_buffer_size
                )

        if triggered_tags:
            triggered_columns = [extract_tag_name(tag) for tag in triggered_tags]
            data = data.drop(columns=triggered_columns)

        return data

    @staticmethod
    def _check_thresholds(
        data: pd.DataFrame,
        low_threshold: Optional[int],
        high_threshold: Optional[int],
        n_samples_threshold: int,
    ) -> pd.DataFrame:
        # TODO better docstring
        if isinstance(low_threshold, int) and isinstance(high_threshold, int):
            if low_threshold >= high_threshold:
                raise ConfigException(
                    "Low threshold need to be larger than high threshold"
                )
            logger.info("Applying global min/max filtering")
            mask = ((data > low_threshold) & (data < high_threshold)).all(1)
            data = data[mask]
            logger.info("Shape of data after global min/max filtering: %s", data.shape)
            if len(data) <= n_samples_threshold:
                raise GlobalExtremaEmptyDataError(len(data), n_samples_threshold)
        return data

    def _apply_filter_periods(
        self,
        data: pd.DataFrame,
        filter_periods: Optional[FilterPeriods],
        n_samples_threshold: int,
        process_metadata: bool = True,
    ) -> pd.DataFrame:
        # TODO better docstring
        if filter_periods:
            data, drop_periods, _ = filter_periods.filter_data(data)
            if process_metadata:
                self._metadata["filtered_periods"] = drop_periods
            if len(data) <= n_samples_threshold:
                raise NuisanceEmptyDataError(len(data), n_samples_threshold)
        return data

    def _extract_x_and_y(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        # TODO better docstring
        x_tag_names = [extract_tag_name(tag) for tag in self.tag_list]
        y_tag_names = [extract_tag_name(tag) for tag in self.target_tag_list]

        X = data[x_tag_names]
        y = data[y_tag_names] if self.target_tag_list else None
        return X, y

    @staticmethod
    def _additional_metadata(X: pd.DataFrame, y: Optional[pd.DataFrame]) -> dict:
        metadata = {}
        if X.first_valid_index():
            metadata["train_start_date_actual"] = X.index[0]
            metadata["train_end_date_actual"] = X.index[-1]

        metadata["summary_statistics"] = X.describe().to_dict()
        hists = dict()
        for tag in X.columns:
            step = round((X[tag].max() - X[tag].min()) / 100, 6)
            if step < 9e-07:
                hists[str(tag)] = "{}"
                continue
            outs = pd.cut(
                X[tag],
                bins=np.arange(
                    round(X[tag].min() - step, 6),
                    round(X[tag].max() + step, 6),
                    step,
                ),
                retbins=False,
            )
            hists[str(tag)] = outs.value_counts().sort_index().to_json(orient="index")
        metadata["x_hist"] = hists
        return metadata

    @staticmethod
    def _trigger_tags_metadata(row_filter_tags: list[str]) -> dict:
        return {
            "row_filter_tags": row_filter_tags,
        }

    def _extract_provider_metadata(self):
        # TODO better docstring
        provider_metadata = self.data_provider.get_metadata()
        if provider_metadata:
            self._metadata["data_provider"] = provider_metadata

    def _warning_data_has_gaps(self, columns_with_gaps: list[str]):
        messages: list[str] = []
        for column in columns_with_gaps:
            messages.append('"%s"' % column)
        if messages:
            message_suffix = (
                "Dataset doesn't have enough data, "
                "for training period '%s' to '%s': "
                % (self.train_start_date.isoformat(), self.train_end_date.isoformat())
            )
            message = message_suffix + ", ".join(messages)
            warnings.warn(message, NotEnoughDataWarning)

    def get_client_data(
        self, build_dataset_metadata: dict
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        # TODO better docstring

        row_filter_tags = self._get_row_filter_tags(self.row_filter)
        tag_list, trigger_tag_list = self._get_tag_list(
            row_filter_tags, self.tag_list, self.target_tag_list, self.additional_tags
        )

        tag_names: Set[str] = set()
        for tags in (tag_list, trigger_tag_list):
            tag_names.update(extract_tag_name(tag) for tag in tags)

        sensor_tags = sensor_tags_from_build_metadata(build_dataset_metadata, tag_names)

        tag_list = [sensor_tags[extract_tag_name(tag)] for tag in tag_list]
        trigger_tag_list = [
            sensor_tags[extract_tag_name(tag)] for tag in trigger_tag_list
        ]

        series_iter: Iterable[Tuple[pd.Series, Tag]] = self.data_provider.load_series(
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            tag_list=tag_list,
            resolution=self.resolution,
        )

        data = self._join_timeseries(series_iter)

        self._check_number_of_samples(data, 0)

        data = self._apply_row_filter_and_known_filter_periods(
            data=data,
            known_filter_periods=[],
            row_filter=self.row_filter,
            triggered_tags=trigger_tag_list,
            row_filter_buffer_size=0,
            n_samples_threshold=0,
        )

        X, y = self._extract_x_and_y(data)

        return X, y

    def get_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        row_filter_tags = self._get_row_filter_tags(self.row_filter)
        tag_list, trigger_tag_list = self._get_tag_list(
            row_filter_tags, self.tag_list, self.target_tag_list, self.additional_tags
        )
        self._metadata.update(self._trigger_tags_metadata(row_filter_tags))

        series_iter: Iterable[Tuple[pd.Series, Tag]] = self.data_provider.load_series(
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            tag_list=tag_list,
            resolution=self.resolution,
        )

        data = self._join_timeseries(series_iter)

        self._check_number_of_samples(data, self.n_samples_threshold)

        data = self._apply_row_filter_and_known_filter_periods(
            data=data,
            known_filter_periods=self.known_filter_periods,
            row_filter=self.row_filter,
            triggered_tags=trigger_tag_list,
            row_filter_buffer_size=self.row_filter_buffer_size,
            n_samples_threshold=self.n_samples_threshold,
        )

        data = self._check_thresholds(
            data, self.low_threshold, self.high_threshold, self.n_samples_threshold
        )

        data = self._apply_filter_periods(
            data=data,
            filter_periods=self.filter_periods,
            n_samples_threshold=self.n_samples_threshold,
        )

        X, y = self._extract_x_and_y(data)

        metadata = self._additional_metadata(X, y)
        if metadata:
            self._metadata.update(metadata)

        self._extract_provider_metadata()

        return X, y

    def get_metadata(self):
        return self._metadata.copy()


class RandomDataset(TimeSeriesDataset):
    """
    Get a TimeSeriesDataset backed by
    gordo_core.data_provider.providers.RandomDataProvider
    """

    @compat
    @capture_args
    def __init__(
        self,
        train_start_date: Union[datetime, str],
        train_end_date: Union[datetime, str],
        tag_list: list,
        **kwargs,
    ):
        kwargs.pop("data_provider", None)  # Don't care what you ask for, you get random
        super().__init__(
            data_provider=RandomDataProvider(),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            tag_list=tag_list,
            **kwargs,
        )
