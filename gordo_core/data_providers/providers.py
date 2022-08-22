# -*- coding: utf-8 -*-
import logging
import random
from datetime import datetime
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from cachetools import TTLCache, cached
from influxdb import DataFrameClient

from gordo_core.sensor_tag import Tag, extract_tag_name
from gordo_core.utils import capture_args

from .base import GordoBaseDataProvider

logger = logging.getLogger(__name__)


class NoSuitableDataProviderError(ValueError):
    pass


class InfluxDataProvider(GordoBaseDataProvider):
    @capture_args
    def __init__(
        self,
        measurement: str,
        value_name: str = "Value",
        api_key: str = None,
        api_key_header: str = None,
        client: DataFrameClient = None,
        uri: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        measurement: str
            Name of the measurement to select from in Influx
        value_name: str
            Name of value to select, default to 'Value'
        api_key: str
            Api key to use in header
        api_key_header: str
            Key of header to insert the api key for requests
        uri: str
            Create a client from a URI
            format: <username>:<password>@<host>:<port>/<optional-path>/<db_name>
        kwargs: dict
            These are passed directly to the init args of influxdb.DataFrameClient
        """
        self.measurement = measurement
        self.value_name = value_name
        self.influx_client = client
        if kwargs.pop("threads", None):
            logger.warning(
                "InfluxDataProvider got parameter 'threads' which is not supported, it "
                "will be ignored."
            )

        if self.influx_client is None:
            if uri:

                # Import here to avoid any circular import error caused by
                # importing TimeSeriesDataset, which imports this provider
                # which would have imported Client via traversal of the __init__
                # which would then try to import TimeSeriesDataset again.
                from gordo_core.utils import influx_client_from_uri

                self.influx_client = influx_client_from_uri(  # type: ignore
                    uri,
                    api_key=api_key,
                    api_key_header=api_key_header,
                    dataframe_client=True,
                )
            else:
                if "type" in kwargs:
                    kwargs.pop("type")
                self.influx_client = DataFrameClient(**kwargs)
                if api_key is not None:
                    if not api_key_header:
                        raise ValueError(
                            "If supplying an api key, you must supply the header key to insert it under."
                        )
                    self.influx_client._headers[api_key_header] = api_key

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        """
        See GordoBaseDataProvider for documentation
        """
        if dry_run:
            raise NotImplementedError(
                "Dry run for InfluxDataProvider is not implemented"
            )
        # TODO better way to handle SensorTag here
        return (
            (
                self.read_single_sensor(
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    tag=extract_tag_name(tag),
                    measurement=self.measurement,
                ),
                tag,
            )
            for tag in tag_list
        )

    def read_single_sensor(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag: str,
        measurement: str,
    ) -> pd.Series:
        """
        Parameters
        ----------
            train_start_date: datetime
                Datetime to start querying for data
            train_end_date: datetime
                Datetime to stop query for data
            tag: str
                Name of the tag to match in influx
            measurement: str
                name of the measurement to select from
        Returns
        -------
            One column DataFrame
        """

        logger.info(f"Reading tag: {tag}")
        logger.info(f"Fetching data from {train_start_date} to {train_end_date}")
        query_string = f"""
            SELECT "{self.value_name}" as "{tag}"
            FROM "{measurement}"
            WHERE("tag" =~ /^{tag}$/)
                {f"AND time >= {int(train_start_date.timestamp())}s" if train_start_date else ""}
                {f"AND time <= {int(train_end_date.timestamp())}s" if train_end_date else ""}
        """

        logger.info(f"Query string: {query_string}")
        dataframes = self.influx_client.query(query_string)  # type: ignore

        try:
            df = list(dataframes.values())[0]
            return df[tag]

        except IndexError as e:
            list_of_tags = self._list_of_tags_from_influx()
            if tag not in list_of_tags:
                raise ValueError(f"tag {tag} is not found in influx")
            logger.error(
                f"Unable to find data for tag {tag} in the time range {train_start_date} - {train_end_date}"
            )
            raise e

    def _list_of_tags_from_influx(self):
        query_tags = (
            f"""SHOW TAG VALUES ON {self.influx_client._database} WITH KEY="tag" """
        )
        result = self.influx_client.query(query_tags)
        list_of_tags = []
        for item in list(result.get_points()):
            list_of_tags.append(item["value"])
        return list_of_tags

    @cached(cache=TTLCache(maxsize=10, ttl=600))
    def get_list_of_tags(self) -> list[str]:
        """
        Queries Influx for the list of tags, using a TTL cache of 600 seconds. The
        cache can be cleared with :func:`cache_clear()` as is usual with cachetools.

        Returns
        -------
        typing.list[str]
            The list of tags in Influx

        """
        return self._list_of_tags_from_influx()

    def can_handle_tag(self, tag: Tag):
        tag_name = extract_tag_name(tag)
        return tag_name in self.get_list_of_tags()


class RandomDataProvider(GordoBaseDataProvider):
    """
    Get a GordoBaseDataset which returns unstructed values for X and y. Each instance
    uses the same seed, so should be a function (same input -> same output)
    """

    def can_handle_tag(self, tag: Tag):
        return True  # We can be random about everything

    @capture_args
    def __init__(self, min_size=100, max_size=300):
        self.max_size = max_size
        self.min_size = min_size
        np.random.seed(0)

    # Thanks stackoverflow
    # https://stackoverflow.com/questions/50559078/generating-random-dates-within-a-given-range-in-pandas
    @staticmethod
    def _random_dates(start, end, n=10):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        start_u = start.value // 10**9
        end_u = end.value // 10**9

        return sorted(
            pd.to_datetime(np.random.randint(start_u, end_u, n), unit="s", utc=True)
        )

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        if dry_run:
            raise NotImplementedError(
                "Dry run for RandomDataProvider is not implemented"
            )
        for tag in tag_list:
            nr = random.randint(self.min_size, self.max_size)

            random_index = self._random_dates(train_start_date, train_end_date, n=nr)
            series = pd.Series(
                index=random_index,
                name=extract_tag_name(tag),
                data=np.random.random(size=len(random_index)),
            )
            yield series, tag

    def get_closest_datapoint(
        self, tag: Tag, before_time: datetime, point_max_look_back: pd.Timedelta
    ) -> Optional[pd.Series]:
        """Uses the same logic as method in parent class."""
        raise NotImplementedError()
