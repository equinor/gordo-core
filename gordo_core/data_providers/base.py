# -*- coding: utf-8 -*-

import abc
from copy import copy
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Tuple, cast

import pandas as pd

from gordo_core.exceptions import ConfigException
from gordo_core.import_utils import BackCompatibleLocations, import_location
from gordo_core.sensor_tag import Sensor, Tag, normalize_sensor_tag, unique_tag_names

from ..back_compatibles import DEFAULT_BACK_COMPATIBLES


class GordoBaseDataProvider:

    tags_required_fields = cast(tuple[str, ...], ())

    @abc.abstractmethod
    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        """
        Load the required data as an iterable of series where each
        contains the values of the tag with time index

        Parameters
        ----------
        train_start_date: datetime
            Datetime object representing the start of fetching data
        train_end_date: datetime
            Datetime object representing the end of fetching data
        tag_list: list[Tag]
            List of tags to fetch, where each will end up being its own dataframe
        dry_run: Optional[bool]
            Set to true to perform a "dry run" of the loading.
            Up to the implementations to determine what that means.
        kwargs: dict
            With these - additional data might be passed by data_provider.

        Returns
        -------
        Iterable[Tuple[pd.Series, SensorTag]]
        """
        ...

    @abc.abstractmethod
    def can_handle_tag(self, tag: Tag):
        """
        Returns true if the dataprovider thinks it can possibly read this tag.
        Typically checks if the asset part of the tag is known to the reader.

        Parameters
        ----------
        tag: SensorTag - Dictionary with a "tag" key and optional "asset"

        Returns
        -------
        bool

        """
        ...

    def to_dict(self):
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object after popping 'type' from the dict.

        Returns
        -------
        dict
        """
        if not hasattr(self, "_params"):
            raise AttributeError(
                "Failed to lookup init parameters, ensure the "
                "object's __init__ is decorated with 'capture_args'"
            )
        # Update dict with the class
        params = getattr(self, "_params")
        params_type = self.__module__ + "." + self.__class__.__name__
        params["type"] = params_type
        return params

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        *,
        back_compatibles: Optional[BackCompatibleLocations] = None,
    ) -> "GordoBaseDataProvider":
        if back_compatibles is None:
            back_compatibles = DEFAULT_BACK_COMPATIBLES
        args = copy(config)
        provider_type = args.pop("type", "")
        if not provider_type:
            raise ConfigException("data_provider.type is empty")
        try:
            provider_cls = import_location(
                provider_type,
                import_path="gordo_core.data_providers",
                back_compatibles=back_compatibles,
            )
        except ImportError as e:
            raise ConfigException(
                f"Unable to find data provider '{provider_type}': {str(e)}"
            )
        if not issubclass(provider_cls, GordoBaseDataProvider):
            base_cls_name = (
                GordoBaseDataProvider.__module__ + "." + GordoBaseDataProvider.__name__
            )
            raise ConfigException(
                f'Data provider class "{provider_type}" is not a subclass of "{base_cls_name}"'
            )
        try:
            data_provider = cast(Callable[..., GordoBaseDataProvider], provider_cls)(
                **args
            )
        except TypeError as e:
            raise ConfigException(
                f'Unable to create data provider "{provider_type}": {str(e)}'
            )
        return data_provider

    def get_closest_datapoint(
        self, tag: Tag, before_time: datetime, point_max_look_back: pd.Timedelta
    ) -> Optional[pd.Series]:
        """
        Latest data point of tag from some time in the past till before_time, None if nothing found.
        This function is optional for implementing in the child classes,
        if it's not implemented NotImplementedError will be thrown.
        """
        raise NotImplementedError()

    def get_metadata(self):
        """
        Get metadata about the current state of the data provider
        """
        return dict()

    def tag_normalizer(
        self,
        sensors: list[Sensor],
        **kwargs: Optional[str],
    ) -> list[Tag]:
        """
        Prepare and validate sensors list.
        This function might be useful for overwriting in the extended class
        """
        tag_list = [
            normalize_sensor_tag(sensor, self.tags_required_fields, **kwargs)
            for sensor in sensors
        ]
        unique_tag_names(tag_list)
        return tag_list
