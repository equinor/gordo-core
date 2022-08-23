# -*- coding: utf-8 -*-

import copy
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.exceptions import ConfigException
from gordo_core.import_utils import BackCompatibleLocations, import_location

from .back_compatibles import DEFAULT_BACK_COMPATIBLES

logger = logging.getLogger(__name__)


def import_dataset(
    location: str, *, back_compatibles: Optional[BackCompatibleLocations] = None
):
    """
    Import `GordoBaseDataset` class.

    Parameters
    ----------
    location: str
        Class import location.
    back_compatibles: Optional[BackCompatibleLocations]
        See `gordo_core.import_utils.prepare_back_compatible_locations()` function for reference.

    Returns
    -------
        `GordoBaseDataset` class.

    """
    if not location:
        location = "TimeSeriesDataset"
    if back_compatibles is None:
        back_compatibles = DEFAULT_BACK_COMPATIBLES
    try:
        dataset_cls = import_location(
            location,
            import_path="gordo_core.time_series",
            back_compatibles=back_compatibles,
        )
    except ImportError as e:
        raise ConfigException(f'Dataset type "{location}" is not supported: {str(e)}')
    if not issubclass(dataset_cls, GordoBaseDataset):
        base_cls_name = GordoBaseDataset.__module__ + "." + GordoBaseDataset.__name__
        raise ConfigException(
            f'Dataset class "{location}" is not a subclass of "{base_cls_name}"'
        )
    return cast(Type[GordoBaseDataset], dataset_cls)


def create_with_provider(
    dataset_cls: Type["DatasetWithProvider"],
    config: dict[str, Any],
    *,
    back_compatibles: Optional[BackCompatibleLocations] = None,
    default_data_provider: Optional[str] = None,
):
    """
    Instantiate `DatasetWithProvider`. Call `DatasetWithProvider.with_data_provider()` under the hood.

    Parameters
    ----------
    dataset_cls: Type["DatasetWithProvider"]
        `DatasetWithProvider` class.
    config: dict[str, Any]
        Dataset arguments.
    back_compatibles: Optional[BackCompatibleLocations]
        See `gordo_core.import_utils.prepare_back_compatible_locations()` function for reference.
    default_data_provider: Optional[str]
        Default data provider type. Will be taken if `data_provider.type` is empty.

    Returns
    -------
        `DatasetWithProvider` instance.

    """
    args = copy.copy(config)
    data_provider = args.pop("data_provider", None)
    if isinstance(data_provider, dict):
        provider_type = data_provider.get("type")
        if not provider_type:
            data_provider["type"] = default_data_provider
    elif data_provider is None and default_data_provider:
        data_provider = {"type": default_data_provider}
    dataset_cls = cast(Type[DatasetWithProvider], dataset_cls)
    try:
        return dataset_cls.with_data_provider(
            data_provider, args, back_compatibles=back_compatibles
        )
    except TypeError as e:
        location = dataset_cls.__module__ + "." + dataset_cls.__name__
        raise ConfigException(f'Unable to create dataset "{location}": {str(e)}')


def create_dataset(
    dataset_cls: Type["GordoBaseDataset"],
    args: dict[str, Any],
    *,
    back_compatibles: Optional[BackCompatibleLocations] = None,
    default_data_provider: Optional[str] = None,
) -> "GordoBaseDataset":
    """
    Create GordoBaseDataset instance.

    Parameters
    ----------
    dataset_cls: Type[GordoBaseDataset]
        Dataset class.
    args: dict[str, Any]
        __init__ arguments.
    back_compatibles: Optional[BackCompatibleLocations]
        See `gordo_core.import_utils.prepare_back_compatible_locations()` function for reference.
    default_data_provider: Optional[str]
        Default data provider type. Will be taken if `data_provider.type` is empty.

    Returns
    -------
    """
    if issubclass(dataset_cls, DatasetWithProvider):
        return create_with_provider(
            cast(Type[DatasetWithProvider], dataset_cls),
            args,
            back_compatibles=back_compatibles,
            default_data_provider=default_data_provider,
        )
    else:
        try:
            return cast(Callable[..., GordoBaseDataset], dataset_cls)(**args)
        except TypeError as e:
            location = dataset_cls.__module__ + "." + dataset_cls.__name__
            raise ConfigException(f'Unable to create dataset "{location}": {str(e)}')


class GordoBaseDataset(metaclass=ABCMeta):
    def __init__(self):
        self._metadata: dict[Any, Any] = dict()
        # provided by @capture_args on child's __init__
        if not hasattr(self, "_params"):
            self._params = dict()

    @abstractmethod
    def get_data(
        self,
    ) -> Tuple[
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ]:
        """
        Return X, y data as numpy or pandas' dataframes given current state
        """

    def get_client_data(
        self, build_dataset_metadata: dict
    ) -> Tuple[
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
        Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ]:
        """
        The version of `get_data` used by gordo-client

        Parameters
        ----------
        build_dataset_metadata: dict
            build_metadata.dataset part of the metadata

        Returns
        -------

        """
        return self.get_data()

    def to_dict(self) -> dict:
        """
        Serialize this object into a dict representation, which can be used to
        initialize a new object using :func:`~GordoBaseDataset.from_dict`

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
        for key, value in params.items():
            if hasattr(value, "to_dict"):
                params[key] = value.to_dict()
        return params

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        *,
        back_compatibles: Optional[BackCompatibleLocations] = None,
        default_data_provider: Optional[str] = None,
    ) -> "GordoBaseDataset":
        """
        Construct the dataset using a config from :func:`~GordoBaseDataset.to_dict`
        """
        args = copy.copy(config)
        kind = args.pop("type", "")
        dataset_cls = import_dataset(kind, back_compatibles=back_compatibles)
        return create_dataset(
            dataset_cls,
            args,
            back_compatibles=back_compatibles,
            default_data_provider=default_data_provider,
        )

    def get_metadata(self):
        """
        Get metadata about the current state of the dataset
        """
        return dict()


class DatasetWithProvider(GordoBaseDataset, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def with_data_provider(
        cls,
        data_provider: Optional[Union[dict[str, Any], GordoBaseDataProvider]],
        args: dict[str, Any],
        *,
        back_compatibles: Optional[BackCompatibleLocations] = None,
    ):
        ...

    @abstractmethod
    def get_data_provider(self) -> GordoBaseDataProvider:
        ...
