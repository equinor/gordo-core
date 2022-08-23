# -*- coding: utf-8 -*-
import datetime
import logging

import dateutil.parser
import pandas as pd

from gordo_core.sensor_tag import SensorTag

logger = logging.getLogger(__name__)


class BaseDescriptor:
    """
    Base descriptor class

    New object should override __set__(self, instance, value) method to check
    if 'value' meets required needs.
    """

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        raise NotImplementedError("Setting value not implemented for this Validator!")


class ValidDataset(BaseDescriptor):
    """
    Descriptor for attributes requiring type :class:`gordo.workflow.config_elements.Dataset`
    """

    def __set__(self, instance, value):

        # Avoid circular dependency imports
        from gordo_core.base import GordoBaseDataset

        if not isinstance(value, GordoBaseDataset):
            raise TypeError(
                f"Expected value to be an instance of GordoBaseDataset, found {value}"
            )
        instance.__dict__[self.name] = value


class ValidDatasetKwargs(BaseDescriptor):
    """
    Descriptor for attributes requiring type :class:`gordo.workflow.config_elements.Dataset`
    """

    def _verify_resolution(self, resolution: str):
        """
        Verifies that a resolution string is supported in pandas
        """
        try:
            pd.tseries.frequencies.to_offset(resolution)
        except ValueError:
            raise ValueError(
                'Values for "resolution" must match pandas frequency terms: '
                "http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html"
            )

    def __set__(self, instance, value):
        if not isinstance(value, dict):
            raise TypeError(f"Expected kwargs to be an instance of dict, found {value}")

        # Check that if 'resolution' is defined, it's one of supported pandas resampling frequencies
        if "resolution" in value:
            self._verify_resolution(value["resolution"])
        instance.__dict__[self.name] = value


class ValidDataProvider(BaseDescriptor):
    """
    Descriptor for DataProvider
    """

    def __set__(self, instance, value):

        # Avoid circular dependency imports
        from gordo_core.data_providers.base import GordoBaseDataProvider

        if not isinstance(value, GordoBaseDataProvider):
            raise TypeError(
                f"Expected value to be an instance of GordoBaseDataProvider, "
                f"found {value} "
            )
        instance.__dict__[self.name] = value


class ValidDatetime(BaseDescriptor):
    """
    Descriptor for attributes requiring valid datetime.datetime attribute
    """

    def __set__(self, instance, value):
        datetime_value = None
        if isinstance(value, datetime.datetime):
            datetime_value = value
        elif isinstance(value, str):
            datetime_value = dateutil.parser.isoparse(value)
        else:
            raise ValueError(
                f"'{value}' is not a valid datetime.datetime object or string!"
            )

        if datetime_value.tzinfo is None:
            raise ValueError(f"Provide timezone to timestamp '{value}'")

        instance.__dict__[self.name] = datetime_value


class ValidTagList(BaseDescriptor):
    """
    Descriptor for attributes requiring a non-empty list of strings
    """

    def __set__(self, instance, value):
        if (
            len(value) == 0
            or not isinstance(value, list)
            or not any(isinstance(value[0], inst) for inst in (str, dict, SensorTag))
        ):
            raise ValueError("Requires setting a non-empty list of strings")
        instance.__dict__[self.name] = value
