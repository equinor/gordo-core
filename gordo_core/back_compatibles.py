from typing import Final

from .import_utils import BackCompatibleLocations, prepare_back_compatible_locations

BACK_COMPATIBLE_LOCATIONS: Final[list[tuple[str, str]]] = [
    ("TimeSeriesDataset", "gordo_dataset.time_series.TimeSeriesDataset"),
    (
        "gordo_dataset.datasets.TimeSeriesDataset",
        "gordo_dataset.time_series.TimeSeriesDataset",
    ),
    ("RandomDataset", "gordo_dataset.time_series.RandomDataset"),
    (
        "gordo_dataset.datasets.RandomDataset",
        "gordo_dataset.time_series.RandomDataset",
    ),
    (
        "DataLakeProvider",
        "gordo_dataset.data_providers.dl.providers.DataLakeProvider",
    ),
    (
        "gordo_dataset.data_provider.providers.DataLakeProvider",
        "gordo_dataset.data_providers.dl.providers.DataLakeProvider",
    ),
    (
        "InfluxDataProvider",
        "gordo_dataset.data_providers.providers.InfluxDataProvider",
    ),
    (
        "gordo_dataset.data_provider.providers.InfluxDataProvider",
        "gordo_dataset.data_providers.providers.InfluxDataProvider",
    ),
    (
        "RandomDataProvider",
        "gordo_dataset.data_providers.providers.RandomDataProvider",
    ),
    (
        "gordo_dataset.data_provider.providers.RandomDataProvider",
        "gordo_dataset.data_providers.providers.RandomDataProvider",
    ),
    (
        "gordo_dataset.base.GordoBaseDataset",
        "gordo_core.base.GordoBaseDataset",
    ),
    (
        "gordo_dataset.time_series.TimeSeriesDataset",
        "gordo_core.time_series.TimeSeriesDataset",
    ),
    (
        "gordo_dataset.time_series.RandomDataset",
        "gordo_core.time_series.RandomDataset",
    ),
    (
        "gordo_dataset.data_providers.base.GordoBaseDataProvider",
        "gordo_core.data_providers.base.GordoBaseDataProvider",
    ),
    (
        "gordo_dataset.data_providers.providers.RandomDataProvider",
        "gordo_core.data_providers.providers.RandomDataProvider",
    ),
    (
        "gordo_dataset.data_providers.providers.InfluxDataProvider",
        "gordo_core.data_providers.providers.InfluxDataProvider",
    ),
]

DEFAULT_BACK_COMPATIBLES: Final[
    BackCompatibleLocations
] = prepare_back_compatible_locations(BACK_COMPATIBLE_LOCATIONS)
