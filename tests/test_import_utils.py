import pytest

from typing import Optional

from gordo_core.import_utils import (
    import_location,
    prepare_back_compatible_locations,
    BackCompatibleLocations,
)
from gordo_core.time_series import TimeSeriesDataset


def test_import_utils_succeeded():
    dataset = import_location("gordo_core.time_series.TimeSeriesDataset")
    assert dataset is TimeSeriesDataset
    dataset = import_location("TimeSeriesDataset", import_path="gordo_core.time_series")
    assert dataset is TimeSeriesDataset


def test_import_utils_failed():
    with pytest.raises(ImportError):
        import_location("wrong_model.WrongModel")
    with pytest.raises(ImportError):
        import_location("TimeSeriesDataset")
    with pytest.raises(ImportError):
        import_location("gordo_core.time_series.WrongDataset")


def test_prepare_back_compatible_locations():
    back_compatibles = prepare_back_compatible_locations(
        [
            (
                "gordo_core.datasets.TimeSeriesDataset",
                "gordo_core.time_series.TimeSeriesDataset",
            )
        ]
    )
    assert back_compatibles == {
        ("gordo_core.datasets", "TimeSeriesDataset"): (
            "gordo_core.time_series",
            "TimeSeriesDataset",
        )
    }


def test_prepare_back_compatible_locations_failed():
    locations = [
        ("gordo_core.datasets.TimeSeriesDataset", "TimeSeriesDataset"),
        (
            "gordo_core.datasets.TimeSeriesDataset",
            "gordo_core.time_series.TimeSeriesDataset",
        ),
    ]
    with pytest.raises(ValueError):
        prepare_back_compatible_locations(locations)
    locations = [
        (
            "gordo_core.datasets.TimeSeriesDataset",
            "gordo_core.datasets.TimeSeriesDataset",
        )
    ]
    with pytest.raises(RecursionError):
        prepare_back_compatible_locations(locations)
    locations = [
        ("gordo_core.datasets.TimeSeriesDataset", "TimeSeriesDataset"),
        ("TimeSeriesDataset", "gordo_core.datasets.TimeSeriesDataset"),
    ]
    with pytest.raises(RecursionError):
        prepare_back_compatible_locations(locations)


def test_import_locate_with_back_back_compatibles():
    back_compatibles: Optional[BackCompatibleLocations] = {
        ("gordo_core.datasets", "TimeSeriesDataset"): (
            "gordo_core.time_series",
            "TimeSeriesDataset",
        ),
        (None, "TimeSeriesDataset"): ("gordo_core.time_series", "TimeSeriesDataset"),
    }
    dataset = import_location(
        "gordo_core.datasets.TimeSeriesDataset", back_compatibles=back_compatibles
    )
    assert dataset is TimeSeriesDataset
    dataset = import_location(
        "TimeSeriesDataset",
        import_path="gordo_core.datasets",
        back_compatibles=back_compatibles,
    )
    assert dataset is TimeSeriesDataset
    back_compatibles = {
        (None, "TimeSeriesDataset"): ("gordo_core.datasets", "TimeSeriesDataset"),
        ("gordo_core.datasets", "TimeSeriesDataset"): (
            "gordo_core.time_series",
            "TimeSeriesDataset",
        ),
    }
    dataset = import_location("TimeSeriesDataset", back_compatibles=back_compatibles)
    assert dataset is TimeSeriesDataset


def test_import_locate_with_back_back_compatibles_failed():
    back_compatibles: Optional[BackCompatibleLocations] = {
        ("gordo_core.datasets", "TimeSeriesDataset"): (
            "gordo_core.datasets",
            "TimeSeriesDataset",
        ),
    }
    with pytest.raises(RecursionError):
        import_location(
            "gordo_core.datasets.TimeSeriesDataset",
            back_compatibles=back_compatibles,
        )
