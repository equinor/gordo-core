# -*- coding: utf-8 -*-

import os

import pandas as pd
import pytest

from gordo_core.filters.periods import FilterPeriods


@pytest.fixture
def data():
    data_parquet_path = os.path.join(
        os.path.dirname(__file__), "data", "periods", "data.parquet"
    )
    return pd.read_parquet(data_parquet_path)


def test_data_shape(data):
    assert data.shape == (4890, 1)
    assert data["Tag 1"].mean() == 0.5032590860542199


def test_filter_periods_typerror(data):
    with pytest.raises(TypeError):
        FilterPeriods(granularity="10T", filter_method="abc", n_iqr=1)


def test_filter_periods_quantile(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T",
        filter_method="quantile",
        n_iqr=1,
        quantile_upper=0.8,
        quantile_lower=0.2,
    ).filter_data(data)
    print("Quantile test results:")
    print(sum(predictions["quantile"]["pred"]))
    print(len(drop_periods["quantile"]))
    print(data_filtered.shape)
    assert sum(predictions["quantile"]["pred"]) == -96
    assert len(drop_periods["quantile"]) == 4
    assert data_filtered.shape == (4794, 1)


def test_filter_periods_median(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="median", n_iqr=1
    ).filter_data(data)

    assert sum(predictions["median"]["pred"]) == -377
    assert len(drop_periods["median"]) == 26
    assert data_filtered.shape == (4513, 1)


def test_filter_periods_iforest(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=False
    ).filter_data(data)

    assert sum(predictions["iforest"]["pred"]) == 4596
    assert len(drop_periods["iforest"]) == 16
    assert data_filtered.shape == (4743, 1)


def test_filter_periods_all(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T",
        filter_method="all",
        n_iqr=1,
        iforest_smooth=False,
        quantile_upper=0.8,
        quantile_lower=0.2,
    ).filter_data(data)

    assert sum(predictions["quantile"]["pred"]) == -96
    assert len(drop_periods["quantile"]) == 4
    assert sum(predictions["median"]["pred"]) == -377
    assert sum(predictions["iforest"]["pred"]) == 4596
    assert len(drop_periods["median"]) == 26
    assert len(drop_periods["iforest"]) == 16
    assert data_filtered.shape == (4352, 1)


def test_filter_periods_iforest_smoothing(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="iforest", iforest_smooth=True
    ).filter_data(data)

    assert sum(predictions["iforest"]["pred"]) == 4178
    assert len(drop_periods["iforest"]) == 16
    assert data_filtered.shape == (4534, 1)


def test_filter_periods_all_smoothing(data):
    data_filtered, drop_periods, predictions = FilterPeriods(
        granularity="10T", filter_method="all", n_iqr=1, iforest_smooth=True
    ).filter_data(data)

    assert sum(predictions["iforest"]["pred"]) == 4178
    assert len(drop_periods["median"]) == 26
    assert len(drop_periods["iforest"]) == 16
    assert data_filtered.shape == (4187, 1)
