from datetime import datetime

import pytest

from gordo_core.data_providers.partition import (
    MonthPartition,
    Partition,
    PartitionBy,
    YearPartition,
    split_by_partitions,
)


def test_find_by_name():
    assert PartitionBy.find_by_name("year") == PartitionBy.YEAR
    assert PartitionBy.find_by_name("solar") is None


def test_year_partition():
    assert YearPartition(2020) < YearPartition(2021)
    assert not YearPartition(2021) < YearPartition(2020)


def test_month_partition():
    assert MonthPartition(2010, 10) < MonthPartition(2011, 10)
    assert MonthPartition(2020, 10) < MonthPartition(2020, 12)
    assert not MonthPartition(2020, 12) < MonthPartition(2020, 10)


def test_split_by_partitions_validation_error():
    with pytest.raises(ValueError):
        list(
            split_by_partitions(
                PartitionBy.YEAR, datetime(2021, 12, 1), datetime(2021, 11, 1)
            )
        )


@pytest.mark.parametrize(
    "partition_by, start_period, end_period, result",
    [
        [
            PartitionBy.YEAR,
            datetime(2020, 1, 1),
            datetime(2021, 12, 1),
            [
                YearPartition(2020),
                YearPartition(2021),
            ],
        ],
        [
            PartitionBy.YEAR,
            datetime(2020, 1, 1),
            datetime(2020, 1, 1),
            [YearPartition(2020)],
        ],
        [
            PartitionBy.MONTH,
            datetime(2020, 1, 1),
            datetime(2020, 3, 1),
            [MonthPartition(2020, 1), MonthPartition(2020, 2), MonthPartition(2020, 3)],
        ],
        [
            PartitionBy.MONTH,
            datetime(2020, 12, 1),
            datetime(2021, 1, 1),
            [MonthPartition(2020, 12), MonthPartition(2021, 1)],
        ],
    ],
)
def test_split_by_partitions(
    partition_by: PartitionBy,
    start_period: datetime,
    end_period: datetime,
    result: list[Partition],
):
    assert list(split_by_partitions(partition_by, start_period, end_period)) == result
