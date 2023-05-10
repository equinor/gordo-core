import pandas as pd

from pathlib import Path
from datatime import datetime

from gordo_core.data_providers.base import GordoBaseDataProvider
from gordo_core.utils import unique_tag_names, capture_args
from gordo_core.sensor_tag import Tag

from typing import Optional, Union, Iterable, Tuple


class CSVDataProvider(GordoBaseDataProvider):
    @capture_args
    def __init__(
        self,
        file_path: Union[str | Path],
        timestamp_column: str,
        sep: str = ",",
        delimiter: Optional[str] = None,
    ):
        self.file_path = file_path
        self.timestamp_column = timestamp_column
        self.sep = sep
        self.delimiter = delimiter

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: list[Tag],
        dry_run: Optional[bool] = False,
        **kwargs,
    ) -> Iterable[Tuple[pd.Series, Tag]]:
        tags = unique_tag_names(tag_list)
        usecols = list(tag_names.keys())
        if self.timestamp_column not in usecols:
            usecols.append(self.timestamp_column)
        df = pd.read_csv(
            self.file_path,
            sep=self.sep,
            delimiter=self.delimiter,
            usecols=usecols,
        )
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], utc=True)
        filtered = df[(df[self.timestamp_column] >= self.train_start_date) & (df[self.timestamp_column] < self.train_end_date)]
        filtered = filtered.set_index(self.timestamp_column)
        for column in filtered:
            yield filtered[column], tags[column]
